# ==============================================================================
#  File: train.py
#  - The main training, merging, and saving orchestration logic.
# ==============================================================================
import logging
import inspect
from dataclasses import fields as dataclass_fields

import torch
from peft import LoraConfig as PeftLoraConfig, PeftModel
from transformers import (
    TrainingArguments,
    EarlyStoppingCallback,
    PreTrainedTokenizer,
    PreTrainedModel,
    AutoModelForCausalLM,
)
from trl import SFTTrainer

from config import ScriptConfig
from utils import Environment
from data import load_and_prepare_dataset, VectorizedCompletionOnlyCollator
from model_setup import load_tokenizer, load_model

logger = logging.getLogger(__name__)


def run_training(config: ScriptConfig, env: Environment, tokenizer: PreTrainedTokenizer, model: PreTrainedModel):
    """Conducts the Supervised Fine-Tuning (SFT) process."""
    dataset = load_and_prepare_dataset(config.data, tokenizer)
    
    lora_config = PeftLoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.lora.target_modules,
    )

    effective_optim = config.training.optim
    if not (env.cuda_available and env.bnb_available):
        logger.warning(f"Paged optimizer '{effective_optim}' requires CUDA and bitsandbytes. Falling back to 'adamw_torch'.")
        effective_optim = "adamw_torch"

    allowed_fields = {f.name for f in dataclass_fields(TrainingArguments)}
    args_dict = {k: v for k, v in config.training.dict().items() if k in allowed_fields}
    if "output_dir" in allowed_fields:
        args_dict["output_dir"] = str(config.training.output_dir)
    if "logging_dir" in allowed_fields:
        args_dict["logging_dir"] = str(config.training.output_dir / "logs")
    if "bf16" in allowed_fields:
        args_dict["bf16"] = env.bf16_supported
    if "fp16" in allowed_fields:
        args_dict["fp16"] = (not env.bf16_supported and env.cuda_available)
    if "optim" in allowed_fields:
        args_dict["optim"] = effective_optim

    wants_best = args_dict.get("load_best_model_at_end", False)
    if wants_best:
        has_eval_strategy_field = ("evaluation_strategy" in allowed_fields) or ("eval_strategy" in allowed_fields)
        if not has_eval_strategy_field:
            logger.warning("Disabling load_best_model_at_end: this transformers version does not expose evaluation_strategy.")
            args_dict["load_best_model_at_end"] = False
        else:
            save_strategy_value = args_dict.get("save_strategy", None)
            if "evaluation_strategy" in allowed_fields and "evaluation_strategy" not in args_dict and save_strategy_value is not None:
                args_dict["evaluation_strategy"] = save_strategy_value
            if "eval_strategy" in allowed_fields and "eval_strategy" not in args_dict and save_strategy_value is not None:
                args_dict["eval_strategy"] = save_strategy_value

    training_args = TrainingArguments(**args_dict)
    
    if training_args.gradient_checkpointing:
        logger.info("Gradient checkpointing enabled. Disabling model cache.")
        model.config.use_cache = False

    data_collator = VectorizedCompletionOnlyCollator(
        tokenizer=tokenizer,
        response_template="Assistant: ",
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["validation"],
        "peft_config": lora_config,
        "data_collator": data_collator,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=3)],
    }

    sft_sig = inspect.signature(SFTTrainer.__init__)
    sft_params = set(sft_sig.parameters.keys())

    if "tokenizer" in sft_params:
        trainer_kwargs["tokenizer"] = tokenizer
    if "dataset_text_field" in sft_params:
        trainer_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in sft_params:
        trainer_kwargs["max_seq_length"] = config.model.max_length

    if "tokenizer" not in sft_params:
        logger.warning("Installed TRL SFTTrainer does not accept 'tokenizer'. Pre-tokenizing dataset.")
        def tok_map(example):
            return tokenizer(example["text"], truncation=True, max_length=config.model.max_length, padding=False)
        tokenized_train = dataset["train"].map(tok_map, remove_columns=list(dataset["train"].features))
        tokenized_eval = dataset["validation"].map(tok_map, remove_columns=list(dataset["validation"].features))
        trainer_kwargs.update({"train_dataset": tokenized_train, "eval_dataset": tokenized_eval})
        trainer_kwargs.pop("dataset_text_field", None)

    trainer = SFTTrainer(**trainer_kwargs)
    logger.info("Starting model training...")
    trainer.train()

    adapter_save_path = config.training.output_dir / "final_adapter"
    logger.info(f"Training complete. Saving final adapter to {adapter_save_path}")
    trainer.save_model(str(adapter_save_path))


def merge_and_save_model(config: ScriptConfig, env: Environment):
    """Loads the base model, merges the adapter, and saves the standalone model."""
    logger.info("Starting model merge process...")
    adapter_path = config.training.output_dir / "final_adapter"
    logger.info(f"Reloading base model '{config.model.name}' in high precision for merging...")
    
    high_precision_model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        trust_remote_code=config.model.trust_remote_code,
        torch_dtype=env.compute_dtype,
        device_map="cpu",
        use_safetensors=True,
    )

    logger.info(f"Loading adapter from {adapter_path} and applying to the base model...")
    peft_model = PeftModel.from_pretrained(high_precision_model, adapter_path)

    logger.info("Merging adapter weights into the base model...")
    merged_model = peft_model.merge_and_unload()

    merged_save_path = config.training.output_dir / "final_merged_model"
    merged_save_path.mkdir(exist_ok=True)
    
    logger.info(f"Saving merged model to {merged_save_path}...")
    merged_model.save_pretrained(str(merged_save_path), safe_serialization=True)
    
    tokenizer = load_tokenizer(config.model)
    tokenizer.save_pretrained(str(merged_save_path))
    
    logger.info("Merged model and tokenizer saved successfully.")


def run_pipeline(config: ScriptConfig):
    """Orchestrates the fine-tuning and merging process."""
    logger.info("Starting fine-tuning script...")
    logger.info(config.model_dump_json(indent=2))

    env = Environment()
    env.setup_backends()
    torch.manual_seed(config.training.seed)

    logger.info("--- Entering Training Phase ---")
    tokenizer = load_tokenizer(config.model)
    quantized_model = load_model(config.model, config.quantization, env)
    run_training(config, env, tokenizer, quantized_model)
    
    del quantized_model
    torch.cuda.empty_cache()

    logger.info("--- Entering Merging Phase ---")
    merge_and_save_model(config, env)
    
    logger.info("Script finished successfully.")
