from typing import Optional, Dict, Any, Callable, List
import logging
import torch
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from .config import ScriptConfig
from .model_setup import load_tokenizer, load_model
from .data import load_and_prepare_dataset, VectorizedCompletionOnlyCollator
from .utils import Environment

logger = logging.getLogger(__name__)

class ProgressCallback(TrainerCallback):
    def __init__(self, on_log: Callable[[Dict[str, Any]], None]):
        self.on_log_cb = on_log
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs:
            self.on_log_cb(dict(logs))

def build_training_args(cfg: ScriptConfig) -> TrainingArguments:
    t = cfg.training
    return TrainingArguments(
        output_dir=t.output_dir,
        seed=t.seed,
        num_train_epochs=t.num_train_epochs,
        per_device_train_batch_size=t.per_device_train_batch_size,
        per_device_eval_batch_size=t.per_device_eval_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        learning_rate=t.learning_rate,
        weight_decay=t.weight_decay,
        max_grad_norm=t.max_grad_norm,
        warmup_ratio=t.warmup_ratio,
        lr_scheduler_type=t.lr_scheduler_type,
        evaluation_strategy=t.evaluation_strategy,
        eval_steps=t.eval_steps,
        save_strategy=t.save_strategy,
        save_steps=t.save_steps,
        save_total_limit=t.save_total_limit,
        logging_steps=t.logging_steps,
        group_by_length=t.group_by_length,
        gradient_checkpointing=t.gradient_checkpointing,
        report_to=t.report_to,
        load_best_model_at_end=t.load_best_model_at_end,
        metric_for_best_model=t.metric_for_best_model,
        greater_is_better=t.greater_is_better,
        remove_unused_columns=t.remove_unused_columns,
        dataloader_num_workers=t.dataloader_num_workers,
        fp16=True,  # Windows-friendly mixed precision
    )

def run_training(cfg: ScriptConfig, env: Environment, on_progress: Callable[[Dict[str, Any]], None]) -> None:
    tokenizer = load_tokenizer(cfg.model)
    model = load_model(cfg.model, cfg.quantization, env)

    # If 4-bit, prep for k-bit training
    try:
        model = prepare_model_for_kbit_training(model)
    except Exception as e:
        logger.warning(f"prepare_model_for_kbit_training failed or bnb not enabled: {e}")

    # Apply LoRA
    lora_cfg = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.lora.target_modules,
    )
    model = get_peft_model(model, lora_cfg)

    ds = load_and_prepare_dataset(cfg.data, tokenizer)
    data_collator = VectorizedCompletionOnlyCollator(tokenizer)

    args = build_training_args(cfg)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[ProgressCallback(on_progress)],
    )

    logger.info("Starting HF Trainer.fit() ...")
    trainer.train()
    logger.info("Training complete; saving adapter")
    trainer.save_model(cfg.training.output_dir)
