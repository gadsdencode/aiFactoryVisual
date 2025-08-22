# dpo.py
# Script to generate preference dataset from augmented 'Breaking Better' JSONL and run DPO training to optimize tool selection for Mistral-7B.
# Generates chosen/rejected pairs (correct vs. incorrect/no tool calls) and fine-tunes with TRL's DPOTrainer.
# Usage: python dpo.py --input augmented_data.jsonl --output dpo_preferences.jsonl --model_path mistralai/Mistral-7B-Instruct-v0.3 --training_output_dir ./dpo_output

import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import Dataset
from trl import DPOTrainer
from peft import LoraConfig, get_peft_model
import torch
import random

# Set up verbose logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Known tools from augment_dataset.py and tools.py
KNOWN_TOOLS = [
    "search_web", "calc_tool", "news_tool", "python_repl", "read_file", "write_file",
    "calendar_tool", "task_tracker_tool", "job_search_tool", "get_current_weather", "animal_medical_database"
]

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Loads a JSONL file into a list of dictionaries.
    
    Args:
        file_path: Path to input JSONL.
    
    Returns:
        List of parsed JSON objects.
    """
    if not Path(file_path).exists():
        raise ValueError(f"Input file not found: {file_path}")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                example = json.loads(line.strip())
                if "messages" not in example or not isinstance(example["messages"], list):
                    logger.warning(f"Skipping invalid example at line {line_num}: Missing 'messages'")
                    continue
                data.append(example)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error at line {line_num}: {e}")
                continue
    logger.info(f"Loaded {len(data)} valid examples from {file_path}")
    return data

def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """
    Saves a list of dictionaries to a JSONL file.
    
    Args:
        data: List of JSON-serializable dictionaries.
        file_path: Path to output JSONL file.
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for example in data:
            f.write(json.dumps(example) + '\n')
    logger.info(f"Saved {len(data)} examples to {file_path}")

def generate_preference_pairs(original_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generates DPO preference pairs from augmented dataset.
    Chosen: Correct tool call (from original). Rejected: Incorrect or no tool.
    
    Args:
        original_data: Loaded augmented dataset.
    
    Returns:
        List of preference pairs: {"prompt": str, "chosen": str, "rejected": str}.
    """
    preference_data = []
    
    for example in original_data:
        messages = example.get("messages", [])
        if not messages:
            continue
        
        # Extract user query and assistant response
        user_msg = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
        assistant_response = next((msg["content"] for msg in messages if msg["role"] == "assistant"), "")
        if not user_msg or not assistant_response:
            continue
        
        prompt = f"[INST]{user_msg}[/INST]"
        chosen = assistant_response
        
        # Generate rejected response: Either no tool (if tool used) or wrong tool
        if "tool_call" in assistant_response:
            # No-tool rejection
            rejected = assistant_response.split("Need data:")[0].strip() or "Direct advice without tool."
        else:
            # Wrong tool rejection
            wrong_tool = random.choice([t for t in KNOWN_TOOLS if t not in assistant_response])
            rejected = f"Need data: {json.dumps({'tool_call': {'name': wrong_tool, 'arguments': {'query': user_msg.split()[0]}}})}\nTool result: Mock result.\nIntegrated advice: {assistant_response}"
        
        preference_data.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })
    
    logger.info(f"Generated {len(preference_data)} preference pairs")
    return preference_data

def prepare_dpo_dataset(preference_data: List[Dict[str, Any]]) -> Dataset:
    """
    Converts preference pairs to Hugging Face Dataset.
    
    Args:
        preference_data: List of {"prompt", "chosen", "rejected"} dicts.
    
    Returns:
        Hugging Face Dataset.
    """
    return Dataset.from_list(preference_data)

def run_dpo_training(
    model_path: str,
    dataset: Dataset,
    output_dir: str,
    lora_rank: int = 32,
    max_steps: int = 100,
    learning_rate: float = 5e-6
):
    """
    Runs DPO training to optimize tool selection.
    
    Args:
        model_path: Path to base or fine-tuned Mistral model.
        dataset: DPO preference dataset.
        output_dir: Directory to save DPO model.
        lora_rank: LoRA rank for QLoRA.
        max_steps: Training steps.
        learning_rate: Learning rate for DPO.
    """
    logger.info(f"Starting DPO training with model: {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Add padding token if it's missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        
        # Apply QLoRA
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "lm_head"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        
        # DPO training arguments (replaces DPOConfig)
        training_args = TrainingArguments(
            output_dir=output_dir,
            max_steps=max_steps,
            learning_rate=learning_rate,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            remove_unused_columns=False,
            optim="adamw_torch",
            logging_steps=10,
            seed=42
        )
        
        # Initialize DPO trainer
        trainer = DPOTrainer(
            model=model,
            ref_model=None,  # Use model as reference (implicit)
            args=training_args,
            beta=0.1, # DPO loss hyperparam
            train_dataset=dataset,
            # The tokenizer argument is removed to fix the error
        )
        
        # Train
        trainer.train()
        trainer.save_model(output_dir)
        logger.info(f"DPO model saved to {output_dir}")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        logger.error(f"DPO training error: {e}")
        raise

def main():
    """CLI entry point for DPO preference dataset generation and training."""
    parser = argparse.ArgumentParser(description="Generate DPO dataset and train for tool selection.")
    parser.add_argument("--input", required=True, help="Path to augmented JSONL dataset")
    parser.add_argument("--output", required=True, help="Path to save DPO preference JSONL")
    parser.add_argument("--model_path", required=True, help="Path to base or fine-tuned Mistral model")
    parser.add_argument("--training_output_dir", required=True, help="Directory for DPO model output")
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--max_steps", type=int, default=100, help="Max training steps")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    
    args = parser.parse_args()
    
    try:
        # Load and generate preference dataset
        original_data = load_jsonl(args.input)
        preference_data = generate_preference_pairs(original_data)
        save_jsonl(preference_data, args.output)
        
        # Prepare dataset and run DPO
        dataset = prepare_dpo_dataset(preference_data)
        run_dpo_training(
            args.model_path,
            dataset,
            args.training_output_dir,
            args.lora_rank,
            args.max_steps,
            args.learning_rate
        )
        
    except Exception as e:
        logger.error(f"Main execution error: {e}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
