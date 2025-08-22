# main.py
# CLI entry point for QLoRA fine-tuning, merging, DPO training, and inference with tools on the 'Breaking Better' dataset for Mistral-7B.
# Updated to include DPO training phase and robust inference with tools from inference_with_tools.py.
# Usage: python -m training.main --config-path training/config.yaml --run_inference --example_queries "Advise on fitness habits using latest research."

import logging
from pathlib import Path
import yaml
import typer
from typing_extensions import Annotated
from typing import List, Optional
import hashlib
import concurrent.futures
import json
import re
import requests
import ast
import torch
from transformers import pipeline

# Import pipeline components (assume in same package)
from config import ScriptConfig
from train import run_training, merge_and_save_model
from utils import Environment
from model_setup import load_tokenizer, load_model
from data import load_and_prepare_dataset, VectorizedCompletionOnlyCollator
from inference_with_tools import agent_loop
from dpo import load_jsonl, generate_preference_pairs, prepare_dpo_dataset, run_dpo_training

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

def load_config_from_yaml(path: Path) -> ScriptConfig:
    """Loads and validates configuration from YAML."""
    logger.info(f"Loading configuration from {path}")
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at {path}")
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    base_dir = path.parent.resolve()
    data_section = config_dict.get("data", {})
    for key in ("train_file", "validation_file"):
        if key in data_section:
            candidate_path = Path(data_section[key])
            if not candidate_path.is_absolute():
                data_section[key] = str((base_dir / candidate_path).resolve())
    config_dict["data"] = data_section
    
    training_section = config_dict.get("training", {})
    if "output_dir" in training_section:
        output_dir_path = Path(training_section["output_dir"])
        if not output_dir_path.is_absolute():
            training_section["output_dir"] = str((base_dir / output_dir_path).resolve())
    config_dict["training"] = training_section
    
    return ScriptConfig(**config_dict)

def run_inference_phase(config: ScriptConfig, example_queries: List[str]) -> None:
    """Runs inference with tools using DPO-trained model if available."""
    dpo_path = config.training.output_dir / "dpo_model"
    merged_path = config.training.output_dir / "final_merged_model"
    model_path = dpo_path if dpo_path.exists() else merged_path
    
    if not model_path.exists():
        raise FileNotFoundError(f"No model found at {model_path}")
    
    logger.info(f"Running inference phase with model: {model_path}")
    try:
        model_pipe = pipeline(
            "text-generation",
            model=str(model_path),
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        raise
    
    for query in example_queries:
        logger.info(f"Processing query: {query}")
        try:
            response = agent_loop(query, model_pipe)
            logger.info(f"Response: {response[:200]}...")
            print(f"Query: {query}\nResponse: {response}\n{'-'*50}")
        except Exception as e:
            logger.error(f"Inference error for query '{query}': {e}")
            print(f"Error for query '{query}': {str(e)}")

def run_pipeline(config: ScriptConfig, run_inference: bool = False, example_queries: List[str] = []) -> None:
    """Orchestrates training, merging, DPO training, and optional inference."""
    logger.info("Starting pipeline")
    env = Environment()
    env.setup_backends()
    torch.manual_seed(config.training.seed)
    
    logger.info("--- Training Phase ---")
    try:
        tokenizer = load_tokenizer(config.model)
        quantized_model = load_model(config.model, config.quantization, env)
        run_training(config, env, tokenizer, quantized_model)
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise
    
    logger.info("Clearing memory post-training")
    del quantized_model
    torch.cuda.empty_cache()
    
    logger.info("--- Merging Phase ---")
    try:
        merge_and_save_model(config, env)
    except Exception as e:
        logger.error(f"Merging error: {e}")
        raise
    
    logger.info("--- DPO Phase ---")
    try:
        augmented_data = load_jsonl(config.data.train_file)
        pref_data = generate_preference_pairs(augmented_data)
        pref_dataset = prepare_dpo_dataset(pref_data)
        dpo_base_path = config.training.output_dir / "final_merged_model"
        dpo_model_path = str(dpo_base_path) if dpo_base_path.exists() else str(config.model.name)
        run_dpo_training(
            model_path=dpo_model_path,
            dataset=pref_dataset,
            output_dir=str(config.training.output_dir / "dpo_model"),
            lora_rank=32,
            max_steps=100,
            learning_rate=5e-6
        )
    except Exception as e:
        logger.error(f"DPO phase error: {e}")
        raise
    
    if run_inference:
        logger.info("--- Inference Phase ---")
        try:
            run_inference_phase(config, example_queries)
        except Exception as e:
            logger.error(f"Inference phase error: {e}")
            raise
    
    logger.info("Pipeline completed successfully")

def main(
    config_path: Annotated[Path, typer.Option(..., help="Path to configuration YAML file.")],
    run_inference: Annotated[bool, typer.Option(help="Run inference after DPO training")] = False,
    example_queries: Annotated[List[str], typer.Option(
        help="Example queries for inference. Defaults: fitness advice, calculation, task tracking."
    )] = [
        "Advise on fitness habits using latest research.",
        "Calculate 5 * (3 + 7) / 2",
        "Add a task to read a book this week."
    ]
) -> None:
    """
    Run the QLoRA fine-tuning, merging, DPO training, and inference pipeline.
    """
    try:
        config = load_config_from_yaml(config_path)
        run_pipeline(config, run_inference, example_queries)
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        raise

if __name__ == "__main__":
    typer.run(main)