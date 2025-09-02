import yaml
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
import torch

# Pydantic models for type-safe configuration

class QuantizationConfig(BaseModel):
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = False

    @validator('bnb_4bit_compute_dtype')
    def validate_compute_dtype(cls, v):
        if v not in ["float16", "bfloat16"]:
            raise ValueError("bnb_4bit_compute_dtype must be 'float16' or 'bfloat16'")
        return getattr(torch, v)

class LoRAConfig(BaseModel):
    r: int = 64
    alpha: int = 16
    dropout: float = 0.1
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: Optional[List[str]] = None

class TrainingConfig(BaseModel):
    output_dir: str = "./results"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    optim: str = "paged_adamw_32bit"
    save_steps: int = 25
    logging_steps: int = 25
    evaluation_strategy: str = "no"  # "no", "steps", "epoch"
    eval_steps: int = 100
    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
    max_grad_norm: float = 0.3
    max_steps: int = -1
    warmup_ratio: float = 0.03
    group_by_length: bool = True
    lr_scheduler_type: str = "constant"
    report_to: str = "tensorboard"
    max_seq_length: Optional[int] = None
    packing: bool = False
    device_map: str | Dict[str, Any] = "auto"
    enable_torch_compile: bool = False

class HuggingFaceConfig(BaseModel):
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_token: Optional[str] = None

class AppConfig(BaseModel):
    project_name: str
    base_model: str
    new_model_name: str
    dataset_name: str
    dataset_split: str
    text_column: str
    # Data source configuration
    data_source: str = "hf"  # 'hf' or 'local'
    local_train_path: Optional[str] = None
    local_validation_path: Optional[str] = None
    data_format: Optional[str] = None  # 'json', 'csv', 'parquet'
    quantization: QuantizationConfig
    lora: LoRAConfig
    training: TrainingConfig
    huggingface: HuggingFaceConfig

def load_config(config_path: str = "config.yaml") -> AppConfig:
    """
    Loads the configuration from a YAML file and validates it using Pydantic.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        AppConfig: A validated configuration object.
    """
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return AppConfig(**config_dict)
    except FileNotFoundError:
        raise Exception(f"Configuration file not found at: {config_path}")
    except yaml.YAMLError as e:
        raise Exception(f"Error parsing YAML file: {e}")
    except Exception as e:
        raise Exception(f"Error loading or validating configuration: {e}")

# Example usage:
if __name__ == '__main__':
    # This allows you to run this file directly to test config loading
    try:
        config = load_config()
        print("Configuration loaded successfully!")
        print("\nProject Name:", config.project_name)
        print("Base Model:", config.base_model)
        print("\nTraining arguments:")
        print(config.training.dict())
    except Exception as e:
        print(e)
