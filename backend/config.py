# ==============================================================================
#  File: config.py
#  - Defines the configuration structure using Pydantic for validation
#   and type safety.
# ==============================================================================
import logging
import yaml
from pathlib import Path
from typing import List, Literal, Optional, Dict, Any

from pydantic import BaseModel, Field, field_validator

# Configure logging at the root level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DataConfig(BaseModel):
    """Configuration for data loading and processing."""
    train_file: str
    validation_file: str


class ModelConfig(BaseModel):
    """Configuration for the model and tokenizer."""
    name: str = Field("mistralai/Mistral-7B-Instruct-v0.3", description="The model identifier from Hugging Face Hub.")
    max_length: int = Field(8192, description="Maximum sequence length for the tokenizer.")
    attn_implementation: Optional[Literal["eager", "flash_attention_2", "sdpa"]] = Field(
        "flash_attention_2", description="Attention implementation to use. 'flash_attention_2' is recommended for supported hardware."
    )
    trust_remote_code: bool = True


class QuantizationConfig(BaseModel):
    """Configuration for BitsAndBytes quantization during training."""
    enabled: bool = Field(True, description="Enable 4-bit quantization for training.")
    quant_type: str = Field("nf4", description="Quantization type (e.g., 'nf4', 'fp4').")
    use_double_quant: bool = True


class LoraConfigModel(BaseModel):
    """Configuration for PEFT LoRA."""
    r: int = 32
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]


class TrainingConfig(BaseModel):
    """Configuration for Hugging Face TrainingArguments."""
    output_dir: str
    seed: int = 42
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    optim: str = "paged_adamw_8bit"
    learning_rate: float = 0.0002
    weight_decay: float = 0.001
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    evaluation_strategy: str = "steps"
    eval_steps: int = 50
    save_strategy: str = "steps"
    save_steps: int = 50
    save_total_limit: int = 2
    logging_steps: int = 10
    group_by_length: bool = True
    gradient_checkpointing: bool = True
    report_to: str = "none"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    remove_unused_columns: bool = False
    dataloader_num_workers: int = 4


class DPOConfig(BaseModel):
    """Configuration for DPO training."""
    output_dir: str
    learning_rate: float = 0.000005
    beta: float = 0.1
    max_steps: int = 100
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    optim: str = "paged_adamw_8bit"
    lr_scheduler_type: str = "cosine"
    eval_steps: int = 50
    save_steps: int = 50
    save_total_limit: int = 2
    logging_steps: int = 10
    gradient_checkpointing: bool = True
    warmup_ratio: float = 0.03


class ScriptConfig(BaseModel):
    """Root configuration model for the entire script."""
    data: DataConfig
    model: ModelConfig
    quantization: QuantizationConfig
    lora: LoraConfigModel
    training: TrainingConfig
    dpo: Optional[DPOConfig] = None


def load_config_from_yaml(yaml_path: str = "config.yaml") -> ScriptConfig:
    """Load configuration from YAML file."""
    try:
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert paths to absolute if needed
        if 'data' in config_dict:
            for key in ['train_file', 'validation_file']:
                if key in config_dict['data']:
                    path = Path(config_dict['data'][key])
                    if not path.is_absolute():
                        config_dict['data'][key] = str(Path.cwd() / path)
                    else:
                        config_dict['data'][key] = str(path)
        
        if 'training' in config_dict and 'output_dir' in config_dict['training']:
            output_path = Path(config_dict['training']['output_dir'])
            if not output_path.is_absolute():
                config_dict['training']['output_dir'] = str(Path.cwd() / output_path)
            else:
                config_dict['training']['output_dir'] = str(output_path)
        
        if 'dpo' in config_dict and 'output_dir' in config_dict['dpo']:
            dpo_path = Path(config_dict['dpo']['output_dir'])
            if not dpo_path.is_absolute():
                config_dict['dpo']['output_dir'] = str(Path.cwd() / dpo_path)
            else:
                config_dict['dpo']['output_dir'] = str(dpo_path)
        
        return ScriptConfig(**config_dict)
    
    except FileNotFoundError:
        logger.warning(f"Config file {yaml_path} not found, using defaults")
        return get_default_config()
    except Exception as e:
        logger.error(f"Error loading config from {yaml_path}: {e}")
        return get_default_config()


def get_default_config() -> ScriptConfig:
    """Get default configuration values."""
    return ScriptConfig(
        data=DataConfig(
            train_file="./data/icdu_training_data_v2.jsonl",
            validation_file="./data/icdu_validation_data_v2.jsonl"
        ),
        model=ModelConfig(
            name="mistralai/Mistral-7B-Instruct-v0.3",
            max_length=8192
        ),
        quantization=QuantizationConfig(),
        lora=LoraConfigModel(),
        training=TrainingConfig(
            output_dir="./nomadic-icdu-v2"
        ),
        dpo=DPOConfig(
            output_dir="./nomadic-mind-v2/dpo_model"
        )
    )