# ==============================================================================
#  File: config.py
#  - Defines the configuration structure using Pydantic for validation
#   and type safety.
# ==============================================================================
import logging
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

# Configure logging at the root level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DataConfig(BaseModel):
    """Configuration for data loading and processing."""
    train_file: Path
    validation_file: Path

    @field_validator("train_file", "validation_file")
    @classmethod
    def file_must_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise FileNotFoundError(f"Data file not found: {v}")
        return v


class ModelConfig(BaseModel):
    """Configuration for the model and tokenizer."""
    name: str = Field(..., description="The model identifier from Hugging Face Hub.")
    max_length: int = Field(32000, description="Maximum sequence length for the tokenizer.")
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
    output_dir: Path
    seed: int = 42
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    optim: str = "paged_adamw_8bit"
    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "constant"
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
    dataloader_num_workers: int = 2


class ScriptConfig(BaseModel):
    """Root configuration model for the entire script."""
    data: DataConfig
    model: ModelConfig
    quantization: QuantizationConfig
    lora: LoraConfigModel
    training: TrainingConfig
