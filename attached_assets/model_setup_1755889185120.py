# ==============================================================================
#  File: model_setup.py
#  - Functions for loading the tokenizer and model with appropriate configs.
# ==============================================================================
import logging
import importlib.util
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from config import ModelConfig, QuantizationConfig
from utils import Environment

logger = logging.getLogger(__name__)


def load_tokenizer(config: ModelConfig) -> PreTrainedTokenizer:
    """Loads and configures the tokenizer."""
    logger.info(f"Loading tokenizer: {config.name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.name,
        trust_remote_code=config.trust_remote_code,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = config.max_length
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(
    model_config: ModelConfig,
    quant_config: QuantizationConfig,
    env: Environment
) -> PreTrainedModel:
    """Loads the model with quantization and other configurations."""
    logger.info(f"Loading base model: {model_config.name}")
    
    bnb_config = None
    if quant_config.enabled:
        if not (env.cuda_available and env.bnb_available):
            logger.warning(
                "Quantization is enabled but CUDA or bitsandbytes is not available. "
                "Disabling quantization."
            )
        else:
            logger.info("4-bit quantization enabled for training. Configuring BitsAndBytes.")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quant_config.quant_type,
                bnb_4bit_compute_dtype=env.compute_dtype,
                bnb_4bit_use_double_quant=quant_config.use_double_quant,
            )

    effective_attn_impl = model_config.attn_implementation
    if effective_attn_impl in ("flash_attention_2", "flash_attention_3"):
        has_flash_attn = importlib.util.find_spec("flash_attn") is not None
        if not has_flash_attn:
            logger.warning(
                f"Requested attention '{effective_attn_impl}' but 'flash_attn' is not installed. Falling back to 'sdpa'."
            )
            effective_attn_impl = "sdpa"

    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": model_config.trust_remote_code,
        "use_safetensors": True,
        "attn_implementation": effective_attn_impl,
    }
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
    # Use environment-selected compute dtype (on Windows we prefer float16)
    model_kwargs["torch_dtype"] = env.compute_dtype
    # Avoid creating meta tensors on some setups by disabling low_cpu_mem_usage
    model_kwargs["low_cpu_mem_usage"] = False
        
    model = AutoModelForCausalLM.from_pretrained(model_config.name, **model_kwargs)
    return model
