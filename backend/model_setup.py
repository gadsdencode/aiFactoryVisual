import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model
from backend.config import AppConfig
import streamlit as st

def setup_model_and_tokenizer(config: AppConfig):
    """
    Sets up the model and tokenizer for training, including quantization and LoRA.

    Args:
        config (AppConfig): The application configuration.

    Returns:
        tuple: A tuple containing the model and tokenizer.
    """
    try:
        # --- Quantization Configuration (optional) ---
        quantization_kwargs = {}
        if bool(getattr(config.quantization, 'load_in_4bit', False)):
            try:
                from transformers import BitsAndBytesConfig  # type: ignore
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=config.quantization.bnb_4bit_quant_type,
                    bnb_4bit_compute_dtype=config.quantization.bnb_4bit_compute_dtype,
                    bnb_4bit_use_double_quant=config.quantization.bnb_4bit_use_double_quant,
                )
                quantization_kwargs['quantization_config'] = bnb_config
            except Exception:
                st.warning("bitsandbytes not available; continuing without 4-bit quantization")

        # --- Load Base Model ---
        st.info(f"Loading base model: {config.base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            device_map=config.training.device_map,
            **quantization_kwargs,
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1
        st.success("Base model loaded successfully.")

        # --- Load Tokenizer ---
        st.info(f"Loading tokenizer for: {config.base_model}")
        tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        st.success("Tokenizer loaded successfully.")

        # --- LoRA Configuration ---
        st.info("Setting up LoRA configuration...")
        peft_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            bias=config.lora.bias,
            task_type=config.lora.task_type,
            target_modules=config.lora.target_modules,
        )
        
        # --- Apply PEFT to the model ---
        model = get_peft_model(model, peft_config)
        st.success("LoRA configuration applied to the model.")
        
        model.print_trainable_parameters()

        return model, tokenizer

    except Exception as e:
        st.error(f"An error occurred during model setup: {e}")
        return None, None

# Example usage:
if __name__ == '__main__':
    from backend.config import load_config
    
    st.title("Model Setup Test")
    
    try:
        config = load_config()
        st.write("Configuration loaded.")
        
        if st.button("Setup Model and Tokenizer"):
            with st.spinner("Setting up model... This may take a while."):
                model, tokenizer = setup_model_and_tokenizer(config)
            
            if model and tokenizer:
                st.success("Model and tokenizer are ready!")
                st.write("Model Architecture:")
                st.text(str(model))
    except Exception as e:
        st.error(f"Failed to load configuration: {e}")
