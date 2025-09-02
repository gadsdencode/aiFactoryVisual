import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model
try:
    # Prepare model for k-bit training (enables input requires_grad, upcasts layer norms, etc.)
    from peft import prepare_model_for_kbit_training  # type: ignore
except Exception:
    prepare_model_for_kbit_training = None  # type: ignore
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
        # Log basic hardware for transparency
        try:
            import torch
            device_info = "cuda" if torch.cuda.is_available() else "cpu"
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-"
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
            st.info(f"Device: {device_info}, GPU: {gpu_name}, VRAM: {vram:.1f}GB")
        except Exception:
            pass
        # --- Device/DType Strategy ---
        import torch
        use_cuda = torch.cuda.is_available()
        torch_dtype = torch.float16 if use_cuda else torch.float32
        # For training with LoRA + 4-bit on Windows, avoid 'auto' dispatch which can offload to CPU
        if 'quantization_config' in quantization_kwargs:
            requested_map = getattr(config.training, 'device_map', 'auto')
            device_map = ({"": 0} if use_cuda else "cpu") if (requested_map == 'auto') else requested_map
        else:
            device_map = {"": 0} if use_cuda else "cpu"

        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            device_map=device_map,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            **quantization_kwargs,
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1
        st.success("Base model loaded successfully.")

        # Enable performance-friendly CUDA settings on Ampere (RTX 4070)
        try:
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore
            torch.set_float32_matmul_precision("high")  # type: ignore
            # Prefer SDPA mem-efficient attention on Windows (use non-deprecated APIs)
            try:
                # Global backend toggles (preferred; not a context manager)
                torch.backends.cuda.enable_flash_sdp(False)  # type: ignore
                torch.backends.cuda.enable_math_sdp(False)  # type: ignore
                torch.backends.cuda.enable_mem_efficient_sdp(True)  # type: ignore
            except Exception:
                # Fallback to new context manager if available, else ignore
                try:
                    from torch.nn.attention import sdpa_kernel as _sdpa_kernel  # type: ignore
                    # No-op context just to ensure compatibility; main benefit comes from backend toggles above
                    with _sdpa_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):  # type: ignore
                        pass
                except Exception:
                    pass
        except Exception:
            pass

        # --- Load Tokenizer ---
        st.info(f"Loading tokenizer for: {config.base_model}")
        tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        if getattr(config.training, 'max_seq_length', None):
            try:
                tokenizer.model_max_length = int(config.training.max_seq_length)
            except Exception:
                pass
        st.success("Tokenizer loaded successfully.")

        # Optionally prepare model for 4-bit training
        try:
            if prepare_model_for_kbit_training is not None and bool(getattr(config.quantization, 'load_in_4bit', False)):
                model = prepare_model_for_kbit_training(model)
                st.info("Prepared model for k-bit (4-bit) training.")
        except Exception:
            pass

        # --- LoRA Configuration ---
        st.info("Setting up LoRA configuration...")
        # Determine target modules dynamically if not explicitly set
        target_modules = getattr(config.lora, 'target_modules', None)
        try:
            if not target_modules:
                model_type = str(getattr(model.config, 'model_type', '')).lower()
                if model_type in ("llama", "mistral", "qwen", "qwen2", "gemma"):
                    target_modules = [
                        "q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"
                    ]
                else:
                    # Fallback: collect common linear projection names
                    import torch.nn as nn  # type: ignore
                    names = set()
                    for name, module in model.named_modules():
                        if isinstance(module, nn.Linear):
                            leaf = name.rsplit('.', 1)[-1]
                            if (leaf.endswith("proj") or leaf in ("in_proj", "out_proj", "fc1", "fc2")):
                                names.add(leaf)
                    target_modules = sorted(names) if names else ["q_proj", "k_proj", "v_proj", "o_proj"]
        except Exception:
            # Conservative fallback
            if not target_modules:
                target_modules = [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ]

        peft_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            bias=config.lora.bias,
            task_type=config.lora.task_type,
            target_modules=target_modules,
        )
        
        # --- Apply PEFT to the model ---
        model = get_peft_model(model, peft_config)
        st.success("LoRA configuration applied to the model.")
        
        if getattr(config.training, 'gradient_checkpointing', False):
            try:
                model.gradient_checkpointing_enable()
                st.info("Gradient checkpointing enabled on model.")
            except Exception:
                pass

        # --- Optional: torch.compile for performance (PyTorch 2.x) ---
        try:
            if bool(getattr(config.training, 'enable_torch_compile', False)):
                import torch
                if getattr(torch, '__version__', '2.0.0').startswith('2'):
                    try:
                        model = torch.compile(model)  # type: ignore
                        st.info("Model compiled with torch.compile for improved performance.")
                    except Exception as e:
                        st.warning(f"torch.compile failed: {e}")
        except Exception:
            pass

        try:
            model.print_trainable_parameters()
        except Exception:
            pass

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
