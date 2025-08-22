import streamlit as st
import json
from backend.training_manager import get_training_manager

def render_configuration():
    st.title("⚙️ Configuration")
    st.markdown("Manage training parameters and model configurations")
    
    # Training Configuration
    st.subheader("🎯 Training Parameters")
    
    col1, col2 = st.columns(2)
    
    # Get current configuration from training manager
    training_manager = get_training_manager()
    config = training_manager.get_config()
    
    with col1:
        learning_rate = st.number_input(
            "Learning Rate",
            min_value=0.0001,
            max_value=0.1,
            value=config['learning_rate'],
            step=0.0001,
            format="%.4f"
        )
        
        batch_size = st.selectbox(
            "Batch Size",
            options=[1, 2, 4, 8, 16, 32],
            index=[1, 2, 4, 8, 16, 32].index(config['batch_size'])
        )
        
        max_epochs = st.slider(
            "Maximum Epochs",
            min_value=10,
            max_value=200,
            value=config['max_epochs'],
            step=10
        )
    
    with col2:
        model_options = [
            "mistralai/Mistral-7B-Instruct-v0.3",
            "meta-llama/Llama-2-7b-hf", 
            "meta-llama/Llama-2-13b-hf",
            "codellama/CodeLlama-7b-Python-hf"
        ]
        try:
            model_index = model_options.index(config['model_name'])
        except ValueError:
            model_index = 0
        
        model_name = st.selectbox(
            "Model Architecture",
            options=model_options,
            index=model_index
        )
        
        optimizer_options = ["paged_adamw_8bit", "adamw_torch", "adafactor"]
        try:
            optimizer_index = optimizer_options.index(config['optimizer'])
        except ValueError:
            optimizer_index = 0
        
        optimizer = st.selectbox(
            "Optimizer",
            options=optimizer_options,
            index=optimizer_index
        )
        
        warmup_steps = st.number_input(
            "Warmup Steps",
            min_value=0,
            max_value=5000,
            value=config['warmup_steps'],
            step=100
        )
    
    st.markdown("---")
    
    # Advanced Configuration
    st.subheader("🔧 Advanced Settings")
    
    with st.expander("Data Configuration"):
        yaml_config = training_manager.get_yaml_config()
        st.text_input("Training Data Path", value=yaml_config.data.train_file, disabled=True)
        st.text_input("Validation Data Path", value=yaml_config.data.validation_file, disabled=True) 
        st.number_input("Max Sequence Length", value=yaml_config.model.max_length, disabled=True)
    
    with st.expander("Hardware Configuration"):
        st.selectbox("GPU Count", [1, 2, 4, 8], index=0, disabled=True)
        st.checkbox("Quantization Enabled", value=yaml_config.quantization.enabled, disabled=True)
        st.checkbox("Gradient Checkpointing", value=yaml_config.training.gradient_checkpointing, disabled=True)
    
    with st.expander("Logging Configuration"):
        st.number_input("Log Every N Steps", value=yaml_config.training.logging_steps, disabled=True)
        st.number_input("Save Every N Steps", value=yaml_config.training.save_steps, disabled=True)
        st.text_input("Report To", value=yaml_config.training.report_to, disabled=True)
    
    st.markdown("---")
    
    # Configuration actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save Configuration", type="primary", use_container_width=True):
            new_config = {
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'max_epochs': max_epochs,
                'model_name': model_name,
                'optimizer': optimizer,
                'warmup_steps': warmup_steps
            }
            if training_manager.update_config(new_config):
                st.success("Configuration saved successfully!")
            else:
                st.error("Cannot update configuration during active training!")
    
    with col2:
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            default_config = {
                'learning_rate': 0.0002,
                'batch_size': 2,
                'max_epochs': 1,
                'model_name': 'mistralai/Mistral-7B-Instruct-v0.3',
                'optimizer': 'paged_adamw_8bit',
                'warmup_steps': 1000
            }
            if training_manager.update_config(default_config):
                st.info("Configuration reset to defaults!")
                st.rerun()
            else:
                st.error("Cannot reset configuration during active training!")
    
    with col3:
        if st.button("📤 Export Config", use_container_width=True):
            config_json = json.dumps(config, indent=2)
            st.download_button(
                label="Download JSON",
                data=config_json,
                file_name="training_config.json",
                mime="application/json"
            )
    
    # Configuration preview
    st.subheader("📋 Current Configuration")
    
    config_display = {
        "Training Parameters": {
            "Learning Rate": config['learning_rate'],
            "Batch Size": config['batch_size'],
            "Max Epochs": config['max_epochs'],
            "Gradient Accumulation Steps": yaml_config.training.gradient_accumulation_steps,
            "Weight Decay": yaml_config.training.weight_decay,
            "LR Scheduler": yaml_config.training.lr_scheduler_type
        },
        "Model Configuration": {
            "Model Name": config['model_name'],
            "Max Length": yaml_config.model.max_length,
            "Optimizer": config['optimizer'],
            "Attention Implementation": yaml_config.model.attn_implementation
        },
        "LoRA Configuration": {
            "Rank (r)": yaml_config.lora.r,
            "Alpha": yaml_config.lora.alpha,
            "Dropout": yaml_config.lora.dropout
        },
        "Quantization": {
            "Enabled": yaml_config.quantization.enabled,
            "Type": yaml_config.quantization.quant_type,
            "Double Quantization": yaml_config.quantization.use_double_quant
        }
    }
    
    st.json(config_display)
    
    # Configuration validation
    st.subheader("✅ Configuration Validation")
    
    validation_results = []
    yaml_config = training_manager.get_yaml_config()
    
    # Check learning rate
    if 0.00001 <= config['learning_rate'] <= 0.001:
        validation_results.append(("✅", "Learning rate is within recommended range for fine-tuning"))
    else:
        validation_results.append(("⚠️", "Learning rate may be too high or too low for fine-tuning"))
    
    # Check batch size for memory constraints
    if 1 <= config['batch_size'] <= 4:
        validation_results.append(("✅", "Batch size is appropriate for QLoRA training"))
    else:
        validation_results.append(("⚠️", "Batch size may cause memory issues with QLoRA"))
    
    # Check quantization
    if yaml_config.quantization.enabled:
        validation_results.append(("✅", "4-bit quantization enabled for memory efficiency"))
    else:
        validation_results.append(("⚠️", "Quantization disabled - may require more VRAM"))
    
    # Check LoRA configuration
    if yaml_config.lora.r == 32 and yaml_config.lora.alpha == 32:
        validation_results.append(("✅", "LoRA rank and alpha are optimally configured"))
    else:
        validation_results.append(("ℹ️", "Consider standard LoRA r=32, alpha=32 configuration"))
    
    for icon, message in validation_results:
        st.write(f"{icon} {message}")
