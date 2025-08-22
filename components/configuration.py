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
        model_name = st.selectbox(
            "Model Architecture",
            options=model_options,
            index=model_options.index(config['model_name']) if config['model_name'] in model_options else 0
        )
        
        optimizer_options = ["paged_adamw_8bit", "adamw_torch", "adafactor"]
        optimizer = st.selectbox(
            "Optimizer",
            options=optimizer_options,
            index=optimizer_options.index(config['optimizer']) if config['optimizer'] in optimizer_options else 0
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
        data_path = st.text_input("Dataset Path", value="/data/training_data.json")
        validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2, 0.05)
        max_sequence_length = st.number_input("Max Sequence Length", 512, 4096, 2048, 128)
    
    with st.expander("Hardware Configuration"):
        gpu_count = st.selectbox("GPU Count", [1, 2, 4, 8], index=0)
        mixed_precision = st.checkbox("Mixed Precision (FP16)", value=True)
        gradient_checkpointing = st.checkbox("Gradient Checkpointing", value=False)
    
    with st.expander("Logging Configuration"):
        log_frequency = st.number_input("Log Every N Steps", 1, 100, 10, 1)
        save_frequency = st.number_input("Save Every N Epochs", 1, 20, 5, 1)
        wandb_logging = st.checkbox("Enable Weights & Biases", value=False)
    
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
                'learning_rate': 2e-4,
                'batch_size': 4,
                'max_epochs': 100,
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
            "Warmup Steps": config['warmup_steps']
        },
        "Model Configuration": {
            "Model Name": config['model_name'],
            "Optimizer": config['optimizer']
        }
    }
    
    st.json(config_display)
    
    # Configuration validation
    st.subheader("✅ Configuration Validation")
    
    validation_results = []
    
    # Check learning rate
    if 0.0001 <= config['learning_rate'] <= 0.01:
        validation_results.append(("✅", "Learning rate is within recommended range"))
    else:
        validation_results.append(("⚠️", "Learning rate may be too high or too low"))
    
    # Check batch size
    if config['batch_size'] >= 4:
        validation_results.append(("✅", "Batch size is adequate for stable training"))
    else:
        validation_results.append(("⚠️", "Small batch size may lead to unstable training"))
    
    # Check epoch count
    if 50 <= config['max_epochs'] <= 150:
        validation_results.append(("✅", "Epoch count is reasonable"))
    else:
        validation_results.append(("ℹ️", "Consider adjusting epoch count based on dataset size"))
    
    for icon, message in validation_results:
        st.write(f"{icon} {message}")
