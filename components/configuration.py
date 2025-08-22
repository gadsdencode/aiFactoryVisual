import streamlit as st
import json
from backend.training_manager import get_training_manager
from backend.huggingface_integration import get_hf_manager

def render_configuration():
    st.title("⚙️ Configuration")
    st.markdown("Manage training parameters and model configurations")
    st.info("🤗 **HuggingFace Integration Enabled**: You can now use any compatible LLM from the HuggingFace model hub!")
    
    # HuggingFace Authentication Section
    with st.expander("🔐 HuggingFace Authentication (Optional)"):
        st.markdown("""**Why provide a token?**
        - Access private/gated models that require approval
        - Higher API rate limits for model validation and search
        - Access to your organization's private models""")
        
        col_token1, col_token2 = st.columns([3, 1])
        
        with col_token1:
            hf_token = st.text_input(
                "HuggingFace Access Token",
                type="password",
                help="Get your token from https://huggingface.co/settings/tokens",
                placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
            )
        
        with col_token2:
            if st.button("💾 Save Token", use_container_width=True):
                if hf_token:
                    st.session_state['hf_token'] = hf_token
                    st.success("Token saved!")
                else:
                    if 'hf_token' in st.session_state:
                        del st.session_state['hf_token']
                    st.info("Token cleared!")
        
        # Token status
        if 'hf_token' in st.session_state and st.session_state['hf_token']:
            st.success("✅ Authentication token is active")
            if st.button("🗑️ Clear Token"):
                del st.session_state['hf_token']
                st.info("Token cleared!")
                st.rerun()
        else:
            st.info("ℹ️ No authentication token set (using public access only)")
    
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
        # HuggingFace Model Selection
        hf_manager = get_hf_manager()
        
        st.subheader("🤗 HuggingFace Model Selection")
        
        # Model input method
        input_method = st.radio(
            "Choose model selection method:",
            ["Popular Models", "Custom Model", "Search Models"],
            index=0
        )
        
        if input_method == "Popular Models":
            popular_models = hf_manager.get_popular_llm_models()
            try:
                model_index = popular_models.index(config['model_name'])
            except ValueError:
                model_index = 0
            
            model_name = st.selectbox(
                "Select from Popular LLM Models",
                options=popular_models,
                index=model_index,
                help="Curated list of popular and well-tested LLM models"
            )
        
        elif input_method == "Custom Model":
            model_name = st.text_input(
                "Enter HuggingFace Model Name",
                value=config['model_name'],
                help="Enter the full model path (e.g., 'microsoft/DialoGPT-medium')",
                placeholder="org_name/model_name"
            )
            
            # Validate custom model
            if model_name and model_name != config['model_name']:
                with st.spinner("Validating model..."):
                    token = st.session_state.get('hf_token', None)
                    is_valid, message, model_info = hf_manager.validate_model(model_name, token)
                    
                    if is_valid:
                        st.success(f"✅ {message}")
                        
                        # Display model information
                        if model_info:
                            col_info1, col_info2 = st.columns(2)
                            with col_info1:
                                st.metric("Downloads", f"{model_info.get('downloads', 0):,}")
                                st.metric("Likes", model_info.get('likes', 0))
                            with col_info2:
                                st.info(f"**Type:** {model_info.get('model_type', 'unknown')}")
                                st.info(f"**Size:** {model_info.get('size_estimate', 'unknown')}")
                            
                            # Suitability check
                            suitable, suit_message = hf_manager.is_model_suitable_for_fine_tuning(model_info)
                            if suitable:
                                if "Warning" in suit_message:
                                    st.warning(suit_message)
                                else:
                                    st.success(suit_message)
                            else:
                                st.error(f"❌ {suit_message}")
                    else:
                        st.error(f"❌ {message}")
        
        else:  # Search Models
            search_query = st.text_input(
                "Search HuggingFace Models",
                placeholder="Enter search terms (e.g., 'llama', 'mistral', 'code')"
            )
            
            if search_query:
                with st.spinner("Searching models..."):
                    token = st.session_state.get('hf_token', None)
                    search_results = hf_manager.search_models(search_query, limit=15, token=token)
                    
                    if search_results:
                        # Create options with additional info
                        model_options = []
                        model_details = {}
                        
                        for result in search_results:
                            name = result['name']
                            downloads = result.get('downloads', 0)
                            likes = result.get('likes', 0)
                            display_name = f"{name} ({downloads:,} downloads, {likes} likes)"
                            model_options.append(display_name)
                            model_details[display_name] = name
                        
                        selected_display = st.selectbox(
                            "Select from Search Results",
                            options=model_options,
                            help="Models sorted by download count"
                        )
                        
                        model_name = model_details.get(selected_display, config['model_name'])
                    else:
                        st.warning("No models found for your search query.")
                        model_name = config['model_name']
            else:
                model_name = config['model_name']
        
        optimizer_options = ["paged_adamw_8bit", "adamw_torch", "adafactor"]
        try:
            optimizer_index = optimizer_options.index(config['optimizer'])
        except ValueError:
            optimizer_index = 0
        
        optimizer = st.selectbox(
            "Optimizer",
            options=optimizer_options,
            index=optimizer_index,
            help="paged_adamw_8bit is recommended for QLoRA training"
        )
        
        warmup_steps = st.number_input(
            "Warmup Steps",
            min_value=0,
            max_value=5000,
            value=config['warmup_steps'],
            step=100,
            help="Number of steps for learning rate warmup"
        )
    
    st.markdown("---")
    
    # Advanced Configuration
    st.subheader("🔧 Advanced Settings")
    
    with st.expander("📊 Data & Model Configuration"):
        yaml_config = training_manager.get_yaml_config()
        
        col_data1, col_data2 = st.columns(2)
        with col_data1:
            st.text_input("Training Data Path", value=yaml_config.data.train_file, disabled=True)
            st.text_input("Validation Data Path", value=yaml_config.data.validation_file, disabled=True)
        
        with col_data2: 
            st.number_input("Max Sequence Length", value=yaml_config.model.max_length, disabled=True)
            st.text_input("Attention Implementation", value=yaml_config.model.attn_implementation, disabled=True)
    
    with st.expander("🖥️ Hardware & Memory Configuration"):
        col_hw1, col_hw2 = st.columns(2)
        
        with col_hw1:
            st.selectbox("GPU Count", [1, 2, 4, 8], index=0, disabled=True)
            st.checkbox("4-bit Quantization", value=yaml_config.quantization.enabled, disabled=True, help="Enables QLoRA for memory efficiency")
            st.text_input("Quantization Type", value=yaml_config.quantization.quant_type, disabled=True)
        
        with col_hw2:
            st.checkbox("Gradient Checkpointing", value=yaml_config.training.gradient_checkpointing, disabled=True, help="Trades compute for memory")
            st.checkbox("Double Quantization", value=yaml_config.quantization.use_double_quant, disabled=True)
            st.number_input("LoRA Rank (r)", value=yaml_config.lora.r, disabled=True)
    
    with st.expander("📝 Logging Configuration"):
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
            hf_token = st.session_state.get('hf_token', None)
            if training_manager.update_config(new_config, hf_token):
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
            hf_token = st.session_state.get('hf_token', None)
            if training_manager.update_config(default_config, hf_token):
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
