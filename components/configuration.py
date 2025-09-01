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
            if st.button("💾 Save Token", width='stretch'):
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
        
        # The app expects epochs up to 200 in UI; config may have small values
        max_epochs = st.slider(
            "Maximum Epochs",
            min_value=1,
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
        
        optimizer_options = ["paged_adamw_32bit", "paged_adamw_8bit", "adamw_torch", "adafactor"]
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
            data_source = st.selectbox("Data Source", ["hf", "local"], index=["hf","local"].index("local" if str(yaml_config.data.train_file).startswith((".", "/", "\\")) else "hf"))
            train_path = st.text_input("Training Data Path", value=yaml_config.data.train_file if data_source == "local" else "", placeholder="path/to/train.jsonl or dataset dir")
            val_path = st.text_input("Validation Data Path", value=yaml_config.data.validation_file if data_source == "local" else "", placeholder="optional path/to/val.jsonl")
        
        with col_data2: 
            max_seq_length_adv = st.number_input("Max Sequence Length", value=yaml_config.model.max_length or 0, min_value=0, step=16)
            attn_impl_adv = st.text_input("Attention Implementation", value=yaml_config.model.attn_implementation)
    
    with st.expander("🖥️ Hardware & Memory Configuration"):
        col_hw1, col_hw2 = st.columns(2)
        
        with col_hw1:
            st.selectbox("GPU Count", [1, 2, 4, 8], index=0, disabled=True)
            q_enabled_adv = st.checkbox("4-bit Quantization", value=yaml_config.quantization.enabled, help="Enables QLoRA for memory efficiency")
            quant_types = ["nf4", "fp4"]
            q_type_current = str(yaml_config.quantization.quant_type or "nf4")
            q_type_index = quant_types.index(q_type_current) if q_type_current in quant_types else 0
            q_type_adv = st.selectbox("Quantization Type", options=quant_types, index=q_type_index)

            compute_types = ["float16", "bfloat16"]
            q_compute_current = str(yaml_config.quantization.compute_dtype or "float16")
            if q_compute_current.startswith("torch."):
                q_compute_current = q_compute_current.split(".", 1)[1]
            q_compute_index = compute_types.index(q_compute_current) if q_compute_current in compute_types else 0
            q_compute_adv = st.selectbox("Compute DType", options=compute_types, index=q_compute_index)
        
        with col_hw2:
            grad_ckpt_adv = st.checkbox("Gradient Checkpointing", value=yaml_config.training.gradient_checkpointing, help="Trades compute for memory")
            q_double_adv = st.checkbox("Double Quantization", value=yaml_config.quantization.use_double_quant)
            lora_r_adv = st.number_input("LoRA Rank (r)", value=yaml_config.lora.r, min_value=1)
            lora_alpha_adv = st.number_input("LoRA Alpha", value=yaml_config.lora.alpha, min_value=1)
            lora_dropout_adv = st.number_input("LoRA Dropout", value=float(yaml_config.lora.dropout), min_value=0.0, max_value=0.9, step=0.05, format="%.2f")
            grad_accum_adv = st.number_input("Gradient Accumulation Steps", value=int(getattr(yaml_config.training, 'gradient_accumulation_steps', 1)), min_value=1)
            lora_targets_str_default = ",".join(getattr(yaml_config.lora, 'target_modules', []) or [])
            lora_target_modules_adv = st.text_input("LoRA Target Modules (comma-separated)", value=lora_targets_str_default, help="Leave empty to keep current; comma-separated module names like q_proj,k_proj,v_proj,o_proj")
    
    with st.expander("📝 Logging Configuration"):
        logging_steps_adv = st.number_input("Log Every N Steps", value=yaml_config.training.logging_steps, min_value=1)
        save_steps_adv = st.number_input("Save Every N Steps", value=yaml_config.training.save_steps, min_value=1)
        report_to_adv = st.text_input("Report To", value=yaml_config.training.report_to)
        eval_enabled = st.checkbox("Enable Evaluation", value=(getattr(yaml_config.training, 'evaluation_strategy', 'no') != 'no'))
        eval_steps_ui = st.number_input("Eval Every N Steps", value=int(getattr(yaml_config.training, 'eval_steps', 100)), min_value=10, step=10)
    
    st.markdown("---")
    
    # Configuration actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save Configuration", type="primary", width='stretch'):
            new_config = {
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'max_epochs': max_epochs,
                'model_name': model_name,
                'optimizer': optimizer,
                'warmup_steps': warmup_steps,
                # Advanced settings
                'adv_max_seq_length': int(max_seq_length_adv) if max_seq_length_adv else None,
                'adv_attn_impl': attn_impl_adv,
                'adv_quant_enabled': bool(q_enabled_adv),
                'adv_quant_type': q_type_adv,
                'adv_quant_double': bool(q_double_adv),
                'adv_quant_compute_dtype': q_compute_adv,
                'adv_lora_r': int(lora_r_adv),
                'adv_lora_alpha': int(lora_alpha_adv),
                'adv_lora_dropout': float(lora_dropout_adv),
                'adv_lora_target_modules': lora_target_modules_adv,
                'adv_logging_steps': int(logging_steps_adv),
                'adv_save_steps': int(save_steps_adv),
                'adv_report_to': report_to_adv,
                'adv_gradient_checkpointing': bool(grad_ckpt_adv),
                'adv_gradient_accumulation_steps': int(grad_accum_adv),
                # Data source settings
                'data_source': data_source,
                'local_train_path': train_path if data_source == 'local' else None,
                'local_validation_path': val_path if data_source == 'local' else None,
                # Eval settings
                'adv_evaluation_strategy': ('steps' if eval_enabled else 'no'),
                'adv_eval_steps': int(eval_steps_ui),
            }
            hf_token = st.session_state.get('hf_token', None)
            if training_manager.update_config(new_config, hf_token):
                st.success("Configuration saved successfully!")
            else:
                st.error("Cannot update configuration during active training!")
    
    with col2:
        if st.button("🔄 Reset to Defaults", width='stretch'):
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
        if st.button("📤 Export Config", width='stretch'):
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


def configuration_sidebar():
    """
    Backwards-compatible alias used by app.py.
    """
    return render_configuration()