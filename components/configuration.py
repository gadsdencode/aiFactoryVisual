import streamlit as st
import json
from backend.training_manager import get_training_manager
from typing import Optional
try:
    from backend.config import AppConfig as _AppConfig  # for type hints only
except Exception:
    _AppConfig = None  # type: ignore
from backend.huggingface_integration import get_hf_manager

def _popover(trigger_label: str, body_md: str) -> None:
    """Render a small inline popover with markdown content.

    The trigger renders as a compact button; clicking shows contextual help.
    """
    try:
        with st.popover(trigger_label):
            st.markdown(body_md)
    except Exception:
        # Fallback if popover is unavailable in the current Streamlit version
        with st.expander(trigger_label):
            st.markdown(body_md)

def _render_configuration_compact():
    """Guided configuration using expandable sections and clean layout.

    Preserves existing config keys; does not remove any features.
    """
    st.header("🔬 Model Training Configuration")
    st.write("Set up parameters for your training job. Hover over the (?) for more information.")

    training_manager = get_training_manager()
    cfg = training_manager.get_config()
    yaml_cfg = training_manager.get_yaml_config()

    with st.expander("🔌 Model & Dataset Configuration", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            # Base model selection
            base_model = st.text_input(
                "Base Model",
                value=cfg.get('model_name', ''),
                help="Select or enter the foundational model to fine-tune (HF repo id)."
            )
        with col2:
            # Dataset name or local path
            data_source = cfg.get('data_source', 'hf')
            ds_source = st.selectbox("Data Source", options=["hf", "local"], index=["hf","local"].index(data_source))
            if ds_source == 'hf':
                dataset_name = st.text_input(
                    "Dataset Name",
                    value=str(getattr(yaml_cfg.data, 'train_file', '') if not str(getattr(yaml_cfg.data, 'train_file', '')).startswith((".", "/", "\\")) else ""),
                    help="Hugging Face dataset id (optional if using local files)."
                )
            else:
                dataset_name = st.text_input(
                    "Training Data Path",
                    value=str(getattr(yaml_cfg.data, 'train_file', '')),
                    help="Local path to training data (e.g., JSONL)."
                )

    with st.expander("⚙️ Training Hyperparameters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            learning_rate = st.number_input(
                "Learning Rate",
                min_value=1e-6, max_value=1e-1,
                value=float(cfg.get('learning_rate', 2e-4)),
                format="%.1e",
                help="Optimizer step size. Typical 1e-5 – 1e-3 for fine-tuning."
            )
            max_epochs = st.number_input(
                "Number of Epochs",
                min_value=1, max_value=200,
                value=int(cfg.get('max_epochs', 1)),
                step=1,
                help="Total training epochs to perform."
            )
        with col2:
            batch_size = st.select_slider(
                "Batch Size",
                options=[1, 2, 4, 8, 16, 32],
                value=int(cfg.get('batch_size', 2)),
                help="Samples per optimizer step."
            )
            optimizer = st.selectbox(
                "Optimizer",
                ["paged_adamw_32bit", "paged_adamw_8bit", "adamw_torch", "adafactor"],
                index=max(0, ["paged_adamw_32bit", "paged_adamw_8bit", "adamw_torch", "adafactor"].index(str(cfg.get('optimizer', 'paged_adamw_8bit'))) if str(cfg.get('optimizer', 'paged_adamw_8bit')) in ["paged_adamw_32bit", "paged_adamw_8bit", "adamw_torch", "adafactor"] else 1),
                help="Optimization algorithm."
            )

    with st.expander("Advanced Options"):
        st.write("Further configuration options can be added here as needed.")
        # surfaced advanced settings preview
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.caption("Quantization")
            st.write(f"4-bit: {getattr(yaml_cfg.quantization, 'enabled', True)}")
        with col_b:
            st.caption("LoRA")
            st.write(f"r={getattr(yaml_cfg.lora, 'r', 32)}, α={getattr(yaml_cfg.lora, 'alpha', 32)}")
        with col_c:
            st.caption("Logging")
            st.write(f"Every {getattr(yaml_cfg.training, 'logging_steps', 25)} steps")

    st.divider()

    col_btn, _ = st.columns([1, 2])
    with col_btn:
        if st.button("🚀 Start Training", use_container_width=True):
            new_cfg = {
                'model_name': base_model,
                'learning_rate': float(learning_rate),
                'batch_size': int(batch_size),
                'max_epochs': int(max_epochs),
                'optimizer': optimizer,
                'data_source': ds_source,
            }
            # Save updates then start training
            hf_token = st.session_state.get('hf_token', None)
            training_manager.update_config(new_cfg, hf_token)
            if training_manager.start_training():
                st.session_state.training_started = True
                st.toast("🚀 Training started!", icon="✅")
                st.rerun()


def render_configuration():
    st.title("⚙️ Configuration")
    st.markdown("Manage training parameters and model configurations")
    st.info("🤗 **HuggingFace Integration Enabled**: You can now use any compatible LLM from the HuggingFace model hub!")

    # Layout selector: keep advanced as default to preserve familiarity
    layout_mode = st.radio("Layout", ["Advanced", "Guided"], horizontal=True, index=0)
    if layout_mode == "Guided":
        return _render_configuration_compact()

    # Get current configuration and helpers
    training_manager = get_training_manager()
    config = training_manager.get_config()
    yaml_config = training_manager.get_yaml_config()

    # Wizard-style tabs
    tab_basic, tab_hparams, tab_exec = st.tabs([
        "1. Model & Dataset",
        "2. Hyperparameters",
        "3. Execution Settings",
    ])

    with tab_basic:
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
                    help="Purpose: Authenticates with HuggingFace to access private/gated models and higher rate limits. Range: A valid personal access token string. Pipeline role: Enables model discovery/validation and gated model downloads during preparation.",
                    placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
                )

            with col_token2:
                if st.button("💾 Save Token", width='stretch', help="Purpose: Persist the entered HF token in session. Range: Click once after entering token. Pipeline role: Saves credentials used for model search/validation and gated downloads."):
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
                if st.button("🗑️ Clear Token", help="Purpose: Remove the stored HF token. Range: Click to clear. Pipeline role: Disables authenticated model operations, reverting to public access."):
                    del st.session_state['hf_token']
                    st.info("Token cleared!")
                    st.rerun()
            else:
                st.info("ℹ️ No authentication token set (using public access only)")

        # Model selection
        hf_manager = get_hf_manager()
        st.subheader("🤗 HuggingFace Model Selection")

        input_method = st.radio(
            "Choose model selection method:",
            ["Popular Models", "Custom Model", "Search Models"],
            index=0,
            help="Purpose: Select how to specify the base LLM. Options: curated list, explicit repo id, or search. Pipeline role: Defines the base checkpoint to fine-tune."
        )
        _popover(
            "ℹ️ Model Input Method",
            "Purpose: Select how to specify the base LLM.\n\n- Options: Popular (curated), Custom (repo id), Search (query HF)\n- Pipeline role: Defines checkpoint, tokenizer, and architecture to fine-tune.\n- Tip: Prefer Instruct-tuned bases for instruction datasets."
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
                help="Purpose: Pick a widely used checkpoint for reliability. Typical: 7B/8x7B instruct models. Pipeline role: Base model defines tokenizer, architecture, and weights for fine-tuning."
            )
            _popover(
                "ℹ️ Popular Models",
                "Purpose: Choose a reliable, widely used base.\n\n- Typical: 7B Instruct variants for single-GPU fine-tuning\n- Pipeline role: Sets tokenizer/model weights.\n- Tip: Check license/usage terms for each base model."
            )
        elif input_method == "Custom Model":
            model_name = st.text_input(
                "Enter HuggingFace Model Name",
                value=config['model_name'],
                help="Purpose: Provide any HF repo id (e.g., 'microsoft/DialoGPT-medium'). Range: Must be a valid model repo. Pipeline role: Custom base checkpoint for fine-tuning.",
                placeholder="org_name/model_name"
            )
            _popover(
                "ℹ️ Custom Model",
                "Purpose: Provide any HF repo id (e.g., org/model).\n\n- Range: Public or gated repos you have access to\n- Pipeline role: Custom base checkpoint for niche tasks\n- Tip: Validate compatibility and size before training."
            )

            if model_name and model_name != config['model_name']:
                with st.spinner("Validating model..."):
                    token = st.session_state.get('hf_token', None)
                    is_valid, message, model_info = hf_manager.validate_model(model_name, token)
                    if is_valid:
                        st.success(f"✅ {message}")
                        if model_info:
                            col_info1, col_info2 = st.columns(2)
                            with col_info1:
                                st.metric("Downloads", f"{model_info.get('downloads', 0):,}")
                                st.metric("Likes", model_info.get('likes', 0))
                            with col_info2:
                                st.info(f"**Type:** {model_info.get('model_type', 'unknown')}")
                                st.info(f"**Size:** {model_info.get('size_estimate', 'unknown')}")
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
        else:
            search_query = st.text_input(
                "Search HuggingFace Models",
                placeholder="Enter search terms (e.g., 'llama', 'mistral', 'code')",
                help="Purpose: Query the model hub by keywords. Range: Any text; refine for better results. Pipeline role: Discover candidate base models before selection."
            )
            _popover(
                "ℹ️ Search Models",
                "Purpose: Discover model candidates by keywords.\n\n- Tip: Filter by 'instruct', 'chat', 'code', etc.\n- Pipeline role: Shortlist base models for selection."
            )

            if search_query:
                with st.spinner("Searching models..."):
                    token = st.session_state.get('hf_token', None)
                    search_results = hf_manager.search_models(search_query, limit=15, token=token)
                    if search_results:
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
                            help="Purpose: Choose a model from search results (sorted by downloads). Pipeline role: Sets the base checkpoint to fine-tune."
                        )
                        _popover(
                            "ℹ️ Search Results",
                            "Purpose: Select a model returned by the search.\n\n- Sorted by downloads (popularity proxy)\n- Pipeline role: Determines base checkpoint for fine-tuning."
                        )
                        model_name = model_details.get(selected_display, config['model_name'])
                    else:
                        st.warning("No models found for your search query.")
                        model_name = config['model_name']
            else:
                model_name = config['model_name']

        with st.expander("📊 Data & Model Configuration"):
            col_data1, col_data2 = st.columns(2)
            with col_data1:
                data_source = st.selectbox("Data Source", ["hf", "local"], index=["hf","local"].index("local" if str(yaml_config.data.train_file).startswith((".", "/", "\\")) else "hf"), help="Purpose: Choose dataset location. 'hf' for hub datasets, 'local' for files on disk. Pipeline role: Controls data ingestion during preprocessing.")
                train_path = st.text_input("Training Data Path", value=yaml_config.data.train_file if data_source == "local" else "", placeholder="path/to/train.jsonl or dataset dir", help="Purpose: Path or HF dataset name for training data when using local. Range: Valid path or directory. Pipeline role: Primary corpus for fine-tuning.")
                val_path = st.text_input("Validation Data Path", value=yaml_config.data.validation_file if data_source == "local" else "", placeholder="optional path/to/val.jsonl", help="Purpose: Optional validation split for periodic evaluation. Range: Valid path or empty. Pipeline role: Monitors generalization during training.")
                _popover(
                    "ℹ️ Data Source & Paths",
                    "Purpose: Configure where data comes from and paths.\n\n- 'hf' uses hub datasets; 'local' uses files/dirs\n- Train path: required for 'local'\n- Val path: optional for evaluation."
                )
            with col_data2:
                max_seq_length_adv = st.number_input("Max Sequence Length", value=yaml_config.model.max_length or 0, min_value=0, step=16, help="Purpose: Maximum tokens per sample fed to the model. Typical range: 512-4096 depending on model. Pipeline role: Affects memory usage and truncation.")
                attn_impl_adv = st.text_input("Attention Implementation", value=yaml_config.model.attn_implementation, help="Purpose: Backend implementation for attention (e.g., flash, sdpa). Range: Depending on model/backend. Pipeline role: Performance/memory trade-offs during forward/backward passes.")
                _popover(
                    "ℹ️ Model Limits",
                    "Purpose: Control sequence length and attention backend.\n\n- Max seq length: 512–4096 typical\n- Attention impl: auto/flash/sdpa depending on hardware."
                )

    with tab_hparams:
        st.subheader("🎯 Training Parameters")
        col1, col2 = st.columns(2)
        with col1:
            learning_rate = st.number_input(
                "Learning Rate",
                min_value=0.0001,
                max_value=0.1,
                value=config['learning_rate'],
                step=0.0001,
                format="%.4f",
                help="Purpose: Controls step size for optimizer updates. Typical range: 1e-5 to 1e-3 for LoRA/QLoRA fine-tuning. Pipeline role: Affects training stability and convergence during optimization."
            )
            _popover(
                "ℹ️ Learning Rate",
                "Purpose: Controls step size for optimizer updates.\n\n- Typical range: 1e-5 – 1e-3 (LoRA/QLoRA)\n- Pipeline role: Affects stability and convergence speed during optimization.\n- Tip: Start lower for small datasets or unstable loss."
            )
            batch_size = st.selectbox(
                "Batch Size",
                options=[1, 2, 4, 8, 16, 32],
                index=[1, 2, 4, 8, 16, 32].index(config['batch_size']),
                help="Purpose: Number of samples per optimizer step. Typical range: 1-4 for QLoRA on 7B models; higher if VRAM permits. Pipeline role: Impacts memory usage and gradient noise; combined with gradient accumulation for effective batch size."
            )
            _popover(
                "ℹ️ Batch Size",
                "Purpose: Samples per optimizer step.\n\n- Typical range: 1 – 4 for 7B QLoRA (raise if VRAM allows)\n- Pipeline role: Impacts VRAM use and gradient noise.\n- Tip: Effective batch = batch_size × grad_accumulation."
            )
            max_epochs = st.slider(
                "Maximum Epochs",
                min_value=1,
                max_value=200,
                value=config['max_epochs'],
                step=10,
                help="Purpose: Upper bound on full passes over the training dataset. Typical range: 1-5 for instruction tuning; up to 10+ for small datasets. Pipeline role: Drives total training compute and risk of overfitting."
            )
            _popover(
                "ℹ️ Max Epochs",
                "Purpose: Upper bound on full dataset passes.\n\n- Typical range: 1 – 5 (instruction tuning), higher for tiny datasets\n- Pipeline role: Drives total training compute and overfitting risk.\n- Tip: Prefer more steps with early stopping over very high epochs."
            )
        with col2:
            optimizer_options = ["paged_adamw_32bit", "paged_adamw_8bit", "adamw_torch", "adafactor"]
            try:
                optimizer_index = optimizer_options.index(config['optimizer'])
            except ValueError:
                optimizer_index = 0
            optimizer = st.selectbox(
                "Optimizer",
                options=optimizer_options,
                index=optimizer_index,
                help="Purpose: Optimization algorithm for training. Typical: paged_adamw_8bit for QLoRA, adamw_torch for full precision; adafactor for large-scale memory efficiency. Pipeline role: Governs parameter updates."
            )
            _popover(
                "ℹ️ Optimizer",
                "Purpose: Algorithm for parameter updates.\n\n- Typical: paged_adamw_8bit (QLoRA), adamw_torch (fp16/bf16), adafactor (memory-efficient)\n- Pipeline role: Convergence behavior and memory footprint."
            )
            warmup_steps = st.number_input(
                "Warmup Steps",
                min_value=0,
                max_value=5000,
                value=config['warmup_steps'],
                step=100,
                help="Purpose: Gradually ramps LR from 0 to target to stabilize early training. Typical range: 0-2000 depending on dataset/steps. Pipeline role: Reduces divergence at start of training."
            )
            _popover(
                "ℹ️ Warmup Steps",
                "Purpose: Ramp LR from 0 to target to stabilize early training.\n\n- Typical range: 0 – 2000 (depends on total steps)\n- Pipeline role: Reduces divergence at start."
            )

        # Quick Glossary (compact)
        with st.expander("📚 Quick Glossary"):
            st.markdown(
                """
| Parameter | Purpose | Typical Range | Pipeline Role |
|---|---|---|---|
| Learning Rate | Step size for updates | 1e-5 – 1e-3 | Convergence/stability |
| Batch Size | Samples per step | 1 – 4 (7B QLoRA) | VRAM, gradient noise |
| Max Epochs | Full passes over data | 1 – 5 | Total compute, overfit risk |
| Optimizer | Update algorithm | AdamW/Adafactor | Convergence, memory |
| Warmup Steps | LR ramp-up | 0 – 2000 | Stabilize early training |
| Max Seq Length | Tokens per sample | 512 – 4096 | Memory, truncation |
| 4-bit Quant | Compress base weights | On for QLoRA | Fit large models on GPU |
| Quant Type | 4-bit format | nf4 / fp4 | Accuracy vs speed |
| Compute DType | Precision | float16 / bfloat16 | Stability/throughput |
| Grad Checkpoint | Save memory | On when VRAM limited | Larger batch/seq |
| LoRA r/alpha | Adapter capacity/scale | r=8–64, α≈r | Quality vs compute |
| LoRA dropout | Regularization | 0.0 – 0.1 | Overfitting control |
| Grad Accum | Virtual batch size | 1 – 16 | Effective batch size |
| Log/Save Steps | Logging/ckpt cadence | 10–100 / 100–1000 | Observability, recovery |
| Eval Steps | Validation cadence | 50 – 500 | Generalization tracking |
                """
            )

    with tab_exec:
        st.subheader("🔧 Execution Settings")
        with st.expander("🖥️ Hardware & Memory Configuration"):
            col_hw1, col_hw2 = st.columns(2)
            with col_hw1:
                st.selectbox("GPU Count", [1, 2, 4, 8], index=0, disabled=True, help="Purpose: Number of GPUs for training. Currently fixed in UI. Pipeline role: Determines data/model parallelism potential.")
                q_enabled_adv = st.checkbox("4-bit Quantization", value=yaml_config.quantization.enabled, help="Purpose: Enable QLoRA 4-bit quantization for base weights to reduce VRAM. Pipeline role: Allows fine-tuning large models on limited hardware.")
                quant_types = ["nf4", "fp4"]
                q_type_current = str(yaml_config.quantization.quant_type or "nf4")
                q_type_index = quant_types.index(q_type_current) if q_type_current in quant_types else 0
                q_type_adv = st.selectbox("Quantization Type", options=quant_types, index=q_type_index, help="Purpose: Choose 4-bit quantization format. Typical: nf4 for accuracy, fp4 for speed. Pipeline role: Impacts quantization error and performance.")
                compute_types = ["float16", "bfloat16"]
                q_compute_current = str(yaml_config.quantization.compute_dtype or "float16")
                if q_compute_current.startswith("torch."):
                    q_compute_current = q_compute_current.split(".", 1)[1]
                q_compute_index = compute_types.index(q_compute_current) if q_compute_current in compute_types else 0
                q_compute_adv = st.selectbox("Compute DType", options=compute_types, index=q_compute_index, help="Purpose: Precision for compute (activations/gradients). Typical: bfloat16 on Ampere+/TPUs; float16 otherwise. Pipeline role: Stability and throughput during training.")
                _popover(
                    "ℹ️ Quantization & Precision",
                    "Purpose: Configure memory/perf trade-offs.\n\n- 4-bit quant reduces VRAM; nf4 for accuracy, fp4 for speed\n- Compute dtype impacts stability (bf16 preferred if supported)."
                )
            with col_hw2:
                grad_ckpt_adv = st.checkbox("Gradient Checkpointing", value=yaml_config.training.gradient_checkpointing, help="Purpose: Save activations selectively to reduce memory at extra compute cost. Pipeline role: Enables larger batch/seq lengths on limited VRAM.")
                q_double_adv = st.checkbox("Double Quantization", value=yaml_config.quantization.use_double_quant, help="Purpose: Apply second quantization to further compress. Pipeline role: Reduces memory at slight accuracy/perf trade-offs.")
                lora_r_adv = st.number_input("LoRA Rank (r)", value=yaml_config.lora.r, min_value=1, help="Purpose: Low-rank adapter size. Typical: 8-64 (commonly 32). Pipeline role: Capacity of LoRA adapters; higher r increases compute/memory.")
                lora_alpha_adv = st.number_input("LoRA Alpha", value=yaml_config.lora.alpha, min_value=1, help="Purpose: Scaling factor for LoRA updates. Typical: equals r (e.g., 32). Pipeline role: Balances update magnitude vs stability.")
                lora_dropout_adv = st.number_input("LoRA Dropout", value=float(yaml_config.lora.dropout), min_value=0.0, max_value=0.9, step=0.05, format="%.2f", help="Purpose: Regularization on LoRA adapters. Typical: 0.0-0.1. Pipeline role: Mitigates overfitting.")
                grad_accum_adv = st.number_input("Gradient Accumulation Steps", value=int(getattr(yaml_config.training, 'gradient_accumulation_steps', 1)), min_value=1, help="Purpose: Accumulate gradients across steps to simulate larger batch size. Typical: 1-16. Pipeline role: Effective batch size = batch_size * accumulation.")
                lora_targets_str_default = ",".join(getattr(yaml_config.lora, 'target_modules', []) or [])
                lora_target_modules_adv = st.text_input("LoRA Target Modules (comma-separated)", value=lora_targets_str_default, help="Purpose: Which layers receive LoRA adapters (e.g., q_proj,k_proj,v_proj,o_proj). Pipeline role: Controls where adaptation capacity is placed.")
                # Tokenizer workers
                num_proc_default = 1
                try:
                    import os
                    num_proc_default = min(os.cpu_count() or 1, 8)
                except Exception:
                    pass
                tokenizer_num_proc = st.number_input("Tokenizer Workers (num_proc)", value=int(num_proc_default), min_value=1, max_value=64, help="Purpose: Number of parallel processes for tokenization. Typical: CPU_count up to 8. Pipeline role: Speeds up data preprocessing.")
                _popover(
                    "ℹ️ LoRA & Training Strategy",
                    "Purpose: Tune adapter capacity and virtual batch size.\n\n- LoRA r/alpha: 8–64, α≈r\n- Dropout: 0.0–0.1\n- Grad accum: 1–16 (effective batch size)."
                )

        with st.expander("📝 Logging Configuration"):
            logging_steps_adv = st.number_input("Log Every N Steps", value=yaml_config.training.logging_steps, min_value=1, help="Purpose: Interval for logging metrics. Typical: 10-100 steps. Pipeline role: Observability during training.")
            save_steps_adv = st.number_input("Save Every N Steps", value=yaml_config.training.save_steps, min_value=1, help="Purpose: Checkpoint frequency. Typical: 100-1000 steps depending on run length. Pipeline role: Controls checkpointing cadence.")
            report_to_adv = st.text_input("Report To", value=yaml_config.training.report_to, help="Purpose: Comma-separated destinations (e.g., tensorboard, wandb). Pipeline role: Integrations for experiment tracking.")
            eval_enabled = st.checkbox("Enable Evaluation", value=(getattr(yaml_config.training, 'evaluation_strategy', 'no') != 'no'), help="Purpose: Toggle periodic evaluation. Pipeline role: Enables eval loop to compute metrics/validate overfitting.")
            eval_steps_ui = st.number_input("Eval Every N Steps", value=int(getattr(yaml_config.training, 'eval_steps', 100)), min_value=10, step=10, help="Purpose: Evaluation interval when enabled. Typical: 50-500 steps. Pipeline role: Frequency of validation runs.")
            _popover(
                "ℹ️ Logging & Evaluation",
                "Purpose: Control observability and validation cadence.\n\n- Log: 10–100 steps; Save: 100–1000 steps\n- Eval: 50–500 steps when enabled\n- Pipeline role: Monitoring, checkpoints, early detection of overfitting."
            )

        st.markdown("---")
        # Actions
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("💾 Save Configuration", type="primary", width='stretch', help="Purpose: Persist all settings above to active training configuration. Pipeline role: Applies parameters for subsequent training runs."):
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
                    'adv_tokenizer_num_proc': int(tokenizer_num_proc),
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
            if st.button("🔄 Reset to Defaults", width='stretch', help="Purpose: Restore recommended default settings. Pipeline role: Quick reset to a safe baseline."):
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
            if st.button("📤 Export Config", width='stretch', help="Purpose: Prepare current config for download as JSON. Pipeline role: Share or archive configuration for reproducibility."):
                config_json = json.dumps(config, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=config_json,
                    file_name="training_config.json",
                    mime="application/json",
                    help="Purpose: Download the configuration file. Pipeline role: Export for reuse or audit."
                )

        # Preview & Validation
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

        st.subheader("✅ Configuration Validation")
        validation_results = []
        yaml_config = training_manager.get_yaml_config()
        if 0.00001 <= config['learning_rate'] <= 0.001:
            validation_results.append(("✅", "Learning rate is within recommended range for fine-tuning"))
        else:
            validation_results.append(("⚠️", "Learning rate may be too high or too low for fine-tuning"))
        if 1 <= config['batch_size'] <= 4:
            validation_results.append(("✅", "Batch size is appropriate for QLoRA training"))
        else:
            validation_results.append(("⚠️", "Batch size may cause memory issues with QLoRA"))
        if yaml_config.quantization.enabled:
            validation_results.append(("✅", "4-bit quantization enabled for memory efficiency"))
        else:
            validation_results.append(("⚠️", "Quantization disabled - may require more VRAM"))
        if yaml_config.lora.r == 32 and yaml_config.lora.alpha == 32:
            validation_results.append(("✅", "LoRA rank and alpha are optimally configured"))
        else:
            validation_results.append(("ℹ️", "Consider standard LoRA r=32, alpha=32 configuration"))
        for icon, message in validation_results:
            st.write(f"{icon} {message}")

        st.markdown("---")
        st.subheader("🚀 Actions")
        # Start training from configuration page for convenience
        if st.button("▶️ Start Training", type="primary"):
            if training_manager.start_training():
                st.toast("🚀 Training has started! Navigate to the Dashboard to monitor progress.", icon="✅")
                st.session_state['training_started'] = True
            else:
                st.warning("Training is already active!")


def configuration_sidebar():
    """
    Backwards-compatible alias used by app.py.
    """
    return render_configuration()