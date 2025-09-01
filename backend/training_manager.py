import streamlit as st
from threading import Thread, Event, Lock
import queue
import pandas as pd
import yaml
from types import SimpleNamespace

from backend.config import load_config
from backend.data import load_and_prepare_dataset, load_validation_dataset
from backend.model_setup import setup_model_and_tokenizer
from backend.trainer import run_training_in_thread

class TrainingManager:
    """
    Manages the state and execution of the model training process.
    """
    def __init__(self):
        self.training_thread = None
        self.log_queue = queue.Queue()
        self.metrics_queue = queue.Queue()
        self.is_training = False
        self.paused = False
        self.stop_event: Event | None = None
        self.current_epoch = 0
        self.progress = 0.0
        self.max_epochs = 1
        self.total_steps: int | None = None
        self.steps_per_epoch: int | None = None
        self.last_step_time = None
        self.last_step = None
        self.smoothed_sps = None  # steps per second EMA
        self.training_data = pd.DataFrame(
            columns=[
                'epoch', 'train_loss', 'val_loss', 'train_accuracy',
                'val_accuracy', 'learning_rate', 'timestamp'
            ]
        )
        self._cached_config = None
        self._lock: Lock = Lock()

    def start_training(self):
        """
        Initializes and starts the training process in a separate thread.
        """
        if self.is_training:
            st.warning("Training is already in progress.")
            return False

        try:
            # Load configurations and data
            config = load_config("config.yaml")
            # sync basic progress constraints
            try:
                with self._lock:
                    self.max_epochs = int(config.training.num_train_epochs)
            except Exception:
                with self._lock:
                    self.max_epochs = 1
            dataset = load_and_prepare_dataset(config)
            eval_dataset = load_validation_dataset(config)
            if dataset is None:
                st.error("Dataset loading failed. Aborting training.")
                return False
                
            model, tokenizer = setup_model_and_tokenizer(config)
            if model is None or tokenizer is None:
                st.error("Model setup failed. Aborting training.")
                return False

            # Start training in a background thread
            with self._lock:
                self.is_training = True
            # create a stop signal for the background trainer
            self.stop_event = Event()
            self.training_thread = Thread(
                target=run_training_in_thread,
                args=(config, model, tokenizer, dataset, eval_dataset, self.log_queue, self.metrics_queue, self.stop_event)
            )
            self.training_thread.daemon = True
            self.training_thread.start()
            
            st.success("Training process started in the background.")
            st.session_state.training_started = True
            # Initialize progress metadata heuristics
            try:
                # Approximate steps per epoch using dataset size and batch size (ignores grad_accum)
                per_device_bs = int(getattr(config.training, 'per_device_train_batch_size', 1) or 1)
                steps_per_epoch = max(1, int((len(dataset) + per_device_bs - 1) / per_device_bs))
                max_epochs = int(getattr(config.training, 'num_train_epochs', 1) or 1)
                with self._lock:
                    self.steps_per_epoch = steps_per_epoch
                    self.total_steps = max(1, steps_per_epoch * max(1, max_epochs))
            except Exception:
                with self._lock:
                    self.steps_per_epoch = None
                    self.total_steps = None
            return True

        except Exception as e:
            st.error(f"Failed to start training: {e}")
            self.is_training = False
            st.session_state.training_started = False
            return False

    def stop_training(self):
        """
        Stops the training process.
        Note: Forcefully stopping a thread is not recommended. This is a placeholder
        for a more graceful shutdown mechanism if the training library supports it.
        """
        if not self.is_training and (self.training_thread is None or not self.training_thread.is_alive()):
            # Allow idempotent stop: if a recent start failed or finished, just log
            st.warning("No training is currently in progress.")
            return False

        # Signal the background thread to request a graceful stop at the next step boundary
        if self.stop_event is not None:
            self.stop_event.set()
        self.log_queue.put("--- Stop request received. Trainer will stop at the next step. ---")
        st.info("Training will stop after the current epoch/step. The thread cannot be forcefully terminated.")
        return True

    def pause_training(self):
        """
        Toggle pause/resume state for UI purposes (no-op for HF Trainer).
        """
        if not self.is_training:
            return False
        self.paused = not self.paused
        return True

    def reset_training(self):
        """
        Reset UI-visible training progress and buffers.
        """
        with self._lock:
            self.current_epoch = 0
            self.progress = 0.0
            self.training_data = pd.DataFrame(columns=self.training_data.columns)
        with self.log_queue.mutex:
            self.log_queue.queue.clear()
        with self.metrics_queue.mutex:
            self.metrics_queue.queue.clear()
        # clear stop signal
        self.stop_event = None


    def get_logs(self):
        """
        Retrieves all available logs from the queue.
        """
        logs = []
        while not self.log_queue.empty():
            log_entry = self.log_queue.get()
            if log_entry is None: # End signal
                with self._lock:
                    self.is_training = False
                st.session_state.training_started = False
                break
            # Accept dict META messages and convert to simple string, also update manager state
            try:
                if isinstance(log_entry, dict) and log_entry.get('meta') == 'progress':
                    if 'total_steps' in log_entry:
                        # Authoritative total steps from trainer; override heuristics
                        with self._lock:
                            self.total_steps = int(log_entry['total_steps'])
                    logs.append(f"[META] total_steps={log_entry.get('total_steps')} log_int={log_entry.get('logging_steps')}")
                else:
                    logs.append(log_entry)
            except Exception:
                logs.append(str(log_entry))
        return logs

    def get_metrics(self):
        """
        Retrieves all available metrics from the queue and concatenates them.
        """
        standardized_frames = []
        expected_cols = [
            'step', 'epoch', 'train_loss', 'val_loss',
            'train_accuracy', 'val_accuracy', 'learning_rate', 'timestamp'
        ]
        while not self.metrics_queue.empty():
            metric_df = self.metrics_queue.get()
            # Normalize incoming logs (which may contain: step, loss, learning_rate, epoch)
            try:
                df = pd.DataFrame(metric_df)
            except Exception:
                df = metric_df if isinstance(metric_df, pd.DataFrame) else pd.DataFrame()

            normalized = pd.DataFrame()
            normalized['step'] = df.get('step')
            normalized['epoch'] = df.get('epoch')
            # Map generic loss -> train_loss; validation metrics may be absent
            normalized['train_loss'] = df.get('train_loss', df.get('loss'))
            normalized['val_loss'] = df.get('val_loss')
            normalized['train_accuracy'] = df.get('train_accuracy')
            normalized['val_accuracy'] = df.get('val_accuracy')
            normalized['learning_rate'] = df.get('learning_rate')
            normalized['grad_norm'] = df.get('grad_norm')
            normalized['timestamp'] = pd.Timestamp.utcnow()

            # Ensure column order and presence
            normalized = normalized.reindex(columns=expected_cols)
            standardized_frames.append(normalized)

        if not standardized_frames:
            return pd.DataFrame(columns=expected_cols)

        out = pd.concat(standardized_frames).reset_index(drop=True)
        return out

    def get_status(self):
        """
        Returns a dict the dashboard expects.
        """
        # incorporate any newly arrived metrics into the cached training_data
        new_metrics = self.get_metrics()
        if not new_metrics.empty:
            with self._lock:
                self.training_data = pd.concat([self.training_data, new_metrics]).reset_index(drop=True)
            # try to update epoch/progress heuristically
            if 'epoch' in new_metrics.columns and pd.notnull(new_metrics['epoch']).any():
                try:
                    with self._lock:
                        self.current_epoch = int(new_metrics['epoch'].dropna().iloc[-1])
                except Exception:
                    pass
            # Step-based progress tracking
            try:
                latest_step = int(new_metrics['step'].dropna().iloc[-1]) if 'step' in new_metrics.columns else None
            except Exception:
                latest_step = None
            if latest_step is not None and latest_step >= 0:
                # Steps/sec estimate using simple EMA to stabilize
                import time
                now = time.time()
                if self.last_step is not None and self.last_step_time is not None and latest_step > self.last_step:
                    dt = max(1e-3, now - self.last_step_time)
                    dstep = latest_step - self.last_step
                    inst_sps = dstep / dt
                    with self._lock:
                        if self.smoothed_sps is None:
                            self.smoothed_sps = inst_sps
                        else:
                            self.smoothed_sps = 0.8 * self.smoothed_sps + 0.2 * inst_sps
                with self._lock:
                    self.last_step = latest_step
                    self.last_step_time = now
                # Progress from steps if total known, else epoch fraction
                with self._lock:
                    if self.total_steps:
                        self.progress = min(1.0, float(latest_step) / float(self.total_steps))
                    elif self.max_epochs > 0:
                        self.progress = min(1.0, self.current_epoch / float(self.max_epochs))
            elif self.max_epochs > 0:
                with self._lock:
                    self.progress = min(1.0, self.current_epoch / float(self.max_epochs))

        thread_alive = bool(self.training_thread and self.training_thread.is_alive())
        # Snapshot under lock for consistent return
        with self._lock:
            is_training = bool(self.is_training or thread_alive)
            paused = self.paused
            current_epoch = int(self.current_epoch)
            progress = float(self.progress)
            max_epochs = int(self.max_epochs)
            training_data_copy = self.training_data.copy()
            total_steps = int(self.total_steps) if self.total_steps else None
            current_step = int(self.last_step) if self.last_step is not None else None
            sps = float(self.smoothed_sps) if self.smoothed_sps is not None else None

        # Compute ETA
        eta_seconds = None
        if sps and sps > 0 and total_steps and current_step is not None:
            remaining = max(0, total_steps - current_step)
            eta_seconds = remaining / sps

        return {
            'active': is_training,
            'paused': paused,
            'current_epoch': current_epoch,
            'progress': progress,
            'max_epochs': max_epochs,
            'training_data': training_data_copy,
            'total_steps': total_steps,
            'current_step': current_step,
            'steps_per_second': sps,
            'eta_seconds': float(eta_seconds) if eta_seconds is not None else None,
        }

    def get_yaml_config(self):
        """
        Load config and expose a shape expected by the UI code.
        Returns an object with namespaces: data, model, quantization, training, lora
        """
        app_cfg = load_config("config.yaml")
        data_ns = SimpleNamespace(
            train_file=(app_cfg.local_train_path if getattr(app_cfg, 'data_source', 'hf') == 'local' else f"hf:{app_cfg.dataset_name}:{app_cfg.dataset_split}"),
            validation_file=(app_cfg.local_validation_path if getattr(app_cfg, 'data_source', 'hf') == 'local' else f"hf:{app_cfg.dataset_name}:validation")
        )
        model_ns = SimpleNamespace(
            max_length=app_cfg.training.max_seq_length or 0,
            attn_implementation="auto",
        )
        # Normalize compute dtype to a plain string ("float16" or "bfloat16") for the UI
        dtype_val = app_cfg.quantization.bnb_4bit_compute_dtype
        try:
            import torch  # type: ignore
            if isinstance(dtype_val, torch.dtype):
                if dtype_val == torch.bfloat16:
                    dtype_str = 'bfloat16'
                else:
                    dtype_str = 'float16'
            else:
                dtype_str = str(dtype_val)
        except Exception:
            dtype_str = str(dtype_val)
        if dtype_str.startswith('torch.'):
            dtype_str = dtype_str.split('.', 1)[1]

        quant_ns = SimpleNamespace(
            enabled=app_cfg.quantization.load_in_4bit,
            quant_type=app_cfg.quantization.bnb_4bit_quant_type,
            use_double_quant=app_cfg.quantization.bnb_4bit_use_double_quant,
            compute_dtype=dtype_str,
        )
        training_ns = SimpleNamespace(
            learning_rate=app_cfg.training.learning_rate,
            per_device_train_batch_size=app_cfg.training.per_device_train_batch_size,
            num_train_epochs=app_cfg.training.num_train_epochs,
            max_seq_length=app_cfg.training.max_seq_length,
            packing=app_cfg.training.packing,
            device_map=app_cfg.training.device_map,
            optim=app_cfg.training.optim,
            gradient_accumulation_steps=app_cfg.training.gradient_accumulation_steps,
            weight_decay=app_cfg.training.weight_decay,
            lr_scheduler_type=app_cfg.training.lr_scheduler_type,
            gradient_checkpointing=False,
            logging_steps=app_cfg.training.logging_steps,
            save_steps=app_cfg.training.save_steps,
            report_to=app_cfg.training.report_to,
            tokenizer_num_proc=int(getattr(app_cfg.training, 'tokenizer_num_proc', 1)),
        )
        lora_ns = SimpleNamespace(
            r=app_cfg.lora.r,
            alpha=app_cfg.lora.alpha,
            dropout=app_cfg.lora.dropout,
        )
        return SimpleNamespace(
            data=data_ns,
            model=model_ns,
            quantization=quant_ns,
            training=training_ns,
            lora=lora_ns,
        )

    def _get_cached_yaml(self):
        try:
            with open("config.yaml", 'r') as f:
                return yaml.safe_load(f)
        except Exception:
            return None

    def get_config(self):
        """
        Return a simplified dict used by the configuration UI.
        Maps fields from Pydantic AppConfig to the expected keys.
        """
        app_cfg = load_config("config.yaml")
        # derive warmup_steps from warmup_ratio if possible (heuristic for UI input)
        try:
            warmup_steps = int(round(float(app_cfg.training.warmup_ratio) * 10000))
        except Exception:
            warmup_steps = 1000
        try:
            batch_size = int(app_cfg.training.per_device_train_batch_size)
        except Exception:
            batch_size = 2
        try:
            max_epochs = int(app_cfg.training.num_train_epochs)
        except Exception:
            max_epochs = 1
        return {
            'learning_rate': float(getattr(app_cfg.training, 'learning_rate', 2e-4)),
            'batch_size': batch_size,
            'max_epochs': max_epochs,
            'model_name': str(getattr(app_cfg, 'base_model', 'mistralai/Mistral-7B-Instruct-v0.3')),
            'optimizer': str(getattr(app_cfg.training, 'optim', 'paged_adamw_32bit')),
            'warmup_steps': warmup_steps,
        }

    def update_config(self, new_config: dict, hf_token: str | None = None) -> bool:
        """
        Update config.yaml with values from the simplified UI config.
        Returns False if training is active.
        """
        if self.is_training:
            return False

        yaml_cfg = self._get_cached_yaml() or {}
        # ensure nested structure exists
        yaml_cfg.setdefault('training', {})
        yaml_cfg.setdefault('quantization', {})
        yaml_cfg.setdefault('lora', {})

        # map fields
        training = yaml_cfg['training']
        training['learning_rate'] = float(new_config.get('learning_rate', training.get('learning_rate', 2e-4)))
        training['per_device_train_batch_size'] = int(new_config.get('batch_size', training.get('per_device_train_batch_size', 4)))
        training['num_train_epochs'] = int(new_config.get('max_epochs', training.get('num_train_epochs', 1)))
        training['optim'] = str(new_config.get('optimizer', training.get('optim', 'paged_adamw_32bit')))
        # advanced toggles
        if 'adv_gradient_checkpointing' in new_config:
            training['gradient_checkpointing'] = bool(new_config['adv_gradient_checkpointing'])

        # convert warmup_steps back into a ratio heuristic (divide by 10000.0)
        warmup_steps = int(new_config.get('warmup_steps', 1000))
        training['warmup_ratio'] = float(max(0.0, min(1.0, warmup_steps / 10000.0)))

        # advanced training/logging fields
        if 'adv_logging_steps' in new_config and new_config['adv_logging_steps']:
            training['logging_steps'] = int(new_config['adv_logging_steps'])
        if 'adv_save_steps' in new_config and new_config['adv_save_steps']:
            training['save_steps'] = int(new_config['adv_save_steps'])
        if 'adv_report_to' in new_config and new_config['adv_report_to']:
            training['report_to'] = str(new_config['adv_report_to'])
        if 'adv_evaluation_strategy' in new_config:
            training['evaluation_strategy'] = str(new_config['adv_evaluation_strategy'])
        if 'adv_eval_steps' in new_config and new_config['adv_eval_steps']:
            training['eval_steps'] = int(new_config['adv_eval_steps'])
        if 'adv_gradient_accumulation_steps' in new_config and new_config['adv_gradient_accumulation_steps']:
            training['gradient_accumulation_steps'] = int(new_config['adv_gradient_accumulation_steps'])
        if 'adv_tokenizer_num_proc' in new_config and new_config['adv_tokenizer_num_proc']:
            training['tokenizer_num_proc'] = int(new_config['adv_tokenizer_num_proc'])

        # base model is stored at top-level key 'base_model'
        if 'model_name' in new_config and new_config['model_name']:
            yaml_cfg['base_model'] = str(new_config['model_name'])

        # advanced model/data fields
        if 'adv_max_seq_length' in new_config:
            training['max_seq_length'] = (int(new_config['adv_max_seq_length'])
                                          if new_config['adv_max_seq_length'] else None)
        if 'adv_attn_impl' in new_config and new_config['adv_attn_impl']:
            # store as hint under training section for now
            training['attn_implementation'] = str(new_config['adv_attn_impl'])

        # quantization section
        quant = yaml_cfg['quantization']
        if 'adv_quant_enabled' in new_config:
            quant['load_in_4bit'] = bool(new_config['adv_quant_enabled'])
        if 'adv_quant_type' in new_config and new_config['adv_quant_type']:
            quant['bnb_4bit_quant_type'] = str(new_config['adv_quant_type'])
        if 'adv_quant_double' in new_config:
            quant['bnb_4bit_use_double_quant'] = bool(new_config['adv_quant_double'])
        if 'adv_quant_compute_dtype' in new_config and new_config['adv_quant_compute_dtype']:
            # store raw string; pydantic validator will map to torch dtype
            quant['bnb_4bit_compute_dtype'] = str(new_config['adv_quant_compute_dtype'])

        # lora section
        lora = yaml_cfg['lora']
        if 'adv_lora_r' in new_config and new_config['adv_lora_r']:
            lora['r'] = int(new_config['adv_lora_r'])
        if 'adv_lora_alpha' in new_config and new_config['adv_lora_alpha']:
            lora['alpha'] = int(new_config['adv_lora_alpha'])
        if 'adv_lora_dropout' in new_config and new_config['adv_lora_dropout'] is not None:
            lora['dropout'] = float(new_config['adv_lora_dropout'])
        if 'adv_lora_target_modules' in new_config:
            raw = (new_config['adv_lora_target_modules'] or '').strip()
            if raw:
                # store as list of trimmed strings
                lora['target_modules'] = [s.strip() for s in raw.split(',') if s.strip()]

        # optionally store HF token in session only
        if hf_token:
            st.session_state['hf_token'] = hf_token

        # data source persistence
        if 'data_source' in new_config:
            yaml_cfg['data_source'] = str(new_config['data_source'])
        if 'local_train_path' in new_config and new_config['local_train_path']:
            yaml_cfg['local_train_path'] = str(new_config['local_train_path'])
        if 'local_validation_path' in new_config and new_config['local_validation_path']:
            yaml_cfg['local_validation_path'] = str(new_config['local_validation_path'])

        # write back to file
        try:
            with open("config.yaml", 'w') as f:
                yaml.safe_dump(yaml_cfg, f, sort_keys=False)
            # refresh cached values used by UI
            self._cached_config = None
            return True
        except Exception as e:
            st.error(f"Failed to update configuration: {e}")
            return False


def get_training_manager() -> TrainingManager:
    """
    Returns a singleton TrainingManager stored in Streamlit session state.
    """
    if 'training_manager' not in st.session_state or not isinstance(st.session_state.training_manager, TrainingManager):
        st.session_state.training_manager = TrainingManager()
    return st.session_state.training_manager
