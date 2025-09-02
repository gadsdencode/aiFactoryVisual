import os
import math
import pandas as pd
import numpy as np
from transformers import TrainingArguments
import torch
from transformers import DataCollatorForLanguageModeling
from trl import SFTConfig
from trl import SFTTrainer
from transformers import Trainer
from backend.config import AppConfig
from backend.huggingface_integration import hf_login, push_to_hub
from backend.metrics import compute_token_accuracy
import streamlit as st
from threading import Thread, Event
import queue
from transformers import TrainerCallback, TrainingArguments
from backend.data import tokenize_and_cache_dataset

class TrainingProgressCallback(TrainerCallback):
    """Callback that forwards logs/metrics to a TrainingManager instance."""
    def __init__(self, manager):
        self.manager = manager

    def on_log(self, args: TrainingArguments, state, control, logs=None, **kwargs):
        try:
            if logs and getattr(state, 'is_world_process_zero', True):
                step = int(getattr(state, 'global_step', logs.get('step', 0)) or 0)
                metrics = {
                    'step': step,
                    'epoch': logs.get('epoch'),
                    'train_loss': logs.get('loss'),
                    'val_loss': logs.get('eval_loss'),
                    'val_accuracy': logs.get('eval_accuracy'),
                    'learning_rate': logs.get('learning_rate'),
                    'grad_norm': logs.get('grad_norm'),
                }
                metrics = {k: v for k, v in metrics.items() if v is not None}
                self.manager.add_metrics(metrics)
                # concise human log
                parts = [f"step={step}"]
                if 'epoch' in metrics:
                    try:
                        parts.append(f"epoch={float(metrics['epoch']):.2f}")
                    except Exception:
                        parts.append(f"epoch={metrics['epoch']}")
                if 'train_loss' in metrics:
                    try:
                        parts.append(f"loss={float(metrics['train_loss']):.4f}")
                    except Exception:
                        parts.append(f"loss={metrics['train_loss']}")
                if 'val_loss' in metrics:
                    try:
                        parts.append(f"val_loss={float(metrics['val_loss']):.4f}")
                    except Exception:
                        parts.append(f"val_loss={metrics['val_loss']}")
                if 'learning_rate' in metrics:
                    try:
                        parts.append(f"lr={float(metrics['learning_rate']):.6f}")
                    except Exception:
                        parts.append(f"lr={metrics['learning_rate']}")
                self.manager.add_log(" | ".join(parts))
        except Exception:
            pass

class StreamlitCallback(TrainerCallback):
    """TrainerCallback that streams logs/metrics to Streamlit via queues and supports stopping."""
    def __init__(self, log_queue: queue.Queue, metrics_queue: queue.Queue, stop_event: "Event | None" = None):
        self.log_queue = log_queue
        self.metrics_queue = metrics_queue
        self.stop_event = stop_event

    def on_train_begin(self, args: TrainingArguments, state, control, **kwargs):
        self.log_queue.put("--- Training started ---")

    def on_log(self, args: TrainingArguments, state, control, logs=None, **kwargs):
        if logs:
            # Push structured metrics
            metric_data = {
                'step': state.global_step,
                'loss': logs.get('loss'),
                'val_loss': logs.get('eval_loss'),
                'val_accuracy': logs.get('eval_accuracy'),
                'learning_rate': logs.get('learning_rate'),
                'epoch': logs.get('epoch'),
                'grad_norm': logs.get('grad_norm'),
            }
            metric_data = {k: v for k, v in metric_data.items() if v is not None}
            if 'loss' in metric_data or 'val_loss' in metric_data:
                self.metrics_queue.put(pd.DataFrame([metric_data]))

            # Emit concise human-readable log line
            try:
                parts = [f"step={state.global_step}"]
                if 'epoch' in metric_data:
                    parts.append(f"epoch={metric_data['epoch']:.2f}")
                if 'loss' in metric_data:
                    parts.append(f"loss={float(metric_data['loss']):.4f}")
                if 'val_loss' in metric_data:
                    parts.append(f"val_loss={float(metric_data['val_loss']):.4f}")
                if 'grad_norm' in metric_data:
                    parts.append(f"grad_norm={float(metric_data['grad_norm']):.3f}")
                if 'learning_rate' in metric_data:
                    parts.append(f"lr={float(metric_data['learning_rate']):.6f}")
                self.log_queue.put(" | ".join(parts))
            except Exception:
                # Fallback to raw dict if formatting fails
                self.log_queue.put(f"step={state.global_step} | {logs}")
        # Stop check on each log
        if self.stop_event is not None and self.stop_event.is_set():
            control.should_training_stop = True

    def on_epoch_begin(self, args: TrainingArguments, state, control, **kwargs):
        try:
            self.log_queue.put(f"Epoch {int(state.epoch) if state.epoch is not None else 0} begin")
        except Exception:
            self.log_queue.put("Epoch begin")

    def on_step_begin(self, args: TrainingArguments, state, control, **kwargs):
        # Emit a per-step start log to indicate liveness
        try:
            self.log_queue.put(f"Step {state.global_step + 1} begin")
            self.log_queue.put("[training_step] forward begin")
        except Exception:
            self.log_queue.put("Step begin")

    def on_step_end(self, args: TrainingArguments, state, control, **kwargs):
        # Emit lightweight step heartbeat to drive UI progress
        try:
            hb = {
                'step': int(state.global_step),
                'epoch': float(state.epoch) if state.epoch is not None else None,
            }
            self.metrics_queue.put(pd.DataFrame([hb]))
        except Exception:
            pass
        try:
            self.log_queue.put("[training_step] forward/backward done")
        except Exception:
            pass
        # Respect stop request
        if self.stop_event is not None and self.stop_event.is_set():
            control.should_training_stop = True

    def on_train_end(self, args: TrainingArguments, state, control, **kwargs):
        self.log_queue.put("--- Training finished ---")


def run_training_in_thread(config, model, tokenizer, dataset, eval_dataset, log_queue=None, metrics_queue=None, stop_event=None, manager=None):
    """
    This function runs the training process and is intended to be called in a separate thread.
    """
    success = True
    final_eval: dict | None = None
    try:
        def _emit_log(message: str):
            try:
                if manager is not None:
                    manager.add_log(str(message))
                elif log_queue is not None:
                    log_queue.put(message)
            except Exception:
                pass

        # --- Hugging Face Login ---
        if config.huggingface.push_to_hub:
            hf_login(config.huggingface.hub_token)

        # --- Training Arguments ---
        # Only pass fields supported by TrainingArguments. Some items in our
        # config (e.g., max_seq_length, packing, device_map) are used by the
        # model or SFTTrainer, not by TrainingArguments.

        # Validate/normalize optimizer choice for this environment
        chosen_optim = getattr(config.training, 'optim', 'adamw_torch') or 'adamw_torch'
        needs_bnb = ('8bit' in chosen_optim) or ('paged' in chosen_optim)
        if needs_bnb:
            try:
                import bitsandbytes as _bnb  # type: ignore
            except Exception:
                log_queue.put(f"Optimizer '{chosen_optim}' requires bitsandbytes. Falling back to 'adamw_torch'.")
                chosen_optim = 'adamw_torch'

        # Normalize core numeric hyperparameters
        try:
            lr_val = float(getattr(config.training, 'learning_rate', 2e-4) or 2e-4)
        except Exception:
            lr_val = 2e-4
        try:
            warmup_ratio_val = float(getattr(config.training, 'warmup_ratio', 0.0) or 0.0)
        except Exception:
            warmup_ratio_val = 0.0
        try:
            weight_decay_val = float(getattr(config.training, 'weight_decay', 0.0) or 0.0)
        except Exception:
            weight_decay_val = 0.0
        try:
            max_grad_norm_val = float(getattr(config.training, 'max_grad_norm', 1.0) or 1.0)
        except Exception:
            max_grad_norm_val = 1.0
        try:
            num_train_epochs_val = int(getattr(config.training, 'num_train_epochs', 1) or 1)
        except Exception:
            num_train_epochs_val = 1
        try:
            per_device_bs_val = int(getattr(config.training, 'per_device_train_batch_size', 1) or 1)
        except Exception:
            per_device_bs_val = 1
        try:
            grad_accum_val = int(getattr(config.training, 'gradient_accumulation_steps', 1) or 1)
        except Exception:
            grad_accum_val = 1
        try:
            logging_steps_val = int(getattr(config.training, 'logging_steps', 10) or 10)
        except Exception:
            logging_steps_val = 10
        try:
            save_steps_val = int(getattr(config.training, 'save_steps', 1000) or 1000)
        except Exception:
            save_steps_val = 1000
        try:
            max_steps_val = int(getattr(config.training, 'max_steps', -1) if getattr(config.training, 'max_steps', -1) is not None else -1)
        except Exception:
            max_steps_val = -1

        _emit_log(f"Using optimizer: {chosen_optim}; lr: {lr_val}")
        _emit_log(
            f"Resolved hparams -> epochs:{num_train_epochs_val}, per_device_bs:{per_device_bs_val}, "
            f"grad_accum:{grad_accum_val}, warmup_ratio:{warmup_ratio_val}, weight_decay:{weight_decay_val}, "
            f"max_grad_norm:{max_grad_norm_val}, logging_steps:{logging_steps_val}, save_steps:{save_steps_val}"
        )
        # --- Derive total optimizer steps if max_steps <= 0 to avoid None in schedulers ---
        try:
            train_size = int(len(dataset))
        except Exception:
            try:
                train_size = int(len(tokenized_train))
            except Exception:
                train_size = 0
        steps_per_epoch = max(1, math.ceil(train_size / max(1, per_device_bs_val)))
        opt_steps_per_epoch = max(1, math.floor(steps_per_epoch / max(1, grad_accum_val)))
        derived_total_steps = max(1, opt_steps_per_epoch * max(1, num_train_epochs_val))
        if max_steps_val is None or (isinstance(max_steps_val, int) and max_steps_val <= 0):
            final_max_steps = derived_total_steps
        else:
            final_max_steps = int(max_steps_val)
        _emit_log(f"Derived steps_per_epoch={steps_per_epoch}, opt_steps_per_epoch={opt_steps_per_epoch}, total_steps={derived_total_steps}, using max_steps={final_max_steps}")

        # Consolidate training args into SFTConfig
        sft_max_seq_len = int(getattr(config.training, 'max_seq_length', 1024) or 1024)
        sft_config = SFTConfig(
            output_dir=config.training.output_dir,
            num_train_epochs=num_train_epochs_val,
            per_device_train_batch_size=per_device_bs_val,
            gradient_accumulation_steps=grad_accum_val,
            optim=chosen_optim,
            save_steps=save_steps_val,
            logging_steps=logging_steps_val,
            learning_rate=lr_val,
            weight_decay=weight_decay_val,
            fp16=getattr(config.training, 'fp16', False),
            bf16=getattr(config.training, 'bf16', False),
            gradient_checkpointing=getattr(config.training, 'gradient_checkpointing', False),
            max_grad_norm=max_grad_norm_val,
            max_steps=final_max_steps,
            warmup_ratio=warmup_ratio_val,
            group_by_length=getattr(config.training, 'group_by_length', False),
            lr_scheduler_type=getattr(config.training, 'lr_scheduler_type', 'constant'),
            report_to=getattr(config.training, 'report_to', 'none'),
            remove_unused_columns=False,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            dataloader_persistent_workers=False,
            logging_first_step=True,
            # SFT-specific
            dataset_text_field=getattr(config, 'text_column', 'text'),
            max_seq_length=sft_max_seq_len,
            packing=getattr(config.training, 'packing', False),
        )

        # --- Initialize Trainer ---
        streamlit_callback = StreamlitCallback(metrics_queue if metrics_queue is not None else queue.Queue(), metrics_queue if metrics_queue is not None else queue.Queue(), stop_event)

        # Default max_seq_length when not provided
        max_seq_len = sft_max_seq_len

        # compute_token_accuracy imported from backend.metrics

        # Early log to indicate tokenization/preparation can take time
        _emit_log("Preparing/tokenizing datasets...")

        # Stable single-thread CPU to avoid Windows deadlocks
        try:
            torch.set_num_threads(1)
            os.environ.setdefault('OMP_NUM_THREADS', '1')
            os.environ.setdefault('MKL_NUM_THREADS', '1')
        except Exception:
            pass

        # Tokenize datasets with on-disk caching
        tokenized_train = tokenize_and_cache_dataset(config, tokenizer, dataset)
        tokenized_eval = tokenize_and_cache_dataset(config, tokenizer, eval_dataset) if eval_dataset is not None else None

        # Ensure torch format handled by caching function; keep a safe fallback
        try:
            tokenized_train = tokenized_train.with_format("torch", columns=["input_ids","attention_mask","labels"])  # type: ignore
            if tokenized_eval is not None:
                tokenized_eval = tokenized_eval.with_format("torch", columns=["input_ids","attention_mask","labels"])  # type: ignore
        except Exception:
            pass

        # Use simple collator to reduce CPU overhead and avoid padding-induced stalls
        try:
            from transformers import DefaultDataCollator
            data_collator = DefaultDataCollator()
        except Exception:
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        # Enable cudnn benchmarking for faster first iteration on fixed shapes
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

        # Ensure model in training mode and gradients enabled for LoRA adapters
        try:
            model.train()
            for p in model.parameters():
                # Do not force enable grads for frozen base when using PEFT; only ensure trainable params require grad
                if getattr(p, 'requires_grad', None) is False and hasattr(p, 'is_lora_param') and p.is_lora_param:  # type: ignore
                    p.requires_grad = True
        except Exception:
            pass

        # Standard SFTTrainer with StreamlitCallback for logging
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=data_collator,
            compute_metrics=compute_token_accuracy if tokenized_eval is not None and getattr(config.training, 'evaluation_strategy', 'no') != 'no' else None,
            callbacks=[streamlit_callback],
            tokenizer=tokenizer,
        )

        _emit_log("Trainer initialized. Starting training loop...")
        # Emit initial META with expected totals (mirrors TrainingManager heuristics)
        try:
            meta = {
                'meta': 'progress',
                'total_steps': final_max_steps,
                'logging_steps': logging_steps_val,
            }
            _emit_log(meta)
        except Exception:
            pass

        # --- Start Training ---
        _emit_log("--- Starting Training ---")
        if stop_event is None:
            # Choose callback based on manager/queues
            if manager is not None:
                callbacks = [TrainingProgressCallback(manager)]
                try:
                    trainer.callback_handler.callbacks = callbacks  # type: ignore[attr-defined]
                except Exception:
                    pass
            trainer.train()
        else:
            # Train with periodic stop checks; ensure scheduler has concrete total steps
            try:
                trainer.create_optimizer_and_scheduler(num_training_steps=final_max_steps)
            except Exception:
                pass
            if manager is not None:
                callbacks = [TrainingProgressCallback(manager)]
                try:
                    trainer.callback_handler.callbacks = callbacks  # type: ignore[attr-defined]
                except Exception:
                    pass
            train_result = trainer.train(resume_from_checkpoint=None)
            # Built-in callbacks honor control.should_training_stop based on our StreamlitCallback
            if stop_event.is_set():
                try:
                    trainer.control.should_training_stop = True
                except Exception:
                    pass
        _emit_log("--- Training Finished ---")

        # --- Evaluate ---
        try:
            if tokenized_eval is not None:
                final_eval = trainer.evaluate()
                if isinstance(final_eval, dict):
                    _emit_log(f"Final evaluation: {final_eval}")
        except Exception:
            pass

        # --- Save Model ---
        final_output_dir = os.path.join(config.training.output_dir, "final_model")
        _emit_log(f"Saving final model to {final_output_dir}")
        trainer.model.save_pretrained(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)
        _emit_log("Model saved successfully.")

        # --- Push to Hub ---
        if config.huggingface.push_to_hub:
            _emit_log("Pushing model to Hugging Face Hub...")
            push_to_hub(config, model, tokenizer)
            _emit_log("Push to Hub complete.")
            
    except Exception as e:
        try:
            import traceback
            tb = traceback.format_exc()
            if manager is not None:
                manager.add_log(f"--- TRAINING ERROR ---\n{tb}")
            elif log_queue is not None:
                log_queue.put(f"--- TRAINING ERROR ---\n{tb}")
        except Exception:
            if manager is not None:
                manager.add_log(f"--- TRAINING ERROR --- \n{e}")
            elif log_queue is not None:
                log_queue.put(f"--- TRAINING ERROR --- \n{e}")
        success = False
    finally:
        if manager is None and log_queue is not None:
            log_queue.put(None) # Signal that training is complete

    return success, (final_eval or {})

