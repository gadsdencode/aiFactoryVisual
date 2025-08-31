import os
import pandas as pd
import numpy as np
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig
from backend.config import AppConfig
from backend.huggingface_integration import hf_login, push_to_hub
import streamlit as st
from threading import Thread
import queue
from transformers import TrainerCallback, TrainingArguments

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
            self.log_queue.put(f"Step: {state.global_step} | Log: {logs}")
            metric_data = {
                'step': state.global_step,
                'loss': logs.get('loss'),
                'val_loss': logs.get('eval_loss'),
                'val_accuracy': logs.get('eval_accuracy'),
                'learning_rate': logs.get('learning_rate'),
                'epoch': logs.get('epoch'),
            }
            metric_data = {k: v for k, v in metric_data.items() if v is not None}
            if 'loss' in metric_data:
                self.metrics_queue.put(pd.DataFrame([metric_data]))
        # Stop check on each log
        if self.stop_event is not None and self.stop_event.is_set():
            control.should_training_stop = True

    def on_step_end(self, args: TrainingArguments, state, control, **kwargs):
        # Heartbeat to ensure UI stays active even if no logs in this step
        if state.global_step % max(1, args.logging_steps) != 0:
            self.log_queue.put(f"Heartbeat step {state.global_step}")
        if self.stop_event is not None and self.stop_event.is_set():
            control.should_training_stop = True

    def on_train_end(self, args: TrainingArguments, state, control, **kwargs):
        self.log_queue.put("--- Training finished ---")


def run_training_in_thread(config, model, tokenizer, dataset, eval_dataset, log_queue, metrics_queue, stop_event=None):
    """
    This function runs the training process and is intended to be called in a separate thread.
    """
    try:
        # Avoid tokenizers parallelism deadlocks on Windows/threads
        try:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        except Exception:
            pass
        # --- Hugging Face Login ---
        if config.huggingface.push_to_hub:
            hf_login(config.huggingface.hub_token)

        # --- Training Arguments ---
        # Only pass fields supported by TrainingArguments. Some items in our
        # config (e.g., max_seq_length, packing, device_map) are used by the
        # model or SFTTrainer, not by TrainingArguments.
        training_args = TrainingArguments(
            output_dir=config.training.output_dir,
            num_train_epochs=config.training.num_train_epochs,
            per_device_train_batch_size=config.training.per_device_train_batch_size,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            optim=config.training.optim,
            save_steps=config.training.save_steps,
            logging_steps=config.training.logging_steps,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            fp16=config.training.fp16,
            bf16=config.training.bf16,
            gradient_checkpointing=config.training.gradient_checkpointing,
            max_grad_norm=config.training.max_grad_norm,
            max_steps=config.training.max_steps,
            warmup_ratio=config.training.warmup_ratio,
            group_by_length=config.training.group_by_length,
            lr_scheduler_type=config.training.lr_scheduler_type,
            report_to=config.training.report_to,
            remove_unused_columns=False,
            dataloader_num_workers=0,
            dataloader_pin_memory=True,
            dataloader_persistent_workers=False,
            logging_first_step=True,
        )

        # --- Initialize Trainer ---
        streamlit_callback = StreamlitCallback(log_queue, metrics_queue, stop_event)

        # Build SFTConfig to avoid deprecations and set SFT-specific options
        # Default max_seq_length when not provided
        sft_max_seq_len = config.training.max_seq_length or 1024
        sft_config = SFTConfig(
            output_dir=config.training.output_dir,
            num_train_epochs=config.training.num_train_epochs,
            per_device_train_batch_size=config.training.per_device_train_batch_size,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            lr_scheduler_type=config.training.lr_scheduler_type,
            warmup_ratio=config.training.warmup_ratio,
            logging_steps=config.training.logging_steps,
            save_steps=config.training.save_steps,
            eval_strategy=getattr(config.training, 'evaluation_strategy', 'no'),
            eval_steps=getattr(config.training, 'eval_steps', 100),
            report_to=config.training.report_to,
            dataset_text_field=config.text_column,
            max_seq_length=sft_max_seq_len,
            packing=config.training.packing,
            dataset_num_proc=1,
        )

        # Compute token-level accuracy over non-ignored labels (-100)
        def compute_token_accuracy(eval_pred):
            predictions = eval_pred.predictions
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            labels = eval_pred.label_ids
            try:
                # predictions: (batch, seq_len, vocab)
                pred_ids = np.argmax(predictions, axis=-1)
            except Exception:
                return {}
            mask = labels != -100
            total = np.maximum(mask.sum(), 1)
            correct = (pred_ids[mask] == labels[mask]).sum()
            acc = float(correct) / float(total)
            return {"accuracy": acc}

        # Early log to indicate tokenization/preparation can take time
        log_queue.put("Preparing/tokenizing datasets (this can take a few minutes on first run)...")

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            args=sft_config,
            compute_metrics=compute_token_accuracy if eval_dataset is not None and getattr(config.training, 'evaluation_strategy', 'no') != 'no' else None,
            callbacks=[streamlit_callback],
        )

        log_queue.put("Trainer initialized. Starting training loop...")

        # --- Start Training ---
        log_queue.put("--- Starting Training ---")
        if stop_event is None:
            trainer.train()
        else:
            # Train with periodic stop checks
            trainer.create_optimizer_and_scheduler(num_training_steps=None)
            train_result = trainer.train(resume_from_checkpoint=None)
            # HF Trainer does not expose per-step loop here; rely on built-in hooks honoring control.should_training_stop
            # When stop_event is set, ask trainer to stop next step
            if stop_event.is_set():
                trainer.control.should_training_stop = True
        log_queue.put("--- Training Finished ---")

        # --- Save Model ---
        final_output_dir = os.path.join(config.training.output_dir, "final_model")
        log_queue.put(f"Saving final model to {final_output_dir}")
        trainer.model.save_pretrained(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)
        log_queue.put("Model saved successfully.")

        # --- Push to Hub ---
        if config.huggingface.push_to_hub:
            log_queue.put("Pushing model to Hugging Face Hub...")
            push_to_hub(config, model, tokenizer)
            log_queue.put("Push to Hub complete.")
            
    except Exception as e:
        log_queue.put(f"--- TRAINING ERROR --- \n{e}")
    finally:
        log_queue.put(None) # Signal that training is complete

