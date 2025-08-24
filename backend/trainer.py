import os
import pandas as pd
from transformers import TrainingArguments
from trl import SFTTrainer
from backend.config import AppConfig
from backend.huggingface_integration import hf_login, push_to_hub
import streamlit as st
from threading import Thread
import queue

class StreamlitCallback:
    """
    A custom callback to stream training logs and metrics to the Streamlit UI.
    """
    def __init__(self, log_queue, metrics_queue):
        self.log_queue = log_queue
        self.metrics_queue = metrics_queue

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            log_str = f"Step: {state.global_step} | Log: {logs}"
            self.log_queue.put(log_str)
            
            # Create a DataFrame from the logs for metrics plotting
            metric_data = {
                'step': state.global_step,
                'loss': logs.get('loss'),
                'learning_rate': logs.get('learning_rate'),
                'epoch': logs.get('epoch')
            }
            # Filter out None values
            metric_data = {k: v for k, v in metric_data.items() if v is not None}
            if 'loss' in metric_data: # Only send if there's something to plot
                self.metrics_queue.put(pd.DataFrame([metric_data]))


def run_training_in_thread(config, model, tokenizer, dataset, log_queue, metrics_queue):
    """
    This function runs the training process and is intended to be called in a separate thread.
    """
    try:
        # --- Hugging Face Login ---
        if config.huggingface.push_to_hub:
            hf_login(config.huggingface.hub_token)

        # --- Training Arguments ---
        training_args = TrainingArguments(**config.training.dict())

        # --- Initialize Trainer ---
        streamlit_callback = StreamlitCallback(log_queue, metrics_queue)
        
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=model.peft_config['default'],
            dataset_text_field=config.text_column,
            max_seq_length=config.training.max_seq_length,
            tokenizer=tokenizer,
            args=training_args,
            packing=config.training.packing,
            callbacks=[streamlit_callback],
        )

        # --- Start Training ---
        log_queue.put("--- Starting Training ---")
        trainer.train()
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

