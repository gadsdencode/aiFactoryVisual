"""
Training Manager - Handles training state and operations for the Streamlit dashboard
"""
import logging
import threading
import time
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class TrainingManager:
    """Manages training state and operations for the dashboard"""
    
    def __init__(self):
        self.training_active = False
        self.training_paused = False
        self.current_epoch = 0
        self.max_epochs = 100
        self.training_thread: Optional[threading.Thread] = None
        self.training_data = pd.DataFrame()
        self.lock = threading.Lock()
        self.start_time: Optional[datetime] = None
        self.config = self._default_config()
        
    def _default_config(self) -> Dict[str, Any]:
        """Default training configuration"""
        return {
            'learning_rate': 2e-4,
            'batch_size': 4,
            'max_epochs': 100,
            'model_name': 'mistralai/Mistral-7B-Instruct-v0.3',
            'optimizer': 'paged_adamw_8bit',
            'warmup_steps': 1000,
            'output_dir': './training_output',
            'gradient_accumulation_steps': 1,
            'weight_decay': 0.001,
            'max_grad_norm': 0.3,
            'warmup_ratio': 0.03,
            'lr_scheduler_type': 'constant'
        }
    
    def start_training(self) -> bool:
        """Start training process"""
        with self.lock:
            if self.training_active:
                return False
            
            self.training_active = True
            self.training_paused = False
            self.current_epoch = 0
            self.start_time = datetime.now()
            self.training_data = pd.DataFrame()
            
            # Start training thread
            self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
            self.training_thread.start()
            
            logger.info("Training started")
            return True
    
    def pause_training(self) -> bool:
        """Pause/resume training"""
        with self.lock:
            if not self.training_active:
                return False
            
            self.training_paused = not self.training_paused
            logger.info(f"Training {'paused' if self.training_paused else 'resumed'}")
            return True
    
    def stop_training(self) -> bool:
        """Stop training process"""
        with self.lock:
            if not self.training_active:
                return False
            
            self.training_active = False
            self.training_paused = False
            logger.info("Training stopped")
            return True
    
    def reset_training(self) -> bool:
        """Reset training state"""
        with self.lock:
            self.training_active = False
            self.training_paused = False
            self.current_epoch = 0
            self.training_data = pd.DataFrame()
            self.start_time = None
            logger.info("Training reset")
            return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get current training status"""
        with self.lock:
            elapsed_time = None
            if self.start_time:
                elapsed_time = (datetime.now() - self.start_time).total_seconds()
            
            return {
                'active': self.training_active,
                'paused': self.training_paused,
                'current_epoch': self.current_epoch,
                'max_epochs': self.max_epochs,
                'progress': self.current_epoch / self.max_epochs if self.max_epochs > 0 else 0,
                'elapsed_time': elapsed_time,
                'training_data': self.training_data.copy() if not self.training_data.empty else pd.DataFrame()
            }
    
    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """Update training configuration"""
        with self.lock:
            if self.training_active:
                return False  # Can't update config during training
            
            self.config.update(new_config)
            self.max_epochs = new_config.get('max_epochs', self.max_epochs)
            logger.info("Training configuration updated")
            return True
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        with self.lock:
            return self.config.copy()
    
    def _training_loop(self):
        """Simulated training loop - replace with actual training logic"""
        logger.info("Starting training loop")
        
        while self.training_active and self.current_epoch < self.max_epochs:
            if not self.training_paused:
                # Simulate one epoch of training
                self._simulate_epoch()
                
                with self.lock:
                    self.current_epoch += 1
                
                time.sleep(2)  # Simulate epoch duration
            else:
                time.sleep(0.1)  # Check pause state frequently
        
        with self.lock:
            if self.current_epoch >= self.max_epochs:
                self.training_active = False
                logger.info("Training completed")
    
    def _simulate_epoch(self):
        """Simulate training metrics for one epoch"""
        epoch = self.current_epoch + 1
        
        # Generate realistic training metrics
        np.random.seed(42 + epoch)  # Consistent but varying seed
        
        # Base loss starts high and decreases
        base_train_loss = 4.0 * np.exp(-0.05 * epoch) + 0.3
        base_val_loss = 4.2 * np.exp(-0.04 * epoch) + 0.4
        
        # Add noise
        train_loss = base_train_loss + np.random.normal(0, 0.05)
        val_loss = base_val_loss + np.random.normal(0, 0.08)
        
        # Accuracy improves over time
        train_accuracy = min(0.95, 1 - np.exp(-0.1 * epoch) * 0.8 + np.random.normal(0, 0.01))
        val_accuracy = min(0.92, 1 - np.exp(-0.08 * epoch) * 0.85 + np.random.normal(0, 0.015))
        
        # Learning rate schedule (decay)
        learning_rate = self.config['learning_rate'] * np.exp(-0.02 * epoch)
        
        # Add to training data
        new_row = pd.DataFrame({
            'epoch': [epoch],
            'train_loss': [max(0.1, train_loss)],  # Minimum loss
            'val_loss': [max(0.1, val_loss)],
            'train_accuracy': [max(0, min(1, train_accuracy))],  # Clamp to [0,1]
            'val_accuracy': [max(0, min(1, val_accuracy))],
            'learning_rate': [learning_rate],
            'timestamp': [datetime.now()]
        })
        
        with self.lock:
            if self.training_data.empty:
                self.training_data = new_row
            else:
                self.training_data = pd.concat([self.training_data, new_row], ignore_index=True)

# Global training manager instance
training_manager = TrainingManager()

def get_training_manager() -> TrainingManager:
    """Get the global training manager instance"""
    return training_manager