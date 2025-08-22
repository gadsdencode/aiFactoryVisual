import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_training_data(epochs=0):
    """Generate mock training data for demonstration purposes as requested in success criteria."""
    if epochs == 0:
        return pd.DataFrame()
    
    np.random.seed(42)  # For reproducible results
    
    epochs_range = range(1, epochs + 1)
    
    # Generate realistic loss curves with some noise
    base_train_loss = 4.0 * np.exp(-0.1 * np.array(epochs_range)) + 0.5
    base_val_loss = 4.2 * np.exp(-0.08 * np.array(epochs_range)) + 0.6
    
    train_loss = base_train_loss + np.random.normal(0, 0.05, len(epochs_range))
    val_loss = base_val_loss + np.random.normal(0, 0.08, len(epochs_range))
    
    # Generate accuracy curves (inverse of loss with some adjustments)
    train_accuracy = 1 - np.exp(-0.15 * np.array(epochs_range)) * 0.8 + np.random.normal(0, 0.01, len(epochs_range))
    val_accuracy = 1 - np.exp(-0.12 * np.array(epochs_range)) * 0.85 + np.random.normal(0, 0.015, len(epochs_range))
    
    # Ensure accuracy is between 0 and 1
    train_accuracy = np.clip(train_accuracy, 0, 1)
    val_accuracy = np.clip(val_accuracy, 0, 1)
    
    # Generate learning rate schedule (decay)
    learning_rate = 0.001 * np.exp(-0.02 * np.array(epochs_range))
    
    data = pd.DataFrame({
        'epoch': epochs_range,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
        'learning_rate': learning_rate,
        'timestamp': [datetime.now() - timedelta(minutes=i) for i in reversed(range(len(epochs_range)))]
    })
    
    return data

def update_training_data(current_data, current_epoch):
    """Update training data with new epoch information."""
    if current_epoch == 0:
        return pd.DataFrame()
    
    # If we don't have data yet or need to extend it
    if current_data.empty or current_epoch > len(current_data):
        return generate_training_data(current_epoch)
    
    return current_data

def generate_model_comparison_data():
    """Generate mock model comparison data for demonstration purposes as requested in success criteria."""
    np.random.seed(123)
    
    models = [
        "llama-2-7b", "llama-2-13b", "mistral-7b", "codellama-7b", 
        "vicuna-7b", "alpaca-7b", "gpt-3.5-turbo-ft", "claude-instant-ft"
    ]
    
    data = []
    
    for model in models:
        # Generate realistic but varied performance metrics
        if "7b" in model:
            base_params = 7e9
            base_memory = 14
            base_time = np.random.uniform(2, 4)
        elif "13b" in model:
            base_params = 13e9
            base_memory = 26
            base_time = np.random.uniform(4, 7)
        else:
            base_params = np.random.uniform(6e9, 15e9)
            base_memory = np.random.uniform(12, 30)
            base_time = np.random.uniform(2, 6)
        
        final_loss = np.random.uniform(0.8, 1.5)
        final_accuracy = 1 - (final_loss / 4.0) + np.random.uniform(-0.1, 0.1)
        final_accuracy = np.clip(final_accuracy, 0.6, 0.95)
        
        data.append({
            'model_name': model,
            'parameters': int(base_params),
            'final_loss': final_loss,
            'final_accuracy': final_accuracy,
            'training_time': base_time,
            'memory_usage': base_memory,
            'convergence_epoch': np.random.randint(30, 80),
            'best_val_loss': final_loss + np.random.uniform(0.05, 0.2)
        })
    
    return pd.DataFrame(data)
