from datasets import load_dataset, Dataset
from backend.config import AppConfig
import streamlit as st

@st.cache_data
def load_and_prepare_dataset(config: AppConfig) -> Dataset:
    """
    Loads a dataset from the Hugging Face Hub.

    Args:
        config (AppConfig): The application configuration object.

    Returns:
        Dataset: The loaded dataset.
    """
    try:
        dataset = load_dataset(config.dataset_name, split=config.dataset_split)
        
        # Verify that the specified text column exists
        if config.text_column not in dataset.column_names:
            raise ValueError(
                f"The specified text_column '{config.text_column}' was not found in the dataset. "
                f"Available columns are: {dataset.column_names}"
            )
            
        st.success(f"Successfully loaded dataset '{config.dataset_name}' with {len(dataset)} examples.")
        return dataset
    except Exception as e:
        st.error(f"Failed to load dataset '{config.dataset_name}': {e}")
        return None

# Example usage:
if __name__ == '__main__':
    from backend.config import load_config
    
    # This demonstrates how to use the function.
    # It requires a valid config.yaml file in the root directory.
    try:
        config = load_config()
        dataset = load_and_prepare_dataset(config)
        if dataset:
            print("\nDataset loaded.")
            print("First 5 rows:")
            print(dataset[:5])
            print("\nDataset features:")
            print(dataset.features)
    except Exception as e:
        print(f"An error occurred: {e}")
