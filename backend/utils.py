import os
import streamlit as st

def get_available_models(results_dir: str = "./results"):
    """
    Scans the results directory for saved models.

    Args:
        results_dir (str): The directory where training results are saved.

    Returns:
        list: A list of model directory names.
    """
    if not os.path.isdir(results_dir):
        return []
    
    try:
        # List all entries in the directory and filter for directories
        models = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
        return models
    except Exception as e:
        st.error(f"Error reading model directory '{results_dir}': {e}")
        return []

def load_model_for_inference(model_path: str):
    """
    Loads a fine-tuned model for inference.
    (This is a placeholder for the actual implementation)
    
    Args:
        model_path (str): The path to the saved model directory.

    Returns:
        A tuple of (model, tokenizer) or (None, None) if loading fails.
    """
    st.info(f"Loading model from: {model_path}")
    # In a real application, you would use:
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    #
    # try:
    #     model = AutoModelForCausalLM.from_pretrained(model_path)
    #     tokenizer = AutoTokenizer.from_pretrained(model_path)
    #     st.success("Model and tokenizer loaded for inference.")
    #     return model, tokenizer
    # except Exception as e:
    #     st.error(f"Failed to load model: {e}")
    #     return None, None
    
    # For now, we'll just simulate it
    if os.path.exists(model_path):
        return "mock_model", "mock_tokenizer"
    return None, None

# Example usage
if __name__ == '__main__':
    st.title("Backend Utilities Test")
    
    st.subheader("Available Models")
    # Create a dummy directory for testing
    if not os.path.exists("./results/test_model_1"):
        os.makedirs("./results/test_model_1")
    
    models = get_available_models()
    if models:
        st.write(models)
    else:
        st.write("No models found in './results'.")

