import streamlit as st
from huggingface_hub import HfApi, HfFolder, login
from backend.config import AppConfig


class HuggingFaceManager:
    """
    Lightweight helper for querying Hugging Face Hub, tailored for the UI.
    """
    def __init__(self):
        self._api = HfApi()

    def get_popular_llm_models(self):
        # Curated list; adjust as needed
        return [
            "mistralai/Mistral-7B-Instruct-v0.3",
            "meta-llama/Llama-3.1-8B-Instruct",
            "google/gemma-2-9b-it",
            "Qwen/Qwen2.5-7B-Instruct",
        ]

    def validate_model(self, model_name: str, token: str | None = None):
        try:
            api = HfApi(token=token) if token else self._api
            info = api.model_info(model_name)
            # Build a minimal info dict the UI can render
            model_info = {
                'name': model_name,
                'downloads': getattr(info, 'downloads', 0) or 0,
                'likes': getattr(info, 'likes', 0) or 0,
                'model_type': getattr(info, 'pipeline_tag', 'unknown') or 'unknown',
                'size_estimate': getattr(info, 'sha', 'unknown')[:8] if getattr(info, 'sha', None) else 'unknown',
            }
            return True, "Model is available on Hugging Face", model_info
        except Exception as e:
            return False, f"Model validation failed: {e}", None

    def search_models(self, query: str, limit: int = 15, token: str | None = None):
        try:
            api = HfApi(token=token) if token else self._api
            results = api.list_models(search=query, limit=limit)
            out = []
            for r in results:
                out.append({
                    'name': getattr(r, 'modelId', None) or getattr(r, 'id', ''),
                    'downloads': getattr(r, 'downloads', 0) or 0,
                    'likes': getattr(r, 'likes', 0) or 0,
                })
            # Sort by downloads desc if present
            out.sort(key=lambda x: x.get('downloads', 0), reverse=True)
            return out
        except Exception:
            return []

    def is_model_suitable_for_fine_tuning(self, model_info: dict):
        # Simple heuristic based on name/type
        name = model_info.get('name', '').lower()
        model_type = model_info.get('model_type', '').lower()
        if any(x in name for x in ["instruct", "chat", "qwen", "llama", "mistral", "gemma"]):
            return True, "Suitable model family for instruction tuning"
        if model_type in ["text-generation", "text2text-generation", "causal-lm"]:
            return True, "Suitable pipeline for fine-tuning"
        return False, "Model may not be ideal for causal LM fine-tuning"


def get_hf_manager() -> HuggingFaceManager:
    if 'hf_manager' not in st.session_state or not isinstance(st.session_state.hf_manager, HuggingFaceManager):
        st.session_state.hf_manager = HuggingFaceManager()
    return st.session_state.hf_manager

def hf_login(token: str):
    """
    Logs into the Hugging Face Hub.

    Args:
        token (str): The Hugging Face API token.
    """
    if not token:
        st.warning("No Hugging Face token provided. Skipping login.")
        return
    try:
        login(token=token, add_to_git_credential=True)
        st.success("Successfully logged into Hugging Face Hub.")
    except Exception as e:
        st.error(f"Hugging Face login failed: {e}")

def push_to_hub(config: AppConfig, model, tokenizer):
    """
    Pushes the fine-tuned model and tokenizer to the Hugging Face Hub.

    Args:
        config (AppConfig): The application configuration.
        model: The fine-tuned model object.
        tokenizer: The tokenizer object.
    """
    if not config.huggingface.push_to_hub:
        st.info("Push to Hub is disabled in the configuration.")
        return

    if not config.huggingface.hub_model_id:
        st.error("Cannot push to Hub: 'hub_model_id' is not set in the configuration.")
        return

    try:
        st.info(f"Pushing model to Hub repository: {config.huggingface.hub_model_id}")
        
        # Push model and tokenizer
        model.push_to_hub(config.huggingface.hub_model_id, use_temp_dir=False)
        tokenizer.push_to_hub(config.huggingface.hub_model_id, use_temp_dir=False)
        
        st.success(f"Model and tokenizer successfully pushed to {config.huggingface.hub_model_id}")

    except Exception as e:
        st.error(f"Failed to push model to Hub: {e}")

# Example usage (for demonstration purposes)
if __name__ == '__main__':
    # This is a conceptual example. It won't run without a valid token and trained model.
    st.title("Hugging Face Integration Test")

    hf_token = st.text_input("Enter your Hugging Face Token:", type="password")

    if st.button("Login to Hugging Face"):
        if hf_token:
            hf_login(hf_token)
        else:
            st.warning("Please enter a token.")

    # You would typically call push_to_hub from your training script after training is complete.
