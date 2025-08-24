import streamlit as st
import pandas as pd
import time
from backend.training_manager import TrainingManager
from components.configuration import render_configuration
from components.training_dashboard import render_training_dashboard
from components.model_comparison import render_model_comparison
from utils.styles import load_css

def main():
    """
    Main function to run the Streamlit application.
    """
    # --- Page Configuration ---
    st.set_page_config(
        page_title="AI Factory Visualizer",
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # --- Load Custom CSS ---
    load_css()

    # --- Initialize Session State ---
    if "training_manager" not in st.session_state:
        st.session_state.training_manager = TrainingManager()
    if "training_started" not in st.session_state:
        st.session_state.training_started = False
    if "training_logs" not in st.session_state:
        st.session_state.training_logs = []
    if "metrics_data" not in st.session_state:
        st.session_state.metrics_data = pd.DataFrame()
    # Initialize model comparison dataset if missing
    if "model_data" not in st.session_state:
        st.session_state.model_data = pd.DataFrame({
            'model_name': [
                'mistralai/Mistral-7B-Instruct-v0.3',
                'meta-llama/Llama-3.1-8B-Instruct',
                'Qwen/Qwen2.5-7B-Instruct',
            ],
            'final_loss': [1.85, 1.72, 1.78],
            'final_accuracy': [0.74, 0.77, 0.76],
            'training_time': [3.5, 4.2, 3.8],
            'memory_usage': [14.0, 18.5, 16.0],
            'parameters': [7e9, 8e9, 7e9],
        })

    # --- Sidebar Navigation ---
    st.sidebar.title("🏭 AI Factory LLM Trainer")
    st.sidebar.markdown("A visual interface for fine-tuning and monitoring LLMs.")
    page = st.sidebar.radio(
        "Navigation",
        [
            "📊 Training Dashboard",
            "⚙️ Configuration",
            "⚖️ Model Comparison",
        ],
        index=0,
    )

    # --- Main Content Area (Single Page Renderer) ---
    if page == "📊 Training Dashboard":
        render_training_dashboard()
    elif page == "⚙️ Configuration":
        render_configuration()
    elif page == "⚖️ Model Comparison":
        render_model_comparison()

    # --- Live Update Logic (if training is running) ---
    # --- Live Update Logic (only on Dashboard when training is active) ---
    if 'training_manager' in st.session_state and page == "📊 Training Dashboard":
        manager = st.session_state.training_manager
        status = manager.get_status() if hasattr(manager, 'get_status') else {"active": False}
        if status.get('active', False):
            new_logs = manager.get_logs()
            if new_logs:
                st.session_state.training_logs.extend(new_logs)

            new_metrics = manager.get_metrics()
            if not new_metrics.empty:
                st.session_state.metrics_data = pd.concat(
                    [st.session_state.metrics_data, new_metrics]
                ).drop_duplicates(subset=['step'], keep='last')

            time.sleep(5)
            st.rerun()


if __name__ == "__main__":
    main()
