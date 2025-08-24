import streamlit as st
import pandas as pd
import time
from backend.training_manager import TrainingManager
from components.configuration import configuration_sidebar
from components.training_dashboard import training_dashboard
from components.model_comparison import model_comparison_tab
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

    # --- App Title ---
    st.title("🏭 AI Factory LLM Trainer")
    st.markdown("A visual interface for fine-tuning and monitoring Language Models.")

    # --- Sidebar for Configuration ---
    configuration_sidebar()

    # --- Main Content Area ---
    tab1, tab2 = st.tabs(["📊 Training Dashboard", "⚖️ Model Comparison"])

    with tab1:
        # Pass the placeholder to the dashboard function
        log_placeholder = st.empty()
        metrics_placeholder = st.empty()
        training_dashboard(log_placeholder, metrics_placeholder)

    with tab2:
        model_comparison_tab()

    # --- Live Update Logic (if training is running) ---
    if st.session_state.training_started:
        # This part of the script will re-run periodically if training is active
        # We need a mechanism to fetch live data from the TrainingManager
        # For now, we simulate this with a loop and sleep
        # In a real scenario, this would involve checking a queue or a file
        
        # This is a placeholder for a more robust live update mechanism
        # A background thread in TrainingManager would be a better approach
        
        # Get latest logs and metrics from the manager
        manager = st.session_state.training_manager
        
        new_logs = manager.get_logs()
        if new_logs:
            st.session_state.training_logs.extend(new_logs)

        new_metrics = manager.get_metrics()
        if not new_metrics.empty:
            st.session_state.metrics_data = pd.concat(
                [st.session_state.metrics_data, new_metrics]
            ).drop_duplicates(subset=['step'], keep='last')

        # Rerun to update the UI
        time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    main()
