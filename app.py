import streamlit as st
import pandas as pd
import time
from backend.training_manager import TrainingManager
from components.configuration import render_configuration
from components.training_dashboard import render_training_dashboard, display_training_dashboard
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

    # --- Live Refresh Control (Dashboard only) ---
    if page == "📊 Training Dashboard":
        refresh_labels = ["Off", "0.5s", "1s", "2s", "5s"]
        refresh_values = [0.0, 0.5, 1.0, 2.0, 5.0]
        try:
            default_idx = refresh_labels.index("1s")
        except ValueError:
            default_idx = 2
        sel_label = st.sidebar.selectbox("Live Refresh", options=refresh_labels, index=default_idx, help="How often the dashboard refreshes while training")
        st.session_state['_auto_refresh_interval'] = refresh_values[refresh_labels.index(sel_label)]

    # --- Main Content Area (Single Page Renderer) ---
    if page == "📊 Training Dashboard":
        display_training_dashboard()
    elif page == "⚙️ Configuration":
        render_configuration()
    elif page == "⚖️ Model Comparison":
        render_model_comparison()

    # --- Live Update Logic (Dashboard-only; periodic gentle refresh while active) ---
    if 'training_manager' in st.session_state and page == "📊 Training Dashboard":
        manager = st.session_state.training_manager
        status_snapshot = manager.get_status_snapshot() if hasattr(manager, 'get_status_snapshot') else {"active": False}
        if status_snapshot.get('active', False):
            # Periodic refresh independent of queues to ensure UI keeps up
            import time
            now = time.time()
            last = st.session_state.get('_last_auto_refresh_ts', 0.0)
            interval = float(st.session_state.get('_auto_refresh_interval', 1.0))
            # Refresh only if interval is set (> 0)
            if interval > 0.0 and (now - last) > interval:
                st.session_state['_last_auto_refresh_ts'] = now
                st.rerun()


if __name__ == "__main__":
    main()
