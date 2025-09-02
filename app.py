import streamlit as st
import pandas as pd
import time
from backend.training_manager import TrainingManager
from components.configuration import render_configuration
from components.training_dashboard import render_training_dashboard, display_training_dashboard
from components.model_comparison import render_model_comparison
from utils.styles import load_css
from utils.chart_themes import setup_altair_theme
from streamlit_option_menu import option_menu
from components.home import render_home

def main():
    """
    Main function to run the Streamlit application.
    """
    # --- Page Configuration ---
    st.set_page_config(
        page_title="AI Factory - LLM Training Suite",
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # --- Theme & Styles ---
    # Default to dark theme for a modern aesthetic unless user chose otherwise
    if 'theme' not in st.session_state:
        st.session_state['theme'] = 'dark'
    load_css(theme=st.session_state['theme'])
    try:
        setup_altair_theme()
    except Exception:
        pass

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
    with st.sidebar:
        st.image("https://i.imgur.com/MhW3f0i.png", width=80)
        st.title("AI Factory")
        st.info(
            """
            Welcome to the AI Factory, your all-in-one suite for fine-tuning Large Language Models. 
            
            **Follow these steps:**
            1.  **Configure:** Set up your model, dataset, and training parameters.
            2.  **Train:** Start the training process and monitor its progress.
            3.  **Compare:** Analyze the performance of different models.
            """
        )

        with st.expander("Current Configuration"):
            try:
                mgr = st.session_state.training_manager
                cfg = mgr.get_config() if hasattr(mgr, 'get_config') else None
                if cfg:
                    st.markdown(f"**Model:** `{cfg.get('model_name', '-')}`")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("LR", f"{cfg.get('learning_rate', '-')}")
                    with col_b:
                        st.metric("Batch", f"{cfg.get('batch_size', '-')}")
                    col_c, col_d = st.columns(2)
                    with col_c:
                        st.metric("Epochs", f"{cfg.get('max_epochs', '-')}")
                    with col_d:
                        st.metric("Optimizer", str(cfg.get('optimizer', '-')))
                else:
                    st.info("No configuration loaded yet.")
            except Exception:
                st.info("No configuration loaded yet.")

        page = option_menu(
            menu_title="Navigation",
            options=["Home", "Training Dashboard", "Configuration", "Model Comparison"],
            icons=["house", "clipboard-data", "gear", "bar-chart-steps"],
            menu_icon="cast",
            default_index=0,
        )

        with st.expander("Settings"):
            # Theme selector
            theme_label_to_key = {"🌙 Dark": "dark", "🔆 Light": "light"}
            current_theme_key = st.session_state.get('theme', 'dark')
            current_label = "🌙 Dark" if current_theme_key == 'dark' else "🔆 Light"
            chosen_label = st.selectbox("Theme", options=["🌙 Dark", "🔆 Light"], index=["🌙 Dark", "🔆 Light"].index(current_label))
            new_theme = theme_label_to_key[chosen_label]
            if new_theme != current_theme_key:
                st.session_state['theme'] = new_theme
                load_css(theme=new_theme)
                st.rerun()

        # Live snapshot visible on all pages when training is active
        with st.expander("Live Snapshot"):
            try:
                mgr = st.session_state.training_manager
                status = mgr.get_status_snapshot() if hasattr(mgr, 'get_status_snapshot') else {"active": False}
                if status.get('active', False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Epoch", f"{status.get('current_epoch', 0)}")
                    with col2:
                        step = status.get('current_step')
                        total = status.get('total_steps')
                        st.metric("Step", f"{step or 0}/{total or '-'}")
                    # Losses
                    current_loss = None
                    val_loss = None
                    acc = None
                    eta_txt = None
                    try:
                        df = status.get('training_data')
                        if df is not None and not df.empty:
                            if 'train_loss' in df.columns and df['train_loss'].notna().any():
                                current_loss = float(df['train_loss'].dropna().iloc[-1])
                            if 'val_loss' in df.columns and df['val_loss'].notna().any():
                                val_loss = float(df['val_loss'].dropna().iloc[-1])
                            # pick best available accuracy
                            if 'val_accuracy' in df.columns and df['val_accuracy'].notna().any():
                                acc = float(df['val_accuracy'].dropna().iloc[-1])
                            elif 'train_accuracy' in df.columns and df['train_accuracy'].notna().any():
                                acc = float(df['train_accuracy'].dropna().iloc[-1])
                    except Exception:
                        pass
                    col3, col4 = st.columns(2)
                    with col3:
                        st.metric("Loss", f"{current_loss:.4f}" if current_loss is not None else "-")
                    with col4:
                        st.metric("Val Loss", f"{val_loss:.4f}" if val_loss is not None else "-")
                    # Accuracy and ETA
                    try:
                        eta_seconds = status.get('eta_seconds')
                        if eta_seconds:
                            import math
                            m, s = divmod(int(math.ceil(eta_seconds)), 60)
                            h, m = divmod(m, 60)
                            eta_txt = f"{h:02d}:{m:02d}:{s:02d}"
                    except Exception:
                        eta_txt = None
                    col5, col6 = st.columns(2)
                    with col5:
                        st.metric("Accuracy", f"{acc*100:.2f}%" if acc is not None else "-")
                    with col6:
                        st.metric("ETA", eta_txt or "-")
                else:
                    st.info("Training not active.")
            except Exception:
                st.info("No live data yet.")

        if page == "Training Dashboard":
            with st.expander("Dashboard Controls"):
                refresh_labels = ["Off", "0.5s", "1s", "2s", "5s"]
                refresh_values = [0.0, 0.5, 1.0, 2.0, 5.0]
                try:
                    default_idx = refresh_labels.index("1s")
                except ValueError:
                    default_idx = 2
                sel_label = st.selectbox("Live Refresh", options=refresh_labels, index=default_idx, help="How often the dashboard refreshes while training")
                st.session_state['_auto_refresh_interval'] = refresh_values[refresh_labels.index(sel_label)]

    # --- Main Content Area (Single Page Renderer) ---
    if page == "Home":
        st.header("🏠 Welcome to the AI Factory LLM Trainer")
        render_home()
    elif page == "Training Dashboard":
        st.header("📊 Training Dashboard")
        display_training_dashboard()
    elif page == "Configuration":
        st.header("⚙️ Configuration")
        render_configuration()
    elif page == "Model Comparison":
        st.header("⚖️ Model Comparison")
        render_model_comparison()

    # --- Live Update Logic (Dashboard-only; periodic gentle refresh while active) ---
    if 'training_manager' in st.session_state and page == "Training Dashboard":
        manager = st.session_state.training_manager
        status_snapshot = manager.get_status_snapshot() if hasattr(manager, 'get_status_snapshot') else {"active": False}
        # Notify when training completes
        if not status_snapshot.get('active', False) and st.session_state.get('training_started', False):
            st.success("🎉 Training complete! You can now view the final results.")
            st.session_state['training_started'] = False
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
