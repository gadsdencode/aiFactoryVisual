import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

# Import custom components
from components.training_dashboard import render_training_dashboard
from components.model_comparison import render_model_comparison
from components.configuration import render_configuration
from utils.data_generator import generate_model_comparison_data
from utils.styles import apply_custom_styles
from backend.training_manager import get_training_manager

# Configure page
st.set_page_config(
    page_title="LLM Training Pipeline Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styles
apply_custom_styles(st.session_state.get('theme', 'light'))

# Initialize session state
if 'model_data' not in st.session_state:
    st.session_state.model_data = generate_model_comparison_data()
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Training Dashboard'

def main():
    # Sidebar navigation
    st.sidebar.title("🤖 LLM Training Dashboard")
    st.sidebar.markdown("---")
    
    # Theme toggle
    theme_col1, theme_col2 = st.sidebar.columns(2)
    with theme_col1:
        if st.button("🌙 Dark", disabled=st.session_state.theme == 'dark'):
            st.session_state.theme = 'dark'
            st.rerun()
    with theme_col2:
        if st.button("☀️ Light", disabled=st.session_state.theme == 'light'):
            st.session_state.theme = 'light'
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # Navigation
    page_options = ["Training Dashboard", "Model Comparison", "Configuration"]
    current_index = page_options.index(st.session_state.current_page) if st.session_state.current_page in page_options else 0
    
    page = st.sidebar.selectbox(
        "Navigate to:",
        page_options,
        index=current_index
    )
    
    # Update session state if page changed
    if page != st.session_state.current_page:
        st.session_state.current_page = page
    
    # Pipeline status in sidebar
    training_manager = get_training_manager()
    status = training_manager.get_status()
    config = training_manager.get_config()
    
    st.sidebar.markdown("### Pipeline Status")
    if status['active'] and not status['paused']:
        st.sidebar.success("🟢 Training Active")
    elif status['paused']:
        st.sidebar.warning("🟡 Training Paused")
    else:
        st.sidebar.info("🔵 Training Idle")
    
    st.sidebar.markdown(f"**Current Epoch:** {status['current_epoch']}")
    st.sidebar.markdown(f"**Model:** {config['model_name']}")
    st.sidebar.markdown(f"**Learning Rate:** {config['learning_rate']}")
    
    # Render selected page
    if page == "Training Dashboard":
        render_training_dashboard()
    elif page == "Model Comparison":
        render_model_comparison()
    elif page == "Configuration":
        render_configuration()
    
    # Auto-refresh for real-time updates when training is active
    if status['active'] and not status['paused']:
        time.sleep(2)
        st.rerun()

if __name__ == "__main__":
    main()
