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
from utils.data_generator import generate_training_data, generate_model_comparison_data
from utils.styles import apply_custom_styles

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
if 'training_active' not in st.session_state:
    st.session_state.training_active = False
if 'training_paused' not in st.session_state:
    st.session_state.training_paused = False
if 'current_epoch' not in st.session_state:
    st.session_state.current_epoch = 0
if 'training_data' not in st.session_state:
    st.session_state.training_data = generate_training_data()
if 'model_data' not in st.session_state:
    st.session_state.model_data = generate_model_comparison_data()
if 'config' not in st.session_state:
    st.session_state.config = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'max_epochs': 100,
        'model_name': 'llama-2-7b',
        'optimizer': 'AdamW',
        'warmup_steps': 1000
    }
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

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
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["Training Dashboard", "Model Comparison", "Configuration"]
    )
    
    # Pipeline status in sidebar
    st.sidebar.markdown("### Pipeline Status")
    if st.session_state.training_active and not st.session_state.training_paused:
        st.sidebar.success("🟢 Training Active")
    elif st.session_state.training_paused:
        st.sidebar.warning("🟡 Training Paused")
    else:
        st.sidebar.info("🔵 Training Idle")
    
    st.sidebar.markdown(f"**Current Epoch:** {st.session_state.current_epoch}")
    st.sidebar.markdown(f"**Model:** {st.session_state.config['model_name']}")
    st.sidebar.markdown(f"**Learning Rate:** {st.session_state.config['learning_rate']}")
    
    # Render selected page
    if page == "Training Dashboard":
        render_training_dashboard()
    elif page == "Model Comparison":
        render_model_comparison()
    elif page == "Configuration":
        render_configuration()
    
    # Auto-refresh for real-time updates when training is active
    if st.session_state.training_active and not st.session_state.training_paused:
        time.sleep(1)
        st.session_state.current_epoch += 1
        if st.session_state.current_epoch >= st.session_state.config['max_epochs']:
            st.session_state.training_active = False
            st.success("Training completed!")
        st.rerun()

if __name__ == "__main__":
    main()
