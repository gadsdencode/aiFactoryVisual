import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from utils.chart_themes import apply_chart_theme, get_chart_theme
from backend.training_manager import get_training_manager

def render_training_dashboard():
    st.title("🚀 Training Dashboard")
    st.markdown("Real-time monitoring of LLM training pipeline")
    
    # Get training manager
    training_manager = get_training_manager()
    
    # Control buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("▶️ Start Training", type="primary", use_container_width=True):
            if training_manager.start_training():
                st.success("Training started!")
            else:
                st.warning("Training is already active!")
    
    with col2:
        if st.button("⏸️ Pause Training", use_container_width=True):
            if training_manager.pause_training():
                status = training_manager.get_status()
                if status['paused']:
                    st.warning("Training paused!")
                else:
                    st.info("Training resumed!")
            else:
                st.warning("No active training to pause/resume!")
    
    with col3:
        if st.button("⏹️ Stop Training", use_container_width=True):
            if training_manager.stop_training():
                st.error("Training stopped!")
            else:
                st.warning("No active training to stop!")
    
    with col4:
        if st.button("🔄 Reset", use_container_width=True):
            training_manager.reset_training()
            st.info("Training reset!")
    
    st.markdown("---")
    
    # Get current training status
    status = training_manager.get_status()
    training_data = status['training_data']
    
    # Progress metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Epoch", 
            status['current_epoch'],
            delta=1 if status['active'] and not status['paused'] else 0
        )
    
    with col2:
        current_loss = training_data['train_loss'].iloc[-1] if not training_data.empty else 0.0
        st.metric(
            "Training Loss", 
            f"{current_loss:.4f}",
            delta=f"{-0.001:.4f}" if status['active'] else None
        )
    
    with col3:
        current_acc = training_data['train_accuracy'].iloc[-1] if not training_data.empty else 0.0
        st.metric(
            "Training Accuracy", 
            f"{current_acc:.2%}",
            delta=f"{0.005:.3f}" if status['active'] else None
        )
    
    with col4:
        st.metric(
            "Progress", 
            f"{status['progress']:.1%}",
            delta=f"{1/status['max_epochs']:.3f}" if status['active'] else None
        )
    
    # Progress bar
    st.progress(status['progress'])
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Loss Curves")
        
        # Loss chart
        fig_loss = go.Figure()
        theme = get_chart_theme()
        
        if not training_data.empty:
            fig_loss.add_trace(go.Scatter(
                x=training_data['epoch'],
                y=training_data['train_loss'],
                mode='lines',
                name='Training Loss',
                line=dict(color=theme['line_colors'][0], width=3)
            ))
            
            fig_loss.add_trace(go.Scatter(
                x=training_data['epoch'],
                y=training_data['val_loss'],
                mode='lines',
                name='Validation Loss',
                line=dict(color=theme['line_colors'][1], width=3)
            ))
        
        fig_loss.update_layout(
            title="Training & Validation Loss",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            height=400,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig_loss = apply_chart_theme(fig_loss)
        
        st.plotly_chart(fig_loss, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Accuracy Metrics")
        
        # Accuracy chart
        fig_acc = go.Figure()
        
        if not training_data.empty:
            fig_acc.add_trace(go.Scatter(
                x=training_data['epoch'],
                y=training_data['train_accuracy'],
                mode='lines',
                name='Training Accuracy',
                line=dict(color=theme['line_colors'][2], width=3)
            ))
            
            fig_acc.add_trace(go.Scatter(
                x=training_data['epoch'],
                y=training_data['val_accuracy'],
                mode='lines',
                name='Validation Accuracy',
                line=dict(color=theme['line_colors'][3], width=3)
            ))
        
        fig_acc.update_layout(
            title="Training & Validation Accuracy",
            xaxis_title="Epoch",
            yaxis_title="Accuracy",
            height=400,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig_acc = apply_chart_theme(fig_acc)
        
        st.plotly_chart(fig_acc, use_container_width=True)
    
    # Learning rate schedule
    st.subheader("📊 Learning Rate Schedule")
    
    fig_lr = go.Figure()
    
    if not training_data.empty:
        fig_lr.add_trace(go.Scatter(
            x=training_data['epoch'],
            y=training_data['learning_rate'],
            mode='lines',
            name='Learning Rate',
            line=dict(color=theme['line_colors'][4], width=3),
            fill='tozeroy',
            fillcolor='rgba(239, 68, 68, 0.1)'
        ))
    
    fig_lr.update_layout(
        title="Learning Rate Schedule",
        xaxis_title="Epoch",
        yaxis_title="Learning Rate",
        height=300,
        showlegend=False
    )
    
    fig_lr = apply_chart_theme(fig_lr)
    
    st.plotly_chart(fig_lr, use_container_width=True)
    
    # Training logs
    st.subheader("📝 Training Logs")
    
    if status['active'] and not status['paused']:
        st.info(f"🔄 Epoch {status['current_epoch']}: Training in progress...")
    elif status['paused']:
        st.warning(f"⏸️ Epoch {status['current_epoch']}: Training paused")
    elif status['current_epoch'] > 0:
        st.success(f"✅ Last completed epoch: {status['current_epoch']}")
    else:
        st.info("⏳ Training not started yet")
    
    # Display recent training data
    if not training_data.empty:
        display_data = training_data.tail(10).copy()
        # Format columns for better display
        if 'timestamp' in display_data.columns:
            display_data['timestamp'] = display_data['timestamp'].dt.strftime('%H:%M:%S')
        st.dataframe(
            display_data,
            use_container_width=True
        )
