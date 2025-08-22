import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from utils.data_generator import update_training_data

def render_training_dashboard():
    st.title("🚀 Training Dashboard")
    st.markdown("Real-time monitoring of LLM training pipeline")
    
    # Control buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("▶️ Start Training", type="primary", use_container_width=True):
            st.session_state.training_active = True
            st.session_state.training_paused = False
            st.success("Training started!")
    
    with col2:
        if st.button("⏸️ Pause Training", use_container_width=True):
            if st.session_state.training_active:
                st.session_state.training_paused = not st.session_state.training_paused
                if st.session_state.training_paused:
                    st.warning("Training paused!")
                else:
                    st.info("Training resumed!")
    
    with col3:
        if st.button("⏹️ Stop Training", use_container_width=True):
            st.session_state.training_active = False
            st.session_state.training_paused = False
            st.error("Training stopped!")
    
    with col4:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.current_epoch = 0
            st.session_state.training_active = False
            st.session_state.training_paused = False
            st.info("Training reset!")
    
    st.markdown("---")
    
    # Update training data if training is active
    if st.session_state.training_active:
        st.session_state.training_data = update_training_data(
            st.session_state.training_data, 
            st.session_state.current_epoch
        )
    
    # Progress metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Epoch", 
            st.session_state.current_epoch,
            delta=1 if st.session_state.training_active and not st.session_state.training_paused else 0
        )
    
    with col2:
        current_loss = st.session_state.training_data['train_loss'].iloc[-1] if not st.session_state.training_data.empty else 0.0
        st.metric(
            "Training Loss", 
            f"{current_loss:.4f}",
            delta=f"{-0.001:.4f}" if st.session_state.training_active else None
        )
    
    with col3:
        current_acc = st.session_state.training_data['train_accuracy'].iloc[-1] if not st.session_state.training_data.empty else 0.0
        st.metric(
            "Training Accuracy", 
            f"{current_acc:.2%}",
            delta=f"{0.005:.3f}" if st.session_state.training_active else None
        )
    
    with col4:
        progress = st.session_state.current_epoch / st.session_state.config['max_epochs']
        st.metric(
            "Progress", 
            f"{progress:.1%}",
            delta=f"{1/st.session_state.config['max_epochs']:.3f}" if st.session_state.training_active else None
        )
    
    # Progress bar
    st.progress(progress)
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Loss Curves")
        
        # Loss chart
        fig_loss = go.Figure()
        
        if not st.session_state.training_data.empty:
            fig_loss.add_trace(go.Scatter(
                x=st.session_state.training_data['epoch'],
                y=st.session_state.training_data['train_loss'],
                mode='lines',
                name='Training Loss',
                line=dict(color='#6366F1', width=3)
            ))
            
            fig_loss.add_trace(go.Scatter(
                x=st.session_state.training_data['epoch'],
                y=st.session_state.training_data['val_loss'],
                mode='lines',
                name='Validation Loss',
                line=dict(color='#8B5CF6', width=3)
            ))
        
        fig_loss.update_layout(
            title="Training & Validation Loss",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            height=400,
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_loss, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Accuracy Metrics")
        
        # Accuracy chart
        fig_acc = go.Figure()
        
        if not st.session_state.training_data.empty:
            fig_acc.add_trace(go.Scatter(
                x=st.session_state.training_data['epoch'],
                y=st.session_state.training_data['train_accuracy'],
                mode='lines',
                name='Training Accuracy',
                line=dict(color='#10B981', width=3)
            ))
            
            fig_acc.add_trace(go.Scatter(
                x=st.session_state.training_data['epoch'],
                y=st.session_state.training_data['val_accuracy'],
                mode='lines',
                name='Validation Accuracy',
                line=dict(color='#F59E0B', width=3)
            ))
        
        fig_acc.update_layout(
            title="Training & Validation Accuracy",
            xaxis_title="Epoch",
            yaxis_title="Accuracy",
            height=400,
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_acc, use_container_width=True)
    
    # Learning rate schedule
    st.subheader("📊 Learning Rate Schedule")
    
    fig_lr = go.Figure()
    
    if not st.session_state.training_data.empty:
        fig_lr.add_trace(go.Scatter(
            x=st.session_state.training_data['epoch'],
            y=st.session_state.training_data['learning_rate'],
            mode='lines',
            name='Learning Rate',
            line=dict(color='#EF4444', width=3),
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
    
    st.plotly_chart(fig_lr, use_container_width=True)
    
    # Training logs
    st.subheader("📝 Training Logs")
    
    if st.session_state.training_active and not st.session_state.training_paused:
        st.info(f"🔄 Epoch {st.session_state.current_epoch}: Training in progress...")
    elif st.session_state.training_paused:
        st.warning(f"⏸️ Epoch {st.session_state.current_epoch}: Training paused")
    elif st.session_state.current_epoch > 0:
        st.success(f"✅ Last completed epoch: {st.session_state.current_epoch}")
    else:
        st.info("⏳ Training not started yet")
    
    # Display recent training data
    if not st.session_state.training_data.empty:
        st.dataframe(
            st.session_state.training_data.tail(10),
            use_container_width=True
        )
