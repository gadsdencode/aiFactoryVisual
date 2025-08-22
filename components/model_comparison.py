import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from utils.chart_themes import apply_chart_theme, get_chart_theme

def render_model_comparison():
    st.title("🔍 Model Comparison")
    st.markdown("Compare performance across different model configurations")
    
    # Model selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_models = st.multiselect(
            "Select models to compare:",
            options=st.session_state.model_data['model_name'].unique(),
            default=st.session_state.model_data['model_name'].unique()[:3]
        )
    
    with col2:
        metric_to_compare = st.selectbox(
            "Primary metric:",
            ["final_loss", "final_accuracy", "training_time", "memory_usage"]
        )
    
    if not selected_models:
        st.warning("Please select at least one model to compare.")
        return
    
    # Filter data for selected models
    filtered_data = st.session_state.model_data[
        st.session_state.model_data['model_name'].isin(selected_models)
    ]
    
    st.markdown("---")
    
    # Performance comparison metrics
    st.subheader("📊 Performance Overview")
    
    cols = st.columns(len(selected_models))
    for i, model in enumerate(selected_models):
        model_data = filtered_data[filtered_data['model_name'] == model].iloc[0]
        
        with cols[i]:
            st.markdown(f"### {model}")
            st.metric("Final Loss", f"{model_data['final_loss']:.4f}")
            st.metric("Final Accuracy", f"{model_data['final_accuracy']:.2%}")
            st.metric("Training Time", f"{model_data['training_time']:.1f}h")
            st.metric("Memory Usage", f"{model_data['memory_usage']:.1f}GB")
    
    st.markdown("---")
    
    # Comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Model Performance Comparison")
        
        # Bar chart comparing the selected metric
        fig_bar = go.Figure()
        theme = get_chart_theme()
        
        fig_bar.add_trace(go.Bar(
            x=filtered_data['model_name'],
            y=filtered_data[metric_to_compare],
            marker_color=theme['line_colors'][:len(selected_models)],
            text=filtered_data[metric_to_compare].round(4),
            textposition='auto'
        ))
        
        fig_bar.update_layout(
            title=f"{metric_to_compare.replace('_', ' ').title()} Comparison",
            xaxis_title="Model",
            yaxis_title=metric_to_compare.replace('_', ' ').title(),
            height=400
        )
        
        fig_bar = apply_chart_theme(fig_bar)
        
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        st.subheader("⚡ Efficiency Analysis")
        
        # Scatter plot: Accuracy vs Training Time
        fig_scatter = go.Figure()
        
        for model in selected_models:
            model_data = filtered_data[filtered_data['model_name'] == model]
            fig_scatter.add_trace(go.Scatter(
                x=model_data['training_time'],
                y=model_data['final_accuracy'],
                mode='markers+text',
                name=model,
                text=model,
                textposition='top center',
                marker=dict(size=15, opacity=0.8)
            ))
        
        fig_scatter.update_layout(
            title="Accuracy vs Training Time",
            xaxis_title="Training Time (hours)",
            yaxis_title="Final Accuracy",
            height=400,
            showlegend=False
        )
        
        fig_scatter = apply_chart_theme(fig_scatter)
        
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Detailed comparison table
    st.subheader("📋 Detailed Comparison")
    
    # Prepare comparison data
    comparison_df = filtered_data[['model_name', 'final_loss', 'final_accuracy', 
                                  'training_time', 'memory_usage', 'parameters']].copy()
    
    # Format the data
    comparison_df['final_loss'] = comparison_df['final_loss'].round(4)
    comparison_df['final_accuracy'] = (comparison_df['final_accuracy'] * 100).round(2)
    comparison_df['training_time'] = comparison_df['training_time'].round(1)
    comparison_df['memory_usage'] = comparison_df['memory_usage'].round(1)
    comparison_df['parameters'] = comparison_df['parameters'].apply(lambda x: f"{x/1e9:.1f}B")
    
    # Rename columns
    comparison_df.columns = ['Model', 'Final Loss', 'Accuracy (%)', 
                            'Training Time (h)', 'Memory (GB)', 'Parameters']
    
    st.dataframe(comparison_df, use_container_width=True)
    
    # Performance heatmap
    st.subheader("🌡️ Performance Heatmap")
    
    # Prepare data for heatmap
    heatmap_data = filtered_data[['model_name', 'final_loss', 'final_accuracy', 
                                 'training_time', 'memory_usage']].set_index('model_name')
    
    # Normalize data for better visualization
    heatmap_normalized = heatmap_data.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_normalized.values,
        x=['Final Loss', 'Final Accuracy', 'Training Time', 'Memory Usage'],
        y=heatmap_normalized.index,
        colorscale='RdYlBu_r',
        showscale=True
    ))
    
    fig_heatmap.update_layout(
        title="Normalized Performance Metrics Heatmap",
        height=400
    )
    
    fig_heatmap = apply_chart_theme(fig_heatmap)
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Best model recommendation
    st.subheader("🏅 Model Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        best_accuracy = filtered_data.loc[filtered_data['final_accuracy'].idxmax()]
        st.success(f"**Best Accuracy**: {best_accuracy['model_name']}")
        st.write(f"Accuracy: {best_accuracy['final_accuracy']:.2%}")
    
    with col2:
        best_efficiency = filtered_data.loc[filtered_data['training_time'].idxmin()]
        st.info(f"**Most Efficient**: {best_efficiency['model_name']}")
        st.write(f"Training Time: {best_efficiency['training_time']:.1f}h")
    
    with col3:
        best_memory = filtered_data.loc[filtered_data['memory_usage'].idxmin()]
        st.warning(f"**Lowest Memory**: {best_memory['model_name']}")
        st.write(f"Memory Usage: {best_memory['memory_usage']:.1f}GB")
