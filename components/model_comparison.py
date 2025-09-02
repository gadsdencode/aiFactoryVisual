import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import altair as alt
from utils.chart_themes import apply_chart_theme, get_chart_theme

def _render_model_comparison_side_by_side():
    st.header("⚖️ Model Comparison")
    if 'model_data' not in st.session_state or st.session_state.model_data is None or st.session_state.model_data.empty:
        st.warning("No models have been trained yet. Complete a training run to compare models.", icon="⚠️")
        return

    df = st.session_state.model_data.copy()
    # Map to option labels
    opts = {f"{m} (Run {i+1})": m for i, m in enumerate(df['model_name'].tolist())}
    selected = st.multiselect("Select Models to Compare", options=list(opts.keys()), default=list(opts.keys())[:2])
    if len(selected) < 2:
        st.info("Please select at least two models to compare.")
        return
    st.divider()

    # Side-by-side metrics
    chosen = [opts[n] for n in selected]
    sub = df[df['model_name'].isin(chosen)]
    cols = st.columns(len(chosen))
    for i, name in enumerate(chosen):
        row = sub[sub['model_name'] == name].iloc[0]
        with cols[i]:
            st.subheader(selected[i])
            st.metric("Final Loss", f"{row['final_loss']:.4f}")
            st.metric("Final Accuracy", f"{row['final_accuracy']:.2%}")
            st.metric("Training Time", f"{row['training_time']:.1f}h")

    st.divider()
    st.subheader("📈 Combined Performance Charts")
    # Synthetic per-epoch chart data isn't persisted; demonstrate with Altair lines using available columns
    alt_df = pd.DataFrame({
        'epoch': list(range(1, len(chosen)+1)) * 2,
        'value': list(sub['final_loss']) + list(sub['final_accuracy']),
        'metric': ['loss'] * len(chosen) + ['accuracy'] * len(chosen),
        'Model': list(sub['model_name']) + list(sub['model_name'])
    })
    loss_chart = alt.Chart(alt_df[alt_df['metric'] == 'loss']).mark_line(point=True).encode(
        x=alt.X('epoch:Q', title='Epoch'),
        y=alt.Y('value:Q', title='Loss', scale=alt.Scale(zero=False)),
        color='Model:N',
        tooltip=['epoch', 'value', 'Model']
    ).properties(title='Loss Comparison').interactive()
    acc_chart = alt.Chart(alt_df[alt_df['metric'] == 'accuracy']).mark_line(point=True).encode(
        x=alt.X('epoch:Q', title='Epoch'),
        y=alt.Y('value:Q', title='Accuracy', scale=alt.Scale(zero=False)),
        color='Model:N',
        tooltip=['epoch', 'value', 'Model']
    ).properties(title='Accuracy Comparison').interactive()
    st.altair_chart(loss_chart, use_container_width=True)
    st.altair_chart(acc_chart, use_container_width=True)


def render_model_comparison():
    st.title("🔍 Model Comparison")
    st.markdown("Compare performance across different model configurations")

    layout_mode = st.radio("Layout", ["Advanced", "Side-by-side"], horizontal=True, index=0)
    if layout_mode == "Side-by-side":
        return _render_model_comparison_side_by_side()

    # Guard: ensure model_data exists and has required columns
    if 'model_data' not in st.session_state or st.session_state.model_data is None or st.session_state.model_data.empty:
        st.info("No model comparison data available yet. Start a training run to populate results.")
        return

    # Tabs for Quantitative vs Qualitative
    tab_quant, tab_qual = st.tabs(["📊 Quantitative Comparison", "💬 Qualitative Playground"])

    with tab_quant:
        # Interactive filters
        with st.sidebar:
            st.header("Filter Options")
            try:
                min_accuracy = st.slider("Minimum Accuracy", 0.0, 1.0, 0.7)
            except Exception:
                min_accuracy = 0.0
            try:
                max_time = st.slider("Maximum Training Time", 0.0, 10.0, 5.0)
            except Exception:
                max_time = 10.0
        # Model selection
        col1, col2 = st.columns([2, 1])
        with col1:
            try:
                unique_models = st.session_state.model_data['model_name'].unique()
            except Exception:
                st.warning("Model data format unexpected: missing 'model_name' column.")
                return
            selected_models = st.multiselect(
                "Select models to compare:",
                options=unique_models,
                default=unique_models[:3]
            )
        with col2:
            metric_to_compare = st.selectbox(
                "Primary metric:",
                ["final_loss", "final_accuracy", "training_time", "memory_usage"]
            )
        if not selected_models:
            st.warning("Please select at least one model to compare.")
            return
        # Apply sidebar filters before selection visuals
        base_df = st.session_state.model_data.copy()
        if 'final_accuracy' in base_df.columns and 'training_time' in base_df.columns:
            base_df = base_df[(base_df['final_accuracy'] >= float(min_accuracy)) & (base_df['training_time'] <= float(max_time))]
        filtered_data = base_df[base_df['model_name'].isin(selected_models)]
        st.markdown("---")
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
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🏆 Model Performance Comparison")
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
            st.plotly_chart(fig_bar, width='stretch')
        with col2:
            st.subheader("⚡ Efficiency Analysis")
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
            st.plotly_chart(fig_scatter, width='stretch')
        st.subheader("📋 Detailed Comparison")
        comparison_df = filtered_data[['model_name', 'final_loss', 'final_accuracy', 
                                      'training_time', 'memory_usage', 'parameters']].copy()
        # Sorting option
        sort_by = st.selectbox("Sort by", options=list(comparison_df.columns))
        comparison_df = comparison_df.sort_values(by=sort_by)
        comparison_df['final_loss'] = comparison_df['final_loss'].round(4)
        comparison_df['final_accuracy'] = (comparison_df['final_accuracy'] * 100).round(2)
        comparison_df['training_time'] = comparison_df['training_time'].round(1)
        comparison_df['memory_usage'] = comparison_df['memory_usage'].round(1)
        comparison_df['parameters'] = comparison_df['parameters'].apply(lambda x: f"{x/1e9:.1f}B")
        comparison_df.columns = ['Model', 'Final Loss', 'Accuracy (%)', 'Training Time (h)', 'Memory (GB)', 'Parameters']
        st.dataframe(comparison_df, width='stretch')
        st.subheader("🌡️ Performance Heatmap")
        heatmap_data = filtered_data[['model_name', 'final_loss', 'final_accuracy', 'training_time', 'memory_usage']].set_index('model_name')
        heatmap_normalized = heatmap_data.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_normalized.values,
            x=['Final Loss', 'Final Accuracy', 'Training Time', 'Memory Usage'],
            y=heatmap_normalized.index,
            colorscale='RdYlBu_r',
            showscale=True
        ))
        fig_heatmap.update_layout(title="Normalized Performance Metrics Heatmap", height=400)
        fig_heatmap = apply_chart_theme(fig_heatmap)
        st.plotly_chart(fig_heatmap, width='stretch')
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

    with tab_qual:
        st.markdown("Compare model responses side-by-side.")
        try:
            unique_models = st.session_state.model_data['model_name'].unique()
        except Exception:
            unique_models = []
        selected_models_qual = st.multiselect(
            "Select models for qualitative comparison:", options=unique_models, default=list(unique_models[:2])
        )
        prompt = st.text_area("Enter a prompt", height=120, placeholder="Ask a question or provide an instruction for the models...")
        generate = st.button("Generate Responses", type="primary")
        if not selected_models_qual:
            st.info("Select at least one model to compare.")
            return
        cols = st.columns(max(1, len(selected_models_qual)))
        if generate and prompt:
            # Placeholder UI: backend inference not wired; keep UI ready without breaking
            for i, model in enumerate(selected_models_qual):
                with cols[i]:
                    st.markdown(f"#### {model}")
                    st.info("Inference not connected. Displaying placeholder response.")
                    st.code(f"[Placeholder response from {model} for prompt: '{prompt[:60]}...']")
        else:
            for i, model in enumerate(selected_models_qual):
                with cols[i]:
                    st.markdown(f"#### {model}")
                    st.write("Awaiting prompt and Generate click…")


def model_comparison_tab():
    """
    Backwards-compatible alias used by app.py.
    """
    return render_model_comparison()