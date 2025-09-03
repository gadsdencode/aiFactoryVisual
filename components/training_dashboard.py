import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import time
from utils.chart_themes import (
    apply_chart_theme,
    get_chart_theme,
    build_line_chart,
    build_loss_figure_from_df,
    build_accuracy_figure_from_df,
)
from backend.training_manager import get_training_manager

def _render_training_dashboard_compact():
    st.header("📊 Training Dashboard")
    manager = get_training_manager()
    if 'training_complete' not in st.session_state:
        st.session_state.training_complete = False

    if not st.session_state.training_complete:
        with st.spinner("Training in progress... Please wait."):
            manager.start_training()
        st.session_state.training_complete = True
        st.success("🎉 Training complete!")
        if st.button("Compare Your Model", key="btn_compare_postrun"):
            st.session_state['_nav_target'] = 'Model Comparison'
            st.rerun()
        try:
            st.balloons()
        except Exception:
            pass

    st.subheader("🏁 Final Results")
    df = manager.get_metrics_df()
    if not df.empty:
        # Key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            last_loss = df['train_loss'].dropna().iloc[-1] if 'train_loss' in df.columns and df['train_loss'].notna().any() else None
            st.metric("Final Loss", f"{last_loss:.4f}" if last_loss is not None else "-")
        with col2:
            last_acc = df['val_accuracy'].dropna().iloc[-1] if 'val_accuracy' in df.columns and df['val_accuracy'].notna().any() else None
            st.metric("Final Accuracy", f"{last_acc:.2%}" if last_acc is not None else "-")
        with col3:
            epochs = int(df['epoch'].dropna().max()) if 'epoch' in df.columns and df['epoch'].notna().any() else None
            st.metric("Total Epochs", f"{epochs or '-'}")

        st.divider()
        st.subheader("📈 Performance Charts")
        col_a, col_b = st.columns(2)
        with col_a:
            if 'epoch' in df.columns and 'train_loss' in df.columns:
                fig_loss_c = build_loss_figure_from_df(df, height=360)
                st.plotly_chart(fig_loss_c, use_container_width=True)
        with col_b:
            if 'epoch' in df.columns and 'val_accuracy' in df.columns:
                fig_acc_c = build_accuracy_figure_from_df(df, height=360)
                st.plotly_chart(fig_acc_c, use_container_width=True)

        with st.expander("📜 View Training Logs"):
            logs = manager.get_logs()
            st.text_area("Logs", logs or "", height=300, disabled=True)
    else:
        st.info("No metrics captured yet.")


def render_training_dashboard():
    st.title("🚀 Training Dashboard")
    st.markdown("Real-time monitoring of LLM training pipeline")
    mode = st.radio("Layout", ["Advanced", "Guided"], horizontal=True, index=0)
    if mode == "Guided":
        return _render_training_dashboard_compact()
    
    # Get training manager
    training_manager = get_training_manager()

    # Tiny hardware panel
    try:
        import torch  # type: ignore
        try:
            import bitsandbytes as _bnb  # type: ignore
            bnb_ok = True
        except Exception:
            _bnb = None  # type: ignore
            bnb_ok = False
        cuda_ok = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "-"
        vram_gb = (torch.cuda.get_device_properties(0).total_memory / (1024**3)) if cuda_ok else 0.0
        hw_cols = st.columns(4)
        with hw_cols[0]:
            st.metric("CUDA", "Yes" if cuda_ok else "No")
        with hw_cols[1]:
            st.metric("bitsandbytes", "Yes" if (bnb_ok and cuda_ok) else "No")
        with hw_cols[2]:
            st.metric("GPU", gpu_name)
        with hw_cols[3]:
            st.metric("VRAM", f"{vram_gb:.1f} GB" if cuda_ok else "-")
    except Exception:
        pass
    
    # Controls and views organized into tabs
    tabs = st.tabs(["📈 Metrics", "📝 Training Logs", "📊 System"])

    # Global control bar at top
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)
    with ctrl1:
        if st.button("▶️ Start Training", type="primary", width='stretch'):
            if training_manager.start_training():
                st.success("Training started!")
                try:
                    st.toast("🚀 Training has started! Navigate to the Dashboard to monitor progress.", icon="✅")
                except Exception:
                    pass
                st.session_state['training_started'] = True
            else:
                st.warning("Training is already active!")
    with ctrl2:
        if st.button("⏸️ Pause/Resume", width='stretch'):
            if training_manager.pause_training():
                status = training_manager.get_status_snapshot()
                if status['paused']:
                    st.warning("Training paused!")
                else:
                    st.info("Training resumed!")
            else:
                st.warning("No active training to pause/resume!")
    with ctrl3:
        if st.button("⏹️ Stop Training", width='stretch'):
            if training_manager.stop_training():
                st.error("Training stopped!")
            else:
                st.warning("No active training to stop!")
    with ctrl4:
        if st.button("🔄 Reset", width='stretch'):
            training_manager.reset_training()
            st.info("Training reset!")

    st.markdown("---")
    
    # Get current training status
    status = training_manager.get_status_snapshot()
    training_data = status['training_data']

    # Overview section with gauge + KPIs
    st.subheader("📌 Training Overview")
    ov1, ov2 = st.columns([2, 3])
    with ov1:
        try:
            prog = float(status.get('current_step', 0) / status.get('total_steps', 1)) if status.get('total_steps') else float(status['progress'])
            expected = max(0.0, min(1.0, prog)) * 100.0
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=expected,
                number={"suffix": "%"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": get_chart_theme()['line_colors'][0]}}
            ))
            gauge.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10))
            gauge = apply_chart_theme(gauge)
            st.plotly_chart(gauge, width='stretch')
        except Exception:
            st.progress(status['progress'])
    with ov2:
        k1, k2, k3 = st.columns(3)
        with k1:
            step = status.get('current_step')
            total_steps = status.get('total_steps')
            st.metric("Step", f"{step or 0}/{total_steps or '-'}")
        with k2:
            sps = status.get('steps_per_second')
            st.metric("Speed", f"{sps:.2f} steps/s" if sps else "-")
        with k3:
            eta = status.get('eta_seconds')
            if eta:
                import math
                m, s = divmod(int(math.ceil(eta)), 60)
                h, m = divmod(m, 60)
                st.metric("ETA", f"{h:02d}:{m:02d}:{s:02d}")
            else:
                st.metric("ETA", "-")
    
    # Progress metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Epoch", 
            status['current_epoch'],
            delta=1 if status['active'] and not status['paused'] else 0
        )
    
    with col2:
        current_loss = (
            training_data['train_loss'].dropna().iloc[-1]
            if (not training_data.empty and 'train_loss' in training_data.columns and training_data['train_loss'].notna().any())
            else 0.0
        )
        st.metric(
            "Training Loss", 
            f"{current_loss:.4f}",
            delta=None
        )
    
    with col3:
        current_acc = (
            training_data['train_accuracy'].dropna().iloc[-1]
            if (not training_data.empty and 'train_accuracy' in training_data.columns and training_data['train_accuracy'].notna().any())
            else 0.0
        )
        st.metric(
            "Training Accuracy", 
            f"{current_acc:.2%}",
            delta=f"{0.005:.3f}" if status['active'] else None
        )
    
    with col4:
        # Prefer step-based indicator if available
        step = status.get('current_step')
        total_steps = status.get('total_steps')
        if step is not None and total_steps:
            st.metric("Progress", f"{(step/total_steps):.1%}")
        else:
            st.metric("Progress", f"{status['progress']:.1%}")
    
    # Progress bar (step-based if possible)
    if status.get('current_step') is not None and status.get('total_steps'):
        st.progress(min(1.0, float(status['current_step'])/float(status['total_steps'])))
    else:
        st.progress(status['progress'])
    
    st.markdown("---")
    
    # KPIs row 2: Steps/sec and ETA
    k1, k2, k3 = st.columns(3)
    with k1:
        sps = status.get('steps_per_second')
        st.metric("Speed", f"{sps:.2f} steps/s" if sps else "-")
    with k2:
        eta = status.get('eta_seconds')
        if eta:
            # pretty ETA
            import math
            m, s = divmod(int(math.ceil(eta)), 60)
            h, m = divmod(m, 60)
            pretty = f"{h:02d}:{m:02d}:{s:02d}"
            st.metric("ETA", pretty)
        else:
            st.metric("ETA", "-")
    with k3:
        st.metric("Total Steps", str(status.get('total_steps') or '-'))

    # Charts inside the Metrics tab
    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 Loss Curves")
            fig_loss = build_loss_figure_from_df(training_data, height=400)
            st.plotly_chart(fig_loss, width='stretch')
        with col2:
            st.subheader("🎯 Accuracy Metrics")
            fig_acc = build_accuracy_figure_from_df(training_data, height=400)
            st.plotly_chart(fig_acc, width='stretch')
    
    # Learning rate schedule & Grad Norm
    with tabs[0]:
        st.subheader("📊 Learning Rate Schedule")
        fig_lr = build_line_chart(
            series_list=(
                [
                    {
                        'x': training_data['epoch'],
                        'y': training_data['learning_rate'],
                        'name': 'Learning Rate',
                        'color_index': 4,
                        'fill': 'tozeroy',
                        'fillcolor': 'rgba(239, 68, 68, 0.1)',
                    }
                ]
                if (not training_data.empty and 'epoch' in training_data.columns and 'learning_rate' in training_data.columns)
                else []
            ),
            title="Learning Rate Schedule",
            xaxis_title="Epoch",
            yaxis_title="Learning Rate",
            height=300,
            showlegend=False,
        )
        st.plotly_chart(fig_lr, width='stretch')

    # Grad Norm chart (if available)
    with tabs[0]:
        if not training_data.empty and 'grad_norm' in training_data.columns and training_data['grad_norm'].notna().any():
            st.subheader("📐 Gradient Norm")
            x_vals = training_data['epoch'] if 'epoch' in training_data.columns else training_data.index
            fig_g = build_line_chart(
                [
                    {
                        'x': x_vals,
                        'y': training_data['grad_norm'],
                        'name': 'Grad Norm',
                        'color_index': 5,
                    }
                ],
                title="Gradient Norm",
                xaxis_title="Epoch",
                yaxis_title="Norm",
                height=300,
                showlegend=False,
            )
            st.plotly_chart(fig_g, width='stretch')
    
    # Logs and recent steps organized into the Logs tab
    with tabs[1]:
        st.subheader("🧭 Live View")
        if status['active'] and not status['paused']:
            st.info(f"🔄 {status.get('phase', 'RUNNING')} • Epoch {status['current_epoch']}: Training in progress...")
        elif status['paused']:
            st.warning(f"⏸️ Epoch {status['current_epoch']}: Training paused")
        elif status['current_epoch'] > 0:
            st.success(f"✅ Last completed epoch: {status['current_epoch']}")
        else:
            st.info("⏳ Training not started yet")

        # Recent metrics preview table
        if not training_data.empty:
            cols = [c for c in ['step','epoch','train_loss','val_loss','grad_norm','learning_rate','timestamp'] if c in training_data.columns]
            mini = training_data[cols].tail(8).copy()
            if 'timestamp' in mini.columns:
                mini['timestamp'] = mini['timestamp'].dt.strftime('%H:%M:%S')
            st.dataframe(mini, width='stretch')

        # Interactive log viewer
        st.subheader("📝 Training Logs")
        log_level = st.selectbox("Filter by log level", ["ALL", "INFO", "WARNING", "ERROR"], index=0)

        logs_text = training_manager.get_logs() or ""
        if log_level != "ALL":
            level_prefix = {
                'INFO': ['INFO', '[INFO]'],
                'WARNING': ['WARNING', 'WARN', '[WARN]', '[WARNING]'],
                'ERROR': ['ERROR', 'ERR', '[ERROR]']
            }[log_level]
            filtered_lines = []
            for line in logs_text.splitlines():
                upper = line.upper()
                if any(token in upper for token in level_prefix):
                    filtered_lines.append(line)
            logs_text = "\n".join(filtered_lines)

        log_container = st.empty()
        with log_container:
            st.text_area("Logs", logs_text, height=300, key="logs_tab_textarea")

    # System tab: hardware snapshot and final metrics
    with tabs[2]:
        st.subheader("🖥️ System & Session")
        try:
            import platform
            import sys
            st.markdown(f"**Python:** {sys.version.split()[0]}  •  **Platform:** {platform.platform()}")
        except Exception:
            pass
        st.markdown("---")
        st.subheader("📦 Final Metrics Snapshot")
        df_final = training_manager.get_metrics_df()
        if not df_final.empty:
            st.dataframe(df_final.tail(100), width='stretch')
        else:
            st.info("No metrics captured yet.")


def training_dashboard():
    """
    Backwards-compatible alias used by app.py.
    """
    return render_training_dashboard()


def display_training_dashboard():
    """
    New live-polling dashboard using placeholders and manager methods.
    """
    st.title("🚀 Training Dashboard (Live)")
    manager = get_training_manager()

    # Placeholders for dynamic sections
    status_placeholder = st.empty()
    metrics_container = st.container()
    log_placeholder = st.empty()

    # Simple controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start Training", type="primary"):
            manager.start_training()
    with col2:
        if st.button("⏹️ Stop Training"):
            manager.stop_training()

    # Poll while running
    while manager.get_status() == "RUNNING":
        with status_placeholder:
            st.status("Training in progress...", state="running")

        df = manager.get_metrics_df()
        logs_text = manager.get_logs()

        with metrics_container:
            st.subheader("Live Metrics")
            if not df.empty:
                # KPIs
                k1, k2, k3 = st.columns(3)
                with k1:
                    v = df['train_loss'].dropna().iloc[-1] if 'train_loss' in df.columns and df['train_loss'].notna().any() else None
                    st.metric("Loss", f"{v:.4f}" if v is not None else "-")
                with k2:
                    lr = df['learning_rate'].dropna().iloc[-1] if 'learning_rate' in df.columns and df['learning_rate'].notna().any() else None
                    st.metric("LR", f"{lr:.6f}" if lr is not None else "-")
                with k3:
                    ep = df['epoch'].dropna().iloc[-1] if 'epoch' in df.columns and df['epoch'].notna().any() else None
                    st.metric("Epoch", f"{ep:.2f}" if ep is not None else "-")

                # Chart
                plot_df = df.copy()
                if 'step' in plot_df.columns and 'train_loss' in plot_df.columns:
                    plot_df = plot_df.dropna(subset=['step']).set_index('step')
                    st.line_chart(plot_df[['train_loss']].fillna(method='ffill'))
            else:
                st.info("Awaiting first metrics...")

        with log_placeholder:
            st.text_area("Logs", logs_text or "", height=220, key="logs_live_textarea")

        time.sleep(5)

    # Final state
    final_status = manager.get_status()
    if final_status == "COMPLETED":
        status_placeholder.success("Training completed!")
        try:
            st.balloons()
        except Exception:
            pass
    elif final_status == "FAILED":
        status_placeholder.error("Training failed. Check logs below.")
        with st.expander("Show Error Details"):
            st.code(manager.get_logs(), language='bash')
    else:
        status_placeholder.info("Training is not running.")

    # Final snapshot
    final_df = manager.get_metrics_df()
    if not final_df.empty and 'step' in final_df.columns and 'train_loss' in final_df.columns:
        show_df = final_df.dropna(subset=['step']).set_index('step')
        st.line_chart(show_df[['train_loss']].fillna(method='ffill'))
    st.text_area("Final Logs", manager.get_logs() or "", height=240, key="logs_final_textarea")