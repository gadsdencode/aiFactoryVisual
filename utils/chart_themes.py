import streamlit as st
import altair as alt
import plotly.graph_objects as go

def get_chart_theme():
    """Get chart styling based on current theme."""
    theme = st.session_state.get('theme', 'light')
    
    if theme == 'dark':
        return {
            'bg_color': '#1E293B',
            'paper_bgcolor': '#1E293B',
            'plot_bgcolor': '#1E293B',
            'text_color': '#F1F5F9',
            'grid_color': '#334155',
            'line_colors': ['#6366F1', '#8B5CF6', '#10B981', '#F59E0B', '#EF4444'],
        }
    else:
        return {
            'bg_color': '#FFFFFF',
            'paper_bgcolor': '#FFFFFF',
            'plot_bgcolor': '#FFFFFF',
            'text_color': '#1E293B',
            'grid_color': '#E2E8F0',
            'line_colors': ['#6366F1', '#8B5CF6', '#10B981', '#F59E0B', '#EF4444'],
        }

def apply_chart_theme(fig):
    """Apply theme styling to a Plotly figure."""
    theme = get_chart_theme()

    fig.update_layout(
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        font=dict(color=theme['text_color'], family="Inter, sans-serif", size=13),
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(
            gridcolor=theme['grid_color'],
            color=theme['text_color']
        ),
        yaxis=dict(
            gridcolor=theme['grid_color'],
            color=theme['text_color']
        ),
        legend=dict(
            font=dict(color=theme['text_color'])
        )
    )

    return fig


def setup_altair_theme():
    """Register and enable Altair themes for light/dark, switch based on session theme."""
    font = "Inter"
    common_range = [
        "#4F8BF9",
        "#FF6B6B",
        "#42D6A4",
        "#FFC107",
        "#9A5BEF",
        "#2D3748",
    ]

    urban_light_theme = {
        "config": {
            "title": {"fontSize": 18, "font": font, "anchor": "start", "color": "#1E1E1E"},
            "axis": {
                "labelFont": font,
                "labelFontSize": 12,
                "titleFont": font,
                "titleFontSize": 14,
                "titleColor": "#333333",
                "gridColor": "#E0E0E0",
                "domainColor": "#666666",
            },
            "legend": {"labelFont": font, "labelFontSize": 12, "titleFont": font, "titleFontSize": 14, "symbolSize": 100},
            "view": {"stroke": "transparent"},
            "range": {"category": common_range},
        }
    }

    urban_dark_theme = {
        "config": {
            "title": {"fontSize": 18, "font": font, "anchor": "start", "color": "#E5E7EB"},
            "axis": {
                "labelFont": font,
                "labelFontSize": 12,
                "labelColor": "#E5E7EB",
                "titleFont": font,
                "titleFontSize": 14,
                "titleColor": "#F3F4F6",
                "gridColor": "#334155",
                "domainColor": "#94A3B8",
            },
            "legend": {"labelFont": font, "labelFontSize": 12, "labelColor": "#E5E7EB", "titleFont": font, "titleFontSize": 14, "titleColor": "#F3F4F6", "symbolSize": 100},
            "view": {"stroke": "transparent", "background": "#0F172A"},
            "background": "#0F172A",
            "range": {"category": common_range},
        }
    }

    try:
        alt.themes.register("urban_light", lambda: urban_light_theme)
        alt.themes.register("urban_dark", lambda: urban_dark_theme)
        current = st.session_state.get('theme', 'light')
        alt.themes.enable("urban_dark" if current == 'dark' else "urban_light")
    except Exception:
        pass


def build_line_chart(
    series_list,
    title:
    str,
    xaxis_title:
    str,
    yaxis_title:
    str,
    height: int = 400,
    hovermode: str = 'x unified',
    showlegend: bool = True,
):
    """Create a Plotly line chart with consistent theming.

    series_list: list of dicts with keys:
      - x: array-like
      - y: array-like
      - name: str
      - color_index: Optional[int]
      - line_width: Optional[int]
      - mode: Optional[str] (e.g., 'lines', 'lines+markers')
      - fill: Optional[str]
      - fillcolor: Optional[str]
    """
    theme = get_chart_theme()
    fig = go.Figure()
    for idx, series in enumerate(series_list):
        color_idx = series.get('color_index', idx) % len(theme['line_colors'])
        line_width = series.get('line_width', 3)
        mode = series.get('mode', 'lines')
        fig.add_trace(
            go.Scatter(
                x=series['x'],
                y=series['y'],
                mode=mode,
                name=series.get('name', f"Series {idx+1}"),
                line=dict(color=theme['line_colors'][color_idx], width=line_width),
                fill=series.get('fill'),
                fillcolor=series.get('fillcolor'),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
        showlegend=showlegend,
        hovermode=hovermode,
    )
    return apply_chart_theme(fig)


def add_colored_annotation(fig: go.Figure, x, y, text: str, color_index: int = 0, ay: int = -30):
    """Add a styled annotation using theme colors."""
    theme = get_chart_theme()
    fig.add_annotation(
        x=x,
        y=y,
        text=text,
        showarrow=True,
        arrowhead=2,
        ay=ay,
        arrowcolor=theme['line_colors'][color_index % len(theme['line_colors'])],
    )
    return fig


def build_bar_chart(
    x_values,
    y_values,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    height: int = 400,
    show_text: bool = True,
):
    """Create a themed bar chart with optional data labels."""
    theme = get_chart_theme()
    colors = theme['line_colors'][: len(x_values)]
    fig = go.Figure(
        data=[
            go.Bar(
                x=x_values,
                y=y_values,
                marker_color=colors,
                text=y_values if show_text else None,
                textposition='auto' if show_text else None,
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
    )
    return apply_chart_theme(fig)


def build_scatter_chart(
    series_list,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    height: int = 400,
    showlegend: bool = False,
):
    """Create a themed multi-series scatter chart.

    series_list: list of dicts with keys:
      - x: array-like
      - y: array-like
      - name: str
      - text: Optional[array-like]
      - textposition: Optional[str]
      - size: Optional[int]
      - opacity: Optional[float]
    """
    fig = go.Figure()
    for series in series_list:
        fig.add_trace(
            go.Scatter(
                x=series['x'],
                y=series['y'],
                mode='markers+text',
                name=series.get('name'),
                text=series.get('text'),
                textposition=series.get('textposition', 'top center'),
                marker=dict(size=series.get('size', 15), opacity=series.get('opacity', 0.85)),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
        showlegend=showlegend,
    )
    return apply_chart_theme(fig)


def build_heatmap(z, x_labels, y_labels, title: str, height: int = 400, colorscale: str = 'RdYlBu_r'):
    """Create a themed heatmap."""
    fig = go.Figure(
        data=go.Heatmap(z=z, x=x_labels, y=y_labels, colorscale=colorscale, showscale=True)
    )
    fig.update_layout(title=title, height=height)
    return apply_chart_theme(fig)


def build_loss_figure_from_df(
    df,
    epoch_col: str = 'epoch',
    train_loss_col: str = 'train_loss',
    val_loss_col: str = 'val_loss',
    height: int = 400,
):
    """Build a standardized loss chart with heuristic annotations if columns exist."""
    if df is None or df.empty or epoch_col not in df.columns or train_loss_col not in df.columns:
        return apply_chart_theme(go.Figure())

    series_list = [
        {
            'x': df[epoch_col],
            'y': df[train_loss_col],
            'name': 'Training Loss',
            'color_index': 0,
        }
    ]
    if val_loss_col in df.columns and df[val_loss_col].notna().any():
        series_list.append(
            {
                'x': df[epoch_col],
                'y': df[val_loss_col],
                'name': 'Validation Loss',
                'color_index': 1,
            }
        )

    fig = build_line_chart(
        series_list,
        title="Training & Validation Loss",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        height=height,
        showlegend=True,
    )

    # Add heuristic annotations
    try:
        tl = df[train_loss_col].dropna()
        el = df.loc[tl.index, epoch_col] if epoch_col in df.columns else tl.index
        if len(tl) >= 2:
            first_tl = float(tl.iloc[0])
            last_tl = float(tl.iloc[-1])
            last_ep = float(el.iloc[-1])
            if last_tl < first_tl * 0.9:
                add_colored_annotation(fig, last_ep, last_tl, "Good! The model is learning.", color_index=0, ay=-30)
        if val_loss_col in df.columns and df[val_loss_col].notna().any() and len(tl) >= 1:
            vl = df[val_loss_col].dropna()
            evl = df.loc[vl.index, epoch_col] if epoch_col in df.columns else vl.index
            if len(vl) >= 1:
                last_vl = float(vl.iloc[-1])
                last_vl_ep = float(evl.iloc[-1])
                last_tl_val = float(tl.iloc[-1])
                if last_tl_val < 0.7 * last_vl:
                    add_colored_annotation(
                        fig,
                        last_vl_ep,
                        last_vl,
                        "Warning: The model might be 'memorizing' your data (overfitting).",
                        color_index=1,
                        ay=-40,
                    )
    except Exception:
        pass

    return fig


def build_accuracy_figure_from_df(
    df,
    epoch_col: str = 'epoch',
    val_accuracy_col: str = 'val_accuracy',
    height: int = 400,
):
    """Build a standardized accuracy chart with heuristic annotations if columns exist."""
    if df is None or df.empty or epoch_col not in df.columns or val_accuracy_col not in df.columns:
        return apply_chart_theme(go.Figure())

    theme = get_chart_theme()
    fig = build_line_chart(
        [
            {
                'x': df[epoch_col],
                'y': df[val_accuracy_col],
                'name': 'Validation Accuracy',
                'color_index': 3,
            }
        ],
        title="Training & Validation Accuracy" if 'val_accuracy' in df.columns else "Accuracy",
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        height=height,
        showlegend=True,
    )

    # Add improvement annotation
    try:
        va = df[val_accuracy_col].dropna()
        ea = df.loc[va.index, epoch_col]
        if len(va) >= 2:
            first_va = float(va.iloc[0])
            last_va = float(va.iloc[-1])
            last_ep = float(ea.iloc[-1])
            if (last_va - first_va) >= max(0.01, first_va * 0.02):
                add_colored_annotation(fig, last_ep, last_va, "Excellent! The model is getting better at its task.", color_index=3, ay=-30)
    except Exception:
        pass

    return fig
