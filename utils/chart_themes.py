import streamlit as st
import altair as alt

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