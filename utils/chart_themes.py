import streamlit as st

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