import streamlit as st

def apply_styles(theme='light'):
    """
    Applies a modern, intuitive, and visually appealing style to the Streamlit app.
    This single function consolidates base styles and brand overrides for easier maintenance.
    """
    if theme == 'dark':
        # Dark theme color palette
        bg_color = '#0b1120'
        text_color = '#e2e8f0' # Light gray for text
        muted_text_color = '#94a3b8' # Muted gray for less important text
        card_bg = '#1e293b'
        border_color = '#334155'
        sidebar_bg = '#1e293b'
        input_bg = '#334155'
        hover_bg = '#334155'
        focus_color = 'rgba(99, 102, 241, 0.3)'
        
        # Alert colors for dark theme
        success_bg = 'rgba(74, 222, 128, 0.1)'
        success_border = '#4ade80'
        success_text = '#a7f3d0'
        info_bg = 'rgba(96, 165, 250, 0.1)'
        info_border = '#60a5fa'
        info_text = '#bfdbfe'
        warning_bg = 'rgba(251, 191, 36, 0.1)'
        warning_border = '#fbbf24'
        warning_text = '#fde68a'
        error_bg = 'rgba(248, 113, 113, 0.1)'
        error_border = '#f87171'
        error_text = '#fecaca'
    else:
        # Light theme color palette
        bg_color = '#f7fafc'
        text_color = '#1a202c' # Dark gray for text
        muted_text_color = '#718096'
        card_bg = '#ffffff'
        border_color = '#e2e8f0'
        sidebar_bg = '#ffffff'
        input_bg = '#ffffff'
        hover_bg = '#f0f2f6'
        focus_color = 'rgba(99, 102, 241, 0.2)'

        # Alert colors for light theme
        success_bg = '#f0fdf4'
        success_border = '#4ade80'
        success_text = '#14532d'
        info_bg = '#eff6ff'
        info_border = '#60a5fa'
        info_text = '#1e3a8a'
        warning_bg = '#fefce8'
        warning_border = '#fbbf24'
        warning_text = '#78350f'
        error_bg = '#fef2f2'
        error_border = '#f87171'
        error_text = '#991b1b'

    st.markdown(f"""
    <style>
        /* Import Inter and JetBrains Mono fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
        
        /* CSS Variables for consistent theming */
        :root {{
            --primary-color: #6366F1;
            --primary-gradient: linear-gradient(90deg, #6366F1, #8B5CF6);
            --secondary-color: #8B5CF6;
            --border-radius-lg: 12px;
            --border-radius-md: 8px;
            --transition-speed: 0.2s;
        }}
        
        /* --- Global & Typography Styles --- */
        .stApp {{
            background-color: {bg_color};
        }}
        
        body, .stApp, .stMarkdown, .stButton, .stTextInput, .stTextArea, .stSelectbox {{
            font-family: 'Inter', sans-serif;
            color: {text_color};
        }}
        
        h1, h2, h3, h4, h5, h6 {{
            font-weight: 600;
            color: {text_color};
        }}

        p, ol, ul, li {{
             color: {text_color};
        }}
        
        code, .stCode, pre {{
            font-family: 'JetBrains Mono', monospace;
        }}
        
        /* --- Layout --- */
        .main .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}

        [data-testid="stSidebar"] {{
            background-color: {sidebar_bg};
            border-right: 1px solid {border_color};
        }}
        
        /* --- Component Styling --- */
        
        /* Buttons */
        .stButton > button {{
            border-radius: var(--border-radius-md);
            font-weight: 600;
            border: none;
            transition: all var(--transition-speed) ease;
        }}
        
        .stButton > button[kind="primary"] {{
            background: var(--primary-gradient);
            color: white;
        }}
        
        .stButton > button:not([kind="primary"]) {{
            background-color: {card_bg};
            border: 1px solid {border_color};
            color: {text_color};
        }}

        /* Input Fields (Text, Number, Select) */
        .stTextInput input, .stNumberInput input, .stSelectbox > div > div, .stTextArea textarea {{
            background-color: {input_bg} !important;
            border: 1px solid {border_color} !important;
            border-radius: var(--border-radius-md) !important;
            color: {text_color} !important;
            transition: all var(--transition-speed) ease;
        }}
        
        .stTextInput input:focus, .stNumberInput input:focus, .stSelectbox > div > div:focus-within, .stTextArea textarea:focus {{
            border-color: var(--primary-color) !important;
            box-shadow: 0 0 0 3px {focus_color} !important;
            outline: none !important;
        }}
        
        /* Dropdown Menus */
        div[data-baseweb="popover"] ul li {{
            color: {text_color} !important;
        }}

        /* Alerts */
        .stAlert {{
            border-radius: var(--border-radius-md);
            border-width: 1px;
            border-style: solid;
            border-left-width: 4px;
        }}
        .stAlert p {{
             color: inherit !important; /* Make sure text inside alert inherits the color */
        }}
        .stAlert.stSuccess {{ background-color: {success_bg}; border-color: {success_border}; color: {success_text}; }}
        .stAlert.stInfo {{ background-color: {info_bg}; border-color: {info_border}; color: {info_text}; }}
        .stAlert.stWarning {{ background-color: {warning_bg}; border-color: {warning_border}; color: {warning_text}; }}
        .stAlert.stError {{ background-color: {error_bg}; border-color: {error_border}; color: {error_text}; }}
        
        /* Expanders */
        .streamlit-expanderHeader {{
            color: {text_color};
        }}
        .streamlit-expanderContent, [data-testid="stExpander"] > div > div {{
            background-color: {card_bg};
        }}
        
        /* --- Deep Color Overrides & Sidebar Fix --- */
        
        /* Force color inheritance in tricky components */
        .streamlit-expanderContent *,
        [data-testid="stExpander"] *,
        .stAlert *,
        [data-testid="stMetric"] * {{
            color: inherit !important;
        }}

        /* === START OF SIDEBAR FIX === */
        /* Explicitly set readable colors for text elements inside the sidebar */
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] li,
        [data-testid="stSidebar"] ol,
        [data-testid="stSidebar"] ul {{
            color: {muted_text_color} !important;
        }}
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {{
             color: {text_color} !important;
        }}
        /* === END OF SIDEBAR FIX === */
        
        /* Custom Card Container */
        .model-card {{
            background-color: {card_bg};
            border: 1px solid {border_color};
            border-radius: var(--border-radius-lg);
            padding: calc(var(--spacing, 20px) * 1.25);
            transition: all var(--transition-speed) ease;
        }}
        
        /* Hide Streamlit branding */
        #MainMenu, footer, header {{
            visibility: hidden;
        }}
    </style>
    """, unsafe_allow_html=True)


def inject_card(title: str | None = None, body_md: str | None = None):
    """Convenience helper to render a styled card container."""
    with st.container():
        if title:
            st.markdown(f"### {title}")
        if body_md:
            st.markdown(body_md)


def render_model_overview_card(
    model_name: str,
    final_loss: float | None,
    final_accuracy: float | None,
    training_time_hours: float | None,
    memory_gb: float | None,
    is_best: bool = False,
    best_badge_text: str | None = None,
    baseline_loss: float | None = None,
    baseline_accuracy: float | None = None,
    baseline_training_time_hours: float | None = None,
    baseline_memory_gb: float | None = None,
):
    """
    Renders a standardized model overview card with a 2x2 metric layout.
    This uses Streamlit metrics and relies on global CSS for card styling.
    """
    container = st.container()
    with container:
        classes = "model-card best" if is_best else "model-card"
        st.markdown(f'<div class="{classes}">', unsafe_allow_html=True)
        if is_best:
            badge = best_badge_text or "BEST"
            st.markdown(f'<div class="card-badge">{badge}</div>', unsafe_allow_html=True)
        st.markdown(f"### {model_name}")
        
        col1, col2 = st.columns(2)
        with col1:
            if final_loss is not None:
                delta = f"{(final_loss - baseline_loss):+.4f}" if baseline_loss is not None else None
                st.metric("Final Loss", f"{final_loss:.4f}", delta=delta, delta_color="inverse")
            else:
                st.metric("Final Loss", "-")
            
            if training_time_hours is not None:
                delta = f"{(training_time_hours - baseline_training_time_hours):+.1f}h" if baseline_training_time_hours is not None else None
                st.metric("Training Time", f"{training_time_hours:.1f}h", delta=delta, delta_color="inverse")
            else:
                st.metric("Training Time", "-")

        with col2:
            if final_accuracy is not None:
                delta = f"{(final_accuracy - baseline_accuracy) * 100.0:+.2f}%" if baseline_accuracy is not None else None
                st.metric("Final Accuracy", f"{final_accuracy:.2%}", delta=delta, delta_color="normal")
            else:
                st.metric("Final Accuracy", "-")

            if memory_gb is not None:
                delta = f"{(memory_gb - baseline_memory_gb):+.1f}GB" if baseline_memory_gb is not None else None
                st.metric("Memory Usage", f"{memory_gb:.1f}GB", delta=delta, delta_color="inverse")
            else:
                st.metric("Memory Usage", "-")
                
        st.markdown('</div>', unsafe_allow_html=True)

def load_css(theme: str | None = None):
    """Backwards-compatible helper to prevent import errors in the main app."""
    chosen_theme = theme or st.session_state.get('theme', 'light')
    apply_styles(chosen_theme)
