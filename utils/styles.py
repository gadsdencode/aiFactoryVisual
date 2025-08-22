import streamlit as st

def apply_custom_styles(theme='light'):
    """Apply custom CSS styles matching the specified design requirements."""
    
    # Define colors based on theme
    if theme == 'dark':
        bg_color = '#0F172A'
        text_color = '#F1F5F9'
        card_bg = '#1E293B'
        border_color = '#334155'
        sidebar_bg = '#1E293B'
        input_bg = '#334155'
    else:  # light theme
        bg_color = '#F8FAFC'
        text_color = '#1E293B'
        card_bg = '#FFFFFF'
        border_color = '#E2E8F0'
        sidebar_bg = '#FFFFFF'
        input_bg = '#FFFFFF'
    
    st.markdown(f"""
    <style>
        /* Import Inter and JetBrains Mono fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600&display=swap');
        
        /* Root variables for consistent theming */
        :root {{
            --primary-color: #6366F1;
            --secondary-color: #8B5CF6;
            --success-color: #10B981;
            --warning-color: #F59E0B;
            --error-color: #EF4444;
            --spacing: 20px;
        }}
        
        /* Global page styling - target the root Streamlit container */
        .stApp {{
            background-color: {bg_color} !important;
            color: {text_color} !important;
        }}
        
        /* Main content area */
        .main {{
            font-family: 'Inter', sans-serif;
            color: {text_color} !important;
            background-color: {bg_color} !important;
        }}
        
        /* Main content block */
        .block-container {{
            background-color: {bg_color} !important;
            color: {text_color} !important;
        }}
        
        /* Headers */
        h1, h2, h3 {{
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            color: {text_color};
        }}
        
        /* Code and monospace text */
        code, .stCode, pre {{
            font-family: 'JetBrains Mono', monospace;
        }}
        
        /* Sidebar styling - comprehensive */
        .css-1d391kg, .css-1cypcdb, .css-17eq0hr, section[data-testid="stSidebar"] {{
            background-color: {sidebar_bg} !important;
            border-right: 1px solid {border_color} !important;
            color: {text_color} !important;
        }}
        
        /* Sidebar content */
        .css-1d391kg * {{
            color: {text_color} !important;
        }}
        
        /* All text elements */
        p, span, div, label, .stMarkdown, .stText {{
            color: {text_color} !important;
        }}
        
        /* Metric cards styling */
        [data-testid="metric-container"] {{
            background-color: {card_bg} !important;
            border: 1px solid {border_color} !important;
            border-radius: 8px;
            padding: var(--spacing);
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }}
        
        [data-testid="metric-container"] > div {{
            color: {text_color} !important;
        }}
        
        /* Button styling */
        .stButton > button {{
            border-radius: 8px;
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            border: none;
            transition: all 0.2s ease;
        }}
        
        .stButton > button[kind="primary"] {{
            background-color: var(--primary-color);
            color: white;
        }}
        
        .stButton > button[kind="primary"]:hover {{
            background-color: #5855EB;
            transform: translateY(-1px);
        }}
        
        .stButton > button:not([kind="primary"]) {{
            background-color: {card_bg} !important;
            color: {text_color} !important;
            border: 1px solid {border_color} !important;
        }}
        
        .stButton > button:not([kind="primary"]):hover {{
            background-color: {bg_color} !important;
            border-color: var(--primary-color) !important;
        }}
        
        /* Selectbox and input styling */
        .stSelectbox > div > div {{
            background-color: {card_bg} !important;
            border: 1px solid {border_color} !important;
            border-radius: 8px;
            color: {text_color} !important;
        }}
        
        .stNumberInput > div > div > input,
        .stTextInput > div > div > input {{
            background-color: {input_bg} !important;
            border: 1px solid {border_color} !important;
            border-radius: 8px;
            font-family: 'Inter', sans-serif;
            color: {text_color} !important;
        }}
        
        /* Progress bar styling */
        .stProgress > div > div > div {{
            background-color: var(--primary-color);
            border-radius: 4px;
        }}
        
        /* Alert styling */
        .stAlert {{
            border-radius: 8px;
            font-family: 'Inter', sans-serif;
        }}
        
        .stSuccess {{
            background-color: #ECFDF5;
            border-color: var(--success-color);
            color: #047857;
        }}
        
        .stInfo {{
            background-color: #EFF6FF;
            border-color: var(--primary-color);
            color: #1D4ED8;
        }}
        
        .stWarning {{
            background-color: #FFFBEB;
            border-color: var(--warning-color);
            color: #92400E;
        }}
        
        .stError {{
            background-color: #FEF2F2;
            border-color: var(--error-color);
            color: #DC2626;
        }}
        
        /* DataFrame styling */
        .stDataFrame {{
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid {border_color} !important;
            background-color: {card_bg} !important;
        }}
        
        /* Expander styling */
        .streamlit-expanderHeader {{
            background-color: {card_bg} !important;
            border: 1px solid {border_color} !important;
            border-radius: 8px;
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            color: {text_color} !important;
        }}
        
        /* Plotly chart container */
        .js-plotly-plot {{
            border-radius: 8px;
        }}
        
        /* Comprehensive text color overrides */
        .stMarkdown, .stText, .stMarkdown p, .stMarkdown div, .stMarkdown span {{
            color: {text_color} !important;
        }}
        
        /* Column containers */
        .css-ocqkz7, .css-1kyxreq, .element-container {{
            background-color: transparent !important;
        }}
        
        /* Widget labels */
        .stSelectbox label, .stNumberInput label, .stTextInput label, .stSlider label {{
            color: {text_color} !important;
        }}
        
        /* Slider styling */
        .stSlider > div > div > div > div {{
            background-color: {card_bg} !important;
        }}
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {{
            background-color: {card_bg} !important;
            border-bottom: 1px solid {border_color} !important;
        }}
        
        /* Multiselect styling */
        .stMultiSelect > div > div {{
            background-color: {card_bg} !important;
            border: 1px solid {border_color} !important;
            color: {text_color} !important;
        }}
        
        /* JSON styling */
        .stJson {{
            background-color: {card_bg} !important;
            border: 1px solid {border_color} !important;
            color: {text_color} !important;
        }}
        
        /* Spacing utilities */
        .space-small {{ margin: calc(var(--spacing) / 2) 0; }}
        .space-medium {{ margin: var(--spacing) 0; }}
        .space-large {{ margin: calc(var(--spacing) * 2) 0; }}
        
        /* Custom grid system */
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: var(--spacing);
            margin: var(--spacing) 0;
        }}
        
        /* Additional comprehensive styling */
        .css-10trblm, .css-16idsys, .css-1inwz65 {{
            background-color: {bg_color} !important;
            color: {text_color} !important;
        }}
        
        /* Plotly charts background */
        .js-plotly-plot .plot-container {{
            background-color: {card_bg} !important;
        }}
        
        /* Hide Streamlit branding */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
        
        /* Responsive adjustments */
        @media (max-width: 768px) {{
            :root {{
                --spacing: 15px;
            }}
            
            .main {{
                padding: var(--spacing);
            }}
        }}
    </style>
    """, unsafe_allow_html=True)