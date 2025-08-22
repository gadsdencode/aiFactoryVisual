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
        hover_bg = '#334155'
        focus_color = 'rgba(99, 102, 241, 0.2)'
        button_shadow = '0 2px 4px rgba(0, 0, 0, 0.3)'
        card_shadow = '0 2px 4px rgba(0, 0, 0, 0.3)'
        # Alert colors for dark theme
        success_bg = '#0F2A1A'
        success_color = '#4ADE80'
        info_bg = '#1E293B'
        info_color = '#93C5FD'
        warning_bg = '#2D1B0F'
        warning_color = '#FCD34D'
        error_bg = '#2D1B1B'
        error_color = '#F87171'
    else:  # light theme
        bg_color = '#F8FAFC'
        text_color = '#1E293B'
        card_bg = '#FFFFFF'
        border_color = '#E2E8F0'
        sidebar_bg = '#FFFFFF'
        input_bg = '#FFFFFF'
        hover_bg = '#F8FAFC'
        focus_color = 'rgba(99, 102, 241, 0.1)'
        button_shadow = '0 1px 2px rgba(0, 0, 0, 0.1)'
        card_shadow = '0 1px 3px rgba(0, 0, 0, 0.1)'
        # Alert colors for light theme
        success_bg = '#ECFDF5'
        success_color = '#047857'
        info_bg = '#EFF6FF'
        info_color = '#1D4ED8'
        warning_bg = '#FFFBEB'
        warning_color = '#92400E'
        error_bg = '#FEF2F2'
        error_color = '#DC2626'
    
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
            box-shadow: {card_shadow};
        }}
        
        [data-testid="metric-container"] > div {{
            color: {text_color} !important;
        }}
        
        /* Enhanced button styling for dark mode */
        .stButton > button {{
            border-radius: 8px;
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            border: none;
            transition: all 0.2s ease;
            box-shadow: {button_shadow};
        }}
        
        .stButton > button[kind="primary"] {{
            background-color: var(--primary-color);
            color: white;
        }}
        
        .stButton > button[kind="primary"]:hover {{
            background-color: #5855EB;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(99, 102, 241, 0.3);
        }}
        
        .stButton > button:not([kind="primary"]) {{
            background-color: {card_bg} !important;
            color: {text_color} !important;
            border: 1px solid {border_color} !important;
        }}
        
        .stButton > button:not([kind="primary"]):hover {{
            background-color: {hover_bg} !important;
            border-color: var(--primary-color) !important;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px {focus_color};
        }}
        
        /* Enhanced form controls for dark mode */
        .stSelectbox > div > div {{
            background-color: {card_bg} !important;
            border: 1px solid {border_color} !important;
            border-radius: 8px;
            color: {text_color} !important;
            transition: all 0.2s ease;
        }}
        
        .stSelectbox > div > div:hover {{
            border-color: #64748B !important;
        }}
        
        .stSelectbox > div > div:focus-within {{
            border-color: var(--primary-color) !important;
            box-shadow: 0 0 0 2px {focus_color} !important;
        }}
        
        .stNumberInput > div > div > input,
        .stTextInput > div > div > input {{
            background-color: {input_bg} !important;
            border: 1px solid {border_color} !important;
            border-radius: 8px;
            font-family: 'Inter', sans-serif;
            color: {text_color} !important;
            transition: all 0.2s ease;
        }}
        
        .stNumberInput > div > div > input:focus,
        .stTextInput > div > div > input:focus {{
            border-color: var(--primary-color) !important;
            box-shadow: 0 0 0 2px {focus_color} !important;
            outline: none !important;
        }}
        
        .stNumberInput > div > div > input:hover,
        .stTextInput > div > div > input:hover {{
            border-color: #64748B !important;
        }}
        
        /* Progress bar styling */
        .stProgress > div > div > div {{
            background-color: var(--primary-color);
            border-radius: 4px;
        }}
        
        /* Enhanced alert styling for dark mode */
        .stAlert {{
            border-radius: 8px;
            font-family: 'Inter', sans-serif;
            border-width: 1px;
            border-style: solid;
        }}
        
        .stSuccess {{
            background-color: {success_bg} !important;
            border-color: var(--success-color) !important;
            color: {success_color} !important;
        }}
        
        .stInfo {{
            background-color: {info_bg} !important;
            border-color: var(--primary-color) !important;
            color: {info_color} !important;
        }}
        
        .stWarning {{
            background-color: {warning_bg} !important;
            border-color: var(--warning-color) !important;
            color: {warning_color} !important;
        }}
        
        .stError {{
            background-color: {error_bg} !important;
            border-color: var(--error-color) !important;
            color: {error_color} !important;
        }}
        
        /* Enhanced DataFrame styling for dark mode */
        .stDataFrame {{
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid {border_color} !important;
            background-color: {card_bg} !important;
            box-shadow: {card_shadow};
        }}
        
        .stDataFrame [data-testid="stDataFrameResizable"] {{
            background-color: {card_bg} !important;
            color: {text_color} !important;
        }}
        
        .stDataFrame table {{
            background-color: {card_bg} !important;
            color: {text_color} !important;
        }}
        
        /* ULTIMATE FIX: All possible expander selectors for dark mode */
        .streamlit-expanderHeader {{
            background-color: {card_bg} !important;
            border: 1px solid {border_color} !important;
            border-radius: 8px;
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            color: {text_color} !important;
            transition: all 0.2s ease;
        }}
        
        .streamlit-expanderHeader:hover {{
            background-color: {hover_bg} !important;
            border-color: #64748B !important;
        }}
        
        /* CRITICAL: Force expander content backgrounds */
        .streamlit-expanderContent,
        [data-testid="stExpander"] > div,
        [data-testid="stExpander"] > div > div,
        [data-testid="stExpander"] > div > div > div,
        [data-testid="stExpander"] div[data-testid="stExpanderDetails"],
        .css-1kyxreq,
        .css-ocqkz7,
        .css-1kyxreq > div,
        .css-ocqkz7 > div,
        .element-container > div > div[style*="padding"] {{
            background-color: {card_bg} !important;
            color: {text_color} !important;
        }}
        
        /* Force ALL expander descendants to have proper colors */
        [data-testid="stExpander"] *,
        .streamlit-expanderContent *,
        .css-1kyxreq *,
        .css-ocqkz7 * {{
            background-color: transparent !important;
            color: {text_color} !important;
        }}
        
        /* Override any white/light backgrounds in expanders */
        [data-testid="stExpander"] div[style*="background-color: rgb(255, 255, 255)"],
        [data-testid="stExpander"] div[style*="background-color: white"],
        [data-testid="stExpander"] div[style*="background-color: #fff"],
        [data-testid="stExpander"] div[style*="background-color: #ffffff"] {{
            background-color: {card_bg} !important;
        }}
        
        /* Specific targeting for the problematic light containers */
        .css-1kyxreq[style*="background"],
        .css-ocqkz7[style*="background"],
        .element-container[style*="background"] {{
            background-color: {card_bg} !important;
        }}
        
        /* Nuclear option: override any element with white background inside expanders */
        [data-testid="stExpander"] div,
        [data-testid="stExpander"] section,
        [data-testid="stExpander"] [class*="css-"] {{
            background-color: {card_bg} !important;
            border-color: {border_color} !important;
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
        
        /* Enhanced slider styling for dark mode */
        .stSlider > div > div > div > div {{
            background-color: {card_bg} !important;
        }}
        
        .stSlider [data-baseweb="slider"] [data-testid="stTickBar"] {{
            background-color: #475569 !important;
        }}
        
        .stSlider [data-baseweb="slider"] [data-testid="stThumb"] {{
            background-color: var(--primary-color) !important;
            border: 2px solid {card_bg} !important;
            box-shadow: {button_shadow} !important;
        }}
        
        /* Enhanced tab styling for dark mode */
        .stTabs [data-baseweb="tab-list"] {{
            background-color: {card_bg} !important;
            border-bottom: 1px solid {border_color} !important;
            border-radius: 8px 8px 0 0;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            color: #94A3B8 !important;
            transition: all 0.2s ease;
        }}
        
        .stTabs [data-baseweb="tab"]:hover {{
            color: {text_color} !important;
        }}
        
        .stTabs [aria-selected="true"] {{
            color: var(--primary-color) !important;
            border-bottom-color: var(--primary-color) !important;
        }}
        
        /* Radio button and checkbox styling for dark mode */
        .stRadio > div {{
            color: {text_color} !important;
        }}
        
        .stRadio > div > label {{
            color: {text_color} !important;
        }}
        
        .stCheckbox > label {{
            color: {text_color} !important;
        }}
        
        .stRadio [data-baseweb="radio"] {{
            background-color: {card_bg} !important;
            border: 1px solid {border_color} !important;
        }}
        
        .stCheckbox [data-testid="stCheckbox"] {{
            background-color: {card_bg} !important;
            border: 1px solid {border_color} !important;
        }}
        
        /* Enhanced multiselect styling for dark mode */
        .stMultiSelect > div > div {{
            background-color: {card_bg} !important;
            border: 1px solid {border_color} !important;
            color: {text_color} !important;
            border-radius: 8px;
            transition: all 0.2s ease;
        }}
        
        .stMultiSelect > div > div:focus-within {{
            border-color: var(--primary-color) !important;
            box-shadow: 0 0 0 2px {focus_color} !important;
        }}
        
        .stMultiSelect [data-baseweb="tag"] {{
            background-color: #475569 !important;
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
        
        /* Configuration Page Specific Dark Mode Fixes */
        
        /* Help text and descriptions */
        .stMarkdown small, .stHelp, [data-testid="stHelp"] {{
            color: {'#94A3B8' if theme == 'dark' else '#64748B'} !important;
        }}
        
        /* Form field containers */
        .stForm {{
            background-color: {card_bg} !important;
            border: 1px solid {border_color} !important;
            border-radius: 8px;
        }}
        
        /* Radio button containers and labels */
        .stRadio > div {{
            background-color: transparent !important;
        }}
        
        .stRadio > div > label > div {{
            color: {text_color} !important;
        }}
        
        .stRadio [data-baseweb="radio"] > div {{
            background-color: {input_bg} !important;
            border: 1px solid {border_color} !important;
        }}
        
        .stRadio [data-baseweb="radio"][aria-checked="true"] > div {{
            background-color: var(--primary-color) !important;
            border-color: var(--primary-color) !important;
        }}
        
        /* Spinner/loading indicators */
        .stSpinner > div {{
            border-color: {border_color} !important;
            border-top-color: var(--primary-color) !important;
        }}
        
        /* Selectbox dropdown options */
        .stSelectbox [role="listbox"] {{
            background-color: {card_bg} !important;
            border: 1px solid {border_color} !important;
            border-radius: 8px;
            box-shadow: {card_shadow};
        }}
        
        .stSelectbox [role="option"] {{
            background-color: {card_bg} !important;
            color: {text_color} !important;
        }}
        
        .stSelectbox [role="option"]:hover {{
            background-color: {hover_bg} !important;
        }}
        
        .stSelectbox [aria-selected="true"] {{
            background-color: {'#475569' if theme == 'dark' else '#F1F5F9'} !important;
        }}
        
        /* Text input placeholders */
        .stTextInput input::placeholder,
        .stNumberInput input::placeholder {{
            color: {'#64748B' if theme == 'dark' else '#94A3B8'} !important;
        }}
        
        /* Widget help tooltips */
        [data-testid="stTooltipHoverTarget"] {{
            color: {text_color} !important;
        }}
        
        /* Validation and error messages in forms */
        .stTextInput .stError,
        .stNumberInput .stError,
        .stSelectbox .stError {{
            color: {error_color} !important;
        }}
        
        /* Password input specific styling */
        .stTextInput input[type="password"] {{
            background-color: {input_bg} !important;
            border: 1px solid {border_color} !important;
            color: {text_color} !important;
        }}
        
        /* Download button styling */
        .stDownloadButton > button {{
            background-color: {card_bg} !important;
            color: {text_color} !important;
            border: 1px solid {border_color} !important;
        }}
        
        .stDownloadButton > button:hover {{
            background-color: {hover_bg} !important;
            border-color: var(--primary-color) !important;
        }}
        
        /* Metric delta colors */
        [data-testid="metric-container"] [data-testid="stMetricDelta"] {{
            color: {text_color} !important;
        }}
        
        /* Code blocks and JSON display */
        .stCodeBlock {{
            background-color: {card_bg} !important;
            border: 1px solid {border_color} !important;
            color: {text_color} !important;
        }}
        
        /* Enhanced JSON display styling for dark mode */
        .stJson {{
            background-color: {card_bg} !important;
            border: 1px solid {border_color} !important;
            border-radius: 8px !important;
            padding: 16px !important;
        }}
        
        .stJson pre {{
            background-color: {card_bg} !important;
            color: {text_color} !important;
            margin: 0 !important;
        }}
        
        .stJson pre code {{
            background-color: {card_bg} !important;
            color: {text_color} !important;
        }}
        
        /* JSON syntax highlighting for better readability */
        .stJson .token.string {{
            color: {'#84CC16' if theme == 'dark' else '#059669'} !important;
        }}
        
        .stJson .token.number {{
            color: {'#F59E0B' if theme == 'dark' else '#D97706'} !important;
        }}
        
        .stJson .token.boolean {{
            color: {'#8B5CF6' if theme == 'dark' else '#7C3AED'} !important;
        }}
        
        .stJson .token.keyword {{
            color: {'#06B6D4' if theme == 'dark' else '#0891B2'} !important;
        }}
        
        /* Ensure all JSON container elements have proper styling */
        [data-testid="stJson"] {{
            background-color: {card_bg} !important;
            color: {text_color} !important;
            border: 1px solid {border_color} !important;
            border-radius: 8px !important;
        }}
        
        [data-testid="stJson"] > div {{
            background-color: {card_bg} !important;
            color: {text_color} !important;
        }}
        
        [data-testid="stJson"] * {{
            color: {text_color} !important;
        }}
        
        /* Widget containers */
        .stWidget {{
            background-color: transparent !important;
        }}
        
        .stWidget > div {{
            color: {text_color} !important;
        }}
        
        /* Status indicators */
        .element-container .stSuccess [data-testid="stMarkdownContainer"] p {{
            color: {success_color} !important;
        }}
        
        .element-container .stInfo [data-testid="stMarkdownContainer"] p {{
            color: {info_color} !important;
        }}
        
        .element-container .stWarning [data-testid="stMarkdownContainer"] p {{
            color: {warning_color} !important;
        }}
        
        .element-container .stError [data-testid="stMarkdownContainer"] p {{
            color: {error_color} !important;
        }}
        
        /* Container backgrounds */
        .stContainer {{
            background-color: transparent !important;
        }}
        
        /* Column dividers */
        .element-container {{
            border-color: transparent !important;
        }}
        
        /* Search results styling */
        .stSelectbox [data-testid="stMarkdownContainer"] {{
            color: {text_color} !important;
        }}
        
        /* Form labels with better contrast */
        .stSelectbox > label,
        .stNumberInput > label,
        .stTextInput > label,
        .stSlider > label,
        .stRadio > label {{
            color: {text_color} !important;
            font-weight: 500;
        }}
        
        /* Disabled input styling */
        .stTextInput input[disabled],
        .stNumberInput input[disabled],
        .stSelectbox select[disabled] {{
            background-color: {'#374151' if theme == 'dark' else '#F3F4F6'} !important;
            color: {'#6B7280' if theme == 'dark' else '#9CA3AF'} !important;
            border-color: {border_color} !important;
        }}
        
        /* Progress indicators */
        .stProgress [data-testid="stProgress"] {{
            background-color: {border_color} !important;
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