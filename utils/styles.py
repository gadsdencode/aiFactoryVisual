import streamlit as st

def apply_custom_styles(theme='light'):
    """Apply custom CSS styles matching the specified design requirements."""
    
    st.markdown("""
    <style>
        /* Import Inter and JetBrains Mono fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600&display=swap');
        
        /* Root variables for consistent theming */
        :root {
            --primary-color: #6366F1;
            --secondary-color: #8B5CF6;
            --success-color: #10B981;
            --warning-color: #F59E0B;
            --error-color: #EF4444;
            --spacing: 20px;
        }
        
        /* Light theme variables */
        .light-theme {
            --background-color: #F8FAFC;
            --text-color: #1E293B;
            --card-background: #FFFFFF;
            --border-color: #E2E8F0;
            --sidebar-background: #FFFFFF;
        }
        
        /* Dark theme variables */
        .dark-theme {
            --background-color: #0F172A;
            --text-color: #F1F5F9;
            --card-background: #1E293B;
            --border-color: #334155;
            --sidebar-background: #1E293B;
        }
        
        /* Main app styling */
        .main {
            font-family: 'Inter', sans-serif;
            color: var(--text-color);
            background-color: var(--background-color);
        }
        
        /* Headers */
        h1, h2, h3 {
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            color: var(--text-color);
        }
        
        /* Code and monospace text */
        code, .stCode, pre {
            font-family: 'JetBrains Mono', monospace;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: var(--sidebar-background);
            border-right: 1px solid var(--border-color);
        }
        
        /* Metric cards styling */
        [data-testid="metric-container"] {
            background-color: var(--card-background);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: var(--spacing);
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        [data-testid="metric-container"] > div {
            color: var(--text-color);
        }
        
        /* Button styling */
        .stButton > button {
            border-radius: 8px;
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            border: none;
            transition: all 0.2s ease;
        }
        
        .stButton > button[kind="primary"] {
            background-color: var(--primary-color);
            color: white;
        }
        
        .stButton > button[kind="primary"]:hover {
            background-color: #5855EB;
            transform: translateY(-1px);
        }
        
        .stButton > button:not([kind="primary"]) {
            background-color: var(--card-background);
            color: var(--text-color);
            border: 1px solid var(--border-color);
        }
        
        .stButton > button:not([kind="primary"]):hover {
            background-color: var(--background-color);
            border-color: var(--primary-color);
        }
        
        /* Selectbox and input styling */
        .stSelectbox > div > div {
            background-color: var(--card-background);
            border: 1px solid var(--border-color);
            border-radius: 8px;
        }
        
        .stNumberInput > div > div > input,
        .stTextInput > div > div > input {
            background-color: var(--card-background);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            font-family: 'Inter', sans-serif;
            color: var(--text-color);
        }
        
        /* Progress bar styling */
        .stProgress > div > div > div {
            background-color: var(--primary-color);
            border-radius: 4px;
        }
        
        /* Alert styling */
        .stAlert {
            border-radius: 8px;
            font-family: 'Inter', sans-serif;
        }
        
        .stSuccess {
            background-color: #ECFDF5;
            border-color: var(--success-color);
            color: #047857;
        }
        
        .stInfo {
            background-color: #EFF6FF;
            border-color: var(--primary-color);
            color: #1D4ED8;
        }
        
        .stWarning {
            background-color: #FFFBEB;
            border-color: var(--warning-color);
            color: #92400E;
        }
        
        .stError {
            background-color: #FEF2F2;
            border-color: var(--error-color);
            color: #DC2626;
        }
        
        /* DataFrame styling */
        .stDataFrame {
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border-color);
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: var(--card-background);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            font-family: 'Inter', sans-serif;
            font-weight: 500;
        }
        
        /* Plotly chart container */
        .js-plotly-plot {
            border-radius: 8px;
        }
        
        /* Spacing utilities */
        .space-small { margin: calc(var(--spacing) / 2) 0; }
        .space-medium { margin: var(--spacing) 0; }
        .space-large { margin: calc(var(--spacing) * 2) 0; }
        
        /* Custom grid system */
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: var(--spacing);
            margin: var(--spacing) 0;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            :root {
                --spacing: 15px;
            }
            
            .main {
                padding: var(--spacing);
            }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Apply theme class to the document body
    theme_class = f"{theme}-theme"
    st.markdown(f"""
    <script>
        document.body.className = '{theme_class}';
    </script>
    """, unsafe_allow_html=True)
