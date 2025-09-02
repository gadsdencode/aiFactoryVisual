import streamlit as st


def render_home():
    # Welcome Tour / Tutorial (dismissible)
    if 'show_welcome_tour' not in st.session_state:
        st.session_state['show_welcome_tour'] = True
    if 'welcome_step' not in st.session_state:
        st.session_state['welcome_step'] = 0

    if st.session_state.get('show_welcome_tour', True):
        step = int(st.session_state.get('welcome_step', 0))
        steps = [
            "Welcome to AI Factory! Let's get you started.",
            "First, go to the 'Configuration' page to set up your model.",
            "Next, head to the 'Training Dashboard' to start and monitor your training.",
            "Finally, you can compare your model with others on the 'Model Comparison' page.",
        ]

        st.info(steps[step])

        col_a, col_b, col_c = st.columns([1, 1, 2])
        with col_a:
            if st.button("Next ➜", key=f"welcome_next_{step}"):
                st.session_state['welcome_step'] = min(step + 1, len(steps) - 1)
                st.rerun()
        with col_b:
            dont_show = st.checkbox("Don't show this again", key=f"dont_show_{step}")
            if dont_show:
                st.session_state['show_welcome_tour'] = False
                st.rerun()
        with col_c:
            # Quick navigation buttons for the highlighted pages
            if step == 1:
                if st.button("Go to Configuration", key="tour_go_config"):
                    st.session_state['_nav_target'] = 'Configuration'
                    st.rerun()
            elif step == 2:
                if st.button("Go to Training Dashboard", key="tour_go_dashboard"):
                    st.session_state['_nav_target'] = 'Training Dashboard'
                    st.rerun()
            elif step == 3:
                if st.button("Go to Model Comparison", key="tour_go_compare"):
                    st.session_state['_nav_target'] = 'Model Comparison'
                    st.rerun()

    st.markdown(
        """
        Welcome to the AI Factory LLM Trainer! This application allows you to fine-tune, monitor, and compare various Large Language Models (LLMs).

        ### **Key Features:**

        - **📊 Training Dashboard:** Monitor the training process in real-time with live metrics and logs.
        - **⚙️ Configuration:** Easily configure your training jobs, including model selection, hyperparameters, and datasets.
        - **⚖️ Model Comparison:** Compare the performance of different models based on key metrics.

        ### **How to Get Started:**

        1. Navigate to the **Configuration** page to set up your training job.
        2. Start the training and monitor the progress on the **Training Dashboard**.
        3. Once the training is complete, you can analyze the results and compare different models on the **Model Comparison** page.
        """
    )


