import streamlit as st


def render_home():
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


