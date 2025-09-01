AI Factory Visual 🏭

🤖AI Factory Visual is a web-based application designed to simplify the process of fine-tuning Hugging Face Transformer models. It provides an intuitive user interface to configure training parameters, select models and datasets, monitor training progress in real-time, and compare the performance of different models.This tool is built for both beginners and experts in machine learning who want to experiment with fine-tuning without writing extensive boilerplate code.

✨ Features

Interactive UI: A user-friendly interface built with Streamlit to manage the entire fine-tuning workflow.

Easy Configuration: Set up your training runs by selecting pre-trained models, datasets from the Hugging Face Hub, and tuning hyperparameters through simple widgets.

Real-time Monitoring: A live dashboard to track key metrics like training/validation loss and accuracy during the fine-tuning process.

Model Comparison: Visualize and compare the performance of different fine-tuned models to easily identify the best-performing one.

Hugging Face Hub Integration: Automatically download models and datasets, and optionally push your fine-tuned models back to the Hub.

🛠️ Tech StackFrontend: StreamlitBackend: PythonML/AI: Hugging Face Transformers, PyTorch, Accelerate, DatasetsPlotting: Altair

📂 Project Structure.

├── app.py                      # Main Streamlit application file
├── backend/                    # Core logic for model training and management
│   ├── config.py               # Handles configuration
│   ├── data.py                 # Data loading and processing
│   ├── huggingface_integration.py # Manages interaction with HF Hub
│   ├── metrics.py              # Metrics calculation
│   ├── model_setup.py          # Model and tokenizer setup
│   ├── trainer.py              # Custom training logic
│   └── training_manager.py     # Orchestrates the training process
├── components/                 # UI components for the Streamlit app
│   ├── configuration.py        # Configuration sidebar UI
│   ├── model_comparison.py     # Model comparison dashboard UI
│   └── training_dashboard.py   # Training dashboard UI
├── utils/                      # Utility scripts
│   ├── chart_themes.py         # Themes for charts
│   └── styles.py               # CSS styles
├── config.yaml                 # Main configuration file for the app
├── pyproject.toml              # Project dependencies
└── README.md                   # This file

🚀 Getting Started

# create env
conda env create -f environment.yml
conda activate aifactory-visual

# GPU users (optional): 
install CUDA-enabled PyTorch

# (Skip this if you want CPU-only. Windows/NVIDIA example:)
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

# Run command
python -m streamlit run app.py --server.port 5000

🔧 ConfigurationYou can modify the config.yaml file to change default settings, such as the default model, dataset, and available choices in the UI.app_config:
  title: "AI Factory Visual"
  layout: "wide"

model_config:
  default_model: "distilbert-base-uncased"

dataset_config:
  default_dataset: "imdb"
  
🤝 Contributing

Contributions are welcome! 

If you have ideas for new features, improvements, or bug fixes, please open an issue or submit a pull request.

Fork the ProjectCreate your Feature Branch (git checkout -b feature/AmazingFeature)Commit your Changes (git commit -m 'Add some AmazingFeature')Push to the Branch (git push origin feature/AmazingFeature)Open a Pull Request📄 LicenseThis project is licensed under the MIT License - see the LICENSE file for details.