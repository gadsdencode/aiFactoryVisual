# AI Factory Visual — ICDU Pipeline Interface

**AI Factory Visual** is a Streamlit-based LLM training and fine-tuning interface for configuring, launching, monitoring, and comparing Hugging Face model training runs.

This repository is the working application layer intended to support ICDU-aware training and evaluation workflows, including model configuration, dataset preparation, supervised fine-tuning, progress monitoring, model comparison, and future integration with ICDU evaluation gates.

---

## What This App Does

AI Factory Visual gives users a browser-based workflow for fine-tuning transformer models without hand-writing the full training loop every time.

The app supports:

- model and dataset configuration
- Hugging Face model loading
- local or Hugging Face dataset workflows
- LoRA / PEFT-style fine-tuning configuration
- supervised fine-tuning through TRL / Transformers
- live training status and metrics
- background training execution
- model comparison dashboards
- optional Hugging Face Hub integration
- Streamlit UI for non-specialist operators

---

## ICDU Context

ICDU stands for **Intent-Conscious Data Unit**.

In the broader ICDU architecture, a model or AI workload should not be promoted simply because it runs. It should be evaluated against explicit intent, governing principles, context, and safety expectations.

The intended direction for this repo is:

```text
Configure model + dataset
   ↓
Fine-tune / generate candidate model
   ↓
Evaluate outputs against ICDU records
   ↓
Score with AI Judge metrics
   ↓
Apply HITL Nuance Grader where needed
   ↓
Stress-test scenario perturbations
   ↓
Promote / escalate / block
```

This repo currently focuses on the model training and visualization layer. ICDU-specific gates, schemas, scoring reports, and stress testing should be integrated as discrete modules rather than buried inside UI code.

---

## Current Application Architecture

```text
.
├── app.py
├── backend/
│   ├── callbacks.py
│   ├── config.py
│   ├── data.py
│   ├── huggingface_integration.py
│   ├── metrics.py
│   ├── model_setup.py
│   ├── trainer.py
│   └── training_manager.py
├── components/
│   ├── configuration.py
│   ├── home.py
│   ├── model_comparison.py
│   └── training_dashboard.py
├── utils/
│   ├── chart_themes.py
│   └── styles.py
├── config.yaml
├── pyproject.toml
└── README.md
```

---

## Major Modules

### `app.py`

Main Streamlit entrypoint.

Responsibilities:

- sets page title, icon, layout, and sidebar behavior
- initializes session state
- initializes the `TrainingManager`
- renders navigation for Home, Training Dashboard, Configuration, and Model Comparison
- displays live training snapshots
- controls dashboard refresh behavior while training is active

Run it with:

```bash
python -m streamlit run app.py --server.port 5000
```

### `backend/config.py`

Defines type-safe application configuration using Pydantic models.

Key config models:

- `QuantizationConfig`
- `LoRAConfig`
- `TrainingConfig`
- `HuggingFaceConfig`
- `AppConfig`

The app loads configuration from:

```text
config.yaml
```

### `backend/training_manager.py`

Orchestrates background training execution and UI-visible training state.

Responsibilities:

- validates configuration
- starts training in a background thread
- loads datasets
- prepares model and tokenizer
- tracks training status, phase, logs, metrics, progress, and final metrics
- supports stop requests through a thread event
- exposes status snapshots for the Streamlit dashboard

Important statuses include:

- `IDLE`
- `RUNNING`
- `COMPLETED`
- `FAILED`

Important phases include:

- `INITIALIZING`
- `LOADING_DATA`
- `TOKENIZING`
- `TRAINING`
- `SAVING_MODEL`
- `ERROR`

### `backend/trainer.py`

Builds and runs the supervised fine-tuning workflow.

Responsibilities:

- creates TRL `SFTConfig`
- normalizes training hyperparameters
- falls back from bitsandbytes optimizers when unavailable
- prepares tokenized datasets with caching
- creates data collators
- builds `SFTTrainer`
- streams logs and metrics through callbacks
- executes training
- runs final evaluation when an evaluation dataset is available

### `components/`

Streamlit UI components.

Expected responsibilities:

- `home.py` — landing page and orientation
- `configuration.py` — model, dataset, and training settings
- `training_dashboard.py` — live training dashboard
- `model_comparison.py` — comparison charts and model performance summaries

### `utils/`

UI support utilities.

Expected responsibilities:

- custom styling
- chart themes
- presentation helpers

---

## Tech Stack

Core app:

- Python 3.11+
- Streamlit
- Pandas
- NumPy
- Pydantic
- PyYAML

ML / training:

- Hugging Face Transformers
- Hugging Face Datasets
- Hugging Face Hub
- PyTorch
- Accelerate
- TRL
- PEFT
- scikit-learn

Visualization:

- Altair
- Plotly
- Matplotlib
- Seaborn

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/gadsdencode/aiFactoryVisual.git
cd aiFactoryVisual
```

### 2. Create an environment

Recommended with `uv`:

```bash
uv sync
```

Alternative with `pip`:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Then install dependencies:

```bash
pip install -e .
```

### 3. Install PyTorch for your hardware

CPU-only installs are simplest, but GPU training requires a CUDA-compatible PyTorch build.

For NVIDIA CUDA environments, install the PyTorch version that matches your driver and CUDA runtime. See the official PyTorch selector before installing.

Example CUDA command pattern:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Use the correct CUDA index for your machine.

### 4. Configure the app

Edit:

```text
config.yaml
```

Typical configuration areas:

- project name
- base model
- output model name
- dataset name or local dataset path
- text column
- quantization settings
- LoRA settings
- training hyperparameters
- Hugging Face Hub settings

### 5. Run the app

```bash
python -m streamlit run app.py --server.port 5000
```

Open the local Streamlit URL shown in the terminal.

---

## Basic Workflow

1. Open the app.
2. Go to **Configuration**.
3. Select or enter a base model.
4. Select a Hugging Face dataset or configure local data.
5. Set training parameters.
6. Start training.
7. Monitor progress from **Training Dashboard**.
8. Review final metrics.
9. Compare results under **Model Comparison**.
10. Export or push the resulting model when configured.

---

## Training Configuration

Training configuration is loaded through `config.yaml` and validated with Pydantic.

Common training fields include:

```yaml
training:
  output_dir: "./results"
  num_train_epochs: 1
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 1
  optim: "paged_adamw_32bit"
  save_steps: 25
  logging_steps: 25
  evaluation_strategy: "no"
  eval_steps: 100
  learning_rate: 0.0002
  weight_decay: 0.001
  fp16: false
  bf16: false
  gradient_checkpointing: false
  max_grad_norm: 0.3
  max_steps: -1
  warmup_ratio: 0.03
  group_by_length: true
  lr_scheduler_type: "constant"
  report_to: "tensorboard"
  packing: false
```

---

## Dataset Configuration

The app is designed to support both Hugging Face datasets and local datasets.

Expected dataset configuration fields include:

```yaml
dataset_name: "imdb"
dataset_split: "train"
text_column: "text"
data_source: "hf"
local_train_path:
local_validation_path:
data_format:
```

For local datasets, ensure the file format and text column match the configuration.

---

## Hugging Face Hub Integration

The app can optionally authenticate and push models to the Hugging Face Hub.

Relevant configuration fields:

```yaml
huggingface:
  push_to_hub: false
  hub_model_id:
  hub_token:
```

Do not commit real tokens to the repository. Use environment variables or local-only config practices for secrets.

---

## Recommended ICDU Integration Points

The cleanest ICDU integration is modular. Recommended future modules:

```text
backend/icdu/
├── schema.py
├── loader.py
├── evaluator.py
├── judge.py
├── hitl.py
├── stress.py
└── reports.py
```

### Suggested Responsibilities

#### `schema.py`

Validate ICDU records against the public ICDU schema.

#### `loader.py`

Load ICDU payloads from local files, object storage, database records, or API requests.

#### `evaluator.py`

Coordinate evaluation runs against trained or candidate models.

#### `judge.py`

Compute quantitative scores such as:

- Intent-Alignment Score
- Principle-Adherence Score
- Application Score

#### `hitl.py`

Provide hooks for human rubric scoring.

#### `stress.py`

Generate controlled scenario perturbations and rerun evaluations.

#### `reports.py`

Emit ICDU-conformant evaluation reports with governance identifiers and pass/fail outcomes.

---

## Proposed ICDU Evaluation Report Shape

A future ICDU report emitted by this application should include:

```json
{
  "icdu_version": "1.0",
  "icdu_governance_id": "example-governance-id",
  "evaluation_timestamp": "2026-07-06T12:00:00Z",
  "evaluator_type": "quantitative",
  "model_id": "local-or-hf-model-id",
  "dataset_id": "dataset-or-scenario-id",
  "perturbation_profile": null,
  "scores": {
    "IAS": 0.86,
    "PAS": 0.92,
    "AS": 0.73
  },
  "pass_fail_outcome": "PASS",
  "notes": "Candidate passed baseline evaluation gates."
}
```

---

## Development Notes

### Run the app locally

```bash
python -m streamlit run app.py --server.port 5000
```

### Useful checks

```bash
python -m compileall .
```

### Suggested branch workflow

```bash
git checkout -b feature/icdu-readme
git add README.md
git commit -m "Improve README for AI Factory Visual pipeline"
git push origin feature/icdu-readme
```

---

## Roadmap

Recommended next steps:

- Add a stable `environment.yml` or tighten `pyproject.toml`.
- Add `.env.example` for local secrets.
- Add sample `config.yaml` variants for CPU, CUDA, and low-VRAM machines.
- Add basic smoke tests for config loading.
- Add model/dataset validation before training starts.
- Add ICDU schema validation module.
- Add AI Judge scoring interface.
- Add HITL rubric capture.
- Add scenario perturbation runner.
- Add evaluation report export.
- Add CI checks for import safety and formatting.

---

## Security and Safety Notes

- Never commit Hugging Face tokens.
- Validate local dataset paths before loading.
- Keep generated model artifacts out of Git unless intentionally tracked through Git LFS or model registry tooling.
- Use small test models before launching long GPU runs.
- Treat evaluation reports as governance artifacts.
- Do not promote a trained model without evaluation gates.

---

## License

This repository currently states that it is licensed under the MIT License. Confirm the final licensing posture before combining this code with ICDU-controlled materials or commercial deployment workflows.

ICDU-related materials may have separate patent, commercial-use, or evaluation-only restrictions.
