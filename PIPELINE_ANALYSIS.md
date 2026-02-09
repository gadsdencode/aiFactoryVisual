# AI Factory Visual — ML Training Pipeline Analysis

**Date:** 2026-02-09
**Scope:** Full codebase review covering code quality, architecture, system design, and ML-pipeline concerns.

---

## Executive Summary

AI Factory Visual is a Streamlit-based web application for fine-tuning Hugging Face Transformer models using LoRA/QLoRA. The codebase is ~2,500 lines of Python across 16 source files organized into `backend/`, `components/`, and `utils/` packages. It provides a config-driven pipeline with real-time dashboard monitoring, HF Hub integration, and model comparison tooling.

**Overall assessment:** The project is a functional prototype with a solid UI layer and reasonable module separation. However, it has significant gaps in training correctness, reproducibility, testing, security, and production readiness that would need to be addressed before any serious ML workload.

### Scoring Summary

| Dimension | Score | Rating |
|---|---|---|
| Pipeline Structure & Modularity | 6/10 | Adequate |
| Training Correctness & Reliability | 4/10 | Needs Work |
| Data & Features | 5/10 | Adequate |
| Experiment Management | 4/10 | Needs Work |
| Performance & Scaling | 4/10 | Needs Work |
| Testing & CI | 1/10 | Critical Gap |
| Observability & Debugging | 5/10 | Adequate |
| Security & Safety | 3/10 | Needs Work |
| Packaging & Dev Experience | 5/10 | Adequate |

---

## 1. Pipeline Structure & Modularity

### Strengths

- **Clear package separation:** `backend/` (ML logic), `components/` (UI), `utils/` (helpers) is a reasonable top-level split.
- **Pydantic configuration models** (`backend/config.py:8-74`): Type-safe, validated configuration with nested `QuantizationConfig`, `LoRAConfig`, `TrainingConfig`, and `HuggingFaceConfig`.
- **Callback abstraction** (`backend/callbacks.py:1-25`): `BaseCallback` provides a framework-agnostic interface that `GenericCallbackAdapter` bridges to transformers' `TrainerCallback`. This is a good pattern for decoupling.
- **TrainingManager as orchestrator** (`backend/training_manager.py:15-579`): Centralizes training lifecycle, state, and config management in a single class.

### Weaknesses

**W1.1 — Backend modules have circular UI dependencies.** Nearly every backend module imports `streamlit` and calls `st.info()`, `st.error()`, `st.success()` directly:

- `backend/data.py:5` — `import streamlit as st`
- `backend/model_setup.py:13` — `import streamlit as st`
- `backend/huggingface_integration.py:1` — `import streamlit as st`

This means backend logic cannot be used outside a Streamlit context (CLI, notebooks, tests, distributed workers). This is the single most impactful architectural issue.

**Recommendation:** Replace all `st.*` calls in backend modules with a logging/event interface. Use Python's `logging` module or pass a callback/observer:

```python
# backend/model_setup.py — BEFORE
st.info(f"Loading base model: {config.base_model}")

# AFTER
import logging
logger = logging.getLogger(__name__)
logger.info(f"Loading base model: {config.base_model}")
```

**W1.2 — TrainingManager is a god object.** At 580 lines, it handles: config loading, YAML I/O, config validation, config transformation for UI namespaces, training thread lifecycle, metrics aggregation, progress estimation, and ETA calculation. This violates single-responsibility and makes it difficult to test individual concerns.

**Recommendation:** Extract into:
- `ConfigManager` — YAML read/write, validation, UI config transforms
- `TrainingOrchestrator` — thread lifecycle, state machine
- `MetricsCollector` — metrics history, progress, ETA

**W1.3 — No dependency injection.** Functions like `setup_model_and_tokenizer()` and `load_and_prepare_dataset()` read config and create objects internally. This makes them impossible to unit test without mocking internals.

**W1.4 — `attached_assets/` contains 10 Python files** that appear to be earlier or alternative implementations (`config_1755889185118.py`, `train_1755889185120.py`, etc.). These are dead code that should be removed or moved to a reference directory outside the package.

---

## 2. Training Correctness & Reliability

### Strengths

- **Seed configuration** exists as `TrainingConfig.seed` with default `42` (`config.py:44` — although this field is actually missing from the current config, only referenced in trainer code).
- **Gradient norm clipping** is configured via `max_grad_norm` (`config.yaml:39`, default `0.3`).
- **Mixed precision** flags (`fp16`/`bf16`) are exposed in config (`config.py:40-41`).
- **Graceful stop mechanism** via `threading.Event` propagated to `control.should_training_stop` in callbacks (`trainer.py:110-111`, `142-143`).

### Weaknesses

**W2.1 — Incomplete seed setting.** The `TrainingConfig` class has no `seed` field in the Pydantic model (`config.py:28-52`). While `torch.manual_seed` may be called somewhere in the training flow, there is no setting of:
- `random.seed()`
- `np.random.seed()`
- `torch.cuda.manual_seed_all()`
- `PYTHONHASHSEED`
- `torch.backends.cudnn.deterministic = True`

The HF `TrainingArguments` does accept a `seed` parameter, but it's never passed in `create_sft_config()` (`trainer.py:190-289`).

**Recommendation:**
```python
# In TrainingConfig
seed: int = 42

# In create_sft_config
return SFTConfig(
    seed=config.training.seed,
    ...
)
```

**W2.2 — No checkpointing/resume support.** While `save_steps` is configured, there is no logic to resume from a checkpoint. `execute_training_loop()` passes `resume_from_checkpoint=None` explicitly (`trainer.py:514`). If training crashes at step 9,000 of 10,000, all progress is lost.

**Recommendation:** Add checkpoint detection and resume logic:
```python
# Find latest checkpoint in output_dir
import glob
checkpoints = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")))
resume_from = checkpoints[-1] if checkpoints else None
trainer.train(resume_from_checkpoint=resume_from)
```

**W2.3 — Labels copy creates train/eval leakage risk.** In `data.py:93`:
```python
enc["labels"] = enc["input_ids"].copy()
```
This sets labels = input_ids for all tokens, meaning the model trains on the entire sequence including the instruction/prompt tokens. For instruction tuning, you typically want to mask instruction tokens (set labels to -100) so the model only learns to predict the response portion.

**W2.4 — `padding="max_length"` wastes compute.** `data.py:90` pads every sample to `max_seq_length` (default 1024). This means short samples waste significant GPU compute on padding tokens. Consider dynamic padding via a proper data collator or `padding=True` with `DataCollatorWithPadding`.

**W2.5 — `torch.set_num_threads(1)` is set globally** in `prepare_tokenized_datasets()` (`trainer.py:300`). This persists beyond tokenization and affects the entire training loop, potentially crippling CPU-side operations like data loading and collation.

**W2.6 — `model.config.pretraining_tp = 1`** (`model_setup.py:132`) is a LLaMA-specific hack that's applied unconditionally to all model architectures.

**W2.7 — Error swallowing.** The codebase has 60+ bare `except Exception: pass` blocks. Many of these silently hide critical failures:

- `model_setup.py:174` — silently fails `prepare_model_for_kbit_training`
- `trainer.py:504` — silently fails `create_optimizer_and_scheduler`
- `data.py:106` — silently fails to save tokenized cache

---

## 3. Data & Features

### Strengths

- **Multi-format support:** JSONL, CSV, Parquet, and Arrow datasets (`data.py:133-141`).
- **Caching with content hashing:** Uses SHA1 of `dataset_desc|split|tokenizer|max_len|column|fingerprint|size` (`data.py:55-56`) for cache invalidation. This is a solid approach.
- **ICDU schema auto-composition:** `_ensure_text_column()` (`data.py:205-257`) automatically constructs training text from structured ICDU fields.
- **Validation dataset support:** Separate loading path for validation data (`data.py:180-202`).

### Weaknesses

**W3.1 — No data validation or schema contracts.** There is no assertion or validation that:
- Required columns exist before tokenization
- Data types are correct
- Text values are non-empty
- No NaN/null values in critical fields

The `_ensure_text_column()` function raises a `ValueError` if the text column is missing and ICDU mapping fails, but there's no validation of data quality within the column.

**Recommendation:** Add a `validate_dataset()` function:
```python
def validate_dataset(dataset: Dataset, text_column: str) -> List[str]:
    """Return list of validation warnings/errors."""
    issues = []
    if text_column not in dataset.column_names:
        issues.append(f"Missing column: {text_column}")
    empty_count = sum(1 for x in dataset[text_column] if not x or not x.strip())
    if empty_count > 0:
        issues.append(f"{empty_count} empty text samples detected")
    return issues
```

**W3.2 — No train/eval preprocessing consistency guarantee.** Tokenization is applied identically to train and eval data (good), but there's no mechanism to verify the same tokenizer version, max_length, or preprocessing pipeline is used across runs. The cache key helps, but there's no manifest or hash stored with the model.

**W3.3 — SHA1 truncated to 16 hex chars** (`data.py:56`) creates a 64-bit collision space. While unlikely for small-scale use, this is unnecessarily risky. Use at least 32 chars (128 bits) or the full hash.

**W3.4 — Cache invalidation is incomplete.** The cache key doesn't include the tokenizer version/hash or the preprocessing logic hash. If you update the `_tokenize_batch` function but the dataset fingerprint doesn't change, stale cached data will be served.

**W3.5 — No data lineage tracking.** There's no record of which exact dataset version, preprocessing config, or tokenizer was used to produce a training run. This makes experiment comparison unreliable.

---

## 4. Experiment Management

### Strengths

- **Config export as JSON** (`components/configuration.py:583-591`): Users can download current config for archival.
- **TensorBoard integration** configured via `report_to: tensorboard` in `config.yaml:44`.
- **Real-time metrics streaming** via `TrainingProgressCallback` and `StreamlitCallback` to queues/manager.

### Weaknesses

**W4.1 — No experiment versioning.** Each training run overwrites `./results/final_model`. There is no run ID, timestamp, or versioning. Previous experiments are lost.

**Recommendation:** Auto-generate run directories:
```python
import datetime
run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(config.training.output_dir, run_id)
```

**W4.2 — No config snapshot with artifacts.** The config used for training is not saved alongside the model. You can't determine what hyperparameters produced a given model checkpoint.

**Recommendation:** Save config.yaml alongside model artifacts:
```python
import shutil
shutil.copy("config.yaml", os.path.join(final_output_dir, "training_config.yaml"))
```

**W4.3 — No W&B/MLflow integration.** TensorBoard is configured but provides limited experiment comparison capabilities. For production ML, consider MLflow or W&B for experiment tracking, artifact management, and model registry.

**W4.4 — Metrics history is in-memory only.** `TrainingManager.metrics_history` is a Python list that's lost when the process ends. For long training runs, this should persist to disk.

**W4.5 — Model comparison uses hardcoded mock data.** `app.py:46-57` initializes `model_data` with hardcoded values, and `utils/data_generator.py` generates fake metrics. There's no mechanism to populate comparison data from actual training runs.

---

## 5. Performance & Scaling

### Strengths

- **4-bit quantization** via BitsAndBytes reduces VRAM requirements significantly.
- **Gradient checkpointing** option available (`model_setup.py:200-205`).
- **TF32 and SDPA optimizations** (`model_setup.py:137-154`): Enables TF32 matmul and mem-efficient SDPA on Ampere+ GPUs.
- **torch.compile support** optionally available (`model_setup.py:208-218`).

### Weaknesses

**W5.1 — `dataloader_num_workers=0`** (`trainer.py:281`). Data loading happens in the main process, meaning the GPU idles while waiting for data. This is a significant throughput bottleneck, especially with on-the-fly tokenization.

**Recommendation:** Set `dataloader_num_workers` to at least 2 (or `min(4, os.cpu_count())`), and enable `dataloader_pin_memory=True` for CUDA:
```python
dataloader_num_workers=min(4, os.cpu_count() or 1),
dataloader_pin_memory=torch.cuda.is_available(),
```

**W5.2 — `dataloader_persistent_workers=False`** (`trainer.py:283`). Worker processes are respawned each epoch, incurring startup overhead.

**W5.3 — `group_by_length=True` combined with `padding="max_length"`** is contradictory. Group-by-length sorts samples to minimize padding, but `max_length` padding overrides this by padding everything to the maximum anyway.

**W5.4 — No profiling or benchmarking.** There's no integration with PyTorch Profiler, `torch.cuda.max_memory_allocated()` tracking, or throughput (samples/sec, tokens/sec) logging.

**W5.5 — No distributed training support.** The pipeline is single-GPU only. There's no `accelerate` launcher integration, no FSDP/DDP configuration, and `device_map={"": 0}` is hardcoded when CUDA is available (`model_setup.py:122`).

**W5.6 — `torch.set_num_threads(1)` persists.** As noted in W2.5, this cripples multi-threaded CPU operations for the entire process lifetime, not just tokenization.

---

## 6. Testing & CI

### Critical Gap

**W6.1 — Zero tests.** There are no test files anywhere in the repository. No unit tests, no integration tests, no smoke tests, no property tests. The only test code found is in `repomix-output.txt` (a repository dump file, not runnable tests).

This is the most significant deficiency in the codebase. Without tests:
- Refactoring is risky (you can't verify behavior is preserved)
- Data transforms may silently produce incorrect outputs
- Configuration changes may break the pipeline
- Regressions are invisible

**Recommendation — Priority test targets:**

```
tests/
├── unit/
│   ├── test_config.py          # Config loading, validation, edge cases
│   ├── test_data.py            # Tokenization, text column composition, caching
│   ├── test_metrics.py         # Token accuracy computation
│   └── test_callbacks.py       # Callback invocation and log formatting
├── integration/
│   ├── test_training_smoke.py  # End-to-end training on tiny dataset (2 samples, 1 step)
│   └── test_model_setup.py     # Model + LoRA setup with small checkpoint
└── conftest.py                 # Fixtures: tiny datasets, mock configs, temp dirs
```

Example critical test:
```python
def test_compute_token_accuracy_ignores_masked():
    """Verify -100 labels are excluded from accuracy calculation."""
    from unittest.mock import MagicMock
    import numpy as np

    eval_pred = MagicMock()
    eval_pred.predictions = np.array([[[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]])
    eval_pred.label_ids = np.array([[1, -100, 0]])

    result = compute_token_accuracy(eval_pred)
    assert result == {"accuracy": 0.5}  # 1/2: token 0 correct (argmax=1==1), token 2 wrong (argmax=1!=0)
```

**W6.2 — No CI automation.** No `.github/workflows/`, no `Makefile`, no `tox.ini`, no pre-commit hooks. There's no automated linting, type checking, or formatting enforcement.

**W6.3 — No type checking.** Despite using type hints in some places, there's no `mypy` or `pyright` configuration. Many functions lack return type annotations.

---

## 7. Observability & Debugging

### Strengths

- **Real-time log streaming** via queue-based `StreamlitCallback` with structured metric data.
- **Step/epoch/loss/LR/grad_norm** are all captured and displayed in the dashboard.
- **ETA estimation** with exponential moving average smoothing (`training_manager.py:258-268`).
- **Phase tracking** (`INITIALIZING`, `LOADING_DATA`, `TOKENIZING`, `TRAINING`, `SAVING_MODEL`, `ERROR`) in TrainingManager.
- **Filterable log viewer** in dashboard with INFO/WARNING/ERROR levels.

### Weaknesses

**W7.1 — No structured logging.** All logging uses either `st.*` calls or plain string `manager.add_log()`. There's no structured format (JSON logs), no log levels, no correlation IDs. Debugging production issues from string-formatted logs is painful.

**W7.2 — Logs are capped to 1000 lines** (`training_manager.py:223`). For long training runs, early logs (including configuration and initialization details) are lost.

**W7.3 — No sanity checks on training metrics.** There's no detection of:
- Loss going to NaN/Inf
- Loss not decreasing after N steps
- Learning rate reaching 0 unexpectedly
- Gradient norm exploding

**Recommendation:** Add a sanity-check callback:
```python
class SanityCheckCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        loss = logs.get('loss')
        if loss is not None and (math.isnan(loss) or math.isinf(loss)):
            control.should_training_stop = True
            raise ValueError(f"Training diverged: loss={loss} at step {state.global_step}")
```

**W7.4 — No GPU memory tracking.** `torch.cuda.max_memory_allocated()` is never logged. OOM errors happen without forewarning.

---

## 8. Security & Safety

### Weaknesses

**W8.1 — HF token in config.yaml.** `config.yaml:57` has `hub_token: ''` — this means users are encouraged to put their HF token in a YAML file. The `.gitignore` does NOT exclude `config.yaml`, so tokens can be committed.

**Recommendation:** Never store tokens in config files. Use environment variables or a secrets manager:
```python
hub_token = os.environ.get("HF_TOKEN") or config.huggingface.hub_token
```

**W8.2 — `trust_remote_code=True`** (`model_setup.py:159`). This allows arbitrary Python code execution from model repositories on Hugging Face Hub. This is a significant security risk for untrusted models.

**Recommendation:** Default to `trust_remote_code=False` and require explicit user opt-in for specific models that need it.

**W8.3 — No dependency pinning.** `requirements.txt` has no version pins:
```
streamlit
torch
transformers
...
```
Any `pip install` may pull breaking changes. This also makes the build non-reproducible.

**Recommendation:** Pin all dependencies with exact versions:
```
streamlit==1.39.0
torch==2.4.1
transformers==4.46.0
```
Or better, use the existing `uv.lock` as the source of truth and document it.

**W8.4 — `hf_login(token, add_to_git_credential=True)`** (`huggingface_integration.py:82`). This persists credentials into the git credential store. This is an inappropriate default for an application that may be running in shared environments.

**W8.5 — `.gitignore` is minimal.** Only `/data` and `/results` are excluded. Missing:
- `.cache/`
- `*.pyc`, `__pycache__/`
- `.env`
- `config.yaml` (if it contains secrets)
- `*.egg-info/`
- `.venv/`
- `wandb/`
- `tensorboard/`

**W8.6 — No input sanitization on file paths.** `config.yaml` `local_train_path` and `local_validation_path` values are passed directly to `load_dataset()` without any path traversal protection. A malicious config could read arbitrary files.

---

## 9. Packaging & Dev Experience

### Strengths

- **Multiple deployment targets:** Docker (`Dockerfile`), Conda (`environment.yml`), Replit (`.replit`), and standard pip (`requirements.txt`).
- **One-command run:** `streamlit run app.py` works as documented.
- **Docker health check** included in `Dockerfile:22`.
- **Clear README** (present, though not reviewed in detail).

### Weaknesses

**W9.1 — Inconsistent Python version targets.** Dockerfile uses Python 3.10, `environment.yml` specifies 3.11, and `.replit` uses 3.11. There's no `python_requires` constraint in `pyproject.toml`.

**W9.2 — Dual dependency specs.** `requirements.txt` and `pyproject.toml` both define dependencies but may drift. Choose one source of truth.

**W9.3 — No Makefile or task runner.** Common operations (run, test, lint, format, build docker) should be scripted:
```makefile
.PHONY: run test lint format docker

run:
    streamlit run app.py --server.port 5000

test:
    pytest tests/ -v --tb=short

lint:
    ruff check .
    mypy backend/

format:
    ruff format .
```

**W9.4 — No development setup documentation.** There's no CONTRIBUTING.md or developer onboarding guide explaining how to set up the development environment, run tests, or contribute.

**W9.5 — Docker `HEALTHCHECK CMD streamlit hello`** (`Dockerfile:22`) runs `streamlit hello`, not a check against the actual application endpoint. This will always succeed even if the app is crashing.

**Recommendation:** Use a proper HTTP health check:
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s CMD curl -f http://localhost:8501/_stcore/health || exit 1
```

---

## 10. Cross-Cutting Concerns

### 10.1 — Exception Handling Anti-Pattern

The most pervasive code quality issue is the blanket `except Exception: pass` pattern. A grep reveals **60+ instances**. Examples:

| Location | What's silenced |
|---|---|
| `model_setup.py:174` | `prepare_model_for_kbit_training` failure |
| `trainer.py:300-303` | `torch.set_num_threads(1)` failure |
| `trainer.py:504` | `create_optimizer_and_scheduler` failure |
| `data.py:106` | Cache save failure |
| `data.py:66-68` | Torch format conversion failure |
| `training_manager.py:88-90` | Epoch count parsing |

**Recommendation:** Replace with specific exception types and at minimum log the error:
```python
# BEFORE
except Exception:
    pass

# AFTER
except (ValueError, RuntimeError) as e:
    logger.warning(f"Non-critical failure in X: {e}")
```

### 10.2 — Excessive Use of `getattr` with Defaults

The code uses `getattr(obj, 'attr', default)` extensively even on Pydantic models where attributes are guaranteed to exist:
```python
# trainer.py:197 — config.training always has 'optim' per Pydantic
chosen_optim = getattr(config.training, 'optim', 'adamw_torch') or 'adamw_torch'
```
This obscures the actual data model and makes it unclear what's optional vs. required. With Pydantic, use direct attribute access and let missing fields fail at config load time.

### 10.3 — Thread Safety Concerns

`TrainingManager` uses a `Lock` for some operations but not consistently:
- `reset_training()` sets `self.stop_event = None` outside the lock (`training_manager.py:188`)
- `get_status_snapshot()` reads `self.last_step` and `self.last_step_time` outside the lock before the time comparison (`training_manager.py:260-261`)
- `add_metrics()` constructs the entry dict outside the lock but should be fine since it only reads from the `metrics` parameter

---

## Prioritized Recommendations

### P0 — Critical (Do First)

1. **Remove `trust_remote_code=True`** or gate it behind explicit config (`model_setup.py:159`)
2. **Add `seed` to `SFTConfig`** to ensure training reproducibility (`trainer.py:261`)
3. **Stop storing tokens in `config.yaml`** — use env vars (`config.yaml:57`, `huggingface_integration.py:82`)
4. **Pin dependency versions** in `requirements.txt`
5. **Expand `.gitignore`** to exclude `.cache/`, `__pycache__/`, `.env`, secrets

### P1 — High Priority

6. **Remove `import streamlit` from backend modules** — decouple ML logic from UI
7. **Add checkpoint resume** — save/restore from latest checkpoint on failure
8. **Fix label masking** — mask instruction tokens in labels for proper SFT
9. **Set `dataloader_num_workers >= 2`** and `pin_memory=True`
10. **Add a basic test suite** — at minimum: config loading, tokenization, metric computation
11. **Add NaN/Inf loss detection callback** to stop training early on divergence

### P2 — Medium Priority

12. **Break up TrainingManager** into ConfigManager, TrainingOrchestrator, MetricsCollector
13. **Add experiment run directories** with timestamped output paths
14. **Save config snapshot** alongside model artifacts
15. **Replace `padding="max_length"` with dynamic padding** via `DataCollatorWithPadding`
16. **Remove `torch.set_num_threads(1)`** or scope it to tokenization only
17. **Replace bare `except Exception: pass`** with specific exceptions and logging
18. **Add Makefile** with common dev operations (run, test, lint)
19. **Use 128-bit cache hashes** instead of 64-bit truncated SHA1

### P3 — Nice to Have

20. **Add W&B or MLflow** integration for experiment tracking
21. **Add distributed training** support via `accelerate` config
22. **Add GPU memory tracking** in callbacks
23. **Add data validation** checks (empty samples, schema conformance)
24. **Clean up `attached_assets/`** — move to a reference directory or remove
25. **Add type checking** with mypy and CI enforcement
26. **Fix Docker HEALTHCHECK** to use HTTP endpoint
27. **Unify Python version** across all deployment configs

---

## Detailed File-Level Findings

### `backend/config.py`
- **Line 3:** Uses deprecated `@validator` (Pydantic v1 API). Should use `@field_validator` for Pydantic v2.
- **Lines 14-18:** `validate_compute_dtype` converts string to `torch.dtype` at validation time. This means serialization (`model.dict()`) will contain a `torch.dtype` object, not a string — breaks JSON serialization.
- **Line 106:** `config.training.dict()` uses deprecated Pydantic v1 method. Should be `config.training.model_dump()`.
- **Missing:** `seed` field in `TrainingConfig`.

### `backend/data.py`
- **Lines 27-30:** Redundant re-import of `hashlib`, `os`, and `load_from_disk` — already imported at module level.
- **Line 56:** SHA1 truncated to 16 hex characters (64 bits).
- **Line 93:** `labels = input_ids.copy()` — no instruction masking.
- **Line 117:** On tokenization failure, returns raw untokenized dataset. This will cause a cryptic error downstream in the trainer.

### `backend/model_setup.py`
- **Line 159:** `trust_remote_code=True` — security risk.
- **Line 132:** `pretraining_tp = 1` applied unconditionally.
- **Line 160:** `tokenizer.pad_token = tokenizer.eos_token` — this is a common but problematic practice. Some models (e.g., LLaMA) have undefined padding behavior with EOS token as pad. Consider using a dedicated pad token.

### `backend/trainer.py`
- **Lines 281-283:** `dataloader_num_workers=0, pin_memory=False, persistent_workers=False` — suboptimal defaults.
- **Line 300:** `torch.set_num_threads(1)` — global side effect.
- **Line 514:** `resume_from_checkpoint=None` — no checkpoint resume.
- **Lines 578-579:** Model saving does not save the config, training args, or any metadata.

### `backend/training_manager.py`
- **Line 188:** `self.stop_event = None` set outside lock.
- **Lines 402-405:** `warmup_steps` estimated as `int(round(warmup_ratio * 10000))` — this is a fragile heuristic that doesn't account for actual training steps.
- **Line 519:** `yaml.safe_dump(yaml_cfg, f, sort_keys=False)` — no error handling for YAML serialization of non-string types (e.g., `torch.dtype` from the validator).

### `backend/metrics.py`
- **Line 24:** `np.maximum(mask.sum(), 1)` prevents division by zero but silently returns 0% accuracy when all labels are -100.

### `components/training_dashboard.py`
- **Lines 400-433:** `display_training_dashboard()` uses a `while` loop with `time.sleep(5)`. This blocks the Streamlit main thread and prevents UI interaction during training.
- **Line 426:** `fillna(method='ffill')` is deprecated in pandas >= 2.0. Use `ffill()` instead.

---

## Suggested Module Boundaries (Target Architecture)

```
backend/
├── config/
│   ├── schema.py          # Pydantic models (no imports beyond stdlib + pydantic)
│   ├── loader.py          # YAML I/O, validation
│   └── transforms.py      # Config -> UI namespace transforms
├── data/
│   ├── loading.py         # Dataset loading (HF Hub, local files)
│   ├── preprocessing.py   # Tokenization, text composition
│   ├── validation.py      # Schema checks, data quality
│   └── caching.py         # On-disk cache management
├── training/
│   ├── setup.py           # Model + tokenizer + LoRA initialization
│   ├── trainer.py         # SFTTrainer construction and config
│   ├── loop.py            # Training execution, checkpointing, resume
│   ├── callbacks.py       # All callback implementations
│   └── metrics.py         # Metric computation functions
├── experiment/
│   ├── tracker.py         # Experiment run management, artifact saving
│   └── comparison.py      # Model comparison data aggregation
└── integrations/
    └── huggingface.py     # HF Hub API interactions
```

Key principles:
- No module imports `streamlit`
- All logging goes through Python `logging`
- Configuration is passed via dependency injection
- Each module can be tested independently
