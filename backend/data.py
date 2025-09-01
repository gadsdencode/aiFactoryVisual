import os
import hashlib
from datasets import load_dataset, Dataset, load_from_disk, DatasetDict
from backend.config import AppConfig
import streamlit as st

# Disk-cached dataset preparation suitable for large corpora
def load_and_prepare_dataset(_config: AppConfig) -> Dataset:
    """
    Loads a dataset from the Hugging Face Hub.

    Args:
        config (AppConfig): The application configuration object.

    Returns:
        Dataset: The loaded dataset.
    """
    try:
        if getattr(_config, 'data_source', 'hf') == 'local':
            # Local files or Arrow dataset directory
            if _config.local_train_path and _config.local_train_path.endswith(('.json', '.jsonl')):
                dataset = load_dataset('json', data_files=_config.local_train_path, split='train')
            elif _config.local_train_path and _config.local_train_path.endswith('.csv'):
                dataset = load_dataset('csv', data_files=_config.local_train_path, split='train')
            elif _config.local_train_path and _config.local_train_path.endswith(('.parquet', '.pq')):
                dataset = load_dataset('parquet', data_files=_config.local_train_path, split='train')
            elif _config.local_train_path and not any(_config.local_train_path.endswith(ext) for ext in ['.json', '.jsonl', '.csv', '.parquet', '.pq']):
                dataset = load_from_disk(_config.local_train_path)
            else:
                dataset = load_dataset(_config.dataset_name, split=_config.dataset_split)
        else:
            dataset = load_dataset(_config.dataset_name, split=_config.dataset_split)
        
        # Ensure a usable text column exists; if not, try to compose from ICDU schema
        dataset = _ensure_text_column(dataset, _config)

        # On-disk caching: key based on data source and text column
        cache_root = os.path.join('.cache', 'datasets')
        os.makedirs(cache_root, exist_ok=True)
        cache_key_src = f"{getattr(_config,'data_source','hf')}|{getattr(_config,'dataset_name','') or getattr(_config,'local_train_path','')}|{getattr(_config,'dataset_split','train')}|{getattr(_config,'text_column','text')}"
        cache_key = hashlib.sha1(cache_key_src.encode('utf-8')).hexdigest()[:16]
        cache_dir = os.path.join(cache_root, cache_key)
        try:
            if not os.path.exists(cache_dir):
                dataset.save_to_disk(cache_dir)
            else:
                # reload ensures memory-map for speed
                dataset = load_from_disk(cache_dir)
        except Exception:
            pass

        # Optional: pre-tokenized shards cache marker dir
        token_cache_root = os.path.join(cache_dir, 'tokenized')
        try:
            os.makedirs(token_cache_root, exist_ok=True)
        except Exception:
            pass
            
        source_desc = _config.local_train_path if getattr(_config, 'data_source', 'hf') == 'local' else _config.dataset_name
        st.success(f"Successfully loaded dataset '{source_desc}' with {len(dataset)} examples.")
        return dataset
    except Exception as e:
        source_desc = getattr(_config, 'local_train_path', None) or _config.dataset_name
        st.error(f"Failed to load dataset '{source_desc}': {e}")
        return None


def load_validation_dataset(_config: AppConfig) -> Dataset | None:
    try:
        if getattr(_config, 'data_source', 'hf') == 'local':
            val_path = getattr(_config, 'local_validation_path', None)
            if not val_path:
                return None
            if val_path.endswith(('.json', '.jsonl')):
                ds = load_dataset('json', data_files=val_path, split='train')
            elif val_path.endswith('.csv'):
                ds = load_dataset('csv', data_files=val_path, split='train')
            elif val_path.endswith(('.parquet', '.pq')):
                ds = load_dataset('parquet', data_files=val_path, split='train')
            else:
                ds = load_from_disk(val_path)
            return _ensure_text_column(ds, _config)
        # HF path: try explicit 'validation' split if present, else None
        try:
            ds = load_dataset(_config.dataset_name, split='validation')
            return _ensure_text_column(ds, _config)
        except Exception:
            return None
    except Exception:
        return None


def _ensure_text_column(dataset: Dataset, _config: AppConfig) -> Dataset:
    """
    Ensure the dataset has the configured text column. If missing, try to
    construct it from ICDU fields.
    """
    target_col = getattr(_config, 'text_column', 'text') or 'text'
    if target_col in dataset.column_names:
        return dataset

    # Detect ICDU schema columns
    icdu_cols = set([
        'persona_archetype', 'governing_principle', 'capability_layer', 'user_intent',
        'context_summary', 'application_prompt', 'ideal_response_final',
        'ideal_response_attributes', 'ideal_response_cot'
    ])
    if icdu_cols.intersection(set(dataset.column_names)):
        def compose_batch(batch):
            out = []
            for i in range(len(next(iter(batch.values())))):
                instr = (batch.get('application_prompt', [None])[i]
                         or batch.get('user_intent', [None])[i])
                ctx = batch.get('context_summary', [None])[i]
                persona_bits = []
                for key in ['persona_archetype', 'governing_principle', 'capability_layer']:
                    vals = batch.get(key, None)
                    val = vals[i] if vals is not None else None
                    if val:
                        persona_bits.append(f"{key.replace('_',' ').title()}: {val}")
                persona = "\n".join(persona_bits) if persona_bits else None
                resp = (batch.get('ideal_response_final', [None])[i]
                        or batch.get('ideal_response_cot', [None])[i])

                parts = []
                if instr:
                    parts.append(f"### Instruction\n{instr}")
                if ctx:
                    parts.append(f"### Context\n{ctx}")
                if persona:
                    parts.append(f"### Persona\n{persona}")
                if resp:
                    parts.append(f"### Response\n{resp}")
                out.append("\n\n".join(parts) if parts else "")
            return {target_col: out}

        dataset = dataset.map(compose_batch, batched=True, batch_size=1000, num_proc=1)
        # If the configured text column was not 'text', still ensure Trainer sees it
        return dataset

    # If we reach here, we couldn't build the text field
    raise ValueError(
        f"The specified text_column '{target_col}' was not found in the dataset and automatic ICDU mapping failed. "
        f"Available columns are: {dataset.column_names}"
    )

# Example usage:
if __name__ == '__main__':
    from backend.config import load_config
    
    # This demonstrates how to use the function.
    # It requires a valid config.yaml file in the root directory.
    try:
        config = load_config()
        dataset = load_and_prepare_dataset(config)
        if dataset:
            print("\nDataset loaded.")
            print("First 5 rows:")
            print(dataset[:5])
            print("\nDataset features:")
            print(dataset.features)
    except Exception as e:
        print(f"An error occurred: {e}")
