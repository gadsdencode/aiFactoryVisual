import numpy as np
from typing import Any, Dict


def compute_token_accuracy(eval_pred: Any) -> Dict[str, float]:
    """
    Compute token-level accuracy ignoring labels marked as -100.

    Accepts Hugging Face's EvalPrediction-like object with attributes:
    - predictions: logits array or tuple whose first element is logits, shape (B, T, V)
    - label_ids: ground-truth token ids, shape (B, T)
    """
    try:
        predictions = getattr(eval_pred, 'predictions', None)
        labels = getattr(eval_pred, 'label_ids', None)
        if predictions is None or labels is None:
            return {}

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        pred_ids = np.argmax(predictions, axis=-1)
        mask = labels != -100
        total = int(np.maximum(mask.sum(), 1))
        correct = int((pred_ids[mask] == labels[mask]).sum())
        acc = float(correct) / float(total)
        return {"accuracy": acc}
    except Exception:
        return {}


