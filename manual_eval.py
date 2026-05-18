"""
Module 7 Week A Stretch — Manual Evaluation.

This file evaluates the fine-tuned sentiment classifier manually.
It does not use Trainer.predict, sklearn metrics, or Hugging Face evaluate.
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def manual_predict(model, tokenizer, texts: list[str], batch_size: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """
    Manual PyTorch inference. No Trainer.predict.

    Returns:
      preds: shape (N,), int class indices
      probs: shape (N, num_classes), probabilities after softmax
    """
    all_probs = []
    all_preds = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]

            encoded = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt",
            )

            encoded = {key: value.to(device) for key, value in encoded.items()}

            outputs = model(**encoded)
            logits = outputs.logits

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_probs)


def compute_classification_report_from_arrays(y_true, y_pred) -> dict:
    """
    Compute accuracy, per-class precision/recall/F1, and macro-F1 using only numpy.
    No sklearn metric helpers, no Hugging Face evaluate.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    per_class = {}

    accuracy = float(np.mean(y_true == y_pred))

    f1_scores = []

    for label in labels:
        true_positive = np.sum((y_true == label) & (y_pred == label))
        false_positive = np.sum((y_true != label) & (y_pred == label))
        false_negative = np.sum((y_true == label) & (y_pred != label))

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[int(label)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

        f1_scores.append(f1)

    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }


def load_eval_data(path: str = "data/app_reviews_eval.csv"):
    """
    Load evaluation data.

    The code supports common column names:
    text/review/content and label/labels.
    """
    if not os.path.exists(path):
        path = "app_reviews_eval.csv"

    df = pd.read_csv(path)

    text_col = None
    for candidate in ["text", "review", "content", "sentence"]:
        if candidate in df.columns:
            text_col = candidate
            break

    label_col = None
    for candidate in ["label", "labels", "sentiment"]:
        if candidate in df.columns:
            label_col = candidate
            break

    if text_col is None or label_col is None:
        raise ValueError(f"Could not find text/label columns. Columns found: {list(df.columns)}")

    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].to_numpy()

    if labels.dtype == object:
        mapping = {
            "negative": 0,
            "neutral": 1,
            "positive": 2,
            "LABEL_0": 0,
            "LABEL_1": 1,
            "LABEL_2": 2,
        }
        labels = np.array([mapping[str(label).strip()] for label in labels])

    return texts, labels.astype(int)


def main():
    model_path = "model"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    texts, y_true = load_eval_data()
    y_pred, probs = manual_predict(model, tokenizer, texts)

    report = compute_classification_report_from_arrays(y_true, y_pred)

    with open("manual_eval_results.json", "w") as f:
        json.dump(report, f, indent=2)

    np.save("manual_eval_probs.npy", probs)
    np.save("manual_eval_y_true.npy", y_true)

    print("Manual evaluation complete.")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Macro-F1: {report['macro_f1']:.4f}")

    for label, metrics in report["per_class"].items():
        print(
            f"Class {label}: "
            f"P={metrics['precision']:.4f}, "
            f"R={metrics['recall']:.4f}, "
            f"F1={metrics['f1']:.4f}"
        )


if __name__ == "__main__":
    main()