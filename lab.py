import os
import json
import time
import numpy as np
import pandas as pd

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


MODEL_NAME = "distilbert-base-uncased"
HUB_MODEL_ID = "m7-app-review-sentiment"
SEED = 42
MAX_LENGTH = 128
NUM_LABELS = 3


def get_data_path():
    return os.environ.get("DATA_PATH", "data/app_reviews_train.csv")


def _prepare_labels(df):
    """
    Make sure labels are numeric for Hugging Face Trainer.
    Supports either:
    - numeric labels: 0, 1, 2
    - string labels: negative, neutral, positive
    """

    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column.")

    if "text" not in df.columns:
        raise ValueError("Dataset must contain a 'text' column.")

    label_col = df["label"]

    if pd.api.types.is_numeric_dtype(label_col):
        df["label"] = df["label"].astype(int)

        unique_labels = sorted(df["label"].unique().tolist())

        if unique_labels == [0, 1, 2]:
            id2label = {
                0: "negative",
                1: "neutral",
                2: "positive",
            }
        else:
            id2label = {int(i): f"LABEL_{int(i)}" for i in unique_labels}

        label2id = {v: k for k, v in id2label.items()}
        return df, id2label, label2id

    normalized = df["label"].astype(str).str.lower().str.strip()

    preferred_order = ["negative", "neutral", "positive"]
    unique_values = sorted(normalized.unique().tolist())

    if all(label in unique_values for label in preferred_order):
        label2id = {
            "negative": 0,
            "neutral": 1,
            "positive": 2,
        }
    else:
        label2id = {label: idx for idx, label in enumerate(unique_values)}

    df["label"] = normalized.map(label2id).astype(int)
    id2label = {idx: label for label, idx in label2id.items()}

    return df, id2label, label2id


def prepare_dataset(data_path, test_size=0.2, seed=42):
    """
    Load CSV, convert to Hugging Face Dataset, then split train/test.
    """
    df = pd.read_csv(data_path)
    df, _, _ = _prepare_labels(df)

    dataset = Dataset.from_pandas(df, preserve_index=False)

    ds_dict = dataset.train_test_split(
        test_size=test_size,
        seed=seed,
    )

    return ds_dict


def tokenize_dataset(ds_dict, tokenizer, max_length=128):
    """
    Tokenize the train/test DatasetDict.
    No padding here. DataCollatorWithPadding handles dynamic padding.
    """

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    return ds_dict.map(tokenize_fn, batched=True)


def make_training_args(
    output_dir="model",
    learning_rate=2e-5,
    lr=None,
    epochs=None,
    batch_size=None,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    seed=42,
    **kwargs,
):
    """
    Build TrainingArguments for the Trainer.
    Supports lr, epochs, and batch_size aliases used by tests.
    """

    if lr is not None:
        learning_rate = lr

    if epochs is not None:
        num_train_epochs = epochs

    if batch_size is not None:
        per_device_train_batch_size = batch_size
        per_device_eval_batch_size = batch_size

    return TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=seed,
        report_to="none",
        push_to_hub=True,
        hub_model_id=HUB_MODEL_ID,
    )


def compute_metrics(eval_pred):
    """
    Compute accuracy and macro-F1 during Trainer evaluation.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def train_classifier(
    tokenized_ds: DatasetDict,
    model_name: str,
    training_args: TrainingArguments,
    tokenizer,
    num_labels: int,
) -> Trainer:
    """
    Fine-tune a pre-trained model on the tokenized dataset.
    """

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    return trainer


def evaluate_classifier(trainer: Trainer, tokenized_test) -> dict:
    """
    Evaluate the fine-tuned classifier on the test split.
    Returns accuracy, macro-F1, per-class F1, precision, and recall.
    """

    prediction_output = trainer.predict(tokenized_test)

    logits = prediction_output.predictions
    labels = prediction_output.label_ids
    preds = np.argmax(logits, axis=-1)

    id2label = trainer.model.config.id2label
    label_ids = sorted([int(k) for k in id2label.keys()])
    label_names = [id2label[i] for i in label_ids]

    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")

    per_class_f1_values = f1_score(
        labels,
        preds,
        average=None,
        labels=label_ids,
        zero_division=0,
    )

    per_class_precision_values = precision_score(
        labels,
        preds,
        average=None,
        labels=label_ids,
        zero_division=0,
    )

    per_class_recall_values = recall_score(
        labels,
        preds,
        average=None,
        labels=label_ids,
        zero_division=0,
    )

    per_class_f1 = {
        label_names[i]: float(per_class_f1_values[i])
        for i in range(len(label_names))
    }

    per_class_precision = {
        label_names[i]: float(per_class_precision_values[i])
        for i in range(len(label_names))
    }

    per_class_recall = {
        label_names[i]: float(per_class_recall_values[i])
        for i in range(len(label_names))
    }

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
    }


def _softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(logits)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def write_predictions_csv(trainer, tokenized_test, raw_test, output_path="predictions.csv"):
    """
    Write per-row predictions with full probability distribution.
    """

    prediction_output = trainer.predict(tokenized_test)
    logits = prediction_output.predictions
    labels = prediction_output.label_ids
    preds = np.argmax(logits, axis=-1)
    probs = _softmax(logits)

    id2label = trainer.model.config.id2label
    label_ids = sorted([int(k) for k in id2label.keys()])
    label_names = [id2label[i] for i in label_ids]

    rows = []

    for i in range(len(labels)):
        true_label = id2label[int(labels[i])]
        predicted_label = id2label[int(preds[i])]
        predicted_probability = float(probs[i][int(preds[i])])

        row = {
            "text": raw_test[i]["text"],
            "label": true_label,
            "predicted_label": predicted_label,
            "predicted_probability": predicted_probability,
        }

        for label_id, label_name in zip(label_ids, label_names):
            row[f"prob_{label_name}"] = float(probs[i][label_id])

        rows.append(row)

    pd.DataFrame(rows).to_csv(output_path, index=False)


def write_confusion_matrix_csv(trainer, tokenized_test, output_path="confusion_matrix.csv"):
    """
    Write confusion matrix as square CSV.
    Rows = true label, columns = predicted label.
    """

    prediction_output = trainer.predict(tokenized_test)

    logits = prediction_output.predictions
    labels = prediction_output.label_ids
    preds = np.argmax(logits, axis=-1)

    id2label = trainer.model.config.id2label
    label_ids = sorted([int(k) for k in id2label.keys()])
    label_names = [id2label[i] for i in label_ids]

    cm = confusion_matrix(labels, preds, labels=label_ids)

    cm_df = pd.DataFrame(
        cm,
        index=label_names,
        columns=label_names,
    )

    cm_df.to_csv(output_path)


def write_evaluation_report(
    metrics,
    data_path,
    training_time_seconds,
    hub_url,
    output_path="evaluation-report.md",
):
    """
    Create a simple one-page evaluation report.
    """

    df = pd.read_csv(data_path)
    df, _, _ = _prepare_labels(df)

    total_examples = len(df)
    label_distribution = df["label"].value_counts().sort_index().to_dict()

    train_size = int(total_examples * 0.8)
    test_size = total_examples - train_size

    predictions_df = pd.read_csv("predictions.csv")
    confusion_df = pd.read_csv("confusion_matrix.csv", index_col=0)

    error_examples = predictions_df[
        predictions_df["label"] != predictions_df["predicted_label"]
    ].head(3)

    if len(error_examples) == 0:
        error_section = (
            "No clear misclassified examples were found in the saved predictions. "
            "The model predicted all sampled test rows correctly.\n"
        )
    else:
        error_lines = []
        for _, row in error_examples.iterrows():
            gold_label = row["label"]
            gold_prob_col = f"prob_{gold_label}"
            gold_prob = row[gold_prob_col] if gold_prob_col in row else "N/A"

            error_lines.append(
                f"- Sentence: {row['text']}\n"
                f"  - Gold label: {gold_label}\n"
                f"  - Predicted label: {row['predicted_label']}\n"
                f"  - Predicted probability for gold label: {gold_prob}\n"
                f"  - Analysis: This example may be difficult because the sentence contains wording that can be interpreted differently depending on context.\n"
            )

        error_section = "\n".join(error_lines)

    per_class_rows = []
    for label_name in metrics["per_class_f1"].keys():
        per_class_rows.append(
            f"| {label_name} | "
            f"{metrics['per_class_f1'][label_name]:.4f} | "
            f"{metrics['per_class_precision'][label_name]:.4f} | "
            f"{metrics['per_class_recall'][label_name]:.4f} |"
        )

    confusion_markdown = confusion_df.to_markdown()

    report = f"""# Module 7 Week A — Lab Evaluation Report

## Dataset

The dataset contains AARSynth app review sentences for sentiment classification. It has {total_examples} examples with this numeric label distribution: {label_distribution}. The split is approximately {train_size} training examples and {test_size} test examples.

## Model and hyperparameters

- Backbone: distilbert-base-uncased
- Number of labels: 3
- Learning rate: 2e-5
- Epochs: 2
- Batch size: 8
- Max length: {MAX_LENGTH}
- Seed: {SEED}
- Training time on my machine: {training_time_seconds:.2f} seconds

## Metrics on the test split

| Metric | Value |
|---|---|
| Accuracy | {metrics["accuracy"]:.4f} |
| Macro-F1 | {metrics["macro_f1"]:.4f} |

## Per-class metrics

| Class | F1 | Precision | Recall |
|---|---:|---:|---:|
{chr(10).join(per_class_rows)}

## Confusion matrix

{confusion_markdown}

## Three qualitative error examples

{error_section}

## Hugging Face Hub model URL

{hub_url}
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)


def main():
    set_seed(SEED)

    start_time = time.time()

    data_path = get_data_path()

    df = pd.read_csv(data_path)
    df, id2label, label2id = _prepare_labels(df)
    num_labels = len(id2label)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    ds = prepare_dataset(data_path)
    tokenized_ds = tokenize_dataset(ds, tokenizer, max_length=MAX_LENGTH)

    training_args = make_training_args(
        output_dir="model",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        seed=SEED,
    )

    trainer = train_classifier(
        tokenized_ds=tokenized_ds,
        model_name=MODEL_NAME,
        training_args=training_args,
        tokenizer=tokenizer,
        num_labels=num_labels,
    )

    trainer.model.config.id2label = id2label
    trainer.model.config.label2id = label2id

    metrics = evaluate_classifier(trainer, tokenized_ds["test"])

    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    write_predictions_csv(
        trainer=trainer,
        tokenized_test=tokenized_ds["test"],
        raw_test=ds["test"],
        output_path="predictions.csv",
    )

    write_confusion_matrix_csv(
        trainer=trainer,
        tokenized_test=tokenized_ds["test"],
        output_path="confusion_matrix.csv",
    )

    trainer.save_model("model")

    tokenizer.save_pretrained("model")

    hub_url = f"https://huggingface.co/alaafalugi/{HUB_MODEL_ID}"

    try:
        trainer.push_to_hub(HUB_MODEL_ID)
        tokenizer.push_to_hub(HUB_MODEL_ID)
    except Exception as e:
        print("Warning: push_to_hub failed.")
        print("You can still run tests locally.")
        print(f"Push error: {e}")

    training_time_seconds = time.time() - start_time

    write_evaluation_report(
        metrics=metrics,
        data_path=data_path,
        training_time_seconds=training_time_seconds,
        hub_url=hub_url,
        output_path="evaluation-report.md",
    )

    print("Done.")
    print("Metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"Hub URL: {hub_url}")


if __name__ == "__main__":
    main()