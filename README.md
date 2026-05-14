# Module 7 Week A — Lab: Fine-Tune DistilBERT for App-Review Sentiment

Fine-tune DistilBERT on a curated 7,472-review subset of the AARSynth app-review corpus (9 apps × 3 sentiment classes — negative / neutral / positive), evaluate, push the model to Hugging Face Hub, and produce an evaluation report.

Dataset:
- `data/app_reviews_train.csv` — 7,472 reviews (lab uses this; internally splits 80/20).
- `data/app_reviews_eval.csv` — 1,867 reviews (independent holdout for instructor review; not consumed by the lab pipeline by default).
- `fixtures/tiny_app_reviews.csv` — 60-row CI smoke fixture.

Full instructions: see the **Applied Lab guide** linked in TalentLMS.

## Quick start

```bash
pip install -r requirements.txt
huggingface-cli login        # one-time; needs a write-scoped token
python lab.py
```

Outputs (committed):
- `metrics.json`
- `predictions.csv`
- `evaluation-report.md` (you write this)

Local-only (gitignored):
- `model/` (~265 MB) — also pushed to your HF Hub.

## Submission

Open a PR from `lab-7a-finetune-sentiment` into `main`. Paste the PR URL into TalentLMS → Module 7 → Applied Lab 7A.

---

## License

This repository is provided for educational use only. See [LICENSE](LICENSE) for terms.

You may clone and modify this repository for personal learning and practice, and reference code you wrote here in your professional portfolio. Redistribution outside this course is not permitted.
# Module 7 Week A — Fine-Tune DistilBERT for Sentiment Classification

This project fine-tunes `distilbert-base-uncased` on the AARSynth app reviews dataset for 3-class sentiment classification.

The model classifies app review text into:

- Negative
- Neutral
- Positive

## Project goal

The goal of this lab is to build a complete fine-tuning pipeline using Hugging Face Transformers. The pipeline loads the dataset, tokenizes the text, trains a DistilBERT classifier, evaluates the model, saves evaluation artifacts, and pushes the trained model to Hugging Face Hub.

## Files produced

This repo includes the main implementation and evaluation outputs:

- `lab.py` — full training and evaluation pipeline
- `metrics.json` — accuracy, macro-F1, and per-class metrics
- `predictions.csv` — test examples with predicted labels and probabilities
- `confusion_matrix.csv` — confusion matrix for the test split
- `training_log.json` — Trainer log history from the training run
- `evaluation-report.md` — one-page evaluation summary

The local `model/` directory is generated during training but should not be committed to GitHub because the model checkpoint is large.

## Model

- Backbone: `distilbert-base-uncased`
- Task: Sentiment classification
- Number of labels: 3
- Max sequence length: 128
- Epochs: 2
- Batch size: 8
- Learning rate: 2e-5
- Seed: 42

## Hugging Face model

The trained model is published here:

https://huggingface.co/alaafalugi/m7-app-review-sentiment

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
