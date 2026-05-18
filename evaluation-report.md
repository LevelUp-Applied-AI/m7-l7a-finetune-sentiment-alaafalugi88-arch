# Module 7 Week A — Lab Evaluation Report

## Dataset

The dataset contains AARSynth app review sentences for sentiment classification. It has 7472 examples with this numeric label distribution: {0: 2519, 1: 2442, 2: 2511}. The split is approximately 5977 training examples and 1495 test examples.

## Model and hyperparameters

- Backbone: distilbert-base-uncased
- Number of labels: 3
- Learning rate: 2e-5
- Epochs: 2
- Batch size: 8
- Max length: 128
- Seed: 42
- Training time on my machine: 1192.21 seconds

## Metrics on the test split

| Metric | Value |
|---|---|
| Accuracy | 0.6321 |
| Macro-F1 | 0.6298 |

## Per-class metrics

| Class | F1 | Precision | Recall |
|---|---:|---:|---:|
| negative | 0.7159 | 0.7123 | 0.7194 |
| neutral | 0.4855 | 0.4671 | 0.5054 |
| positive | 0.6882 | 0.7184 | 0.6604 |

## Confusion matrix

|          |   negative |   neutral |   positive |
|:---------|-----------:|----------:|-----------:|
| negative |        359 |       120 |         20 |
| neutral  |        111 |       234 |        118 |
| positive |         34 |       147 |        352 |

## Three qualitative error examples

- Sentence: good, but slow workflow.
  - Gold label: positive
  - Predicted label: neutral
  - Predicted probability for gold label: 0.2777950167655945
  - Analysis: This example may be difficult because the sentence contains wording that can be interpreted differently depending on context.

- Sentence: nice app to use with friends
  - Gold label: neutral
  - Predicted label: positive
  - Predicted probability for gold label: 0.1456903517246246
  - Analysis: This example may be difficult because the sentence contains wording that can be interpreted differently depending on context.

- Sentence: everthing is tought before its use
  - Gold label: neutral
  - Predicted label: negative
  - Predicted probability for gold label: 0.3785228729248047
  - Analysis: This example may be difficult because the sentence contains wording that can be interpreted differently depending on context.


## Hugging Face Hub model URL

https://huggingface.co/alaafalugi/m7-app-review-sentiment
