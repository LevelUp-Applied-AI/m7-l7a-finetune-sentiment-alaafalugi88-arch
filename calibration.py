"""
Module 7 Week A Stretch — Calibration Analysis.

Builds a reliability diagram and computes Expected Calibration Error manually.
"""

import os

import matplotlib.pyplot as plt
import numpy as np


def reliability_diagram(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10):
    """
    Bin predictions by max predicted probability.

    Returns:
      bucket_centers
      bucket_accuracies
      bucket_counts
    """
    probs = np.asarray(probs)
    y_true = np.asarray(y_true)

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = predictions == y_true

    edges = np.linspace(0, 1, n_bins + 1)
    bucket_centers = (edges[:-1] + edges[1:]) / 2
    bucket_accuracies = np.zeros(n_bins)
    bucket_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lower = edges[i]
        upper = edges[i + 1]

        if i == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)

        bucket_counts[i] = np.sum(mask)

        if bucket_counts[i] > 0:
            bucket_accuracies[i] = np.mean(correct[mask])
        else:
            bucket_accuracies[i] = 0.0

    return bucket_centers, bucket_accuracies, bucket_counts


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """
    ECE = sum over bins of:
    (bucket_count / N) * |bucket_accuracy - bucket_confidence|
    """
    probs = np.asarray(probs)
    y_true = np.asarray(y_true)

    confidences = np.max(probs, axis=1)
    centers, accuracies, counts = reliability_diagram(probs, y_true, n_bins=n_bins)

    edges = np.linspace(0, 1, n_bins + 1)
    total = len(y_true)
    ece = 0.0

    for i in range(n_bins):
        lower = edges[i]
        upper = edges[i + 1]

        if i == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)

        if counts[i] > 0:
            bucket_confidence = np.mean(confidences[mask])
            ece += (counts[i] / total) * abs(accuracies[i] - bucket_confidence)

    return float(ece)


def plot_reliability(centers, accs, counts, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.bar(centers, accs, width=0.08, alpha=0.7, label="Model accuracy")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    probs = np.load("manual_eval_probs.npy")
    y_true = np.load("manual_eval_y_true.npy")

    centers, accuracies, counts = reliability_diagram(probs, y_true, n_bins=10)
    ece = expected_calibration_error(probs, y_true, n_bins=10)

    plot_reliability(centers, accuracies, counts, "figures/reliability-diagram.png")

    with open("calibration_results.txt", "w") as f:
        f.write(f"ECE: {ece:.4f}\n\n")
        f.write("Buckets:\n")
        for center, acc, count in zip(centers, accuracies, counts):
            f.write(f"center={center:.2f}, accuracy={acc:.4f}, count={count}\n")

    print("Calibration analysis complete.")
    print(f"ECE: {ece:.4f}")
    print()
    print("Buckets:")
    for center, acc, count in zip(centers, accuracies, counts):
        print(f"center={center:.2f}, accuracy={acc:.4f}, count={count}")


if __name__ == "__main__":
    main()