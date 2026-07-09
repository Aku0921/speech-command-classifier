from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
def evaluate(
    classifier,
    test_embeddings: np.ndarray,
    test_labels: List[str],
    report_name: str,
) -> None:
    """
    Evaluate the classifier on the test dataset.
    """
    predictions = []
    similarities = []
    for embedding in test_embeddings:
        prediction, similarity = classifier.predict(
            embedding
        )
        predictions.append(prediction)
        similarities.append(similarity)
    accuracy = accuracy_score(
        test_labels,
        predictions,
    )
    report = classification_report(
        test_labels,
        predictions,
    )
    matrix = confusion_matrix(
        test_labels,
        predictions,
    )
    # Out-of-Scope (OOS) evaluation
    known_total = 0
    known_rejected = 0
    unknown_total = 0
    unknown_correct = 0
    for true_label, predicted_label in zip(
        test_labels,
        predictions,
    ):
        if true_label == "unknown":
            unknown_total += 1
            if predicted_label == "unknown":
                unknown_correct += 1
        else:
            known_total += 1
            if predicted_label == "unknown":
                known_rejected += 1
    oos_rejection_rate = (
        unknown_correct / unknown_total
        if unknown_total > 0
        else 0.0
    )
    false_rejection_rate = (
        known_rejected / known_total
        if known_total > 0
        else 0.0
    )
    labels = sorted(set(test_labels) | set(predictions))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels,
    )
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(
        cmap="Blues",
        ax=ax,
        xticks_rotation=45,
        colorbar=False,
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report\n")
    print(report)
    print("\nConfusion Matrix\n")
    print(matrix)
    print("\nOut-of-Scope Evaluation\n")
    print(
        f"OOS Rejection Rate : "
        f"{oos_rejection_rate:.2%}"
    )
    print(
        f"False Rejection Rate : "
        f"{false_rejection_rate:.2%}"
    )
    RESULTS_DIR.mkdir(exist_ok=True)
    report_path = RESULTS_DIR / f"{report_name}_evaluation_report.txt"
    with open(report_path, "w") as file:
        file.write(f"Accuracy: {accuracy:.4f}\n\n")
        file.write("Classification Report\n")
        file.write(report)
        file.write("\n")
        file.write(
            "\nOut-of-Scope Evaluation\n"
        )
        file.write(
            f"OOS Rejection Rate: "
            f"{oos_rejection_rate:.2%}\n"
        )
        file.write(
            f"False Rejection Rate: "
            f"{false_rejection_rate:.2%}\n\n"
        )
        file.write("Confusion Matrix\n")
        file.write(str(matrix))
    print(f"\nEvaluation report saved to: {report_path}")
    figure_path = RESULTS_DIR / f"{report_name}_confusion_matrix.png"
    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Confusion matrix saved to: {figure_path}")