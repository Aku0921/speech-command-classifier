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
) -> None:
    """
    Evaluate the classifier on the test dataset.
    """
    predictions = []
    for embedding in test_embeddings:
        prediction, _ = classifier.predict(
            embedding
        )
        predictions.append(prediction)
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
    RESULTS_DIR.mkdir(exist_ok=True)
    report_path = RESULTS_DIR / "evaluation_report.txt"
    with open(report_path, "w") as file:
        file.write(f"Accuracy: {accuracy:.4f}\n\n")
        file.write("Classification Report\n")
        file.write(report)
        file.write("\n")
        file.write("Confusion Matrix\n")
        file.write(str(matrix))
    print(f"\nEvaluation report saved to: {report_path}")
    figure_path = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Confusion matrix saved to: {figure_path}")