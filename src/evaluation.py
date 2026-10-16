
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_model(df, label_column="is_fraud"):
    """Calculate evaluation metrics for fraud detection."""

    if label_column not in df.columns:
        raise ValueError(
            f"Missing label column: {label_column}"
        )

    if "anomaly_flag" not in df.columns:
        raise ValueError(
            "Missing prediction column: anomaly_flag"
        )

    y_true = df[label_column].astype(int)
    y_pred = df["anomaly_flag"].astype(int)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    accuracy = accuracy_score(y_true, y_pred)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Fraud"],
        zero_division=0,
    )

    true_negatives, false_positives = matrix[0]
    false_negatives, true_positives = matrix[1]

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "confusion_matrix": matrix,
        "classification_report": report,
        "true_negatives": int(true_negatives),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "true_positives": int(true_positives),
    }
