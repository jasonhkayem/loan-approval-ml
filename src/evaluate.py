"""
evaluate.py — reusable evaluation utilities: metrics, plots, and reporting.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import cross_val_score


def missing_values_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return counts and percentages of missing values, sorted descending."""
    counts = df.isnull().sum()
    pcts   = 100 * counts / len(df)
    table  = pd.concat([counts, pcts], axis=1)
    table.columns = ["Missing Values", "% of Total"]
    return table[table["Missing Values"] > 0].sort_values("% of Total", ascending=False).round(1)


def plot_confusion_matrix(y_true, y_pred, title: str = "Confusion Matrix", cmap: str = "Blues") -> None:
    """Annotated heatmap of TP/TN/FP/FN counts."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap,
        xticklabels=["Predicted: Denied", "Predicted: Approved"],
        yticklabels=["Actual: Denied",    "Actual: Approved"],
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_roc_curve(model, X_test: pd.DataFrame, y_test: pd.Series, label: str = "Model") -> float:
    """
    ROC curve with AUC. Preferred over accuracy for imbalanced classification
    because it evaluates performance across all decision thresholds.
    Returns AUC score.
    """
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline (AUC = 0.500)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return roc_auc


def plot_precision_recall_curve(model, X_test: pd.DataFrame, y_test: pd.Series, label: str = "Model") -> float:
    """
    Precision-Recall curve with Average Precision score.
    More informative than ROC when the positive class is the minority class,
    because it focuses on the model's ability to find true positives without
    being misled by the large number of true negatives.
    Returns Average Precision score.
    """
    y_score = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    ap = average_precision_score(y_test, y_score)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, lw=2, label=f"{label} (AP = {ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return ap


def print_cv_results(
    models: dict, X: pd.DataFrame, y: pd.Series, cv: int = 5, scoring: str = "roc_auc"
) -> pd.DataFrame:
    """Cross-validate each model and print a ranked summary table."""
    rows = []
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        rows.append({
            "Model":            name,
            f"Mean {scoring}": round(scores.mean(), 4),
            "Std":              round(scores.std(),  4),
        })
    result = pd.DataFrame(rows).sort_values(f"Mean {scoring}", ascending=False)
    print(result.to_string(index=False))
    return result


def visualize_outliers(pca_data: np.ndarray, preds: np.ndarray, method_name: str) -> None:
    """Scatter plot of 2-D PCA projection coloured by anomaly label (1 = outlier)."""
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(
        pca_data[:, 0], pca_data[:, 1],
        c=preds, cmap="coolwarm", edgecolors="k", linewidths=0.4, s=25,
    )
    plt.colorbar(scatter, label="Anomaly label (1 = outlier)")
    plt.title(f"Anomaly Detection — {method_name}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.tight_layout()
    plt.show()
