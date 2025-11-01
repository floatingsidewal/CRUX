"""
Model Evaluation

Metrics and evaluation framework for misconfiguration detection models.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates model performance on misconfiguration detection.

    Supports:
    - Multi-label classification metrics
    - Per-label performance analysis
    - Macro/Micro averaging
    - Confusion matrices
    - Classification reports
    """

    def __init__(self, label_names: List[str]):
        """
        Initialize the evaluator.

        Args:
            label_names: Names of all labels
        """
        self.label_names = label_names

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate model predictions.

        Args:
            y_true: True label matrix (n_samples, n_labels)
            y_pred: Predicted label matrix (n_samples, n_labels)
            y_proba: Predicted probability matrix (n_samples, n_labels)

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating predictions for {y_true.shape[0]} samples")

        metrics = {}

        # Overall metrics
        metrics["accuracy_samples"] = float(accuracy_score(y_true, y_pred))
        metrics["hamming_loss"] = float(hamming_loss(y_true, y_pred))

        # Precision, Recall, F1 - Macro averaging
        metrics["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        # Precision, Recall, F1 - Micro averaging
        metrics["precision_micro"] = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
        metrics["recall_micro"] = float(recall_score(y_true, y_pred, average="micro", zero_division=0))
        metrics["f1_micro"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))

        # Precision, Recall, F1 - Weighted averaging
        metrics["precision_weighted"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        metrics["recall_weighted"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        # Per-label metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        metrics["per_label"] = {}
        for i, label_name in enumerate(self.label_names):
            metrics["per_label"][label_name] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }

        # Calculate label-wise accuracy (exact match per sample)
        exact_match = np.all(y_true == y_pred, axis=1)
        metrics["exact_match_ratio"] = float(np.mean(exact_match))

        logger.info(f"Evaluation complete: F1 Macro = {metrics['f1_macro']:.3f}, "
                   f"F1 Micro = {metrics['f1_micro']:.3f}")

        return metrics

    def print_report(self, metrics: Dict[str, Any]) -> None:
        """
        Print a formatted evaluation report.

        Args:
            metrics: Metrics dictionary from evaluate()
        """
        print("\n" + "=" * 70)
        print("MODEL EVALUATION REPORT")
        print("=" * 70)

        print("\nOverall Metrics:")
        print(f"  Exact Match Ratio: {metrics['exact_match_ratio']:.3f}")
        print(f"  Sample Accuracy:   {metrics['accuracy_samples']:.3f}")
        print(f"  Hamming Loss:      {metrics['hamming_loss']:.3f}")

        print("\nMacro-Averaged Metrics:")
        print(f"  Precision: {metrics['precision_macro']:.3f}")
        print(f"  Recall:    {metrics['recall_macro']:.3f}")
        print(f"  F1 Score:  {metrics['f1_macro']:.3f}")

        print("\nMicro-Averaged Metrics:")
        print(f"  Precision: {metrics['precision_micro']:.3f}")
        print(f"  Recall:    {metrics['recall_micro']:.3f}")
        print(f"  F1 Score:  {metrics['f1_micro']:.3f}")

        print("\nWeighted-Averaged Metrics:")
        print(f"  Precision: {metrics['precision_weighted']:.3f}")
        print(f"  Recall:    {metrics['recall_weighted']:.3f}")
        print(f"  F1 Score:  {metrics['f1_weighted']:.3f}")

        print("\nPer-Label Performance:")
        print(f"  {'Label':<40} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
        print("  " + "-" * 86)

        for label_name, label_metrics in metrics["per_label"].items():
            print(f"  {label_name:<40} "
                  f"{label_metrics['precision']:<12.3f} "
                  f"{label_metrics['recall']:<12.3f} "
                  f"{label_metrics['f1']:<12.3f} "
                  f"{label_metrics['support']:<10}")

        print("=" * 70 + "\n")

    def save_report(self, metrics: Dict[str, Any], output_path: str) -> None:
        """
        Save evaluation report to JSON file.

        Args:
            metrics: Metrics dictionary from evaluate()
            output_path: Path to save the report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Evaluation report saved to {output_path}")

    def confusion_matrix_per_label(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Calculate confusion matrix for each label.

        Args:
            y_true: True label matrix (n_samples, n_labels)
            y_pred: Predicted label matrix (n_samples, n_labels)

        Returns:
            Dictionary mapping label names to confusion matrices
        """
        confusion_matrices = {}

        for i, label_name in enumerate(self.label_names):
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            confusion_matrices[label_name] = cm

        return confusion_matrices

    def summarize_errors(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Summarize the most common errors.

        Args:
            y_true: True label matrix (n_samples, n_labels)
            y_pred: Predicted label matrix (n_samples, n_labels)
            top_k: Number of top errors to report

        Returns:
            Dictionary of error statistics
        """
        # False positives and false negatives per label
        false_positives = np.sum((y_pred == 1) & (y_true == 0), axis=0)
        false_negatives = np.sum((y_pred == 0) & (y_true == 1), axis=0)

        # Sort by error count
        fp_sorted = sorted(
            enumerate(false_positives),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        fn_sorted = sorted(
            enumerate(false_negatives),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        error_summary = {
            "false_positives": [
                {"label": self.label_names[i], "count": int(count)}
                for i, count in fp_sorted
            ],
            "false_negatives": [
                {"label": self.label_names[i], "count": int(count)}
                for i, count in fn_sorted
            ],
        }

        return error_summary

    def compare_models(
        self,
        metrics_list: List[Dict[str, Any]],
        model_names: List[str],
    ) -> None:
        """
        Print a comparison table of multiple models.

        Args:
            metrics_list: List of metrics dictionaries
            model_names: List of model names
        """
        print("\n" + "=" * 90)
        print("MODEL COMPARISON")
        print("=" * 90)

        print(f"\n{'Model':<20} {'F1 Macro':<12} {'F1 Micro':<12} {'Precision':<12} {'Recall':<12}")
        print("-" * 90)

        for metrics, name in zip(metrics_list, model_names):
            print(f"{name:<20} "
                  f"{metrics['f1_macro']:<12.3f} "
                  f"{metrics['f1_micro']:<12.3f} "
                  f"{metrics['precision_macro']:<12.3f} "
                  f"{metrics['recall_macro']:<12.3f}")

        print("=" * 90 + "\n")


def cross_validate(
    model_class,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
    **model_params,
) -> List[Dict[str, Any]]:
    """
    Perform k-fold cross-validation.

    Args:
        model_class: Model class to instantiate
        X: Feature matrix
        y: Label matrix
        n_folds: Number of folds
        random_state: Random seed
        **model_params: Parameters to pass to model constructor

    Returns:
        List of metrics dictionaries, one per fold
    """
    from sklearn.model_selection import KFold

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
        logger.info(f"Training fold {fold_idx + 1}/{n_folds}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train model
        model = model_class(**model_params)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_val)

        evaluator = ModelEvaluator(label_names=[f"label_{i}" for i in range(y.shape[1])])
        metrics = evaluator.evaluate(y_val, y_pred)
        metrics["fold"] = fold_idx + 1

        fold_metrics.append(metrics)

    return fold_metrics
