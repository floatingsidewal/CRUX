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
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
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


class BinaryEvaluator:
    """
    Evaluates model performance on binary classification.

    Supports:
    - Binary classification metrics (accuracy, precision, recall, F1)
    - ROC AUC and ROC curve
    - Precision-Recall curve
    - Confusion matrix
    - Optimal threshold finding
    """

    def __init__(self, positive_label: str = "misconfiguration"):
        """
        Initialize the binary evaluator.

        Args:
            positive_label: Name of the positive class (for reporting)
        """
        self.positive_label = positive_label

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate binary classification predictions.

        Args:
            y_true: True binary labels (n_samples,) with values 0 or 1
            y_pred: Predicted binary labels (n_samples,) with values 0 or 1
            y_proba: Predicted probabilities (n_samples,) with values in [0, 1]

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating binary predictions for {len(y_true)} samples")

        metrics = {}

        # Basic classification metrics
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        # Extract confusion matrix components
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["true_negatives"] = int(tn)
            metrics["false_positives"] = int(fp)
            metrics["false_negatives"] = int(fn)
            metrics["true_positives"] = int(tp)

            # Specificity (True Negative Rate)
            metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        # Class distribution
        metrics["positive_samples"] = int(np.sum(y_true))
        metrics["negative_samples"] = int(len(y_true) - np.sum(y_true))
        metrics["positive_ratio"] = float(np.mean(y_true))

        # Probability-based metrics (if probabilities provided)
        if y_proba is not None:
            try:
                # ROC AUC
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))

                # ROC curve data
                fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
                metrics["roc_curve"] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "thresholds": roc_thresholds.tolist(),
                }

                # Precision-Recall curve
                precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_proba)
                metrics["pr_curve"] = {
                    "precision": precision_curve.tolist(),
                    "recall": recall_curve.tolist(),
                    "thresholds": pr_thresholds.tolist(),
                }

                # PR AUC
                metrics["pr_auc"] = float(auc(recall_curve, precision_curve))

                # Find optimal threshold (maximizes F1)
                optimal_threshold, optimal_f1 = self._find_optimal_threshold(
                    y_true, y_proba
                )
                metrics["optimal_threshold"] = float(optimal_threshold)
                metrics["optimal_f1"] = float(optimal_f1)

            except ValueError as e:
                logger.warning(f"Could not calculate probability-based metrics: {e}")

        logger.info(f"Evaluation complete: F1 = {metrics['f1']:.3f}, "
                   f"Accuracy = {metrics['accuracy']:.3f}")
        if "roc_auc" in metrics:
            logger.info(f"ROC AUC = {metrics['roc_auc']:.3f}")

        return metrics

    def _find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> tuple[float, float]:
        """
        Find the threshold that maximizes F1 score.

        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities

        Returns:
            Tuple of (optimal_threshold, optimal_f1)
        """
        precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, y_proba)

        # Calculate F1 for each threshold
        # Note: precision_curve and recall_curve have one more element than thresholds
        f1_scores = 2 * (precision_curve[:-1] * recall_curve[:-1]) / (
            precision_curve[:-1] + recall_curve[:-1] + 1e-10
        )

        optimal_idx = np.argmax(f1_scores)
        return thresholds[optimal_idx], f1_scores[optimal_idx]

    def print_report(self, metrics: Dict[str, Any]) -> None:
        """
        Print a formatted evaluation report.

        Args:
            metrics: Metrics dictionary from evaluate()
        """
        print("\n" + "=" * 70)
        print("BINARY CLASSIFICATION EVALUATION REPORT")
        print("=" * 70)

        print("\nDataset Statistics:")
        print(f"  Total samples:      {metrics['positive_samples'] + metrics['negative_samples']}")
        print(f"  Positive samples:   {metrics['positive_samples']} ({metrics['positive_ratio']:.1%})")
        print(f"  Negative samples:   {metrics['negative_samples']} ({1-metrics['positive_ratio']:.1%})")

        print("\nClassification Metrics:")
        print(f"  Accuracy:   {metrics['accuracy']:.3f}")
        print(f"  Precision:  {metrics['precision']:.3f}")
        print(f"  Recall:     {metrics['recall']:.3f}")
        print(f"  F1 Score:   {metrics['f1']:.3f}")
        if "specificity" in metrics:
            print(f"  Specificity: {metrics['specificity']:.3f}")

        if "confusion_matrix" in metrics:
            print("\nConfusion Matrix:")
            print(f"                 Predicted Neg  Predicted Pos")
            print(f"  Actual Neg     {metrics.get('true_negatives', 'N/A'):<14} {metrics.get('false_positives', 'N/A'):<14}")
            print(f"  Actual Pos     {metrics.get('false_negatives', 'N/A'):<14} {metrics.get('true_positives', 'N/A'):<14}")

        if "roc_auc" in metrics:
            print("\nProbability-Based Metrics:")
            print(f"  ROC AUC:         {metrics['roc_auc']:.3f}")
            print(f"  PR AUC:          {metrics['pr_auc']:.3f}")
            print(f"  Optimal Threshold: {metrics['optimal_threshold']:.3f} (F1={metrics['optimal_f1']:.3f})")

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

        logger.info(f"Binary evaluation report saved to {output_path}")


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
