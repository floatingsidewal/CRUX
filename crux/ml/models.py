"""
Baseline Models

XGBoost and Random Forest models for misconfiguration detection.
"""

import json
import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not installed. Install with: pip install xgboost")

logger = logging.getLogger(__name__)


class BaselineModel(ABC):
    """
    Abstract base class for baseline models.

    Handles multi-label classification for misconfiguration detection.
    """

    def __init__(self, model_name: str):
        """
        Initialize the model.

        Args:
            model_name: Name of the model
        """
        self.model_name = model_name
        self.model = None
        self.feature_names: List[str] = []
        self.label_names: List[str] = []
        self.is_fitted = False

    @abstractmethod
    def _create_model(self, **params) -> Any:
        """Create the underlying model instance."""
        pass

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        label_names: Optional[List[str]] = None,
        **fit_params,
    ) -> "BaselineModel":
        """
        Fit the model on training data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Label matrix (n_samples, n_labels)
            feature_names: Names of features
            label_names: Names of labels
            **fit_params: Additional parameters for fit

        Returns:
            self
        """
        logger.info(f"Training {self.model_name} on {X.shape[0]} samples with {X.shape[1]} features")
        logger.info(f"Predicting {y.shape[1]} labels")

        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        self.label_names = label_names or [f"label_{i}" for i in range(y.shape[1])]

        # Train model
        self.model.fit(X, y, **fit_params)
        self.is_fitted = True

        logger.info(f"{self.model_name} training complete")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for samples.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predicted label matrix (n_samples, n_labels)
        """
        if not self.is_fitted:
            raise ValueError(f"{self.model_name} must be fitted before predict")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict label probabilities for samples.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predicted probability matrix (n_samples, n_labels)
        """
        if not self.is_fitted:
            raise ValueError(f"{self.model_name} must be fitted before predict_proba")

        # For multi-output classifiers, get probabilities for each label
        if hasattr(self.model, 'predict_proba'):
            # MultiOutputClassifier returns list of arrays
            probas = self.model.predict_proba(X)
            # Extract probability of positive class for each label
            if isinstance(probas, list):
                return np.column_stack([p[:, 1] for p in probas])
            return probas
        else:
            # Fallback to binary predictions
            return self.predict(X).astype(float)

    def save(self, output_path: str) -> None:
        """
        Save the model to disk.

        Args:
            output_path: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError(f"{self.model_name} must be fitted before saving")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model and metadata
        save_data = {
            "model_name": self.model_name,
            "model": self.model,
            "feature_names": self.feature_names,
            "label_names": self.label_names,
        }

        with open(output_path, "wb") as f:
            pickle.dump(save_data, f)

        logger.info(f"Model saved to {output_path}")

    def load(self, model_path: str) -> "BaselineModel":
        """
        Load a model from disk.

        Args:
            model_path: Path to the saved model

        Returns:
            self
        """
        with open(model_path, "rb") as f:
            save_data = pickle.load(f)

        self.model_name = save_data["model_name"]
        self.model = save_data["model"]
        self.feature_names = save_data["feature_names"]
        self.label_names = save_data["label_names"]
        self.is_fitted = True

        logger.info(f"Model loaded from {model_path}")

        return self

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError(f"{self.model_name} must be fitted before getting feature importance")

        # Try to get feature importance from the underlying model
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'estimators_'):
            # For MultiOutputClassifier, average importance across all estimators
            importances = np.mean(
                [est.feature_importances_ for est in self.model.estimators_],
                axis=0
            )
        else:
            logger.warning("Feature importance not available for this model")
            return {}

        return dict(zip(self.feature_names, importances))


class RandomForestModel(BaselineModel):
    """Random Forest baseline model for multi-label classification."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        """
        Initialize Random Forest model.

        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples in a leaf
            random_state: Random seed
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        super().__init__("RandomForest")

        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "random_state": random_state,
            "n_jobs": n_jobs,
        }

        self.model = self._create_model(**self.params)

    def _create_model(self, **params) -> MultiOutputClassifier:
        """Create a Random Forest multi-output classifier."""
        base_estimator = RandomForestClassifier(**params)
        return MultiOutputClassifier(base_estimator, n_jobs=params.get("n_jobs", -1))


class XGBoostModel(BaselineModel):
    """XGBoost baseline model for multi-label classification."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        """
        Initialize XGBoost model.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns
            random_state: Random seed
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is not installed. Install with: pip install xgboost"
            )

        super().__init__("XGBoost")

        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "tree_method": "hist",  # Fast histogram-based algorithm
        }

        self.model = self._create_model(**self.params)

    def _create_model(self, **params) -> MultiOutputClassifier:
        """Create an XGBoost multi-output classifier."""
        base_estimator = xgb.XGBClassifier(**params)
        return MultiOutputClassifier(base_estimator, n_jobs=params.get("n_jobs", -1))


class LogisticRegressionModel(BaselineModel):
    """Logistic Regression baseline model for multi-label classification."""

    def __init__(
        self,
        penalty: str = "l2",
        C: float = 1.0,
        solver: str = "lbfgs",
        max_iter: int = 1000,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        """
        Initialize Logistic Regression model.

        Args:
            penalty: Regularization type ('l1', 'l2', 'elasticnet', 'none')
            C: Inverse regularization strength (smaller = stronger regularization)
            solver: Optimization algorithm ('lbfgs', 'liblinear', 'saga', etc.)
            max_iter: Maximum number of iterations
            random_state: Random seed
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        super().__init__("LogisticRegression")

        self.params = {
            "penalty": penalty,
            "C": C,
            "solver": solver,
            "max_iter": max_iter,
            "random_state": random_state,
            "n_jobs": n_jobs,
        }

        self.model = self._create_model(**self.params)

    def _create_model(self, **params) -> MultiOutputClassifier:
        """Create a Logistic Regression multi-output classifier."""
        base_estimator = LogisticRegression(**params)
        return MultiOutputClassifier(base_estimator, n_jobs=params.get("n_jobs", -1))

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores (coefficient magnitudes).

        For logistic regression, we use the absolute value of coefficients
        averaged across all output labels as a proxy for feature importance.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError(f"{self.model_name} must be fitted before getting feature importance")

        # For MultiOutputClassifier, average coefficient magnitudes across all estimators
        if hasattr(self.model, 'estimators_'):
            coeffs = np.array([est.coef_[0] for est in self.model.estimators_])
            # Average absolute coefficients across labels
            importances = np.mean(np.abs(coeffs), axis=0)
            return dict(zip(self.feature_names, importances))
        else:
            logger.warning("Feature importance not available for this model")
            return {}
