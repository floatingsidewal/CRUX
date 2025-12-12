"""
Tests for ML components
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from crux.ml.dataset import DatasetLoader
from crux.ml.evaluation import ModelEvaluator, BinaryEvaluator
from crux.ml.features import FeatureExtractor
from crux.ml.models import RandomForestModel, LogisticRegressionModel


class TestFeatureExtractor:
    """Tests for FeatureExtractor."""

    def test_basic_feature_extraction(self):
        """Test basic feature extraction from resources."""
        resources = [
            {
                "type": "Microsoft.Storage/storageAccounts",
                "name": "storage1",
                "properties": {
                    "allowBlobPublicAccess": True,
                    "minimumTlsVersion": "TLS1_2",
                },
            },
            {
                "type": "Microsoft.Storage/storageAccounts",
                "name": "storage2",
                "properties": {
                    "allowBlobPublicAccess": False,
                    "minimumTlsVersion": "TLS1_0",
                },
            },
        ]

        extractor = FeatureExtractor(max_features=20)
        X, resource_ids = extractor.fit_transform(resources)

        # Check output shape
        assert X.shape[0] == 2  # 2 resources
        assert X.shape[1] == len(extractor.feature_names)
        assert len(resource_ids) == 2

        # Check that features are numerical
        assert X.dtype in [np.float32, np.float64]

    def test_feature_extractor_fit_transform(self):
        """Test fit and transform separately."""
        resources_train = [
            {
                "type": "Microsoft.Network/virtualNetworks",
                "properties": {"addressSpace": {"addressPrefixes": ["10.0.0.0/16"]}},
            }
        ]

        resources_test = [
            {
                "type": "Microsoft.Network/virtualNetworks",
                "properties": {"addressSpace": {"addressPrefixes": ["10.1.0.0/16"]}},
            }
        ]

        extractor = FeatureExtractor()
        extractor.fit(resources_train)

        # Transform should work on new data
        X_test, _ = extractor.transform(resources_test)
        assert X_test.shape[1] == len(extractor.feature_names)

    def test_empty_properties(self):
        """Test handling of resources with no properties."""
        resources = [
            {
                "type": "Microsoft.Storage/storageAccounts",
                "name": "storage1",
            },
            {
                "type": "Microsoft.Storage/storageAccounts",
                "name": "storage2",
                "properties": {},
            },
        ]

        extractor = FeatureExtractor()
        X, _ = extractor.fit_transform(resources)

        # Should not crash and should produce features
        assert X.shape[0] == 2
        assert not np.isnan(X).any()


class TestRandomForestModel:
    """Tests for RandomForestModel."""

    def test_model_fit_predict(self):
        """Test model training and prediction."""
        # Create dummy data
        X_train = np.random.rand(100, 10)
        y_train = np.random.randint(0, 2, size=(100, 3))  # 3 labels

        model = RandomForestModel(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        # Test prediction
        X_test = np.random.rand(20, 10)
        y_pred = model.predict(X_test)

        assert y_pred.shape == (20, 3)
        assert np.all((y_pred == 0) | (y_pred == 1))  # Binary predictions

    def test_model_save_load(self):
        """Test model serialization."""
        X_train = np.random.rand(50, 5)
        y_train = np.random.randint(0, 2, size=(50, 2))

        model = RandomForestModel(n_estimators=5, random_state=42)
        model.fit(X_train, y_train, label_names=["label1", "label2"])

        # Save model
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            model_path = f.name

        try:
            model.save(model_path)

            # Load model
            loaded_model = RandomForestModel()
            loaded_model.load(model_path)

            # Verify loaded model
            assert loaded_model.label_names == ["label1", "label2"]
            assert loaded_model.is_fitted

            # Test predictions match
            X_test = np.random.rand(10, 5)
            y_pred_original = model.predict(X_test)
            y_pred_loaded = loaded_model.predict(X_test)

            np.testing.assert_array_equal(y_pred_original, y_pred_loaded)

        finally:
            Path(model_path).unlink(missing_ok=True)

    def test_feature_importance(self):
        """Test feature importance extraction."""
        X_train = np.random.rand(100, 5)
        y_train = np.random.randint(0, 2, size=(100, 2))

        feature_names = ["feat1", "feat2", "feat3", "feat4", "feat5"]

        model = RandomForestModel(n_estimators=10, random_state=42)
        model.fit(X_train, y_train, feature_names=feature_names)

        importance = model.get_feature_importance()

        # Should have importance for each feature
        assert len(importance) == 5
        assert all(feat in importance for feat in feature_names)
        assert all(0 <= val <= 1 for val in importance.values())


class TestLogisticRegressionModel:
    """Tests for LogisticRegressionModel."""

    def test_model_fit_predict(self):
        """Test logistic regression training and prediction."""
        # Create dummy data
        X_train = np.random.rand(100, 10)
        y_train = np.random.randint(0, 2, size=(100, 3))  # 3 labels

        model = LogisticRegressionModel(C=1.0, max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        # Test prediction
        X_test = np.random.rand(20, 10)
        y_pred = model.predict(X_test)

        assert y_pred.shape == (20, 3)
        assert np.all((y_pred == 0) | (y_pred == 1))  # Binary predictions

    def test_model_predict_proba(self):
        """Test probability predictions."""
        X_train = np.random.rand(100, 5)
        y_train = np.random.randint(0, 2, size=(100, 2))

        model = LogisticRegressionModel(random_state=42)
        model.fit(X_train, y_train)

        X_test = np.random.rand(20, 5)
        y_proba = model.predict_proba(X_test)

        assert y_proba.shape == (20, 2)
        assert np.all((y_proba >= 0) & (y_proba <= 1))  # Probabilities in [0, 1]

    def test_feature_importance(self):
        """Test coefficient-based feature importance."""
        X_train = np.random.rand(100, 5)
        y_train = np.random.randint(0, 2, size=(100, 2))

        feature_names = ["feat1", "feat2", "feat3", "feat4", "feat5"]

        model = LogisticRegressionModel(random_state=42)
        model.fit(X_train, y_train, feature_names=feature_names)

        importance = model.get_feature_importance()

        # Should have importance for each feature
        assert len(importance) == 5
        assert all(feat in importance for feat in feature_names)
        assert all(val >= 0 for val in importance.values())  # Absolute values


class TestModelEvaluator:
    """Tests for ModelEvaluator."""

    def test_perfect_predictions(self):
        """Test evaluation with perfect predictions."""
        y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        y_pred = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])

        evaluator = ModelEvaluator(label_names=["L1", "L2", "L3"])
        metrics = evaluator.evaluate(y_true, y_pred)

        # Perfect predictions should have high scores
        assert metrics["f1_macro"] == 1.0
        assert metrics["f1_micro"] == 1.0
        assert metrics["exact_match_ratio"] == 1.0

    def test_random_predictions(self):
        """Test evaluation with random predictions."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, size=(100, 5))
        y_pred = np.random.randint(0, 2, size=(100, 5))

        evaluator = ModelEvaluator(label_names=[f"L{i}" for i in range(5)])
        metrics = evaluator.evaluate(y_true, y_pred)

        # Metrics should exist
        assert "f1_macro" in metrics
        assert "precision_micro" in metrics
        assert "recall_weighted" in metrics
        assert "per_label" in metrics

        # Per-label metrics should exist for each label
        assert len(metrics["per_label"]) == 5

    def test_confusion_matrix(self):
        """Test confusion matrix calculation."""
        y_true = np.array([[1, 0], [0, 1], [1, 1]])
        y_pred = np.array([[1, 0], [0, 0], [1, 1]])

        evaluator = ModelEvaluator(label_names=["L1", "L2"])
        cm = evaluator.confusion_matrix_per_label(y_true, y_pred)

        assert "L1" in cm
        assert "L2" in cm
        assert cm["L1"].shape == (2, 2)  # 2x2 confusion matrix

    def test_error_summary(self):
        """Test error summarization."""
        y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        y_pred = np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]])

        evaluator = ModelEvaluator(label_names=["L1", "L2", "L3"])
        errors = evaluator.summarize_errors(y_true, y_pred)

        assert "false_positives" in errors
        assert "false_negatives" in errors


class TestBinaryEvaluator:
    """Tests for BinaryEvaluator."""

    def test_perfect_binary_predictions(self):
        """Test binary evaluation with perfect predictions."""
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 1, 1, 0])
        y_proba = np.array([0.9, 0.1, 0.8, 0.95, 0.05])

        evaluator = BinaryEvaluator()
        metrics = evaluator.evaluate(y_true, y_pred, y_proba)

        # Perfect predictions
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["roc_auc"] == 1.0

    def test_random_binary_predictions(self):
        """Test binary evaluation with random predictions."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, size=100)
        y_pred = np.random.randint(0, 2, size=100)
        y_proba = np.random.rand(100)

        evaluator = BinaryEvaluator()
        metrics = evaluator.evaluate(y_true, y_pred, y_proba)

        # Metrics should exist
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
        assert "pr_auc" in metrics
        assert "confusion_matrix" in metrics

    def test_confusion_matrix_components(self):
        """Test confusion matrix component extraction."""
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 1, 0])

        evaluator = BinaryEvaluator()
        metrics = evaluator.evaluate(y_true, y_pred)

        # Check confusion matrix components
        assert "true_positives" in metrics
        assert "true_negatives" in metrics
        assert "false_positives" in metrics
        assert "false_negatives" in metrics
        assert "specificity" in metrics

        # Verify counts
        assert metrics["true_positives"] == 2
        assert metrics["true_negatives"] == 2
        assert metrics["false_positives"] == 1
        assert metrics["false_negatives"] == 1

    def test_optimal_threshold_finding(self):
        """Test optimal threshold finding."""
        # Create scenario where optimal threshold is not 0.5
        y_true = np.array([1, 1, 1, 0, 0])
        y_proba = np.array([0.9, 0.7, 0.4, 0.3, 0.1])

        evaluator = BinaryEvaluator()
        metrics = evaluator.evaluate(y_true, (y_proba > 0.5).astype(int), y_proba)

        # Should find optimal threshold
        assert "optimal_threshold" in metrics
        assert "optimal_f1" in metrics
        assert 0 <= metrics["optimal_threshold"] <= 1


class TestDatasetLoader:
    """Tests for DatasetLoader."""

    @pytest.fixture
    def mock_dataset_dir(self, tmp_path):
        """Create a mock dataset directory."""
        dataset_dir = tmp_path / "test-dataset"
        dataset_dir.mkdir()

        # Create baseline directory
        baseline_dir = dataset_dir / "baseline"
        baseline_dir.mkdir()

        baseline_resources = [
            {
                "type": "Microsoft.Storage/storageAccounts",
                "name": "storage1",
                "properties": {"allowBlobPublicAccess": False},
                "_metadata": {"is_mutated": False},
            }
        ]

        with open(baseline_dir / "resources.json", "w") as f:
            json.dump(baseline_resources, f)

        # Create mutated directory
        mutated_dir = dataset_dir / "mutated"
        mutated_dir.mkdir()

        mutated_resources = [
            {
                "type": "Microsoft.Storage/storageAccounts",
                "name": "storage1",
                "id": "/subscriptions/.../storage1",
                "properties": {"allowBlobPublicAccess": True},
                "_metadata": {
                    "is_mutated": True,
                    "mutation_id": "storage_public_blob_access",
                },
            }
        ]

        with open(mutated_dir / "resources.json", "w") as f:
            json.dump(mutated_resources, f)

        # Create labels
        labels = {
            "/subscriptions/.../storage1#storage_public_blob_access": [
                "Storage_PublicAccess",
                "CIS_3.7",
            ]
        }

        with open(dataset_dir / "labels.json", "w") as f:
            json.dump(labels, f)

        # Create metadata
        metadata = {
            "experiment_name": "test-exp",
            "statistics": {"templates_processed": 1},
        }

        with open(dataset_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        return dataset_dir

    def test_load_resources(self, mock_dataset_dir):
        """Test loading resources from dataset."""
        loader = DatasetLoader(str(mock_dataset_dir))

        # Load mutated only
        resources = loader.load_resources(include_baseline=False, include_mutated=True)
        assert len(resources) == 1
        assert resources[0]["_metadata"]["is_mutated"] is True

        # Load both
        resources = loader.load_resources(include_baseline=True, include_mutated=True)
        assert len(resources) == 2

    def test_load_labels(self, mock_dataset_dir):
        """Test loading labels from dataset."""
        loader = DatasetLoader(str(mock_dataset_dir))
        labels = loader.load_labels()

        assert len(labels) == 1
        assert any("Storage_PublicAccess" in v for v in labels.values())

    def test_prepare_training_data(self, mock_dataset_dir):
        """Test preparing training data."""
        loader = DatasetLoader(str(mock_dataset_dir))

        resources = loader.load_resources(include_mutated=True)
        labels_dict = loader.load_labels()

        labeled_resources, label_matrix, label_names = loader.prepare_training_data(
            resources, labels_dict
        )

        assert len(labeled_resources) == 1
        assert label_matrix.shape[0] == 1  # 1 sample
        assert label_matrix.shape[1] == 2  # 2 labels
        assert "Storage_PublicAccess" in label_names
        assert "CIS_3.7" in label_names

    def test_split_data(self, mock_dataset_dir):
        """Test data splitting."""
        # Create larger dataset for splitting
        resources = [{"type": "Test", "name": f"res{i}"} for i in range(100)]
        labels = np.random.randint(0, 2, size=(100, 3))

        loader = DatasetLoader(str(mock_dataset_dir))

        (
            train_res,
            val_res,
            test_res,
            train_labels,
            val_labels,
            test_labels,
        ) = loader.split_data(resources, labels, test_size=0.2, val_size=0.1)

        # Check sizes
        assert len(train_res) + len(val_res) + len(test_res) == 100
        assert train_labels.shape[0] == len(train_res)
        assert val_labels.shape[0] == len(val_res)
        assert test_labels.shape[0] == len(test_res)

    def test_get_binary_labels(self, mock_dataset_dir):
        """Test binary label conversion."""
        loader = DatasetLoader(str(mock_dataset_dir))

        # Create multi-label matrix
        labels_multi = np.array([
            [1, 0, 1],  # Has 2 labels -> 1
            [0, 0, 0],  # No labels -> 0
            [1, 1, 0],  # Has 2 labels -> 1
            [0, 1, 0],  # Has 1 label -> 1
        ])

        # Convert to binary
        binary_labels = loader.get_binary_labels(labels_multi, mode='any')

        # Check results
        expected = np.array([1, 0, 1, 1])
        np.testing.assert_array_equal(binary_labels, expected)

        # Check shape
        assert binary_labels.shape == (4,)
        assert binary_labels.dtype == int

    def test_export_to_csv(self, mock_dataset_dir, tmp_path):
        """Test CSV export functionality."""
        import pandas as pd

        loader = DatasetLoader(str(mock_dataset_dir))
        csv_path = tmp_path / "test_export.csv"

        # Export to CSV
        loader.export_to_csv(
            output_path=str(csv_path),
            include_features=True,
            max_features=20,
            binary_mode='any',
            include_baseline=False,
        )

        # Verify CSV was created
        assert csv_path.exists()

        # Load and check contents
        df = pd.read_csv(csv_path)

        # Check basic columns
        assert 'resource_id' in df.columns
        assert 'resource_type' in df.columns
        assert 'has_misconfiguration' in df.columns
        assert 'is_mutated' in df.columns

        # Check has at least one row
        assert len(df) >= 1

        # Check binary label is 0 or 1
        assert df['has_misconfiguration'].isin([0, 1]).all()

        # Check feature columns exist
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        assert len(feature_cols) > 0

    def test_export_csv_no_features(self, mock_dataset_dir, tmp_path):
        """Test CSV export without features."""
        import pandas as pd

        loader = DatasetLoader(str(mock_dataset_dir))
        csv_path = tmp_path / "test_export_no_features.csv"

        # Export without features
        loader.export_to_csv(
            output_path=str(csv_path),
            include_features=False,
            binary_mode='any',
        )

        # Load and check
        df = pd.read_csv(csv_path)

        # Should have metadata columns
        assert 'resource_id' in df.columns
        assert 'has_misconfiguration' in df.columns

        # Should NOT have feature columns
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        assert len(feature_cols) == 0
