"""
Tests for the scan module
"""

import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

from crux.scan import TemplateScanner, ScanResult
from crux.scan.scanner import ResourceFinding
from crux.ml.features import FeatureExtractor
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier


class TestScanResult:
    """Tests for ScanResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        finding = ResourceFinding(
            resource_id="/test/resource",
            resource_type="Microsoft.Storage/storageAccounts",
            resource_name="storage1",
            risk_score=0.75,
            risk_level="HIGH",
            rule_violations=["Storage_PublicAccess"],
            model_prediction=True,
            model_confidence=0.8,
            recommendations=["Enable soft delete"],
        )

        result = ScanResult(
            template_path="/path/to/template.bicep",
            scan_timestamp="2025-01-25T12:00:00",
            model_name="TestModel",
            model_path="/path/to/model.pkl",
            overall_risk_score=0.75,
            overall_risk_level="HIGH",
            total_resources=5,
            resources_at_risk=2,
            findings=[finding],
            rule_violations_count=1,
            unique_rule_violations=["Storage_PublicAccess"],
            exit_code=0,
            threshold_exceeded=False,
        )

        d = result.to_dict()

        assert d["metadata"]["template_path"] == "/path/to/template.bicep"
        assert d["summary"]["overall_risk_score"] == 0.75
        assert d["summary"]["total_resources"] == 5
        assert len(d["findings"]) == 1
        assert d["findings"][0]["risk_level"] == "HIGH"
        assert d["ci_cd"]["exit_code"] == 0

    def test_to_json(self):
        """Test JSON serialization."""
        result = ScanResult(
            template_path="/path/to/template.bicep",
            scan_timestamp="2025-01-25T12:00:00",
            model_name="TestModel",
            model_path="/path/to/model.pkl",
            overall_risk_score=0.5,
            overall_risk_level="MEDIUM",
            total_resources=3,
            resources_at_risk=1,
        )

        json_str = result.to_json()
        parsed = json.loads(json_str)

        assert parsed["metadata"]["model_name"] == "TestModel"
        assert parsed["summary"]["overall_risk_level"] == "MEDIUM"

    def test_to_text(self):
        """Test human-readable text output."""
        finding = ResourceFinding(
            resource_id="/test/storage",
            resource_type="Microsoft.Storage/storageAccounts",
            resource_name="storage1",
            risk_score=0.8,
            risk_level="CRITICAL",
            rule_violations=["Storage_PublicAccess", "Storage_NoHttps"],
            model_prediction=True,
            model_confidence=0.9,
            recommendations=["Enable soft delete", "Enforce HTTPS"],
        )

        result = ScanResult(
            template_path="/path/to/template.bicep",
            scan_timestamp="2025-01-25T12:00:00",
            model_name="TestModel",
            model_path="/path/to/model.pkl",
            overall_risk_score=0.8,
            overall_risk_level="CRITICAL",
            total_resources=5,
            resources_at_risk=3,
            findings=[finding],
            rule_violations_count=2,
            unique_rule_violations=["Storage_PublicAccess", "Storage_NoHttps"],
        )

        text = result.to_text()

        assert "CRUX Template Scan Report" in text
        assert "CRITICAL" in text
        assert "storage1" in text
        assert "Storage_PublicAccess" in text

    def test_ci_cd_exit_code(self):
        """Test CI/CD exit code handling."""
        # No threshold exceeded
        result_pass = ScanResult(
            template_path="test.bicep",
            scan_timestamp="2025-01-25T12:00:00",
            model_name="Test",
            model_path="test.pkl",
            overall_risk_score=0.5,
            overall_risk_level="MEDIUM",
            total_resources=1,
            resources_at_risk=0,
            exit_code=0,
            threshold_exceeded=False,
        )
        assert result_pass.exit_code == 0
        assert not result_pass.threshold_exceeded

        # Threshold exceeded
        result_fail = ScanResult(
            template_path="test.bicep",
            scan_timestamp="2025-01-25T12:00:00",
            model_name="Test",
            model_path="test.pkl",
            overall_risk_score=0.85,
            overall_risk_level="CRITICAL",
            total_resources=1,
            resources_at_risk=1,
            exit_code=1,
            threshold_exceeded=True,
        )
        assert result_fail.exit_code == 1
        assert result_fail.threshold_exceeded


class TestTemplateScanner:
    """Tests for TemplateScanner."""

    @pytest.fixture
    def mock_model_path(self, tmp_path):
        """Create a mock trained model."""
        # Create dummy feature extractor
        extractor = FeatureExtractor(max_features=20)
        dummy_resources = [
            {
                "type": "Microsoft.Storage/storageAccounts",
                "name": "storage1",
                "properties": {"allowBlobPublicAccess": False},
            }
        ]
        extractor.fit(dummy_resources)

        # Create dummy model
        base_clf = RandomForestClassifier(n_estimators=5, random_state=42)
        model = MultiOutputClassifier(base_clf)

        # Fit with dummy data
        X = np.random.rand(50, len(extractor.feature_names))
        y = np.random.randint(0, 2, size=(50, 2))
        model.fit(X, y)

        # Save model
        model_path = tmp_path / "test_model.pkl"
        model_data = {
            "model_name": "TestRandomForest",
            "model": model,
            "feature_names": extractor.feature_names,
            "label_names": ["Label1", "Label2"],
        }
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        # Save feature extractor
        feature_path = tmp_path / "test_model_features.pkl"
        with open(feature_path, "wb") as f:
            pickle.dump(extractor, f)

        return model_path

    @pytest.fixture
    def mock_arm_template(self, tmp_path):
        """Create a mock ARM template."""
        template = {
            "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
            "contentVersion": "1.0.0.0",
            "resources": [
                {
                    "type": "Microsoft.Storage/storageAccounts",
                    "apiVersion": "2021-04-01",
                    "name": "teststorage123",
                    "location": "eastus",
                    "sku": {"name": "Standard_LRS"},
                    "kind": "StorageV2",
                    "properties": {
                        "allowBlobPublicAccess": True,
                        "supportsHttpsTrafficOnly": False,
                    },
                },
                {
                    "type": "Microsoft.Network/virtualNetworks",
                    "apiVersion": "2021-05-01",
                    "name": "testvnet",
                    "location": "eastus",
                    "properties": {
                        "addressSpace": {"addressPrefixes": ["10.0.0.0/16"]},
                    },
                },
            ],
        }

        template_path = tmp_path / "test_template.json"
        with open(template_path, "w") as f:
            json.dump(template, f)

        return template_path

    @pytest.fixture
    def mock_rules_dir(self, tmp_path):
        """Create a mock rules directory."""
        rules_dir = tmp_path / "rules"
        rules_dir.mkdir()

        rules_content = """
rules:
  - id: storage-public-access
    name: Storage Public Access
    resource_type: Microsoft.Storage/storageAccounts
    severity: high
    condition:
      property: properties.allowBlobPublicAccess
      equals: true
    labels:
      - Storage_PublicAccess
      - CIS_3.7
  - id: storage-no-https
    name: Storage No HTTPS
    resource_type: Microsoft.Storage/storageAccounts
    severity: medium
    condition:
      property: properties.supportsHttpsTrafficOnly
      equals: false
    labels:
      - Storage_NoHttps
"""
        with open(rules_dir / "storage.yaml", "w") as f:
            f.write(rules_content)

        return rules_dir

    def test_scanner_initialization(self, mock_model_path):
        """Test scanner initializes correctly."""
        scanner = TemplateScanner(model_path=str(mock_model_path))

        assert scanner.model_name == "TestRandomForest"
        assert scanner.model is not None
        assert len(scanner.feature_names) > 0
        assert scanner.rule_evaluator is None  # No rules provided

    def test_scanner_with_rules(self, mock_model_path, mock_rules_dir):
        """Test scanner with rules directory."""
        scanner = TemplateScanner(
            model_path=str(mock_model_path),
            rules_dir=str(mock_rules_dir),
        )

        assert scanner.rule_evaluator is not None
        assert len(scanner.rule_evaluator.rules) == 2

    def test_scan_template(self, mock_model_path, mock_arm_template, mock_rules_dir):
        """Test scanning a template."""
        scanner = TemplateScanner(
            model_path=str(mock_model_path),
            rules_dir=str(mock_rules_dir),
        )

        result = scanner.scan(str(mock_arm_template))

        # Check basic result structure
        assert isinstance(result, ScanResult)
        assert result.total_resources == 2
        assert result.template_path == str(mock_arm_template)
        assert result.model_name == "TestRandomForest"

        # Check findings
        assert len(result.findings) == 2

        # Check rule violations detected
        assert result.rule_violations_count > 0
        assert "Storage_PublicAccess" in result.unique_rule_violations

    def test_scan_with_threshold_pass(self, mock_model_path, mock_arm_template):
        """Test scan with threshold that passes."""
        scanner = TemplateScanner(
            model_path=str(mock_model_path),
            fail_threshold=0.99,  # Very high threshold
        )

        result = scanner.scan(str(mock_arm_template))

        assert result.exit_code == 0
        assert not result.threshold_exceeded

    def test_scan_with_threshold_fail(self, mock_model_path, mock_arm_template, mock_rules_dir):
        """Test scan with threshold that fails."""
        scanner = TemplateScanner(
            model_path=str(mock_model_path),
            rules_dir=str(mock_rules_dir),
            fail_threshold=0.01,  # Very low threshold - should fail
        )

        result = scanner.scan(str(mock_arm_template))

        # With rules detecting issues, score should exceed 0.01
        if result.overall_risk_score >= 0.01:
            assert result.exit_code == 1
            assert result.threshold_exceeded

    def test_scan_empty_template(self, mock_model_path, tmp_path):
        """Test scanning template with no resources."""
        empty_template = {
            "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
            "resources": [],
        }

        template_path = tmp_path / "empty.json"
        with open(template_path, "w") as f:
            json.dump(empty_template, f)

        scanner = TemplateScanner(model_path=str(mock_model_path))
        result = scanner.scan(str(template_path))

        assert result.total_resources == 0
        assert result.resources_at_risk == 0
        assert result.overall_risk_score == 0.0
        assert result.overall_risk_level == "LOW"

    def test_scan_nonexistent_template(self, mock_model_path):
        """Test scanning a template that doesn't exist."""
        scanner = TemplateScanner(model_path=str(mock_model_path))

        with pytest.raises(FileNotFoundError):
            scanner.scan("/nonexistent/path/template.json")

    def test_scan_invalid_model_path(self):
        """Test initializing scanner with invalid model path."""
        with pytest.raises(FileNotFoundError):
            TemplateScanner(model_path="/nonexistent/model.pkl")

    def test_score_to_level(self, mock_model_path):
        """Test risk score to level conversion."""
        scanner = TemplateScanner(model_path=str(mock_model_path))

        assert scanner._score_to_level(0.9) == "CRITICAL"
        assert scanner._score_to_level(0.7) == "HIGH"
        assert scanner._score_to_level(0.5) == "MEDIUM"
        assert scanner._score_to_level(0.2) == "LOW"

    def test_generate_recommendations(self, mock_model_path):
        """Test recommendation generation."""
        scanner = TemplateScanner(model_path=str(mock_model_path))

        # Storage recommendations
        recs = scanner._generate_recommendations(
            "Microsoft.Storage/storageAccounts",
            ["Storage_PublicAccess"],
            True,
        )
        assert len(recs) > 0
        assert any("soft delete" in r.lower() for r in recs)

        # VM recommendations
        recs = scanner._generate_recommendations(
            "Microsoft.Compute/virtualMachines",
            ["VM_NoSecureBoot"],
            True,
        )
        assert len(recs) > 0

        # KeyVault recommendations
        recs = scanner._generate_recommendations(
            "Microsoft.KeyVault/vaults",
            ["KeyVault_NoPurgeProtection"],
            True,
        )
        assert len(recs) > 0


class TestScanConvenienceFunction:
    """Test the convenience function."""

    def test_scan_template_function(self, tmp_path):
        """Test scan_template convenience function."""
        from crux.scan.scanner import scan_template

        # Create mock model
        extractor = FeatureExtractor(max_features=10)
        extractor.fit([{"type": "Test", "properties": {}}])

        base_clf = RandomForestClassifier(n_estimators=3, random_state=42)
        model = MultiOutputClassifier(base_clf)
        X = np.random.rand(20, len(extractor.feature_names))
        y = np.random.randint(0, 2, size=(20, 1))
        model.fit(X, y)

        model_path = tmp_path / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({
                "model_name": "Test",
                "model": model,
                "feature_names": extractor.feature_names,
                "label_names": ["TestLabel"],
            }, f)

        with open(tmp_path / "model_features.pkl", "wb") as f:
            pickle.dump(extractor, f)

        # Create template
        template = {
            "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
            "resources": [{"type": "Test", "name": "test1", "properties": {}}],
        }
        template_path = tmp_path / "template.json"
        with open(template_path, "w") as f:
            json.dump(template, f)

        # Test function
        result = scan_template(str(template_path), str(model_path))

        assert isinstance(result, ScanResult)
        assert result.total_resources == 1
