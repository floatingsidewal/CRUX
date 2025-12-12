"""
Unit tests for rule evaluation
"""

import pytest
from crux.rules.evaluator import RuleEvaluator


def test_rule_evaluator_loads_rules():
    """Test that RuleEvaluator loads rules from YAML files."""
    evaluator = RuleEvaluator(rules_dir="rules")

    # Should have loaded storage rules
    assert len(evaluator.rules) > 0

    # Check that rules have required fields
    for rule in evaluator.rules:
        assert "id" in rule
        assert "resource_type" in rule
        assert "condition" in rule
        assert "labels" in rule


def test_rule_evaluation_simple():
    """Test simple rule evaluation."""
    evaluator = RuleEvaluator(rules_dir="rules")

    # Resource with public blob access enabled (should trigger rule)
    resource = {
        "type": "Microsoft.Storage/storageAccounts",
        "name": "teststorage",
        "properties": {
            "allowBlobPublicAccess": True
        }
    }

    labels = evaluator.evaluate(resource)

    # Should have Storage_PublicAccess label
    assert "Storage_PublicAccess" in labels


def test_rule_evaluation_no_match():
    """Test that resources without violations get no labels."""
    evaluator = RuleEvaluator(rules_dir="rules")

    # Resource with public blob access disabled (should NOT trigger rule)
    resource = {
        "type": "Microsoft.Storage/storageAccounts",
        "name": "teststorage",
        "properties": {
            "allowBlobPublicAccess": False
        }
    }

    labels = evaluator.evaluate(resource)

    # Should NOT have Storage_PublicAccess label
    assert "Storage_PublicAccess" not in labels


def test_rule_evaluation_multiple_violations():
    """Test that multiple violations produce multiple labels."""
    evaluator = RuleEvaluator(rules_dir="rules")

    # Resource with multiple violations
    resource = {
        "type": "Microsoft.Storage/storageAccounts",
        "name": "teststorage",
        "properties": {
            "allowBlobPublicAccess": True,
            "minimumTlsVersion": "TLS1_0",
            "supportsHttpsTrafficOnly": False,
        }
    }

    labels = evaluator.evaluate(resource)

    # Should have multiple labels
    assert len(labels) > 1
    assert "Storage_PublicAccess" in labels
    assert "Storage_WeakTLS" in labels
