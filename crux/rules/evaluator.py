"""
Rule Evaluator

Evaluates security rules against resources to generate labels.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
import yaml
from jsonpath_ng import parse as jsonpath_parse

logger = logging.getLogger(__name__)


class RuleEvaluator:
    """Evaluates YAML-defined security rules against resources."""

    def __init__(self, rules_dir: Optional[str] = None):
        """
        Initialize the rule evaluator.

        Args:
            rules_dir: Directory containing YAML rule files
        """
        self.rules_dir = Path(rules_dir) if rules_dir else None
        self.rules: List[Dict[str, Any]] = []

        if self.rules_dir:
            self.load_rules_from_directory(self.rules_dir)

    def load_rules_from_file(self, rules_file: Path) -> List[Dict[str, Any]]:
        """
        Load rules from a single YAML file.

        Args:
            rules_file: Path to YAML rules file

        Returns:
            List of rule dictionaries
        """
        logger.info(f"Loading rules from {rules_file}")

        with open(rules_file, "r") as f:
            data = yaml.safe_load(f)

        rules = data.get("rules", [])
        logger.debug(f"Loaded {len(rules)} rules from {rules_file}")

        return rules

    def load_rules_from_directory(self, rules_dir: Path) -> None:
        """
        Load all rules from YAML files in a directory.

        Args:
            rules_dir: Directory containing YAML rule files
        """
        if not rules_dir.exists():
            logger.warning(f"Rules directory does not exist: {rules_dir}")
            return

        yaml_files = list(rules_dir.glob("*.yaml")) + list(rules_dir.glob("*.yml"))

        for yaml_file in yaml_files:
            rules = self.load_rules_from_file(yaml_file)
            self.rules.extend(rules)

        logger.info(f"Loaded {len(self.rules)} total rules from {rules_dir}")

    def evaluate_rule(self, rule: Dict[str, Any], resource: Dict[str, Any]) -> List[str]:
        """
        Evaluate a single rule against a resource.

        Args:
            rule: Rule dictionary
            resource: Resource dictionary

        Returns:
            List of labels if rule matches, empty list otherwise
        """
        # Check resource type
        resource_type = resource.get("type", "")
        rule_type = rule.get("resource_type", "")

        if resource_type != rule_type:
            return []

        # Evaluate condition
        condition = rule.get("condition", {})
        if not condition:
            logger.warning(f"Rule {rule.get('id')} has no condition")
            return []

        if self._evaluate_condition(condition, resource):
            return rule.get("labels", [])

        return []

    def _evaluate_condition(
        self, condition: Dict[str, Any], resource: Dict[str, Any]
    ) -> bool:
        """
        Evaluate a rule condition against a resource.

        Args:
            condition: Condition dictionary
            resource: Resource dictionary

        Returns:
            True if condition matches, False otherwise
        """
        property_path = condition.get("property")
        if not property_path:
            logger.warning("Condition missing 'property' field")
            return False

        # Get the property value from the resource
        actual_value = self._get_nested_property(resource, property_path)

        # Check various condition types
        if "equals" in condition:
            return actual_value == condition["equals"]

        if "not_equals" in condition:
            return actual_value != condition["not_equals"]

        if "in" in condition:
            return actual_value in condition["in"]

        if "not_in" in condition:
            return actual_value not in condition["not_in"]

        if "greater_than" in condition:
            try:
                return float(actual_value) > float(condition["greater_than"])
            except (TypeError, ValueError):
                return False

        if "less_than" in condition:
            try:
                return float(actual_value) < float(condition["less_than"])
            except (TypeError, ValueError):
                return False

        if "regex_match" in condition:
            import re

            if actual_value is None:
                return False
            return bool(re.match(condition["regex_match"], str(actual_value)))

        if "exists" in condition:
            return (actual_value is not None) == condition["exists"]

        # If none of the above conditions match, return False
        logger.warning(f"Unknown condition type in: {condition}")
        return False

    def _get_nested_property(
        self, resource: Dict[str, Any], property_path: str
    ) -> Any:
        """
        Get a nested property from a resource using dot notation.

        Args:
            resource: Resource dictionary
            property_path: Dot-separated path (e.g., "properties.allowBlobPublicAccess")

        Returns:
            Property value or None if not found
        """
        keys = property_path.split(".")
        current = resource

        for key in keys:
            if isinstance(current, dict):
                current = current.get(key)
            else:
                return None

            if current is None:
                return None

        return current

    def evaluate_all_rules(
        self, resource: Dict[str, Any], rules: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Evaluate all rules against a resource.

        Args:
            resource: Resource dictionary
            rules: Optional list of rules (uses loaded rules if None)

        Returns:
            Combined list of all labels from matching rules
        """
        rules = rules if rules is not None else self.rules
        all_labels = []

        for rule in rules:
            labels = self.evaluate_rule(rule, resource)
            all_labels.extend(labels)

        # Remove duplicates while preserving order
        seen = set()
        unique_labels = []
        for label in all_labels:
            if label not in seen:
                seen.add(label)
                unique_labels.append(label)

        return unique_labels

    def evaluate(self, resource: Dict[str, Any]) -> List[str]:
        """
        Convenience method to evaluate all rules against a resource.
        Alias for evaluate_all_rules().

        Args:
            resource: Resource dictionary

        Returns:
            List of labels from matching rules
        """
        return self.evaluate_all_rules(resource)

    def evaluate_resources(
        self, resources: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """
        Evaluate rules against multiple resources.

        Args:
            resources: List of resource dictionaries

        Returns:
            Dictionary mapping resource IDs to lists of labels
        """
        results = {}

        for resource in resources:
            resource_id = resource.get("id", "unknown")
            labels = self.evaluate_all_rules(resource)
            results[resource_id] = labels

        logger.info(
            f"Evaluated {len(resources)} resources, "
            f"{sum(1 for labels in results.values() if labels)} have issues"
        )

        return results

    def get_rules_by_severity(self, severity: str) -> List[Dict[str, Any]]:
        """Get all rules of a specific severity."""
        return [r for r in self.rules if r.get("severity") == severity]

    def get_rules_by_resource_type(self, resource_type: str) -> List[Dict[str, Any]]:
        """Get all rules for a specific resource type."""
        return [r for r in self.rules if r.get("resource_type") == resource_type]


def evaluate_resources(
    resources: List[Dict[str, Any]], rules_dir: str
) -> Dict[str, List[str]]:
    """
    Convenience function to evaluate rules against resources.

    Args:
        resources: List of resources
        rules_dir: Directory containing YAML rule files

    Returns:
        Dictionary mapping resource IDs to labels
    """
    evaluator = RuleEvaluator(rules_dir=rules_dir)
    return evaluator.evaluate_resources(resources)
