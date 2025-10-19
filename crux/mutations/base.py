"""
Mutation Base Classes

Defines the core mutation framework for applying controlled misconfigurations.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Dict, Any, Optional
import logging
import copy

logger = logging.getLogger(__name__)


@dataclass
class Mutation:
    """
    Represents a single mutation (misconfiguration) that can be applied to a resource.

    Attributes:
        id: Unique identifier for the mutation
        target_type: Azure resource type this mutation applies to
        description: Human-readable description of the mutation
        severity: Severity level (critical, high, medium, low)
        labels: List of misconfiguration labels this mutation produces
        cis_references: CIS Benchmark references (e.g., ["3.7", "3.8"])
        mutate: Function that applies the mutation to a resource
    """

    id: str
    target_type: str
    description: str
    severity: str
    labels: List[str]
    cis_references: List[str] = field(default_factory=list)
    mutate: Callable[[Dict[str, Any]], Dict[str, Any]] = field(repr=False)

    def __post_init__(self):
        """Validate mutation configuration."""
        valid_severities = ["critical", "high", "medium", "low"]
        if self.severity not in valid_severities:
            raise ValueError(
                f"Invalid severity '{self.severity}'. Must be one of {valid_severities}"
            )

    def apply(self, resource: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Apply this mutation to a resource.

        Args:
            resource: Resource dictionary to mutate

        Returns:
            Mutated resource or None if mutation doesn't apply

        Note:
            This method creates a deep copy of the resource to avoid side effects.
        """
        if resource.get("type") != self.target_type:
            logger.debug(
                f"Mutation {self.id} skipped: resource type {resource.get('type')} "
                f"!= target type {self.target_type}"
            )
            return None

        try:
            mutated = copy.deepcopy(resource)
            mutated = self.mutate(mutated)

            # Add metadata about the mutation
            if "_mutation_applied" not in mutated:
                mutated["_mutation_applied"] = []

            mutated["_mutation_applied"].append({
                "mutation_id": self.id,
                "labels": self.labels,
                "severity": self.severity,
            })

            logger.debug(f"Applied mutation {self.id} to resource {resource.get('name')}")
            return mutated

        except Exception as e:
            logger.error(f"Failed to apply mutation {self.id}: {e}")
            return None

    def validate(self, resource: Dict[str, Any]) -> bool:
        """
        Validate that this mutation actually changes the resource.

        Args:
            resource: Resource to validate against

        Returns:
            True if mutation produces a change, False otherwise
        """
        if resource.get("type") != self.target_type:
            return False

        original = copy.deepcopy(resource)
        mutated = self.apply(resource)

        if mutated is None:
            return False

        # Remove mutation metadata for comparison
        mutated_clean = copy.deepcopy(mutated)
        mutated_clean.pop("_mutation_applied", None)

        # Compare properties (the part that matters)
        original_props = original.get("properties", {})
        mutated_props = mutated_clean.get("properties", {})

        return original_props != mutated_props


@dataclass
class MutationResult:
    """
    Result of applying a mutation to a resource.

    Attributes:
        mutation: The mutation that was applied
        resource_id: ID of the resource that was mutated
        resource_type: Type of the resource
        success: Whether the mutation was successful
        original: Original resource (before mutation)
        mutated: Mutated resource (after mutation)
        error: Error message if mutation failed
    """

    mutation: Mutation
    resource_id: str
    resource_type: str
    success: bool
    original: Dict[str, Any]
    mutated: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class MutationEngine:
    """Applies mutations to resources and tracks results."""

    def __init__(self, mutations: List[Mutation]):
        """
        Initialize the mutation engine.

        Args:
            mutations: List of mutations to apply
        """
        self.mutations = mutations
        self.results: List[MutationResult] = []

    def apply_mutations(
        self,
        resources: List[Dict[str, Any]],
        filter_by_type: Optional[str] = None,
    ) -> List[MutationResult]:
        """
        Apply all mutations to a list of resources.

        Args:
            resources: List of resources to mutate
            filter_by_type: Only apply mutations for this resource type

        Returns:
            List of mutation results
        """
        results = []

        for resource in resources:
            resource_type = resource.get("type")
            resource_id = resource.get("id", "unknown")

            # Skip if filtering by type
            if filter_by_type and resource_type != filter_by_type:
                continue

            # Try each mutation
            for mutation in self.mutations:
                if mutation.target_type != resource_type:
                    continue

                try:
                    mutated = mutation.apply(resource)

                    if mutated is not None:
                        result = MutationResult(
                            mutation=mutation,
                            resource_id=resource_id,
                            resource_type=resource_type,
                            success=True,
                            original=resource,
                            mutated=mutated,
                        )
                    else:
                        result = MutationResult(
                            mutation=mutation,
                            resource_id=resource_id,
                            resource_type=resource_type,
                            success=False,
                            original=resource,
                            error="Mutation returned None",
                        )

                except Exception as e:
                    result = MutationResult(
                        mutation=mutation,
                        resource_id=resource_id,
                        resource_type=resource_type,
                        success=False,
                        original=resource,
                        error=str(e),
                    )

                results.append(result)

        self.results.extend(results)
        logger.info(
            f"Applied {len(results)} mutations "
            f"({sum(1 for r in results if r.success)} successful)"
        )
        return results

    def get_results_by_severity(self, severity: str) -> List[MutationResult]:
        """Get all results filtered by severity level."""
        return [r for r in self.results if r.mutation.severity == severity]

    def get_successful_results(self) -> List[MutationResult]:
        """Get all successful mutation results."""
        return [r for r in self.results if r.success]

    def get_failed_results(self) -> List[MutationResult]:
        """Get all failed mutation results."""
        return [r for r in self.results if not r.success]


def create_property_mutation(
    mutation_id: str,
    target_type: str,
    description: str,
    property_path: str,
    value: Any,
    severity: str,
    labels: List[str],
    cis_references: Optional[List[str]] = None,
) -> Mutation:
    """
    Helper function to create a simple property mutation.

    Args:
        mutation_id: Unique ID
        target_type: Azure resource type
        description: Description of the mutation
        property_path: Dot-separated path to the property (e.g., "properties.allowBlobPublicAccess")
        value: Value to set
        severity: Severity level
        labels: Misconfiguration labels
        cis_references: CIS Benchmark references

    Returns:
        Mutation object
    """

    def mutate_func(resource: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the property mutation."""
        keys = property_path.split(".")
        current = resource

        # Navigate to the parent of the target property
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the final property
        current[keys[-1]] = value
        return resource

    return Mutation(
        id=mutation_id,
        target_type=target_type,
        description=description,
        severity=severity,
        labels=labels,
        cis_references=cis_references or [],
        mutate=mutate_func,
    )
