"""
Unit tests for mutations
"""

import pytest
from crux.mutations.base import Mutation, create_property_mutation
from crux.mutations.storage import (
    STORAGE_PUBLIC_BLOB_ACCESS,
    STORAGE_WEAK_TLS,
    ALL_MUTATIONS,
)


def test_mutation_applies_to():
    """Test that mutation.applies_to() works correctly."""
    resource = {
        "type": "Microsoft.Storage/storageAccounts",
        "name": "teststorage",
        "properties": {},
    }

    assert STORAGE_PUBLIC_BLOB_ACCESS.applies_to(resource) is True

    # Different resource type
    other_resource = {
        "type": "Microsoft.KeyVault/vaults",
        "name": "testkv",
    }
    assert STORAGE_PUBLIC_BLOB_ACCESS.applies_to(other_resource) is False


def test_mutation_apply():
    """Test that mutations actually modify resources."""
    resource = {
        "type": "Microsoft.Storage/storageAccounts",
        "name": "teststorage",
        "properties": {"allowBlobPublicAccess": False},
    }

    mutated = STORAGE_PUBLIC_BLOB_ACCESS.apply(resource)

    # Check mutation was applied
    assert mutated is not None
    assert mutated["properties"]["allowBlobPublicAccess"] is True

    # Check metadata was added
    assert "_mutation_applied" in mutated
    assert len(mutated["_mutation_applied"]) == 1
    assert mutated["_mutation_applied"][0]["mutation_id"] == "storage_public_blob_access"


def test_mutation_doesnt_modify_original():
    """Test that mutations create a deep copy and don't modify original."""
    original = {
        "type": "Microsoft.Storage/storageAccounts",
        "name": "teststorage",
        "properties": {"allowBlobPublicAccess": False},
    }

    mutated = STORAGE_PUBLIC_BLOB_ACCESS.apply(original)

    # Original should be unchanged
    assert original["properties"]["allowBlobPublicAccess"] is False

    # Mutated should be changed
    assert mutated["properties"]["allowBlobPublicAccess"] is True


def test_create_property_mutation():
    """Test the create_property_mutation helper function."""
    mutation = create_property_mutation(
        mutation_id="test_mutation",
        target_type="Microsoft.Storage/storageAccounts",
        description="Test mutation",
        property_path="properties.testProperty",
        value="testValue",
        severity="low",
        labels=["Test_Label"],
    )

    resource = {
        "type": "Microsoft.Storage/storageAccounts",
        "name": "teststorage",
        "properties": {},
    }

    mutated = mutation.apply(resource)

    assert mutated is not None
    assert mutated["properties"]["testProperty"] == "testValue"


def test_all_mutations_valid():
    """Test that all storage mutations are valid."""
    assert len(ALL_MUTATIONS) == 10

    for mutation in ALL_MUTATIONS:
        # Check required fields
        assert mutation.id
        assert mutation.target_type == "Microsoft.Storage/storageAccounts"
        assert mutation.description
        assert mutation.severity in ["critical", "high", "medium", "low"]
        assert mutation.labels
        assert callable(mutation.mutate)


def test_mutation_validation():
    """Test that mutations validate severity levels."""
    with pytest.raises(ValueError, match="Invalid severity"):
        Mutation(
            id="test",
            target_type="Microsoft.Storage/storageAccounts",
            description="Test",
            severity="invalid",  # Invalid severity
            labels=["Test"],
            mutate=lambda x: x,
        )
