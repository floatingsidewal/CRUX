"""Tests for mutation functions."""

import pytest
from crux.mutations.storage import (
    STORAGE_PUBLIC_BLOB_ACCESS,
    STORAGE_WEAK_TLS,
    STORAGE_NO_HTTPS,
)


class TestStorageMutations:
    """Test storage account mutations."""

    def test_storage_public_blob_access(self):
        """Test enabling public blob access."""
        resource = {
            "type": "Microsoft.Storage/storageAccounts",
            "name": "test-storage",
            "properties": {"allowBlobPublicAccess": False},
        }

        mutated = STORAGE_PUBLIC_BLOB_ACCESS.mutate(resource.copy())

        assert mutated["properties"]["allowBlobPublicAccess"] is True
        assert mutated["type"] == "Microsoft.Storage/storageAccounts"

    def test_storage_weak_tls(self):
        """Test setting weak TLS version."""
        resource = {
            "type": "Microsoft.Storage/storageAccounts",
            "name": "test-storage",
            "properties": {"minimumTlsVersion": "TLS1_2"},
        }

        mutated = STORAGE_WEAK_TLS.mutate(resource.copy())

        assert mutated["properties"]["minimumTlsVersion"] == "TLS1_0"

    def test_storage_no_https(self):
        """Test disabling HTTPS requirement."""
        resource = {
            "type": "Microsoft.Storage/storageAccounts",
            "name": "test-storage",
            "properties": {"supportsHttpsTrafficOnly": True},
        }

        mutated = STORAGE_NO_HTTPS.mutate(resource.copy())

        assert mutated["properties"]["supportsHttpsTrafficOnly"] is False

    def test_mutation_without_properties(self):
        """Test mutation on resource without properties."""
        resource = {
            "type": "Microsoft.Storage/storageAccounts",
            "name": "test-storage",
        }

        mutated = STORAGE_PUBLIC_BLOB_ACCESS.mutate(resource.copy())

        assert "properties" in mutated
        assert mutated["properties"]["allowBlobPublicAccess"] is True
