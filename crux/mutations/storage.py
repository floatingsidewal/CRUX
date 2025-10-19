"""
Storage Account Mutations

Defines mutations for Azure Storage Account misconfigurations.
"""

from typing import Dict, Any
from .base import Mutation, create_property_mutation

TARGET_TYPE = "Microsoft.Storage/storageAccounts"


# Mutation 1: Enable public blob access
def mutate_public_blob_access(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Enable public blob access on storage account."""
    if "properties" not in resource:
        resource["properties"] = {}
    resource["properties"]["allowBlobPublicAccess"] = True
    return resource


STORAGE_PUBLIC_BLOB_ACCESS = Mutation(
    id="storage_public_blob_access",
    target_type=TARGET_TYPE,
    description="Enable public blob access on storage account",
    severity="high",
    labels=["Storage_PublicAccess", "CIS_3.7"],
    cis_references=["3.7"],
    mutate=mutate_public_blob_access,
)


# Mutation 2: Weak TLS version
STORAGE_WEAK_TLS = create_property_mutation(
    mutation_id="storage_weak_tls",
    target_type=TARGET_TYPE,
    description="Set minimum TLS version to 1.0 (weak)",
    property_path="properties.minimumTlsVersion",
    value="TLS1_0",
    severity="medium",
    labels=["Storage_WeakTLS", "CIS_3.8"],
    cis_references=["3.8"],
)


# Mutation 3: Disable HTTPS-only
STORAGE_HTTP_ALLOWED = create_property_mutation(
    mutation_id="storage_http_allowed",
    target_type=TARGET_TYPE,
    description="Allow HTTP traffic (not HTTPS-only)",
    property_path="properties.supportsHttpsTrafficOnly",
    value=False,
    severity="high",
    labels=["Storage_HTTPAllowed", "CIS_3.1"],
    cis_references=["3.1"],
)


# Mutation 4: Open firewall to all IPs
def mutate_open_firewall(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Open storage account firewall to all IPs."""
    if "properties" not in resource:
        resource["properties"] = {}
    if "networkAcls" not in resource["properties"]:
        resource["properties"]["networkAcls"] = {}

    resource["properties"]["networkAcls"]["defaultAction"] = "Allow"
    return resource


STORAGE_OPEN_FIREWALL = Mutation(
    id="storage_open_firewall",
    target_type=TARGET_TYPE,
    description="Set firewall to allow all network traffic",
    severity="high",
    labels=["Storage_FirewallOpen", "CIS_3.6"],
    cis_references=["3.6"],
    mutate=mutate_open_firewall,
)


# Mutation 5: Disable secure transfer
STORAGE_INSECURE_TRANSFER = create_property_mutation(
    mutation_id="storage_insecure_transfer",
    target_type=TARGET_TYPE,
    description="Disable secure transfer requirement",
    property_path="properties.supportsHttpsTrafficOnly",
    value=False,
    severity="critical",
    labels=["Storage_InsecureTransfer", "CIS_3.1"],
    cis_references=["3.1"],
)


# Mutation 6: Disable blob versioning
STORAGE_NO_BLOB_VERSIONING = create_property_mutation(
    mutation_id="storage_no_blob_versioning",
    target_type=TARGET_TYPE,
    description="Disable blob versioning",
    property_path="properties.isVersioningEnabled",
    value=False,
    severity="low",
    labels=["Storage_NoVersioning"],
    cis_references=[],
)


# Mutation 7: Disable blob soft delete
def mutate_disable_soft_delete(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Disable blob soft delete."""
    if "properties" not in resource:
        resource["properties"] = {}
    if "deleteRetentionPolicy" not in resource["properties"]:
        resource["properties"]["deleteRetentionPolicy"] = {}

    resource["properties"]["deleteRetentionPolicy"]["enabled"] = False
    return resource


STORAGE_NO_SOFT_DELETE = Mutation(
    id="storage_no_soft_delete",
    target_type=TARGET_TYPE,
    description="Disable blob soft delete",
    severity="medium",
    labels=["Storage_NoSoftDelete", "CIS_3.10"],
    cis_references=["3.10"],
    mutate=mutate_disable_soft_delete,
)


# Mutation 8: Allow shared key access
STORAGE_ALLOW_SHARED_KEY = create_property_mutation(
    mutation_id="storage_allow_shared_key",
    target_type=TARGET_TYPE,
    description="Allow shared key access (not just Azure AD)",
    property_path="properties.allowSharedKeyAccess",
    value=True,
    severity="medium",
    labels=["Storage_SharedKeyAccess"],
    cis_references=[],
)


# Mutation 9: Disable infrastructure encryption
STORAGE_NO_INFRASTRUCTURE_ENCRYPTION = create_property_mutation(
    mutation_id="storage_no_infrastructure_encryption",
    target_type=TARGET_TYPE,
    description="Disable infrastructure encryption",
    property_path="properties.encryption.requireInfrastructureEncryption",
    value=False,
    severity="medium",
    labels=["Storage_NoInfraEncryption"],
    cis_references=[],
)


# Mutation 10: Allow cross-tenant replication
STORAGE_CROSS_TENANT_REPLICATION = create_property_mutation(
    mutation_id="storage_cross_tenant_replication",
    target_type=TARGET_TYPE,
    description="Allow cross-tenant replication",
    property_path="properties.allowCrossTenantReplication",
    value=True,
    severity="low",
    labels=["Storage_CrossTenantReplication"],
    cis_references=[],
)


# Export all storage mutations
ALL_MUTATIONS = [
    STORAGE_PUBLIC_BLOB_ACCESS,
    STORAGE_WEAK_TLS,
    STORAGE_HTTP_ALLOWED,
    STORAGE_OPEN_FIREWALL,
    STORAGE_INSECURE_TRANSFER,
    STORAGE_NO_BLOB_VERSIONING,
    STORAGE_NO_SOFT_DELETE,
    STORAGE_ALLOW_SHARED_KEY,
    STORAGE_NO_INFRASTRUCTURE_ENCRYPTION,
    STORAGE_CROSS_TENANT_REPLICATION,
]


def get_mutation_by_id(mutation_id: str) -> Mutation:
    """
    Get a storage mutation by its ID.

    Args:
        mutation_id: The mutation ID

    Returns:
        Mutation object

    Raises:
        KeyError: If mutation ID not found
    """
    for mutation in ALL_MUTATIONS:
        if mutation.id == mutation_id:
            return mutation
    raise KeyError(f"Mutation {mutation_id} not found in storage mutations")


def get_mutations_by_severity(severity: str) -> list[Mutation]:
    """
    Get all storage mutations of a specific severity.

    Args:
        severity: Severity level (critical, high, medium, low)

    Returns:
        List of mutations matching the severity
    """
    return [m for m in ALL_MUTATIONS if m.severity == severity]
