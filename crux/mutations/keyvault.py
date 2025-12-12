"""
Key Vault Mutations

Defines mutations for Azure Key Vault misconfigurations.
Based on real-world incidents: exposed secrets, weak access policies, missing audit trails.

References:
- CIS Azure Foundations Benchmark 8.x
- Capital One breach (2019) - secrets management failures
- Uber breach (2022) - hardcoded credentials
"""

from typing import Dict, Any
from .base import Mutation, create_property_mutation

TARGET_TYPE = "Microsoft.KeyVault/vaults"


# Mutation 1: Disable purge protection
# Real-world: Accidental or malicious permanent deletion of secrets
def mutate_no_purge_protection(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Disable purge protection, allowing permanent secret deletion."""
    if "properties" not in resource:
        resource["properties"] = {}
    resource["properties"]["enablePurgeProtection"] = False
    return resource


KV_NO_PURGE_PROTECTION = Mutation(
    id="kv_no_purge_protection",
    target_type=TARGET_TYPE,
    description="Disable purge protection (allows permanent deletion)",
    severity="high",
    labels=["KeyVault_NoPurgeProtection", "CIS_8.4"],
    mutate=mutate_no_purge_protection,
    cis_references=["8.4"],
)


# Mutation 2: Disable soft delete
# Real-world: No recovery window for accidentally deleted secrets
def mutate_no_soft_delete(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Disable soft delete for Key Vault."""
    if "properties" not in resource:
        resource["properties"] = {}
    resource["properties"]["enableSoftDelete"] = False
    return resource


KV_NO_SOFT_DELETE = Mutation(
    id="kv_no_soft_delete",
    target_type=TARGET_TYPE,
    description="Disable soft delete (no recovery for deleted secrets)",
    severity="high",
    labels=["KeyVault_NoSoftDelete", "CIS_8.5"],
    mutate=mutate_no_soft_delete,
    cis_references=["8.5"],
)


# Mutation 3: Enable public network access
# Real-world: Secrets accessible from internet, major breach vector
def mutate_public_network_access(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Enable public network access to Key Vault."""
    if "properties" not in resource:
        resource["properties"] = {}
    resource["properties"]["publicNetworkAccess"] = "Enabled"
    # Also remove any network ACLs
    if "networkAcls" in resource["properties"]:
        resource["properties"]["networkAcls"]["defaultAction"] = "Allow"
    return resource


KV_PUBLIC_NETWORK_ACCESS = Mutation(
    id="kv_public_network_access",
    target_type=TARGET_TYPE,
    description="Enable public network access to Key Vault",
    severity="critical",
    labels=["KeyVault_PublicAccess", "CIS_8.6"],
    mutate=mutate_public_network_access,
    cis_references=["8.6"],
)


# Mutation 4: Use vault access policy instead of RBAC
# Real-world: Overly permissive access, hard to audit
def mutate_no_rbac(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Disable RBAC, use legacy access policies."""
    if "properties" not in resource:
        resource["properties"] = {}
    resource["properties"]["enableRbacAuthorization"] = False
    return resource


KV_NO_RBAC = Mutation(
    id="kv_no_rbac",
    target_type=TARGET_TYPE,
    description="Disable RBAC authorization (use legacy access policies)",
    severity="medium",
    labels=["KeyVault_NoRBAC", "CIS_8.7"],
    mutate=mutate_no_rbac,
    cis_references=["8.7"],
)


# Mutation 5: Overly permissive access policy
# Real-world: Any principal can read all secrets
def mutate_permissive_access_policy(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Add overly permissive access policy."""
    if "properties" not in resource:
        resource["properties"] = {}
    if "accessPolicies" not in resource["properties"]:
        resource["properties"]["accessPolicies"] = []

    # Add policy that grants all permissions
    permissive_policy = {
        "tenantId": "[subscription().tenantId]",
        "objectId": "*",  # Any principal
        "permissions": {
            "keys": ["all"],
            "secrets": ["all"],
            "certificates": ["all"],
            "storage": ["all"]
        }
    }
    resource["properties"]["accessPolicies"].append(permissive_policy)
    return resource


KV_PERMISSIVE_ACCESS = Mutation(
    id="kv_permissive_access",
    target_type=TARGET_TYPE,
    description="Add overly permissive access policy (all permissions)",
    severity="critical",
    labels=["KeyVault_PermissiveAccess"],
    mutate=mutate_permissive_access_policy,
    cis_references=[],
)


# Mutation 6: Disable Key Vault logging
# Real-world: No audit trail for secret access (compliance violation)
def mutate_no_logging(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Disable diagnostic logging for Key Vault."""
    if "properties" not in resource:
        resource["properties"] = {}
    # Remove any diagnostic settings reference
    resource["properties"]["enabledForDeployment"] = True
    resource["properties"]["enabledForDiskEncryption"] = True
    resource["properties"]["enabledForTemplateDeployment"] = True
    # Note: Actual logging is via Microsoft.Insights/diagnosticSettings
    # This mutation indicates intent to skip logging setup
    return resource


KV_NO_LOGGING = Mutation(
    id="kv_no_logging",
    target_type=TARGET_TYPE,
    description="Key Vault without diagnostic logging configured",
    severity="medium",
    labels=["KeyVault_NoLogging", "CIS_8.1"],
    mutate=mutate_no_logging,
    cis_references=["8.1"],
)


# Mutation 7: Short soft delete retention
# Real-world: Minimal recovery window (7 days minimum, should be 90)
def mutate_short_retention(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Set minimum soft delete retention period."""
    if "properties" not in resource:
        resource["properties"] = {}
    resource["properties"]["softDeleteRetentionInDays"] = 7  # Minimum allowed
    return resource


KV_SHORT_RETENTION = Mutation(
    id="kv_short_retention",
    target_type=TARGET_TYPE,
    description="Set minimum soft delete retention (7 days)",
    severity="low",
    labels=["KeyVault_ShortRetention"],
    mutate=mutate_short_retention,
    cis_references=[],
)


# Mutation 8: Allow Azure services bypass
# Real-world: Any Azure service can access, not just trusted ones
def mutate_allow_services_bypass(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Allow all Azure services to bypass firewall."""
    if "properties" not in resource:
        resource["properties"] = {}
    if "networkAcls" not in resource["properties"]:
        resource["properties"]["networkAcls"] = {}
    resource["properties"]["networkAcls"]["bypass"] = "AzureServices"
    resource["properties"]["networkAcls"]["defaultAction"] = "Deny"
    return resource


KV_SERVICES_BYPASS = Mutation(
    id="kv_services_bypass",
    target_type=TARGET_TYPE,
    description="Allow Azure services to bypass Key Vault firewall",
    severity="medium",
    labels=["KeyVault_ServicesBypass"],
    mutate=mutate_allow_services_bypass,
    cis_references=[],
)


# Mutation 9: No private endpoint
# Real-world: Traffic goes over public internet
def mutate_no_private_endpoint(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Remove private endpoint configuration."""
    if "properties" not in resource:
        resource["properties"] = {}
    resource["properties"]["publicNetworkAccess"] = "Enabled"
    # Indicate no private endpoint by not having networkAcls with virtualNetworkRules
    if "networkAcls" in resource["properties"]:
        resource["properties"]["networkAcls"].pop("virtualNetworkRules", None)
        resource["properties"]["networkAcls"].pop("ipRules", None)
    return resource


KV_NO_PRIVATE_ENDPOINT = Mutation(
    id="kv_no_private_endpoint",
    target_type=TARGET_TYPE,
    description="No private endpoint (traffic over public internet)",
    severity="high",
    labels=["KeyVault_NoPrivateEndpoint"],
    mutate=mutate_no_private_endpoint,
    cis_references=[],
)


# Mutation 10: Standard SKU (no HSM)
# Real-world: Keys not protected by hardware security module
KV_STANDARD_SKU = create_property_mutation(
    mutation_id="kv_standard_sku",
    target_type=TARGET_TYPE,
    description="Use Standard SKU (no HSM protection for keys)",
    property_path="properties.sku.name",
    value="standard",
    severity="low",
    labels=["KeyVault_NoHSM"],
    cis_references=[],
)


# Export all Key Vault mutations
ALL_MUTATIONS = [
    KV_NO_PURGE_PROTECTION,
    KV_NO_SOFT_DELETE,
    KV_PUBLIC_NETWORK_ACCESS,
    KV_NO_RBAC,
    KV_PERMISSIVE_ACCESS,
    KV_NO_LOGGING,
    KV_SHORT_RETENTION,
    KV_SERVICES_BYPASS,
    KV_NO_PRIVATE_ENDPOINT,
    KV_STANDARD_SKU,
]


def get_mutation_by_id(mutation_id: str) -> Mutation:
    """Get a Key Vault mutation by its ID."""
    for mutation in ALL_MUTATIONS:
        if mutation.id == mutation_id:
            return mutation
    raise KeyError(f"Mutation {mutation_id} not found in keyvault mutations")


def get_mutations_by_severity(severity: str) -> list[Mutation]:
    """Get all Key Vault mutations of a specific severity."""
    return [m for m in ALL_MUTATIONS if m.severity == severity]
