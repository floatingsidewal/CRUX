"""
Container Registry Mutations

Defines mutations for Azure Container Registry (ACR) misconfigurations.
Based on real-world incidents: supply chain attacks, public image exposure, admin credential compromise.

References:
- SolarWinds attack (2020) - supply chain compromise
- Codecov breach (2021) - container image tampering
- CIS Azure Foundations Benchmark (Container Security)
"""

from typing import Dict, Any
from .base import Mutation, create_property_mutation

TARGET_TYPE = "Microsoft.ContainerRegistry/registries"


# Mutation 1: Enable admin user
# Real-world: Single point of compromise, shared credentials
def mutate_admin_user_enabled(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Enable admin user account (shared credentials risk)."""
    if "properties" not in resource:
        resource["properties"] = {}
    resource["properties"]["adminUserEnabled"] = True
    return resource


ACR_ADMIN_USER_ENABLED = Mutation(
    id="acr_admin_user_enabled",
    target_type=TARGET_TYPE,
    description="Enable admin user (shared credentials, single point of failure)",
    severity="high",
    labels=["ACR_AdminUserEnabled", "CIS_ACR_1"],
    mutate=mutate_admin_user_enabled,
    cis_references=["ACR_1"],
)


# Mutation 2: Enable public network access
# Real-world: Anyone can pull/push images, supply chain risk
def mutate_public_network_access(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Enable public network access to container registry."""
    if "properties" not in resource:
        resource["properties"] = {}
    resource["properties"]["publicNetworkAccess"] = "Enabled"
    resource["properties"]["networkRuleBypassOptions"] = "AzureServices"
    return resource


ACR_PUBLIC_ACCESS = Mutation(
    id="acr_public_access",
    target_type=TARGET_TYPE,
    description="Enable public network access (images accessible from internet)",
    severity="high",
    labels=["ACR_PublicAccess", "CIS_ACR_2"],
    mutate=mutate_public_network_access,
    cis_references=["ACR_2"],
)


# Mutation 3: Enable anonymous pull
# Real-world: No authentication required to pull images
def mutate_anonymous_pull(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Enable anonymous image pull (no authentication)."""
    if "properties" not in resource:
        resource["properties"] = {}
    resource["properties"]["anonymousPullEnabled"] = True
    return resource


ACR_ANONYMOUS_PULL = Mutation(
    id="acr_anonymous_pull",
    target_type=TARGET_TYPE,
    description="Enable anonymous pull (images publicly accessible)",
    severity="critical",
    labels=["ACR_AnonymousPull", "CIS_ACR_3"],
    mutate=mutate_anonymous_pull,
    cis_references=["ACR_3"],
)


# Mutation 4: Disable content trust
# Real-world: Unsigned images, no verification of image integrity
def mutate_no_content_trust(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Disable content trust (unsigned images allowed)."""
    if "properties" not in resource:
        resource["properties"] = {}
    if "policies" not in resource["properties"]:
        resource["properties"]["policies"] = {}
    if "trustPolicy" not in resource["properties"]["policies"]:
        resource["properties"]["policies"]["trustPolicy"] = {}
    resource["properties"]["policies"]["trustPolicy"]["status"] = "disabled"
    return resource


ACR_NO_CONTENT_TRUST = Mutation(
    id="acr_no_content_trust",
    target_type=TARGET_TYPE,
    description="Disable content trust (allow unsigned images)",
    severity="high",
    labels=["ACR_NoContentTrust", "CIS_ACR_4"],
    mutate=mutate_no_content_trust,
    cis_references=["ACR_4"],
)


# Mutation 5: Use Basic SKU (no security features)
# Real-world: No geo-replication, limited security features
ACR_BASIC_SKU = create_property_mutation(
    mutation_id="acr_basic_sku",
    target_type=TARGET_TYPE,
    description="Use Basic SKU (limited security features)",
    property_path="sku.name",
    value="Basic",
    severity="medium",
    labels=["ACR_BasicSKU"],
    cis_references=[],
)


# Mutation 6: Disable zone redundancy
# Real-world: Single point of failure, availability risk
def mutate_no_zone_redundancy(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Disable zone redundancy."""
    if "properties" not in resource:
        resource["properties"] = {}
    resource["properties"]["zoneRedundancy"] = "Disabled"
    return resource


ACR_NO_ZONE_REDUNDANCY = Mutation(
    id="acr_no_zone_redundancy",
    target_type=TARGET_TYPE,
    description="Disable zone redundancy (single point of failure)",
    severity="low",
    labels=["ACR_NoZoneRedundancy"],
    mutate=mutate_no_zone_redundancy,
    cis_references=[],
)


# Mutation 7: Disable quarantine policy
# Real-world: Vulnerable images deployed without scanning
def mutate_no_quarantine(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Disable quarantine policy (no image scanning)."""
    if "properties" not in resource:
        resource["properties"] = {}
    if "policies" not in resource["properties"]:
        resource["properties"]["policies"] = {}
    if "quarantinePolicy" not in resource["properties"]["policies"]:
        resource["properties"]["policies"]["quarantinePolicy"] = {}
    resource["properties"]["policies"]["quarantinePolicy"]["status"] = "disabled"
    return resource


ACR_NO_QUARANTINE = Mutation(
    id="acr_no_quarantine",
    target_type=TARGET_TYPE,
    description="Disable quarantine policy (vulnerable images not scanned)",
    severity="medium",
    labels=["ACR_NoQuarantine"],
    mutate=mutate_no_quarantine,
    cis_references=[],
)


# Mutation 8: Disable export policy
# Real-world: Images can be exported without control
def mutate_export_enabled(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Enable export policy (images can be exported)."""
    if "properties" not in resource:
        resource["properties"] = {}
    if "policies" not in resource["properties"]:
        resource["properties"]["policies"] = {}
    if "exportPolicy" not in resource["properties"]["policies"]:
        resource["properties"]["policies"]["exportPolicy"] = {}
    resource["properties"]["policies"]["exportPolicy"]["status"] = "enabled"
    return resource


ACR_EXPORT_ENABLED = Mutation(
    id="acr_export_enabled",
    target_type=TARGET_TYPE,
    description="Enable image export (data exfiltration risk)",
    severity="medium",
    labels=["ACR_ExportEnabled"],
    mutate=mutate_export_enabled,
    cis_references=[],
)


# Export all Container Registry mutations
ALL_MUTATIONS = [
    ACR_ADMIN_USER_ENABLED,
    ACR_PUBLIC_ACCESS,
    ACR_ANONYMOUS_PULL,
    ACR_NO_CONTENT_TRUST,
    ACR_BASIC_SKU,
    ACR_NO_ZONE_REDUNDANCY,
    ACR_NO_QUARANTINE,
    ACR_EXPORT_ENABLED,
]


def get_mutation_by_id(mutation_id: str) -> Mutation:
    """Get a Container Registry mutation by its ID."""
    for mutation in ALL_MUTATIONS:
        if mutation.id == mutation_id:
            return mutation
    raise KeyError(f"Mutation {mutation_id} not found in containerregistry mutations")


def get_mutations_by_severity(severity: str) -> list[Mutation]:
    """Get all Container Registry mutations of a specific severity."""
    return [m for m in ALL_MUTATIONS if m.severity == severity]
