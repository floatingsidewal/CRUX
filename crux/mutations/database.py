"""
Database Mutations

Defines mutations for Azure database misconfigurations (Cosmos DB, SQL Server).
Based on real-world incidents: exposed databases, weak authentication, public endpoints.

References:
- MongoDB/Cosmos DB breaches (2017-2023) - public endpoints exposing millions of records
- Azure SQL injection vulnerabilities
- CIS Azure Foundations Benchmark 4.x (SQL), 5.x (Cosmos DB)
"""

from typing import Dict, Any
from .base import Mutation, create_property_mutation

COSMOSDB_TARGET_TYPE = "Microsoft.DocumentDB/databaseAccounts"
SQL_TARGET_TYPE = "Microsoft.Sql/servers"


# ========== Cosmos DB Mutations ==========

# Mutation 1: Enable public network access
# Real-world: Exposed Cosmos DB instances (multiple breaches 2019-2023)
def mutate_cosmosdb_public_access(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Enable public network access for Cosmos DB."""
    if "properties" not in resource:
        resource["properties"] = {}
    resource["properties"]["publicNetworkAccess"] = "Enabled"
    # Remove IP rules
    if "ipRules" in resource["properties"]:
        resource["properties"]["ipRules"] = []
    return resource


COSMOSDB_PUBLIC_ACCESS = Mutation(
    id="cosmosdb_public_access",
    target_type=COSMOSDB_TARGET_TYPE,
    description="Enable public network access (database exposed to internet)",
    severity="critical",
    labels=["CosmosDB_PublicAccess", "CIS_5.1"],
    mutate=mutate_cosmosdb_public_access,
    cis_references=["5.1"],
)


# Mutation 2: Disable local authentication
# Real-world: Key-based auth instead of Azure AD
def mutate_cosmosdb_local_auth(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Enable local authentication (key-based, not Azure AD)."""
    if "properties" not in resource:
        resource["properties"] = {}
    resource["properties"]["disableLocalAuth"] = False
    return resource


COSMOSDB_LOCAL_AUTH = Mutation(
    id="cosmosdb_local_auth",
    target_type=COSMOSDB_TARGET_TYPE,
    description="Enable local authentication (key-based instead of Azure AD)",
    severity="medium",
    labels=["CosmosDB_LocalAuth", "CIS_5.2"],
    mutate=mutate_cosmosdb_local_auth,
    cis_references=["5.2"],
)


# Mutation 3: Disable automatic failover
# Real-world: No high availability, data loss risk
COSMOSDB_NO_FAILOVER = create_property_mutation(
    mutation_id="cosmosdb_no_failover",
    target_type=COSMOSDB_TARGET_TYPE,
    description="Disable automatic failover (availability risk)",
    property_path="properties.enableAutomaticFailover",
    value=False,
    severity="medium",
    labels=["CosmosDB_NoFailover"],
    cis_references=[],
)


# Mutation 4: Allow all Azure datacenter IPs
# Real-world: Overly permissive network rules
def mutate_cosmosdb_allow_azure(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Allow access from all Azure datacenters (0.0.0.0)."""
    if "properties" not in resource:
        resource["properties"] = {}
    # This IP rule allows all Azure services
    resource["properties"]["ipRules"] = [{"ipAddressOrRange": "0.0.0.0"}]
    return resource


COSMOSDB_ALLOW_AZURE = Mutation(
    id="cosmosdb_allow_azure",
    target_type=COSMOSDB_TARGET_TYPE,
    description="Allow access from all Azure datacenters (overly permissive)",
    severity="high",
    labels=["CosmosDB_AllowAllAzure", "CIS_5.3"],
    mutate=mutate_cosmosdb_allow_azure,
    cis_references=["5.3"],
)


# Mutation 5: Disable encryption at rest with customer key
# Real-world: Using Microsoft-managed keys instead of customer-managed
COSMOSDB_NO_CMK = create_property_mutation(
    mutation_id="cosmosdb_no_cmk",
    target_type=COSMOSDB_TARGET_TYPE,
    description="Use Microsoft-managed keys (not customer-managed)",
    property_path="properties.keyVaultKeyUri",
    value=None,
    severity="low",
    labels=["CosmosDB_NoCMK"],
    cis_references=[],
)


# ========== SQL Server Mutations ==========

# Mutation 6: Enable public endpoint
# Real-world: SQL Server exposed to internet (massive breach vector)
def mutate_sql_public_endpoint(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Enable public endpoint for SQL Server."""
    if "properties" not in resource:
        resource["properties"] = {}
    resource["properties"]["publicNetworkAccess"] = "Enabled"
    return resource


SQL_PUBLIC_ENDPOINT = Mutation(
    id="sql_public_endpoint",
    target_type=SQL_TARGET_TYPE,
    description="Enable public endpoint (SQL Server exposed to internet)",
    severity="critical",
    labels=["SQL_PublicEndpoint", "CIS_4.1"],
    mutate=mutate_sql_public_endpoint,
    cis_references=["4.1"],
)


# Mutation 7: Disable auditing
# Real-world: No audit trail for data access (compliance violation)
def mutate_sql_no_auditing(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Disable SQL Server auditing."""
    if "properties" not in resource:
        resource["properties"] = {}
    # Indicate auditing should be disabled
    resource["properties"]["auditingPolicy"] = {
        "state": "Disabled"
    }
    return resource


SQL_NO_AUDITING = Mutation(
    id="sql_no_auditing",
    target_type=SQL_TARGET_TYPE,
    description="Disable SQL Server auditing (no audit trail)",
    severity="high",
    labels=["SQL_NoAuditing", "CIS_4.2"],
    mutate=mutate_sql_no_auditing,
    cis_references=["4.2"],
)


# Mutation 8: Disable threat detection
# Real-world: No alerts for SQL injection, anomalies
def mutate_sql_no_threat_detection(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Disable SQL threat detection."""
    if "properties" not in resource:
        resource["properties"] = {}
    resource["properties"]["threatDetectionPolicy"] = {
        "state": "Disabled"
    }
    return resource


SQL_NO_THREAT_DETECTION = Mutation(
    id="sql_no_threat_detection",
    target_type=SQL_TARGET_TYPE,
    description="Disable SQL threat detection (no anomaly alerts)",
    severity="high",
    labels=["SQL_NoThreatDetection", "CIS_4.3"],
    mutate=mutate_sql_no_threat_detection,
    cis_references=["4.3"],
)


# Mutation 9: Allow Azure services access
# Real-world: Overly permissive, any Azure service can connect
def mutate_sql_allow_azure_services(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Allow all Azure services to access SQL Server."""
    if "properties" not in resource:
        resource["properties"] = {}
    # This creates a firewall rule allowing all Azure IPs
    if "firewallRules" not in resource["properties"]:
        resource["properties"]["firewallRules"] = []
    resource["properties"]["firewallRules"].append({
        "name": "AllowAllAzureIps",
        "startIpAddress": "0.0.0.0",
        "endIpAddress": "0.0.0.0"
    })
    return resource


SQL_ALLOW_AZURE_SERVICES = Mutation(
    id="sql_allow_azure_services",
    target_type=SQL_TARGET_TYPE,
    description="Allow all Azure services access (overly permissive)",
    severity="medium",
    labels=["SQL_AllowAzureServices", "CIS_4.4"],
    mutate=mutate_sql_allow_azure_services,
    cis_references=["4.4"],
)


# Mutation 10: Use SQL authentication instead of Azure AD
# Real-world: Password-based auth, weaker than Azure AD
def mutate_sql_auth(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Use SQL authentication (password-based, not Azure AD only)."""
    if "properties" not in resource:
        resource["properties"] = {}
    resource["properties"]["administrators"] = {
        "azureADOnlyAuthentication": False
    }
    return resource


SQL_AUTH_ENABLED = Mutation(
    id="sql_auth_enabled",
    target_type=SQL_TARGET_TYPE,
    description="Enable SQL authentication (password-based, not Azure AD only)",
    severity="medium",
    labels=["SQL_SQLAuthEnabled", "CIS_4.5"],
    mutate=mutate_sql_auth,
    cis_references=["4.5"],
)


# Mutation 11: Disable TDE (Transparent Data Encryption)
# Real-world: Data at rest not encrypted
def mutate_sql_no_tde(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Disable Transparent Data Encryption."""
    if "properties" not in resource:
        resource["properties"] = {}
    resource["properties"]["transparentDataEncryption"] = {
        "state": "Disabled"
    }
    return resource


SQL_NO_TDE = Mutation(
    id="sql_no_tde",
    target_type=SQL_TARGET_TYPE,
    description="Disable Transparent Data Encryption (data at rest unencrypted)",
    severity="critical",
    labels=["SQL_NoTDE", "CIS_4.6"],
    mutate=mutate_sql_no_tde,
    cis_references=["4.6"],
)


# Mutation 12: Minimum TLS 1.0
# Real-world: Weak TLS, vulnerable to POODLE/BEAST
SQL_WEAK_TLS = create_property_mutation(
    mutation_id="sql_weak_tls",
    target_type=SQL_TARGET_TYPE,
    description="Set minimum TLS version to 1.0 (vulnerable)",
    property_path="properties.minimalTlsVersion",
    value="1.0",
    severity="high",
    labels=["SQL_WeakTLS", "CIS_4.7"],
    cis_references=["4.7"],
)


# Export all Database mutations
ALL_MUTATIONS = [
    # Cosmos DB
    COSMOSDB_PUBLIC_ACCESS,
    COSMOSDB_LOCAL_AUTH,
    COSMOSDB_NO_FAILOVER,
    COSMOSDB_ALLOW_AZURE,
    COSMOSDB_NO_CMK,
    # SQL Server
    SQL_PUBLIC_ENDPOINT,
    SQL_NO_AUDITING,
    SQL_NO_THREAT_DETECTION,
    SQL_ALLOW_AZURE_SERVICES,
    SQL_AUTH_ENABLED,
    SQL_NO_TDE,
    SQL_WEAK_TLS,
]


def get_mutation_by_id(mutation_id: str) -> Mutation:
    """Get a database mutation by its ID."""
    for mutation in ALL_MUTATIONS:
        if mutation.id == mutation_id:
            return mutation
    raise KeyError(f"Mutation {mutation_id} not found in database mutations")


def get_mutations_by_severity(severity: str) -> list[Mutation]:
    """Get all database mutations of a specific severity."""
    return [m for m in ALL_MUTATIONS if m.severity == severity]
