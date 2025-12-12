"""
App Service / Web App Mutations

Defines mutations for Azure App Service (Web Apps, Function Apps) misconfigurations.
Based on real-world incidents: HTTPS bypass, remote debugging exploits, FTP exposure.

References:
- OWASP Top 10 (A02:2021 - Cryptographic Failures)
- CVE-2022-35829 (Azure App Service remote debugging vulnerability)
- CIS Azure Foundations Benchmark 9.x
"""

from typing import Dict, Any
from .base import Mutation, create_property_mutation

TARGET_TYPE = "Microsoft.Web/sites"


# Mutation 1: Disable HTTPS-only
# Real-world: Man-in-the-middle attacks, credential theft
def mutate_http_only(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Disable HTTPS requirement, allow HTTP traffic."""
    if "properties" not in resource:
        resource["properties"] = {}
    resource["properties"]["httpsOnly"] = False
    return resource


WEBAPP_HTTP_ONLY = Mutation(
    id="webapp_http_only",
    target_type=TARGET_TYPE,
    description="Disable HTTPS-only (allow unencrypted HTTP)",
    severity="high",
    labels=["WebApp_HTTPOnly", "CIS_9.2"],
    mutate=mutate_http_only,
    cis_references=["9.2"],
)


# Mutation 2: Weak TLS version
# Real-world: POODLE, BEAST attacks, PCI-DSS violations
def mutate_weak_tls(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Set minimum TLS version to 1.0."""
    if "properties" not in resource:
        resource["properties"] = {}
    if "siteConfig" not in resource["properties"]:
        resource["properties"]["siteConfig"] = {}
    resource["properties"]["siteConfig"]["minTlsVersion"] = "1.0"
    return resource


WEBAPP_WEAK_TLS = Mutation(
    id="webapp_weak_tls",
    target_type=TARGET_TYPE,
    description="Set minimum TLS version to 1.0 (vulnerable)",
    severity="high",
    labels=["WebApp_WeakTLS", "CIS_9.3"],
    mutate=mutate_weak_tls,
    cis_references=["9.3"],
)


# Mutation 3: Enable remote debugging
# Real-world: CVE-2022-35829, production debugging exploits
def mutate_remote_debugging(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Enable remote debugging in production."""
    if "properties" not in resource:
        resource["properties"] = {}
    if "siteConfig" not in resource["properties"]:
        resource["properties"]["siteConfig"] = {}
    resource["properties"]["siteConfig"]["remoteDebuggingEnabled"] = True
    return resource


WEBAPP_REMOTE_DEBUGGING = Mutation(
    id="webapp_remote_debugging",
    target_type=TARGET_TYPE,
    description="Enable remote debugging (security risk in production)",
    severity="critical",
    labels=["WebApp_RemoteDebugging", "CIS_9.7"],
    mutate=mutate_remote_debugging,
    cis_references=["9.7"],
)


# Mutation 4: Enable FTP deployment
# Real-world: Cleartext credentials, legacy protocol attacks
def mutate_ftp_enabled(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Enable FTP deployment (insecure protocol)."""
    if "properties" not in resource:
        resource["properties"] = {}
    if "siteConfig" not in resource["properties"]:
        resource["properties"]["siteConfig"] = {}
    resource["properties"]["siteConfig"]["ftpsState"] = "AllAllowed"
    return resource


WEBAPP_FTP_ENABLED = Mutation(
    id="webapp_ftp_enabled",
    target_type=TARGET_TYPE,
    description="Enable FTP deployment (cleartext credentials)",
    severity="high",
    labels=["WebApp_FTPEnabled", "CIS_9.4"],
    mutate=mutate_ftp_enabled,
    cis_references=["9.4"],
)


# Mutation 5: No managed identity
# Real-world: Hardcoded credentials in app settings, secrets in code
def mutate_no_managed_identity(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Remove managed identity configuration."""
    if "identity" in resource:
        del resource["identity"]
    return resource


WEBAPP_NO_MANAGED_IDENTITY = Mutation(
    id="webapp_no_managed_identity",
    target_type=TARGET_TYPE,
    description="No managed identity (forces credential storage)",
    severity="medium",
    labels=["WebApp_NoManagedIdentity", "CIS_9.5"],
    mutate=mutate_no_managed_identity,
    cis_references=["9.5"],
)


# Mutation 6: Disable client certificate requirement
# Real-world: No mutual TLS, weaker authentication
def mutate_no_client_cert(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Disable client certificate requirement."""
    if "properties" not in resource:
        resource["properties"] = {}
    resource["properties"]["clientCertEnabled"] = False
    resource["properties"]["clientCertMode"] = "Optional"
    return resource


WEBAPP_NO_CLIENT_CERT = Mutation(
    id="webapp_no_client_cert",
    target_type=TARGET_TYPE,
    description="Disable client certificate authentication",
    severity="medium",
    labels=["WebApp_NoClientCert", "CIS_9.6"],
    mutate=mutate_no_client_cert,
    cis_references=["9.6"],
)


# Mutation 7: Allow all network access
# Real-world: No IP restrictions, public exposure
def mutate_public_network(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Allow public network access without restrictions."""
    if "properties" not in resource:
        resource["properties"] = {}
    resource["properties"]["publicNetworkAccess"] = "Enabled"
    if "siteConfig" not in resource["properties"]:
        resource["properties"]["siteConfig"] = {}
    resource["properties"]["siteConfig"]["ipSecurityRestrictions"] = []
    return resource


WEBAPP_PUBLIC_NETWORK = Mutation(
    id="webapp_public_network",
    target_type=TARGET_TYPE,
    description="Allow all public network access (no IP restrictions)",
    severity="medium",
    labels=["WebApp_PublicNetwork"],
    mutate=mutate_public_network,
    cis_references=[],
)


# Mutation 8: Disable HTTPS redirect
# Real-world: Users can access via HTTP, mixed content issues
WEBAPP_NO_HTTPS_REDIRECT = create_property_mutation(
    mutation_id="webapp_no_https_redirect",
    target_type=TARGET_TYPE,
    description="Disable automatic HTTPS redirect",
    property_path="properties.siteConfig.httpLoggingEnabled",
    value=False,
    severity="medium",
    labels=["WebApp_NoHTTPSRedirect"],
    cis_references=[],
)


# Mutation 9: Enable HTTP 2.0 without proper config
# Real-world: HTTP/2 vulnerabilities if misconfigured
def mutate_http20_misconfigured(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Enable HTTP 2.0 without proper security settings."""
    if "properties" not in resource:
        resource["properties"] = {}
    if "siteConfig" not in resource["properties"]:
        resource["properties"]["siteConfig"] = {}
    resource["properties"]["siteConfig"]["http20Enabled"] = True
    resource["properties"]["siteConfig"]["minTlsVersion"] = "1.0"  # Dangerous combo
    return resource


WEBAPP_HTTP20_MISCONFIGURED = Mutation(
    id="webapp_http20_misconfigured",
    target_type=TARGET_TYPE,
    description="HTTP/2 enabled with weak TLS (misconfiguration)",
    severity="medium",
    labels=["WebApp_HTTP20Misconfigured"],
    mutate=mutate_http20_misconfigured,
    cis_references=[],
)


# Mutation 10: Always On disabled
# Real-world: Cold start delays, potential DoS vector
WEBAPP_ALWAYS_ON_DISABLED = create_property_mutation(
    mutation_id="webapp_always_on_disabled",
    target_type=TARGET_TYPE,
    description="Disable Always On (cold start vulnerability)",
    property_path="properties.siteConfig.alwaysOn",
    value=False,
    severity="low",
    labels=["WebApp_AlwaysOnDisabled"],
    cis_references=[],
)


# Mutation 11: Disable authentication
# Real-world: Unauthenticated access to web application
def mutate_no_auth(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Disable App Service authentication."""
    if "properties" not in resource:
        resource["properties"] = {}
    if "siteConfig" not in resource["properties"]:
        resource["properties"]["siteConfig"] = {}
    # Indicates auth is not configured
    resource["properties"]["siteConfig"]["authEnabled"] = False
    return resource


WEBAPP_NO_AUTH = Mutation(
    id="webapp_no_auth",
    target_type=TARGET_TYPE,
    description="Disable App Service authentication",
    severity="high",
    labels=["WebApp_NoAuth", "CIS_9.1"],
    mutate=mutate_no_auth,
    cis_references=["9.1"],
)


# Mutation 12: Use 32-bit worker process
# Real-world: Memory limitations, potential for certain exploits
WEBAPP_32BIT_WORKER = create_property_mutation(
    mutation_id="webapp_32bit_worker",
    target_type=TARGET_TYPE,
    description="Use 32-bit worker process (memory limited)",
    property_path="properties.siteConfig.use32BitWorkerProcess",
    value=True,
    severity="low",
    labels=["WebApp_32BitWorker"],
    cis_references=[],
)


# Export all App Service mutations
ALL_MUTATIONS = [
    WEBAPP_HTTP_ONLY,
    WEBAPP_WEAK_TLS,
    WEBAPP_REMOTE_DEBUGGING,
    WEBAPP_FTP_ENABLED,
    WEBAPP_NO_MANAGED_IDENTITY,
    WEBAPP_NO_CLIENT_CERT,
    WEBAPP_PUBLIC_NETWORK,
    WEBAPP_NO_HTTPS_REDIRECT,
    WEBAPP_HTTP20_MISCONFIGURED,
    WEBAPP_ALWAYS_ON_DISABLED,
    WEBAPP_NO_AUTH,
    WEBAPP_32BIT_WORKER,
]


def get_mutation_by_id(mutation_id: str) -> Mutation:
    """Get an App Service mutation by its ID."""
    for mutation in ALL_MUTATIONS:
        if mutation.id == mutation_id:
            return mutation
    raise KeyError(f"Mutation {mutation_id} not found in appservice mutations")


def get_mutations_by_severity(severity: str) -> list[Mutation]:
    """Get all App Service mutations of a specific severity."""
    return [m for m in ALL_MUTATIONS if m.severity == severity]
