"""
Network Mutations

Defines mutations for Azure Network misconfigurations (NSG, VNet, etc.).
"""

from typing import Dict, Any
from .base import Mutation, create_property_mutation

NSG_TARGET_TYPE = "Microsoft.Network/networkSecurityGroups"
VNET_TARGET_TYPE = "Microsoft.Network/virtualNetworks"


# ========== Network Security Group Mutations ==========

# Mutation 1: Allow all inbound traffic
def mutate_nsg_allow_all_inbound(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Add rule to allow all inbound traffic."""
    if "properties" not in resource:
        resource["properties"] = {}
    if "securityRules" not in resource["properties"]:
        resource["properties"]["securityRules"] = []

    # Add dangerous rule
    dangerous_rule = {
        "name": "AllowAllInbound",
        "properties": {
            "priority": 100,
            "direction": "Inbound",
            "access": "Allow",
            "protocol": "*",
            "sourcePortRange": "*",
            "destinationPortRange": "*",
            "sourceAddressPrefix": "*",
            "destinationAddressPrefix": "*",
        }
    }
    resource["properties"]["securityRules"].insert(0, dangerous_rule)
    return resource


NSG_ALLOW_ALL_INBOUND = Mutation(
    id="nsg_allow_all_inbound",
    target_type=NSG_TARGET_TYPE,
    description="Allow all inbound traffic",
    severity="critical",
    labels=["NSG_AllowAllInbound", "CIS_6.1"],
    mutate=mutate_nsg_allow_all_inbound,
    cis_references=["6.1"],
)


# Mutation 2: Open SSH to internet
def mutate_nsg_open_ssh(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Allow SSH from any source."""
    if "properties" not in resource:
        resource["properties"] = {}
    if "securityRules" not in resource["properties"]:
        resource["properties"]["securityRules"] = []

    ssh_rule = {
        "name": "AllowSSHFromInternet",
        "properties": {
            "priority": 100,
            "direction": "Inbound",
            "access": "Allow",
            "protocol": "Tcp",
            "sourcePortRange": "*",
            "destinationPortRange": "22",
            "sourceAddressPrefix": "*",
            "destinationAddressPrefix": "*",
        }
    }
    resource["properties"]["securityRules"].insert(0, ssh_rule)
    return resource


NSG_OPEN_SSH = Mutation(
    id="nsg_open_ssh",
    target_type=NSG_TARGET_TYPE,
    description="Allow SSH from internet",
    severity="critical",
    labels=["NSG_OpenSSH", "CIS_6.2"],
    mutate=mutate_nsg_open_ssh,
    cis_references=["6.2"],
)


# Mutation 3: Open RDP to internet
def mutate_nsg_open_rdp(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Allow RDP from any source."""
    if "properties" not in resource:
        resource["properties"] = {}
    if "securityRules" not in resource["properties"]:
        resource["properties"]["securityRules"] = []

    rdp_rule = {
        "name": "AllowRDPFromInternet",
        "properties": {
            "priority": 101,
            "direction": "Inbound",
            "access": "Allow",
            "protocol": "Tcp",
            "sourcePortRange": "*",
            "destinationPortRange": "3389",
            "sourceAddressPrefix": "*",
            "destinationAddressPrefix": "*",
        }
    }
    resource["properties"]["securityRules"].insert(0, rdp_rule)
    return resource


NSG_OPEN_RDP = Mutation(
    id="nsg_open_rdp",
    target_type=NSG_TARGET_TYPE,
    description="Allow RDP from internet",
    severity="critical",
    labels=["NSG_OpenRDP", "CIS_6.3"],
    mutate=mutate_nsg_open_rdp,
    cis_references=["6.3"],
)


# Mutation 4: Open database ports
def mutate_nsg_open_database(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Allow database ports from internet."""
    if "properties" not in resource:
        resource["properties"] = {}
    if "securityRules" not in resource["properties"]:
        resource["properties"]["securityRules"] = []

    db_rule = {
        "name": "AllowDatabaseFromInternet",
        "properties": {
            "priority": 102,
            "direction": "Inbound",
            "access": "Allow",
            "protocol": "Tcp",
            "sourcePortRange": "*",
            "destinationPortRanges": ["1433", "3306", "5432"],  # SQL, MySQL, PostgreSQL
            "sourceAddressPrefix": "*",
            "destinationAddressPrefix": "*",
        }
    }
    resource["properties"]["securityRules"].insert(0, db_rule)
    return resource


NSG_OPEN_DATABASE = Mutation(
    id="nsg_open_database",
    target_type=NSG_TARGET_TYPE,
    description="Allow database ports from internet",
    severity="critical",
    labels=["NSG_OpenDatabase", "CIS_6.4"],
    mutate=mutate_nsg_open_database,
    cis_references=["6.4"],
)


# Mutation 5: No default deny rule
def mutate_nsg_no_default_deny(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Remove default deny rules."""
    if "properties" not in resource:
        resource["properties"] = {}
    if "securityRules" not in resource["properties"]:
        resource["properties"]["securityRules"] = []

    # Remove any deny rules
    resource["properties"]["securityRules"] = [
        rule for rule in resource["properties"]["securityRules"]
        if rule.get("properties", {}).get("access") != "Deny"
    ]
    return resource


NSG_NO_DEFAULT_DENY = Mutation(
    id="nsg_no_default_deny",
    target_type=NSG_TARGET_TYPE,
    description="No default deny rule configured",
    severity="high",
    labels=["NSG_NoDefaultDeny"],
    mutate=mutate_nsg_no_default_deny,
    cis_references=[],
)


# Mutation 6: Allow FTP
def mutate_nsg_allow_ftp(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Allow FTP from internet."""
    if "properties" not in resource:
        resource["properties"] = {}
    if "securityRules" not in resource["properties"]:
        resource["properties"]["securityRules"] = []

    ftp_rule = {
        "name": "AllowFTPFromInternet",
        "properties": {
            "priority": 103,
            "direction": "Inbound",
            "access": "Allow",
            "protocol": "Tcp",
            "sourcePortRange": "*",
            "destinationPortRange": "21",
            "sourceAddressPrefix": "*",
            "destinationAddressPrefix": "*",
        }
    }
    resource["properties"]["securityRules"].insert(0, ftp_rule)
    return resource


NSG_ALLOW_FTP = Mutation(
    id="nsg_allow_ftp",
    target_type=NSG_TARGET_TYPE,
    description="Allow FTP from internet",
    severity="high",
    labels=["NSG_OpenFTP"],
    mutate=mutate_nsg_allow_ftp,
    cis_references=[],
)


# ========== Virtual Network Mutations ==========

# Mutation 7: No DDoS protection
VNET_NO_DDOS = create_property_mutation(
    mutation_id="vnet_no_ddos",
    target_type=VNET_TARGET_TYPE,
    description="Disable DDoS protection",
    property_path="properties.enableDdosProtection",
    value=False,
    severity="high",
    labels=["VNet_NoDDoSProtection", "CIS_6.5"],
    cis_references=["6.5"],
)


# Mutation 8: No VM protection
VNET_NO_VM_PROTECTION = create_property_mutation(
    mutation_id="vnet_no_vm_protection",
    target_type=VNET_TARGET_TYPE,
    description="Disable VM protection",
    property_path="properties.enableVmProtection",
    value=False,
    severity="medium",
    labels=["VNet_NoVMProtection"],
    cis_references=[],
)


# Mutation 9: Disable service endpoints
def mutate_vnet_no_service_endpoints(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Remove service endpoints from subnets."""
    if "properties" not in resource:
        resource["properties"] = {}
    if "subnets" not in resource["properties"]:
        return resource

    for subnet in resource["properties"]["subnets"]:
        if "properties" in subnet:
            subnet["properties"].pop("serviceEndpoints", None)

    return resource


VNET_NO_SERVICE_ENDPOINTS = Mutation(
    id="vnet_no_service_endpoints",
    target_type=VNET_TARGET_TYPE,
    description="Remove service endpoints from subnets",
    severity="medium",
    labels=["VNet_NoServiceEndpoints"],
    mutate=mutate_vnet_no_service_endpoints,
    cis_references=[],
)


# Mutation 10: Overly broad address space
def mutate_vnet_broad_address_space(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Use overly broad address space."""
    if "properties" not in resource:
        resource["properties"] = {}

    # Use very large address space
    resource["properties"]["addressSpace"] = {
        "addressPrefixes": ["10.0.0.0/8"]  # Entire 10.x.x.x range
    }
    return resource


VNET_BROAD_ADDRESS_SPACE = Mutation(
    id="vnet_broad_address_space",
    target_type=VNET_TARGET_TYPE,
    description="Use overly broad network address space",
    severity="low",
    labels=["VNet_BroadAddressSpace"],
    mutate=mutate_vnet_broad_address_space,
    cis_references=[],
)


# Mutation 11: No NSG on subnet
def mutate_subnet_no_nsg(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Remove NSG from subnets."""
    if "properties" not in resource:
        resource["properties"] = {}
    if "subnets" not in resource["properties"]:
        return resource

    for subnet in resource["properties"]["subnets"]:
        if "properties" in subnet:
            subnet["properties"].pop("networkSecurityGroup", None)

    return resource


VNET_SUBNET_NO_NSG = Mutation(
    id="vnet_subnet_no_nsg",
    target_type=VNET_TARGET_TYPE,
    description="No NSG configured on subnets",
    severity="high",
    labels=["VNet_SubnetNoNSG", "CIS_6.6"],
    mutate=mutate_subnet_no_nsg,
    cis_references=["6.6"],
)


# Mutation 12: Disable BGP route propagation
def mutate_vnet_no_bgp(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Disable BGP route propagation."""
    if "properties" not in resource:
        resource["properties"] = {}
    if "subnets" not in resource["properties"]:
        return resource

    for subnet in resource["properties"]["subnets"]:
        if "properties" in subnet:
            if "routeTable" in subnet["properties"]:
                subnet["properties"]["routeTable"]["properties"] = {
                    "disableBgpRoutePropagation": True
                }

    return resource


VNET_NO_BGP = Mutation(
    id="vnet_no_bgp",
    target_type=VNET_TARGET_TYPE,
    description="Disable BGP route propagation",
    severity="low",
    labels=["VNet_NoBGP"],
    mutate=mutate_vnet_no_bgp,
    cis_references=[],
)


# Export all Network mutations
ALL_MUTATIONS = [
    # NSG mutations
    NSG_ALLOW_ALL_INBOUND,
    NSG_OPEN_SSH,
    NSG_OPEN_RDP,
    NSG_OPEN_DATABASE,
    NSG_NO_DEFAULT_DENY,
    NSG_ALLOW_FTP,
    # VNet mutations
    VNET_NO_DDOS,
    VNET_NO_VM_PROTECTION,
    VNET_NO_SERVICE_ENDPOINTS,
    VNET_BROAD_ADDRESS_SPACE,
    VNET_SUBNET_NO_NSG,
    VNET_NO_BGP,
]


def get_mutation_by_id(mutation_id: str) -> Mutation:
    """Get a network mutation by its ID."""
    for mutation in ALL_MUTATIONS:
        if mutation.id == mutation_id:
            return mutation
    raise KeyError(f"Mutation {mutation_id} not found in network mutations")


def get_mutations_by_severity(severity: str) -> list[Mutation]:
    """Get all network mutations of a specific severity."""
    return [m for m in ALL_MUTATIONS if m.severity == severity]
