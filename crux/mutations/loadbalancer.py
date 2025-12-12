"""
Load Balancer and Public IP Mutations

Defines mutations for Azure Load Balancer and Public IP misconfigurations.
Based on real-world incidents: DDoS attacks, exposed management ports, availability issues.

References:
- Azure DDoS attack patterns
- CIS Azure Foundations Benchmark 6.x (Network)
"""

from typing import Dict, Any
from .base import Mutation, create_property_mutation

LB_TARGET_TYPE = "Microsoft.Network/loadBalancers"
PIP_TARGET_TYPE = "Microsoft.Network/publicIPAddresses"


# ========== Public IP Mutations ==========

# Mutation 1: Basic SKU (no DDoS protection)
# Real-world: Vulnerable to DDoS, no zone redundancy
PIP_BASIC_SKU = create_property_mutation(
    mutation_id="pip_basic_sku",
    target_type=PIP_TARGET_TYPE,
    description="Use Basic SKU (no DDoS protection or zone redundancy)",
    property_path="sku.name",
    value="Basic",
    severity="high",
    labels=["PIP_BasicSKU", "CIS_6.7"],
    cis_references=["6.7"],
)


# Mutation 2: Dynamic allocation
# Real-world: IP changes on restart, DNS issues, certificate problems
PIP_DYNAMIC_ALLOCATION = create_property_mutation(
    mutation_id="pip_dynamic_allocation",
    target_type=PIP_TARGET_TYPE,
    description="Use dynamic IP allocation (IP changes on restart)",
    property_path="properties.publicIPAllocationMethod",
    value="Dynamic",
    severity="low",
    labels=["PIP_DynamicAllocation"],
    cis_references=[],
)


# Mutation 3: No DDoS protection
# Real-world: Direct DDoS attacks without mitigation
def mutate_pip_no_ddos(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Remove DDoS protection from Public IP."""
    if "properties" not in resource:
        resource["properties"] = {}
    # Remove DDoS protection plan reference
    resource["properties"].pop("ddosSettings", None)
    return resource


PIP_NO_DDOS = Mutation(
    id="pip_no_ddos",
    target_type=PIP_TARGET_TYPE,
    description="No DDoS protection plan configured",
    severity="high",
    labels=["PIP_NoDDoS", "CIS_6.8"],
    mutate=mutate_pip_no_ddos,
    cis_references=["6.8"],
)


# Mutation 4: IPv4 only (no IPv6)
# Real-world: Limited connectivity, future compatibility issues
PIP_IPV4_ONLY = create_property_mutation(
    mutation_id="pip_ipv4_only",
    target_type=PIP_TARGET_TYPE,
    description="IPv4 only (no IPv6 support)",
    property_path="properties.publicIPAddressVersion",
    value="IPv4",
    severity="low",
    labels=["PIP_IPv4Only"],
    cis_references=[],
)


# Mutation 5: No idle timeout configured
# Real-world: Connection issues, resource exhaustion
def mutate_pip_short_timeout(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Set very short idle timeout."""
    if "properties" not in resource:
        resource["properties"] = {}
    resource["properties"]["idleTimeoutInMinutes"] = 4  # Minimum
    return resource


PIP_SHORT_TIMEOUT = Mutation(
    id="pip_short_timeout",
    target_type=PIP_TARGET_TYPE,
    description="Short idle timeout (4 minutes, may cause connection drops)",
    severity="low",
    labels=["PIP_ShortTimeout"],
    mutate=mutate_pip_short_timeout,
    cis_references=[],
)


# ========== Load Balancer Mutations ==========

# Mutation 6: Basic SKU
# Real-world: No availability zones, limited features
LB_BASIC_SKU = create_property_mutation(
    mutation_id="lb_basic_sku",
    target_type=LB_TARGET_TYPE,
    description="Use Basic SKU (no availability zones, limited security)",
    property_path="sku.name",
    value="Basic",
    severity="medium",
    labels=["LB_BasicSKU"],
    cis_references=[],
)


# Mutation 7: No health probes
# Real-world: Traffic sent to unhealthy backends
def mutate_lb_no_health_probes(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Remove health probes from load balancer."""
    if "properties" not in resource:
        resource["properties"] = {}
    resource["properties"]["probes"] = []
    return resource


LB_NO_HEALTH_PROBES = Mutation(
    id="lb_no_health_probes",
    target_type=LB_TARGET_TYPE,
    description="No health probes (traffic sent to unhealthy backends)",
    severity="high",
    labels=["LB_NoHealthProbes"],
    mutate=mutate_lb_no_health_probes,
    cis_references=[],
)


# Mutation 8: Open management ports
# Real-world: RDP/SSH exposed through load balancer
def mutate_lb_open_management(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Add NAT rules for management ports (RDP/SSH)."""
    if "properties" not in resource:
        resource["properties"] = {}
    if "inboundNatRules" not in resource["properties"]:
        resource["properties"]["inboundNatRules"] = []

    # Add dangerous NAT rules
    resource["properties"]["inboundNatRules"].extend([
        {
            "name": "RDP-NAT",
            "properties": {
                "frontendPort": 3389,
                "backendPort": 3389,
                "protocol": "Tcp"
            }
        },
        {
            "name": "SSH-NAT",
            "properties": {
                "frontendPort": 22,
                "backendPort": 22,
                "protocol": "Tcp"
            }
        }
    ])
    return resource


LB_OPEN_MANAGEMENT = Mutation(
    id="lb_open_management",
    target_type=LB_TARGET_TYPE,
    description="NAT rules expose management ports (RDP/SSH) publicly",
    severity="critical",
    labels=["LB_OpenManagement", "CIS_6.9"],
    mutate=mutate_lb_open_management,
    cis_references=["6.9"],
)


# Mutation 9: No outbound rules (implicit SNAT)
# Real-world: Port exhaustion, unpredictable outbound IPs
def mutate_lb_no_outbound_rules(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Remove outbound rules (relies on implicit SNAT)."""
    if "properties" not in resource:
        resource["properties"] = {}
    resource["properties"]["outboundRules"] = []
    return resource


LB_NO_OUTBOUND_RULES = Mutation(
    id="lb_no_outbound_rules",
    target_type=LB_TARGET_TYPE,
    description="No outbound rules (port exhaustion risk with implicit SNAT)",
    severity="medium",
    labels=["LB_NoOutboundRules"],
    mutate=mutate_lb_no_outbound_rules,
    cis_references=[],
)


# Mutation 10: TCP reset on idle disabled
# Real-world: Connections hang indefinitely
def mutate_lb_no_tcp_reset(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Disable TCP reset on idle timeout."""
    if "properties" not in resource:
        resource["properties"] = {}
    if "loadBalancingRules" not in resource["properties"]:
        resource["properties"]["loadBalancingRules"] = []

    for rule in resource["properties"]["loadBalancingRules"]:
        if "properties" in rule:
            rule["properties"]["enableTcpReset"] = False

    return resource


LB_NO_TCP_RESET = Mutation(
    id="lb_no_tcp_reset",
    target_type=LB_TARGET_TYPE,
    description="Disable TCP reset on idle (connections hang)",
    severity="low",
    labels=["LB_NoTCPReset"],
    mutate=mutate_lb_no_tcp_reset,
    cis_references=[],
)


# Mutation 11: Floating IP enabled without proper config
# Real-world: Misconfigured Direct Server Return
def mutate_lb_floating_ip(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Enable floating IP on rules (DSR without proper backend config)."""
    if "properties" not in resource:
        resource["properties"] = {}
    if "loadBalancingRules" not in resource["properties"]:
        resource["properties"]["loadBalancingRules"] = []

    for rule in resource["properties"]["loadBalancingRules"]:
        if "properties" in rule:
            rule["properties"]["enableFloatingIP"] = True

    return resource


LB_FLOATING_IP = Mutation(
    id="lb_floating_ip",
    target_type=LB_TARGET_TYPE,
    description="Floating IP enabled (DSR mode, requires special backend config)",
    severity="low",
    labels=["LB_FloatingIP"],
    mutate=mutate_lb_floating_ip,
    cis_references=[],
)


# Mutation 12: All ports load balancing (HA Ports)
# Real-world: All traffic forwarded, potential security bypass
def mutate_lb_all_ports(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Configure HA Ports rule (all ports forwarded)."""
    if "properties" not in resource:
        resource["properties"] = {}
    if "loadBalancingRules" not in resource["properties"]:
        resource["properties"]["loadBalancingRules"] = []

    # Add HA Ports rule
    resource["properties"]["loadBalancingRules"].append({
        "name": "HAPortsRule",
        "properties": {
            "frontendPort": 0,  # All ports
            "backendPort": 0,   # All ports
            "protocol": "All"
        }
    })
    return resource


LB_ALL_PORTS = Mutation(
    id="lb_all_ports",
    target_type=LB_TARGET_TYPE,
    description="HA Ports rule (all ports forwarded to backend)",
    severity="medium",
    labels=["LB_AllPorts"],
    mutate=mutate_lb_all_ports,
    cis_references=[],
)


# Export all Load Balancer and Public IP mutations
ALL_MUTATIONS = [
    # Public IP
    PIP_BASIC_SKU,
    PIP_DYNAMIC_ALLOCATION,
    PIP_NO_DDOS,
    PIP_IPV4_ONLY,
    PIP_SHORT_TIMEOUT,
    # Load Balancer
    LB_BASIC_SKU,
    LB_NO_HEALTH_PROBES,
    LB_OPEN_MANAGEMENT,
    LB_NO_OUTBOUND_RULES,
    LB_NO_TCP_RESET,
    LB_FLOATING_IP,
    LB_ALL_PORTS,
]


def get_mutation_by_id(mutation_id: str) -> Mutation:
    """Get a load balancer/public IP mutation by its ID."""
    for mutation in ALL_MUTATIONS:
        if mutation.id == mutation_id:
            return mutation
    raise KeyError(f"Mutation {mutation_id} not found in loadbalancer mutations")


def get_mutations_by_severity(severity: str) -> list[Mutation]:
    """Get all load balancer/public IP mutations of a specific severity."""
    return [m for m in ALL_MUTATIONS if m.severity == severity]
