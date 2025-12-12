"""
Virtual Machine Mutations

Defines mutations for Azure Virtual Machine misconfigurations.
"""

from typing import Dict, Any
from .base import Mutation, create_property_mutation

TARGET_TYPE = "Microsoft.Compute/virtualMachines"


# Mutation 1: Disable VM encryption
def mutate_no_vm_encryption(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Disable VM disk encryption."""
    if "properties" not in resource:
        resource["properties"] = {}
    if "storageProfile" not in resource["properties"]:
        resource["properties"]["storageProfile"] = {}
    if "osDisk" not in resource["properties"]["storageProfile"]:
        resource["properties"]["storageProfile"]["osDisk"] = {}

    # Remove encryption settings
    resource["properties"]["storageProfile"]["osDisk"].pop("encryptionSettings", None)
    return resource


VM_NO_ENCRYPTION = Mutation(
    id="vm_no_encryption",
    target_type=TARGET_TYPE,
    description="Disable VM disk encryption",
    severity="critical",
    labels=["VM_NoEncryption", "CIS_7.1"],
    mutate=mutate_no_vm_encryption,
    cis_references=["7.1"],
)


# Mutation 2: Disable managed disk encryption
VM_NO_MANAGED_DISK_ENCRYPTION = create_property_mutation(
    mutation_id="vm_no_managed_disk_encryption",
    target_type=TARGET_TYPE,
    description="Disable managed disk encryption at rest",
    property_path="properties.storageProfile.osDisk.managedDisk.diskEncryptionSet",
    value=None,
    severity="high",
    labels=["VM_NoManagedDiskEncryption"],
    cis_references=[],
)


# Mutation 3: Allow password authentication (SSH)
def mutate_allow_password_auth(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Enable password authentication for SSH."""
    if "properties" not in resource:
        resource["properties"] = {}
    if "osProfile" not in resource["properties"]:
        resource["properties"]["osProfile"] = {}
    if "linuxConfiguration" not in resource["properties"]["osProfile"]:
        resource["properties"]["osProfile"]["linuxConfiguration"] = {}

    resource["properties"]["osProfile"]["linuxConfiguration"]["disablePasswordAuthentication"] = False
    return resource


VM_ALLOW_PASSWORD_AUTH = Mutation(
    id="vm_allow_password_auth",
    target_type=TARGET_TYPE,
    description="Allow password authentication for SSH",
    severity="high",
    labels=["VM_PasswordAuthEnabled", "CIS_7.2"],
    mutate=mutate_allow_password_auth,
    cis_references=["7.2"],
)


# Mutation 4: Disable boot diagnostics
VM_NO_BOOT_DIAGNOSTICS = create_property_mutation(
    mutation_id="vm_no_boot_diagnostics",
    target_type=TARGET_TYPE,
    description="Disable boot diagnostics",
    property_path="properties.diagnosticsProfile.bootDiagnostics.enabled",
    value=False,
    severity="medium",
    labels=["VM_NoBootDiagnostics"],
    cis_references=[],
)


# Mutation 5: Disable OS patching
def mutate_disable_os_patching(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Disable automatic OS patching."""
    if "properties" not in resource:
        resource["properties"] = {}
    if "osProfile" not in resource["properties"]:
        resource["properties"]["osProfile"] = {}
    if "linuxConfiguration" not in resource["properties"]["osProfile"]:
        resource["properties"]["osProfile"]["linuxConfiguration"] = {}

    resource["properties"]["osProfile"]["linuxConfiguration"]["patchSettings"] = {
        "patchMode": "Manual"
    }
    return resource


VM_NO_AUTO_PATCHING = Mutation(
    id="vm_no_auto_patching",
    target_type=TARGET_TYPE,
    description="Disable automatic OS patching",
    severity="medium",
    labels=["VM_NoAutoPatch"],
    mutate=mutate_disable_os_patching,
    cis_references=[],
)


# Mutation 6: Use old VM size (non-secure)
VM_INSECURE_SIZE = create_property_mutation(
    mutation_id="vm_insecure_size",
    target_type=TARGET_TYPE,
    description="Use outdated VM size without security features",
    property_path="properties.hardwareProfile.vmSize",
    value="Standard_A1",  # Old generation
    severity="low",
    labels=["VM_OutdatedSize"],
    cis_references=[],
)


# Mutation 7: Disable VM monitoring
def mutate_disable_monitoring(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Disable VM monitoring extensions."""
    if "resources" in resource:
        # Remove monitoring extensions
        resource["resources"] = [
            ext for ext in resource["resources"]
            if "Microsoft.Insights" not in ext.get("type", "")
        ]
    return resource


VM_NO_MONITORING = Mutation(
    id="vm_no_monitoring",
    target_type=TARGET_TYPE,
    description="Disable VM monitoring and diagnostics",
    severity="medium",
    labels=["VM_NoMonitoring"],
    mutate=mutate_disable_monitoring,
    cis_references=[],
)


# Mutation 8: Use unmanaged disks
def mutate_use_unmanaged_disk(resource: Dict[str, Any]) -> Dict[str, Any]:
    """Use unmanaged disks instead of managed disks."""
    if "properties" not in resource:
        resource["properties"] = {}
    if "storageProfile" not in resource["properties"]:
        resource["properties"]["storageProfile"] = {}
    if "osDisk" not in resource["properties"]["storageProfile"]:
        resource["properties"]["storageProfile"]["osDisk"] = {}

    # Remove managed disk, add VHD
    resource["properties"]["storageProfile"]["osDisk"].pop("managedDisk", None)
    resource["properties"]["storageProfile"]["osDisk"]["vhd"] = {
        "uri": "https://mystorageaccount.blob.core.windows.net/vhds/osdisk.vhd"
    }
    return resource


VM_UNMANAGED_DISK = Mutation(
    id="vm_unmanaged_disk",
    target_type=TARGET_TYPE,
    description="Use unmanaged disks instead of managed disks",
    severity="medium",
    labels=["VM_UnmanagedDisk"],
    mutate=mutate_use_unmanaged_disk,
    cis_references=[],
)


# Mutation 9: Disable secure boot
VM_NO_SECURE_BOOT = create_property_mutation(
    mutation_id="vm_no_secure_boot",
    target_type=TARGET_TYPE,
    description="Disable secure boot feature",
    property_path="properties.securityProfile.uefiSettings.secureBootEnabled",
    value=False,
    severity="high",
    labels=["VM_NoSecureBoot", "CIS_7.5"],
    cis_references=["7.5"],
)


# Mutation 10: Disable vTPM
VM_NO_VTPM = create_property_mutation(
    mutation_id="vm_no_vtpm",
    target_type=TARGET_TYPE,
    description="Disable virtual TPM",
    property_path="properties.securityProfile.uefiSettings.vTpmEnabled",
    value=False,
    severity="high",
    labels=["VM_NoVTPM"],
    cis_references=[],
)


# Mutation 11: No availability set
VM_NO_AVAILABILITY = create_property_mutation(
    mutation_id="vm_no_availability",
    target_type=TARGET_TYPE,
    description="No availability set configured",
    property_path="properties.availabilitySet",
    value=None,
    severity="low",
    labels=["VM_NoAvailability"],
    cis_references=[],
)


# Mutation 12: Disable identity
VM_NO_IDENTITY = create_property_mutation(
    mutation_id="vm_no_identity",
    target_type=TARGET_TYPE,
    description="No managed identity configured",
    property_path="identity",
    value=None,
    severity="medium",
    labels=["VM_NoManagedIdentity"],
    cis_references=[],
)


# Export all VM mutations
ALL_MUTATIONS = [
    VM_NO_ENCRYPTION,
    VM_NO_MANAGED_DISK_ENCRYPTION,
    VM_ALLOW_PASSWORD_AUTH,
    VM_NO_BOOT_DIAGNOSTICS,
    VM_NO_AUTO_PATCHING,
    VM_INSECURE_SIZE,
    VM_NO_MONITORING,
    VM_UNMANAGED_DISK,
    VM_NO_SECURE_BOOT,
    VM_NO_VTPM,
    VM_NO_AVAILABILITY,
    VM_NO_IDENTITY,
]


def get_mutation_by_id(mutation_id: str) -> Mutation:
    """Get a VM mutation by its ID."""
    for mutation in ALL_MUTATIONS:
        if mutation.id == mutation_id:
            return mutation
    raise KeyError(f"Mutation {mutation_id} not found in VM mutations")


def get_mutations_by_severity(severity: str) -> list[Mutation]:
    """Get all VM mutations of a specific severity."""
    return [m for m in ALL_MUTATIONS if m.severity == severity]
