"""
Test script to verify zero-F1 label mutations and rules.

This script tests whether mutations are being applied correctly
and whether rules are detecting them properly.
"""

import json
from crux.mutations.vm import VM_NO_SECURE_BOOT, VM_NO_VTPM, VM_NO_AUTO_PATCHING
from crux.mutations.storage import STORAGE_NO_INFRASTRUCTURE_ENCRYPTION, STORAGE_NO_BLOB_VERSIONING
from crux.rules.evaluator import RuleEvaluator

def test_vm_secure_boot():
    """Test VM_NoSecureBoot mutation and rule."""
    print("\n=== Testing VM_NoSecureBoot ===")

    # Create a baseline VM resource
    resource = {
        "type": "Microsoft.Compute/virtualMachines",
        "id": "/subscriptions/test/resourceGroups/test/providers/Microsoft.Compute/virtualMachines/testvm",
        "name": "testvm",
        "properties": {
            "securityProfile": {
                "uefiSettings": {
                    "secureBootEnabled": True,
                    "vTpmEnabled": True
                }
            }
        }
    }

    print(f"Original secureBootEnabled: {resource['properties']['securityProfile']['uefiSettings']['secureBootEnabled']}")

    # Apply mutation
    mutated = VM_NO_SECURE_BOOT.apply(resource)

    if mutated:
        print(f"Mutated secureBootEnabled: {mutated['properties']['securityProfile']['uefiSettings']['secureBootEnabled']}")

        # Evaluate with rules
        evaluator = RuleEvaluator(rules_dir="rules")
        labels = evaluator.evaluate(mutated)

        print(f"Labels detected: {labels}")
        print(f"Expected: ['VM_NoSecureBoot', 'CIS_7.5']")

        if "VM_NoSecureBoot" in labels:
            print("✓ PASS: Rule correctly detected the mutation")
            return True
        else:
            print("✗ FAIL: Rule did NOT detect the mutation")
            print(f"Full mutated resource: {json.dumps(mutated, indent=2)}")
            return False
    else:
        print("✗ FAIL: Mutation was not applied")
        return False

def test_vm_vtpm():
    """Test VM_NoVTPM mutation and rule."""
    print("\n=== Testing VM_NoVTPM ===")

    resource = {
        "type": "Microsoft.Compute/virtualMachines",
        "id": "/subscriptions/test/resourceGroups/test/providers/Microsoft.Compute/virtualMachines/testvm",
        "name": "testvm",
        "properties": {
            "securityProfile": {
                "uefiSettings": {
                    "secureBootEnabled": True,
                    "vTpmEnabled": True
                }
            }
        }
    }

    print(f"Original vTpmEnabled: {resource['properties']['securityProfile']['uefiSettings']['vTpmEnabled']}")

    mutated = VM_NO_VTPM.apply(resource)

    if mutated:
        print(f"Mutated vTpmEnabled: {mutated['properties']['securityProfile']['uefiSettings']['vTpmEnabled']}")

        evaluator = RuleEvaluator(rules_dir="rules")
        labels = evaluator.evaluate(mutated)

        print(f"Labels detected: {labels}")
        print(f"Expected: ['VM_NoVTPM']")

        if "VM_NoVTPM" in labels:
            print("✓ PASS: Rule correctly detected the mutation")
            return True
        else:
            print("✗ FAIL: Rule did NOT detect the mutation")
            print(f"Full mutated resource: {json.dumps(mutated, indent=2)}")
            return False
    else:
        print("✗ FAIL: Mutation was not applied")
        return False

def test_vm_auto_patch():
    """Test VM_NoAutoPatch mutation and rule."""
    print("\n=== Testing VM_NoAutoPatch ===")

    resource = {
        "type": "Microsoft.Compute/virtualMachines",
        "id": "/subscriptions/test/resourceGroups/test/providers/Microsoft.Compute/virtualMachines/testvm",
        "name": "testvm",
        "properties": {
            "osProfile": {
                "linuxConfiguration": {
                    "disablePasswordAuthentication": True,
                    "patchSettings": {
                        "patchMode": "AutomaticByPlatform"
                    }
                }
            }
        }
    }

    print(f"Original patchMode: {resource['properties']['osProfile']['linuxConfiguration']['patchSettings']['patchMode']}")

    mutated = VM_NO_AUTO_PATCHING.apply(resource)

    if mutated:
        print(f"Mutated patchMode: {mutated['properties']['osProfile']['linuxConfiguration']['patchSettings']['patchMode']}")

        evaluator = RuleEvaluator(rules_dir="rules")
        labels = evaluator.evaluate(mutated)

        print(f"Labels detected: {labels}")
        print(f"Expected: ['VM_NoAutoPatch']")

        if "VM_NoAutoPatch" in labels:
            print("✓ PASS: Rule correctly detected the mutation")
            return True
        else:
            print("✗ FAIL: Rule did NOT detect the mutation")
            print(f"Full mutated resource: {json.dumps(mutated, indent=2)}")
            return False
    else:
        print("✗ FAIL: Mutation was not applied")
        return False

def test_storage_infra_encryption():
    """Test Storage_NoInfraEncryption mutation and rule."""
    print("\n=== Testing Storage_NoInfraEncryption ===")

    resource = {
        "type": "Microsoft.Storage/storageAccounts",
        "id": "/subscriptions/test/resourceGroups/test/providers/Microsoft.Storage/storageAccounts/teststorage",
        "name": "teststorage",
        "properties": {
            "encryption": {
                "requireInfrastructureEncryption": True,
                "services": {
                    "blob": {
                        "enabled": True
                    }
                }
            }
        }
    }

    print(f"Original requireInfrastructureEncryption: {resource['properties']['encryption']['requireInfrastructureEncryption']}")

    mutated = STORAGE_NO_INFRASTRUCTURE_ENCRYPTION.apply(resource)

    if mutated:
        print(f"Mutated requireInfrastructureEncryption: {mutated['properties']['encryption']['requireInfrastructureEncryption']}")

        evaluator = RuleEvaluator(rules_dir="rules")
        labels = evaluator.evaluate(mutated)

        print(f"Labels detected: {labels}")
        print(f"Expected: ['Storage_NoInfraEncryption']")

        if "Storage_NoInfraEncryption" in labels:
            print("✓ PASS: Rule correctly detected the mutation")
            return True
        else:
            print("✗ FAIL: Rule did NOT detect the mutation")
            print(f"Full mutated resource: {json.dumps(mutated, indent=2)}")
            return False
    else:
        print("✗ FAIL: Mutation was not applied")
        return False

def test_storage_versioning():
    """Test Storage_NoVersioning mutation and rule."""
    print("\n=== Testing Storage_NoVersioning ===")

    resource = {
        "type": "Microsoft.Storage/storageAccounts",
        "id": "/subscriptions/test/resourceGroups/test/providers/Microsoft.Storage/storageAccounts/teststorage",
        "name": "teststorage",
        "properties": {
            "isVersioningEnabled": True
        }
    }

    print(f"Original isVersioningEnabled: {resource['properties']['isVersioningEnabled']}")

    mutated = STORAGE_NO_BLOB_VERSIONING.apply(resource)

    if mutated:
        print(f"Mutated isVersioningEnabled: {mutated['properties']['isVersioningEnabled']}")

        evaluator = RuleEvaluator(rules_dir="rules")
        labels = evaluator.evaluate(mutated)

        print(f"Labels detected: {labels}")
        print(f"Expected: ['Storage_NoVersioning']")

        if "Storage_NoVersioning" in labels:
            print("✓ PASS: Rule correctly detected the mutation")
            return True
        else:
            print("✗ FAIL: Rule did NOT detect the mutation")
            print(f"Full mutated resource: {json.dumps(mutated, indent=2)}")
            return False
    else:
        print("✗ FAIL: Mutation was not applied")
        return False

if __name__ == "__main__":
    print("Testing Zero-F1 Label Mutations and Rules")
    print("=" * 60)

    results = []
    results.append(("VM_NoSecureBoot", test_vm_secure_boot()))
    results.append(("VM_NoVTPM", test_vm_vtpm()))
    results.append(("VM_NoAutoPatch", test_vm_auto_patch()))
    results.append(("Storage_NoInfraEncryption", test_storage_infra_encryption()))
    results.append(("Storage_NoVersioning", test_storage_versioning()))

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)

    for label, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {label}")

    total_passed = sum(1 for _, passed in results if passed)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")
