#!/usr/bin/env python3
"""
Verification script for template-level dataset generation.

This script verifies that the mutation resolution and application works correctly,
particularly for scenarios with nested @ references like 'all_mutations'.

Originally created to diagnose a bug where 'baseline' and 'all_mutations' scenarios
showed identical 0% misconfiguration rates. The root cause was a broken import:
compile_bicep_to_arm should have been compile_bicep_file.

Run this script to verify the fix:
    python scripts/diagnose_resolution_bug.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from crux.dataset.template_level_generator import (
    TemplateLevelDatasetGenerator,
    MUTATION_SCENARIOS,
)
from crux.mutations import get_all_mutations


def diagnose_resolution():
    """Diagnose the mutation resolution process."""
    print("=" * 70)
    print("DIAGNOSIS: Template-Level Dataset Generation Bug")
    print("=" * 70)

    # Step 1: Check MUTATION_SCENARIOS structure
    print("\n1. MUTATION_SCENARIOS Structure:")
    print("-" * 50)

    scenarios_with_refs = []
    scenarios_direct = []

    for scenario_id, config in MUTATION_SCENARIOS.items():
        mutations = config.get('mutations', [])
        has_refs = any(m.startswith('@') for m in mutations)

        if has_refs:
            scenarios_with_refs.append(scenario_id)
            print(f"  {scenario_id}: {mutations} (has @ refs)")
        else:
            scenarios_direct.append(scenario_id)

    print(f"\n  Scenarios with @ references: {len(scenarios_with_refs)}")
    print(f"  Scenarios with direct IDs: {len(scenarios_direct)}")

    # Step 2: Check available mutations
    print("\n2. Available Mutations:")
    print("-" * 50)
    all_mutations = get_all_mutations()
    mutation_ids = {m.id for m in all_mutations}
    print(f"  Total mutations loaded: {len(mutation_ids)}")

    # Check if all referenced mutations exist
    all_referenced = set()
    for scenario_id, config in MUTATION_SCENARIOS.items():
        for m in config.get('mutations', []):
            if not m.startswith('@'):
                all_referenced.add(m)

    missing = all_referenced - mutation_ids
    if missing:
        print(f"  WARNING: Missing mutations: {missing}")
    else:
        print(f"  All {len(all_referenced)} referenced mutations exist ✓")

    # Step 3: Initialize generator and check resolution
    print("\n3. Mutation Resolution:")
    print("-" * 50)

    # Create a simple rules directory for testing
    import tempfile
    import yaml

    with tempfile.TemporaryDirectory() as tmpdir:
        rules_path = Path(tmpdir) / "rules"
        rules_path.mkdir()

        # Create minimal rules file
        rules = {"rules": []}
        with open(rules_path / "test.yaml", "w") as f:
            yaml.dump(rules, f)

        generator = TemplateLevelDatasetGenerator(rules_dir=str(rules_path))

        # Check resolution for key scenarios
        scenarios_to_check = ['baseline', 'security_high', 'security_all', 'all_mutations']

        for scenario_id in scenarios_to_check:
            resolved = generator._resolved_scenarios.get(scenario_id, [])
            config_mutations = MUTATION_SCENARIOS[scenario_id]['mutations']

            print(f"\n  {scenario_id}:")
            print(f"    Config mutations: {config_mutations}")
            print(f"    Resolved count: {len(resolved)}")
            print(f"    Resolved: {resolved[:5]}{'...' if len(resolved) > 5 else ''}")

            # Verify all resolved mutations exist
            for m_id in resolved:
                if m_id not in mutation_ids:
                    print(f"    WARNING: Resolved mutation '{m_id}' not in available mutations!")

        # Step 4: Test actual mutation application
        print("\n4. Mutation Application Test:")
        print("-" * 50)

        test_resources = [
            {
                'type': 'Microsoft.Storage/storageAccounts',
                'name': 'testaccount',
                'properties': {
                    'allowBlobPublicAccess': False,
                    'supportsHttpsTrafficOnly': True,
                    'minimumTlsVersion': 'TLS1_2'
                }
            }
        ]

        for scenario_id in ['baseline', 'security_high', 'all_mutations']:
            mutation_ids_list = generator._resolved_scenarios.get(scenario_id, [])
            mutated = generator._apply_mutations(test_resources, mutation_ids_list)

            # Check the storage account properties
            storage = mutated[0]
            public_access = storage.get('properties', {}).get('allowBlobPublicAccess', 'MISSING')
            https_only = storage.get('properties', {}).get('supportsHttpsTrafficOnly', 'MISSING')
            tls = storage.get('properties', {}).get('minimumTlsVersion', 'MISSING')

            print(f"\n  {scenario_id}:")
            print(f"    Mutations to apply: {len(mutation_ids_list)}")
            print(f"    allowBlobPublicAccess: {public_access}")
            print(f"    supportsHttpsTrafficOnly: {https_only}")
            print(f"    minimumTlsVersion: {tls}")

            if '_mutation_applied' in storage:
                print(f"    Mutations applied: {[m['mutation_id'] for m in storage['_mutation_applied']]}")
            else:
                print(f"    Mutations applied: NONE")

        # Step 5: Feature extraction test
        print("\n5. Feature Extraction Test:")
        print("-" * 50)

        import networkx as nx
        empty_graph = nx.DiGraph()

        for scenario_id in ['baseline', 'security_high', 'all_mutations']:
            mutation_ids_list = generator._resolved_scenarios.get(scenario_id, [])
            mutated = generator._apply_mutations(test_resources, mutation_ids_list)
            features = generator._extract_security_features(mutated)

            print(f"\n  {scenario_id}:")
            print(f"    any_public_access: {features.get('any_public_access')}")
            print(f"    any_weak_tls: {features.get('any_weak_tls')}")
            print(f"    any_http_allowed: {features.get('any_http_allowed')}")

    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)


def trace_resolution_step_by_step():
    """Trace the resolution algorithm step by step."""
    print("\n" + "=" * 70)
    print("STEP-BY-STEP RESOLUTION TRACE")
    print("=" * 70)

    scenarios = MUTATION_SCENARIOS

    def trace_resolve(mutation_list, depth=0):
        """Traced version of _resolve_mutations."""
        indent = "  " * depth
        resolved = []

        print(f"{indent}Input: {mutation_list}")

        for item in mutation_list:
            if item.startswith('@'):
                ref_id = item[1:]
                print(f"{indent}  Processing reference: {item} -> looking for '{ref_id}'")

                if ref_id in scenarios:
                    print(f"{indent}    Found! Recursing into {ref_id}...")
                    nested = trace_resolve(scenarios[ref_id]['mutations'], depth + 2)
                    print(f"{indent}    Got from {ref_id}: {len(nested)} mutations")
                    resolved.extend(nested)
                else:
                    print(f"{indent}    NOT FOUND in scenarios! SKIPPING!")
            else:
                print(f"{indent}  Adding direct mutation: {item}")
                resolved.append(item)

        result = list(set(resolved))
        print(f"{indent}Returning: {len(result)} unique mutations")
        return result

    # Trace for all_mutations
    print("\n--- Tracing 'all_mutations' resolution ---")
    all_mutations_config = scenarios['all_mutations']['mutations']
    result = trace_resolve(all_mutations_config)
    print(f"\n==> Final result for 'all_mutations': {len(result)} mutations")
    print(f"==> Mutations: {sorted(result)}")


if __name__ == '__main__':
    diagnose_resolution()
    trace_resolution_step_by_step()
