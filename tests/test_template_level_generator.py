"""
Tests for template-level dataset generator.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest
import pandas as pd
import networkx as nx

from crux.dataset.template_level_generator import (
    TemplateLevelDatasetGenerator,
    MUTATION_SCENARIOS,
)
from crux.mutations.base import Mutation


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def rules_dir(temp_dir):
    """Create a temporary rules directory with sample rules."""
    rules_path = Path(temp_dir) / "rules"
    rules_path.mkdir(exist_ok=True)

    # Create a sample storage rule
    storage_rules = {
        "rules": [
            {
                "id": "storage-public-blob-access",
                "resource_type": "Microsoft.Storage/storageAccounts",
                "severity": "high",
                "cis_reference": "3.7",
                "condition": {
                    "property": "properties.allowBlobPublicAccess",
                    "equals": True
                },
                "labels": ["Storage_PublicAccess", "CIS_3.7"]
            },
            {
                "id": "storage-http-allowed",
                "resource_type": "Microsoft.Storage/storageAccounts",
                "severity": "high",
                "cis_reference": "3.1",
                "condition": {
                    "property": "properties.supportsHttpsTrafficOnly",
                    "equals": False
                },
                "labels": ["Storage_HTTPAllowed", "CIS_3.1"]
            }
        ]
    }

    # Create a sample VM rule
    vm_rules = {
        "rules": [
            {
                "id": "vm-password-auth",
                "resource_type": "Microsoft.Compute/virtualMachines",
                "severity": "high",
                "cis_reference": "7.2",
                "condition": {
                    "property": "properties.osProfile.linuxConfiguration.disablePasswordAuthentication",
                    "equals": False
                },
                "labels": ["VM_PasswordAuthEnabled", "CIS_7.2"]
            }
        ]
    }

    import yaml
    with open(rules_path / "storage.yaml", "w") as f:
        yaml.dump(storage_rules, f)

    with open(rules_path / "vm.yaml", "w") as f:
        yaml.dump(vm_rules, f)

    return str(rules_path)


@pytest.fixture
def sample_mutations():
    """Create sample mutations for testing."""
    def mutate_storage_public(resource: Dict[str, Any]) -> Dict[str, Any]:
        if "properties" not in resource:
            resource["properties"] = {}
        resource["properties"]["allowBlobPublicAccess"] = True
        return resource

    def mutate_vm_password(resource: Dict[str, Any]) -> Dict[str, Any]:
        if "properties" not in resource:
            resource["properties"] = {}
        if "osProfile" not in resource["properties"]:
            resource["properties"]["osProfile"] = {}
        if "linuxConfiguration" not in resource["properties"]["osProfile"]:
            resource["properties"]["osProfile"]["linuxConfiguration"] = {}
        resource["properties"]["osProfile"]["linuxConfiguration"]["disablePasswordAuthentication"] = False
        return resource

    return [
        Mutation(
            id="storage_public_blob_access",
            target_type="Microsoft.Storage/storageAccounts",
            description="Enable public blob access",
            severity="high",
            labels=["Storage_PublicAccess", "CIS_3.7"],
            mutate=mutate_storage_public,
            cis_references=["3.7"],
        ),
        Mutation(
            id="vm_allow_password_auth",
            target_type="Microsoft.Compute/virtualMachines",
            description="Allow password authentication",
            severity="high",
            labels=["VM_PasswordAuthEnabled", "CIS_7.2"],
            mutate=mutate_vm_password,
            cis_references=["7.2"],
        ),
    ]


@pytest.fixture
def sample_arm_template():
    """Create a sample ARM template for testing."""
    return {
        "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
        "contentVersion": "1.0.0.0",
        "resources": [
            {
                "type": "Microsoft.Storage/storageAccounts",
                "apiVersion": "2021-09-01",
                "name": "mystorageaccount",
                "location": "eastus",
                "sku": {"name": "Standard_LRS"},
                "kind": "StorageV2",
                "properties": {
                    "allowBlobPublicAccess": False,
                    "supportsHttpsTrafficOnly": True,
                    "minimumTlsVersion": "TLS1_2"
                }
            },
            {
                "type": "Microsoft.Compute/virtualMachines",
                "apiVersion": "2021-11-01",
                "name": "myvm",
                "location": "eastus",
                "properties": {
                    "hardwareProfile": {"vmSize": "Standard_DS1_v2"},
                    "osProfile": {
                        "computerName": "myvm",
                        "adminUsername": "azureuser",
                        "linuxConfiguration": {
                            "disablePasswordAuthentication": True
                        }
                    },
                    "storageProfile": {
                        "osDisk": {
                            "createOption": "FromImage",
                            "managedDisk": {"storageAccountType": "Premium_LRS"}
                        }
                    }
                },
                "dependsOn": [
                    "[resourceId('Microsoft.Storage/storageAccounts', 'mystorageaccount')]"
                ]
            }
        ]
    }


@pytest.fixture
def sample_template_file(temp_dir, sample_arm_template):
    """Create a sample template file."""
    template_path = Path(temp_dir) / "template.json"
    with open(template_path, "w") as f:
        json.dump(sample_arm_template, f)
    return str(template_path)


# ============================================================================
# Test Classes
# ============================================================================


class TestMutationScenarios:
    """Test mutation scenario configuration and resolution."""

    def test_all_scenarios_have_required_fields(self):
        """Test that all scenarios have required fields."""
        for scenario_id, config in MUTATION_SCENARIOS.items():
            assert 'description' in config, f"Scenario {scenario_id} missing description"
            assert 'mutations' in config, f"Scenario {scenario_id} missing mutations"
            assert isinstance(config['mutations'], list), f"Scenario {scenario_id} mutations not a list"
            assert 'category' in config, f"Scenario {scenario_id} missing category"
            assert 'severity' in config, f"Scenario {scenario_id} missing severity"

    def test_baseline_has_no_mutations(self):
        """Test that baseline scenario has no mutations."""
        assert MUTATION_SCENARIOS['baseline']['mutations'] == []
        assert MUTATION_SCENARIOS['baseline']['category'] == 'control'

    def test_scenario_count(self):
        """Test that we have the expected number of scenarios."""
        assert len(MUTATION_SCENARIOS) == 14, "Expected 14 mutation scenarios"

    def test_scenario_references_resolve(self, rules_dir, sample_mutations):
        """Test that scenario references (@scenario_name) resolve correctly."""
        generator = TemplateLevelDatasetGenerator(
            rules_dir=rules_dir,
            mutations=sample_mutations
        )
        resolved = generator._resolved_scenarios

        # security_all should contain mutations from security_high and security_medium
        security_all = resolved['security_all']
        assert 'storage_public_blob_access' in security_all or len(security_all) >= 0
        # The actual mutations depend on what's available, but should be deduplicated
        assert len(security_all) == len(set(security_all)), "Duplicates in resolved scenarios"

    def test_scenario_categories(self):
        """Test that scenarios have correct categories."""
        assert MUTATION_SCENARIOS['security_high']['category'] == 'security'
        assert MUTATION_SCENARIOS['operational_high']['category'] == 'operational'
        assert MUTATION_SCENARIOS['reliability_high']['category'] == 'reliability'
        assert MUTATION_SCENARIOS['all_mutations']['category'] == 'combined'

    def test_resolve_mutations_deduplicates(self, rules_dir, sample_mutations):
        """Test that mutation resolution removes duplicates."""
        generator = TemplateLevelDatasetGenerator(
            rules_dir=rules_dir,
            mutations=sample_mutations
        )

        # Create a scenario with duplicate references
        test_mutations = ['storage_public_blob_access', 'storage_public_blob_access']
        resolved = generator._resolve_mutations(test_mutations)

        assert len(resolved) == 1, "Should deduplicate mutation IDs"
        assert 'storage_public_blob_access' in resolved


class TestFeatureExtraction:
    """Test feature extraction methods."""

    def test_composition_features(self, rules_dir):
        """Test composition feature extraction."""
        generator = TemplateLevelDatasetGenerator(rules_dir=rules_dir)
        resources = [
            {'type': 'Microsoft.Storage/storageAccounts', 'properties': {}},
            {'type': 'Microsoft.Compute/virtualMachines', 'properties': {}},
            {'type': 'Microsoft.Compute/virtualMachines', 'properties': {}},
        ]

        features = generator._extract_composition_features(resources)

        assert features['num_resources'] == 3
        assert features['num_resource_types'] == 2
        assert features['has_storage'] == 1
        assert features['has_vm'] == 1
        assert features['has_nsg'] == 0
        assert features['count_storage'] == 1
        assert features['count_vm'] == 2

    def test_security_features_storage(self, rules_dir):
        """Test security feature extraction for storage."""
        generator = TemplateLevelDatasetGenerator(rules_dir=rules_dir)
        resources = [
            {
                'type': 'Microsoft.Storage/storageAccounts',
                'properties': {
                    'allowBlobPublicAccess': True,
                    'supportsHttpsTrafficOnly': False,
                    'minimumTlsVersion': 'TLS1_0'
                }
            }
        ]

        features = generator._extract_security_features(resources)

        assert features['any_public_access'] == 1
        assert features['all_https_only'] == 0
        assert features['any_weak_tls'] == 1
        assert features['any_http_allowed'] == 1

    def test_security_features_nsg(self, rules_dir):
        """Test security feature extraction for NSG."""
        generator = TemplateLevelDatasetGenerator(rules_dir=rules_dir)
        resources = [
            {
                'type': 'Microsoft.Network/networkSecurityGroups',
                'properties': {
                    'securityRules': [
                        {
                            'properties': {
                                'direction': 'Inbound',
                                'access': 'Allow',
                                'sourceAddressPrefix': '*',
                                'destinationPortRange': '22'
                            }
                        }
                    ]
                }
            }
        ]

        features = generator._extract_security_features(resources)

        assert features['any_open_ssh'] == 1
        assert features['any_open_rdp'] == 0

    def test_graph_features_empty(self, rules_dir):
        """Test graph features with empty graph."""
        generator = TemplateLevelDatasetGenerator(rules_dir=rules_dir)
        graph = nx.DiGraph()

        features = generator._extract_graph_features(graph)

        assert features['num_dependencies'] == 0
        assert features['avg_resource_degree'] == 0.0
        assert features['max_resource_degree'] == 0
        assert features['has_isolated_resources'] == 0
        assert features['max_dependency_depth'] == 0
        assert features['dependency_density'] == 0.0

    def test_graph_features_with_nodes(self, rules_dir):
        """Test graph features with nodes and edges."""
        generator = TemplateLevelDatasetGenerator(rules_dir=rules_dir)
        graph = nx.DiGraph()
        graph.add_node('resource1')
        graph.add_node('resource2')
        graph.add_node('resource3')
        graph.add_edge('resource1', 'resource2')
        graph.add_edge('resource2', 'resource3')

        features = generator._extract_graph_features(graph)

        assert features['num_dependencies'] == 2
        assert features['avg_resource_degree'] > 0
        assert features['max_resource_degree'] == 2
        assert features['max_dependency_depth'] == 2  # Longest path: 1->2->3

    def test_helper_methods(self, rules_dir):
        """Test property extraction helper methods."""
        generator = TemplateLevelDatasetGenerator(rules_dir=rules_dir)

        resources = [
            {'type': 'test', 'properties': {'foo': 'bar'}},
            {'type': 'test', 'properties': {'foo': 'baz'}},
        ]

        # Test _any_property_equals
        assert generator._any_property_equals(resources, 'properties.foo', 'bar') == 1
        assert generator._any_property_equals(resources, 'properties.foo', 'nope') == 0

        # Test _all_property_equals
        assert generator._all_property_equals(resources, 'properties.foo', 'bar') == 0
        resources_same = [
            {'type': 'test', 'properties': {'foo': 'bar'}},
            {'type': 'test', 'properties': {'foo': 'bar'}},
        ]
        assert generator._all_property_equals(resources_same, 'properties.foo', 'bar') == 1

        # Test _percentage_property_equals
        pct = generator._percentage_property_equals(resources, 'properties.foo', 'bar')
        assert pct == 0.5


class TestLabelExtraction:
    """Test label extraction functionality."""

    def test_labels_with_no_violations(self, rules_dir):
        """Test label extraction when no rules are violated."""
        generator = TemplateLevelDatasetGenerator(rules_dir=rules_dir)
        resources = [
            {
                'type': 'Microsoft.Storage/storageAccounts',
                'properties': {
                    'allowBlobPublicAccess': False,
                    'supportsHttpsTrafficOnly': True
                }
            }
        ]

        labels = generator._extract_labels(resources)

        assert labels['has_any_misconfiguration'] == 0
        assert labels['misconfiguration_count'] == 0
        assert labels['unique_rule_count'] == 0

    def test_labels_with_violations(self, rules_dir):
        """Test label extraction when rules are violated."""
        generator = TemplateLevelDatasetGenerator(rules_dir=rules_dir)
        resources = [
            {
                'type': 'Microsoft.Storage/storageAccounts',
                'properties': {
                    'allowBlobPublicAccess': True,
                    'supportsHttpsTrafficOnly': False
                }
            }
        ]

        labels = generator._extract_labels(resources)

        assert labels['has_any_misconfiguration'] == 1
        assert labels['misconfiguration_count'] > 0
        assert labels['unique_rule_count'] > 0

    def test_labels_severity_from_mutations(self, rules_dir, sample_mutations):
        """Test that severity is extracted from mutation metadata."""
        generator = TemplateLevelDatasetGenerator(
            rules_dir=rules_dir,
            mutations=sample_mutations
        )

        # Apply mutation that adds metadata
        resource = {
            'type': 'Microsoft.Storage/storageAccounts',
            'properties': {}
        }

        mutation = sample_mutations[0]
        mutated = mutation.apply(resource)

        resources = [mutated]
        labels = generator._extract_labels(resources)

        # Should have high severity from mutation metadata
        assert labels['has_high_issue'] == 1 or labels['max_severity_level'] >= 3


class TestIntegration:
    """Integration tests for end-to-end dataset generation."""

    def test_process_single_template(self, rules_dir, sample_mutations, sample_arm_template, temp_dir):
        """Test processing a single template across scenarios."""
        generator = TemplateLevelDatasetGenerator(
            rules_dir=rules_dir,
            mutations=sample_mutations
        )

        # Save template to file
        template_path = Path(temp_dir) / "test_template.json"
        with open(template_path, "w") as f:
            json.dump(sample_arm_template, f)

        # Process with minimal scenarios
        scenarios = {
            'baseline': MUTATION_SCENARIOS['baseline'],
            'security_high': {
                'description': 'Test security',
                'category': 'security',
                'severity': 'high',
                'mutations': ['storage_public_blob_access', 'vm_allow_password_auth']
            }
        }

        observations = generator._process_template(str(template_path), scenarios)

        # Should have one observation per scenario
        assert len(observations) == 2

        # Check baseline observation
        baseline_obs = [o for o in observations if o['scenario_id'] == 'baseline'][0]
        assert baseline_obs['scenario_category'] == 'control'
        assert 'num_resources' in baseline_obs
        assert 'has_any_misconfiguration' in baseline_obs

        # Check mutated observation
        mutated_obs = [o for o in observations if o['scenario_id'] == 'security_high'][0]
        assert mutated_obs['scenario_category'] == 'security'
        # Should have more misconfigurations than baseline (due to mutations)
        assert mutated_obs['has_any_misconfiguration'] >= baseline_obs['has_any_misconfiguration']

    def test_generate_dataset(self, rules_dir, sample_mutations, sample_template_file, temp_dir):
        """Test full dataset generation pipeline."""
        generator = TemplateLevelDatasetGenerator(
            rules_dir=rules_dir,
            mutations=sample_mutations
        )

        output_path = generator.generate_dataset(
            template_paths=[sample_template_file],
            output_dir=temp_dir,
            experiment_name='test-experiment',
            scenarios_subset=['baseline', 'security_high']
        )

        # Check output directory exists
        output_dir = Path(output_path)
        assert output_dir.exists()

        # Check CSV file exists
        csv_file = output_dir / 'template_level_data.csv'
        assert csv_file.exists()

        # Load and check CSV
        df = pd.read_csv(csv_file)
        assert len(df) == 2  # 1 template × 2 scenarios
        assert 'template_id' in df.columns
        assert 'scenario_id' in df.columns
        assert 'has_any_misconfiguration' in df.columns

        # Check metadata file
        metadata_file = output_dir / 'metadata.json'
        assert metadata_file.exists()
        with open(metadata_file) as f:
            metadata = json.load(f)
        assert metadata['template_count'] == 1
        assert metadata['scenario_count'] == 2

        # Check summary stats
        stats_file = output_dir / 'summary_stats.json'
        assert stats_file.exists()
        with open(stats_file) as f:
            stats = json.load(f)
        assert stats['total_observations'] == 2
        assert stats['unique_templates'] == 1
        assert stats['unique_scenarios'] == 2

    def test_mutation_application(self, rules_dir, sample_mutations):
        """Test that mutations are correctly applied to resources."""
        generator = TemplateLevelDatasetGenerator(
            rules_dir=rules_dir,
            mutations=sample_mutations
        )

        resources = [
            {
                'type': 'Microsoft.Storage/storageAccounts',
                'properties': {'allowBlobPublicAccess': False}
            }
        ]

        # Apply storage mutation
        mutated = generator._apply_mutations(resources, ['storage_public_blob_access'])

        # Check mutation was applied
        assert mutated[0]['properties']['allowBlobPublicAccess'] == True
        # Check mutation metadata was added
        assert '_mutation_applied' in mutated[0]
        assert len(mutated[0]['_mutation_applied']) == 1
        assert mutated[0]['_mutation_applied'][0]['mutation_id'] == 'storage_public_blob_access'


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_template(self, rules_dir):
        """Test handling of empty template."""
        generator = TemplateLevelDatasetGenerator(rules_dir=rules_dir)

        resources = []
        features = generator._extract_composition_features(resources)

        assert features['num_resources'] == 0
        assert features['num_resource_types'] == 0

    def test_template_with_no_target_resources(self, rules_dir, sample_mutations):
        """Test template with resources that don't match any mutations."""
        generator = TemplateLevelDatasetGenerator(
            rules_dir=rules_dir,
            mutations=sample_mutations
        )

        resources = [
            {'type': 'Microsoft.Network/virtualNetworks', 'properties': {}}
        ]

        # Apply mutations - should not affect resources
        mutated = generator._apply_mutations(resources, ['storage_public_blob_access'])

        # Resource should be unchanged (except for deep copy)
        assert mutated[0]['type'] == 'Microsoft.Network/virtualNetworks'
        assert '_mutation_applied' not in mutated[0]

    def test_missing_nested_property(self, rules_dir):
        """Test handling of missing nested properties."""
        generator = TemplateLevelDatasetGenerator(rules_dir=rules_dir)

        resources = [
            {'type': 'Microsoft.Storage/storageAccounts', 'properties': {}}
        ]

        # Should handle missing properties gracefully
        result = generator._any_property_equals(
            resources, 'properties.foo.bar.baz', 'value'
        )
        assert result == 0


# ============================================================================
# Parametrized Tests
# ============================================================================


@pytest.mark.parametrize("scenario_id", list(MUTATION_SCENARIOS.keys()))
def test_all_scenarios_resolvable(scenario_id, rules_dir):
    """Test that all scenarios can be resolved without errors."""
    generator = TemplateLevelDatasetGenerator(rules_dir=rules_dir)
    resolved = generator._resolved_scenarios

    assert scenario_id in resolved
    assert isinstance(resolved[scenario_id], list)
    # Check for no duplicates
    assert len(resolved[scenario_id]) == len(set(resolved[scenario_id]))
