"""
Template-Level Dataset Generator for CRUX

Generates aggregated template-level observations with multiple mutation scenarios.
Each observation represents a single template in a specific mutation state.
"""

import copy
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import networkx as nx
import pandas as pd
from tqdm import tqdm

# Import existing CRUX components
from crux.templates.compiler import compile_bicep_to_arm
from crux.templates.extractor import extract_resources
from crux.templates.graph import build_dependency_graph
from crux.mutations.base import Mutation
from crux.rules.evaluator import RuleEvaluator


# Mutation scenarios configuration
MUTATION_SCENARIOS = {
    "baseline": {
        "description": "No mutations applied - template as authored",
        "category": "control",
        "severity": "none",
        "mutations": []
    },

    # Single-category scenarios (high severity)
    "security_high": {
        "description": "High-severity security mutations only",
        "category": "security",
        "severity": "high",
        "mutations": [
            "storage_public_blob_access",
            "storage_http_allowed",
            "nsg_allow_all_inbound",
            "nsg_open_ssh",
            "nsg_open_rdp",
            "vm_no_encryption",
            "vm_allow_password_auth"
        ]
    },
    "security_medium": {
        "description": "Medium-severity security mutations only",
        "category": "security",
        "severity": "medium",
        "mutations": [
            "storage_weak_tls",
            "storage_allow_shared_key",
            "vm_no_secure_boot",
            "vm_no_vtpm",
            "storage_no_infrastructure_encryption"
        ]
    },
    "operational_high": {
        "description": "High-severity operational mutations only",
        "category": "operational",
        "severity": "high",
        "mutations": [
            "vm_no_boot_diagnostics"
        ]
    },
    "operational_medium": {
        "description": "Medium-severity operational mutations only",
        "category": "operational",
        "severity": "medium",
        "mutations": [
            "storage_no_blob_versioning",
            "storage_no_soft_delete",
            "vm_no_identity"
        ]
    },
    "reliability_high": {
        "description": "High-severity reliability mutations only",
        "category": "reliability",
        "severity": "high",
        "mutations": [
            "vm_unmanaged_disk"
        ]
    },
    "reliability_medium": {
        "description": "Medium-severity reliability mutations only",
        "category": "reliability",
        "severity": "medium",
        "mutations": [
            "vnet_no_ddos",
            "vnet_no_service_endpoints",
            "vnet_broad_address_space"
        ]
    },

    # Combined category scenarios
    "security_all": {
        "description": "All security mutations (high + medium)",
        "category": "security",
        "severity": "all",
        "mutations": ["@security_high", "@security_medium"]
    },
    "operational_all": {
        "description": "All operational mutations (high + medium)",
        "category": "operational",
        "severity": "all",
        "mutations": ["@operational_high", "@operational_medium"]
    },
    "reliability_all": {
        "description": "All reliability mutations (high + medium)",
        "category": "reliability",
        "severity": "all",
        "mutations": ["@reliability_high", "@reliability_medium"]
    },

    # Multi-category combinations
    "security_operational": {
        "description": "Security + Operational mutations",
        "category": "combined",
        "severity": "all",
        "mutations": ["@security_all", "@operational_all"]
    },
    "security_reliability": {
        "description": "Security + Reliability mutations",
        "category": "combined",
        "severity": "all",
        "mutations": ["@security_all", "@reliability_all"]
    },
    "operational_reliability": {
        "description": "Operational + Reliability mutations",
        "category": "combined",
        "severity": "all",
        "mutations": ["@operational_all", "@reliability_all"]
    },
    "all_mutations": {
        "description": "All mutations applied - worst case scenario",
        "category": "combined",
        "severity": "all",
        "mutations": ["@security_all", "@operational_all", "@reliability_all"]
    }
}


class TemplateLevelDatasetGenerator:
    """
    Generates template-level dataset with multiple mutation scenarios.

    Each observation represents a single template in a specific mutation state,
    with aggregated features across all resources in that template.
    """

    def __init__(
        self,
        rules_dir: str,
        mutations: Optional[List[Mutation]] = None,
        scenarios: Optional[Dict[str, Dict]] = None
    ):
        """
        Initialize the generator.

        Args:
            rules_dir: Path to directory containing rule definitions
            mutations: List of available mutations (defaults to all)
            scenarios: Mutation scenarios (defaults to MUTATION_SCENARIOS)
        """
        self.rules_evaluator = RuleEvaluator(rules_dir)

        # Import and setup mutations
        if mutations is None:
            from crux.mutations import get_all_mutations
            mutations = get_all_mutations()

        self.mutations = {m.id: m for m in mutations}
        self.scenarios = scenarios or MUTATION_SCENARIOS
        self._resolved_scenarios = self._resolve_all_scenarios()

    def _resolve_all_scenarios(self) -> Dict[str, List[str]]:
        """Resolve scenario references (@scenario_name) to flat mutation lists."""
        resolved = {}
        for scenario_id, config in self.scenarios.items():
            resolved[scenario_id] = self._resolve_mutations(config['mutations'])
        return resolved

    def _resolve_mutations(self, mutation_list: List[str]) -> List[str]:
        """Recursively resolve mutation references."""
        resolved = []
        for item in mutation_list:
            if item.startswith('@'):
                # Reference to another scenario
                ref_id = item[1:]
                if ref_id in self.scenarios:
                    resolved.extend(
                        self._resolve_mutations(self.scenarios[ref_id]['mutations'])
                    )
            else:
                resolved.append(item)
        return list(set(resolved))  # Deduplicate

    def generate_dataset(
        self,
        template_paths: List[str],
        output_dir: str,
        experiment_name: str = "template-level",
        limit: Optional[int] = None,
        scenarios_subset: Optional[List[str]] = None
    ) -> str:
        """
        Generate template-level dataset.

        Args:
            template_paths: List of paths to Bicep/ARM templates
            output_dir: Directory for output files
            experiment_name: Name for this experiment run
            limit: Optional limit on number of templates to process
            scenarios_subset: Optional list of scenario IDs to use

        Returns:
            Path to output directory containing dataset and metadata
        """
        # Apply limits
        if limit:
            template_paths = template_paths[:limit]

        # Filter scenarios if subset specified
        active_scenarios = (
            {k: v for k, v in self.scenarios.items() if k in scenarios_subset}
            if scenarios_subset else self.scenarios
        )

        all_observations = []
        failed_templates = []

        for template_path in tqdm(template_paths, desc="Processing templates"):
            try:
                observations = self._process_template(template_path, active_scenarios)
                all_observations.extend(observations)
            except Exception as e:
                failed_templates.append({
                    'path': template_path,
                    'error': str(e)
                })
                continue

        # Create output directory
        output_path = Path(output_dir) / experiment_name
        output_path.mkdir(parents=True, exist_ok=True)

        # Save dataset
        df = pd.DataFrame(all_observations)
        df.to_csv(output_path / 'template_level_data.csv', index=False)

        # Save metadata
        metadata = self._generate_metadata(
            experiment_name=experiment_name,
            template_count=len(template_paths),
            scenario_count=len(active_scenarios),
            observation_count=len(df),
            failed_count=len(failed_templates),
            scenarios=active_scenarios,
            columns=list(df.columns)
        )
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save failures log
        if failed_templates:
            with open(output_path / 'failed_templates.json', 'w') as f:
                json.dump(failed_templates, f, indent=2)

        # Generate summary statistics
        self._save_summary_stats(df, output_path)

        return str(output_path)

    def _process_template(
        self,
        template_path: str,
        scenarios: Dict[str, Dict]
    ) -> List[Dict[str, Any]]:
        """Process a single template across all scenarios."""
        observations = []

        # Compile and extract baseline
        arm_template = self._compile_template(template_path)
        baseline_resources = extract_resources(arm_template)
        graph = build_dependency_graph(arm_template)

        template_id = self._get_template_id(template_path)
        template_name = Path(template_path).stem

        for scenario_id, scenario_config in scenarios.items():
            # Apply mutations for this scenario
            mutation_ids = self._resolved_scenarios[scenario_id]
            mutated_resources = self._apply_mutations(baseline_resources, mutation_ids)

            # Extract features
            features = self._extract_features(mutated_resources, graph)

            # Extract labels
            labels = self._extract_labels(mutated_resources)

            # Build observation
            observation = {
                'template_id': template_id,
                'template_name': template_name,
                'scenario_id': scenario_id,
                'scenario_category': scenario_config.get('category', 'unknown'),
                'is_mutated': 0 if scenario_id == 'baseline' else 1,
                **features,
                **labels
            }
            observations.append(observation)

        return observations

    def _compile_template(self, template_path: str) -> Dict:
        """Compile Bicep to ARM or load ARM directly."""
        if template_path.endswith('.bicep'):
            return compile_bicep_to_arm(template_path)
        else:
            with open(template_path) as f:
                return json.load(f)

    def _get_template_id(self, template_path: str) -> str:
        """Generate stable template ID from path."""
        path = Path(template_path)
        return str(path.parent.name) + '/' + path.stem

    def _apply_mutations(
        self,
        resources: List[Dict],
        mutation_ids: List[str]
    ) -> List[Dict]:
        """Apply specified mutations to resources."""
        mutated = copy.deepcopy(resources)

        for mutation_id in mutation_ids:
            if mutation_id not in self.mutations:
                continue
            mutation = self.mutations[mutation_id]
            for i, resource in enumerate(mutated):
                if mutation.applies_to(resource):
                    result = mutation.apply(resource)
                    if result is not None:
                        mutated[i] = result

        return mutated

    def _extract_features(
        self,
        resources: List[Dict],
        graph: nx.DiGraph
    ) -> Dict[str, Any]:
        """Extract all template-level features."""
        features = {}

        # Resource composition
        features.update(self._extract_composition_features(resources))

        # Security features
        features.update(self._extract_security_features(resources))

        # Operational features
        features.update(self._extract_operational_features(resources))

        # Reliability features
        features.update(self._extract_reliability_features(resources))

        # Graph features
        features.update(self._extract_graph_features(graph))

        return features

    def _extract_composition_features(self, resources: List[Dict]) -> Dict[str, Any]:
        """Extract resource composition features."""
        features = {}

        features['num_resources'] = len(resources)

        resource_types = [r.get('type', '') for r in resources]
        features['num_resource_types'] = len(set(resource_types))

        # Type mappings
        type_checks = {
            'storage': 'Microsoft.Storage/storageAccounts',
            'vm': 'Microsoft.Compute/virtualMachines',
            'nsg': 'Microsoft.Network/networkSecurityGroups',
            'vnet': 'Microsoft.Network/virtualNetworks',
            'keyvault': 'Microsoft.KeyVault/vaults',
            'sql': 'Microsoft.Sql/servers',
            'webapp': 'Microsoft.Web/sites'
        }

        for short_name, full_type in type_checks.items():
            matching = [r for r in resources if r.get('type') == full_type]
            features[f'has_{short_name}'] = 1 if matching else 0
            features[f'count_{short_name}'] = len(matching)

        return features

    def _extract_security_features(self, resources: List[Dict]) -> Dict[str, Any]:
        """Extract aggregated security configuration features."""
        features = {}

        storage = self._filter_by_type(resources, 'Microsoft.Storage/storageAccounts')
        vms = self._filter_by_type(resources, 'Microsoft.Compute/virtualMachines')
        nsgs = self._filter_by_type(resources, 'Microsoft.Network/networkSecurityGroups')
        vnets = self._filter_by_type(resources, 'Microsoft.Network/virtualNetworks')

        # Storage security
        features['any_public_access'] = self._any_property_equals(
            storage, 'properties.allowBlobPublicAccess', True
        )
        features['all_https_only'] = self._all_property_equals(
            storage, 'properties.supportsHttpsTrafficOnly', True
        )
        features['any_weak_tls'] = self._any_property_in(
            storage, 'properties.minimumTlsVersion', ['TLS1_0', 'TLS1_1']
        )
        features['any_http_allowed'] = self._any_property_equals(
            storage, 'properties.supportsHttpsTrafficOnly', False
        )

        # VM security
        features['pct_secure_boot'] = self._percentage_property_equals(
            vms, 'properties.securityProfile.uefiSettings.secureBootEnabled', True
        )
        features['pct_vtpm_enabled'] = self._percentage_property_equals(
            vms, 'properties.securityProfile.uefiSettings.vTpmEnabled', True
        )
        features['any_no_encryption'] = self._any_property_missing_or_false(
            vms, 'properties.securityProfile.encryptionAtHost'
        )
        features['any_password_auth'] = self._any_property_equals(
            vms, 'properties.osProfile.linuxConfiguration.disablePasswordAuthentication', False
        )

        # NSG security
        features['any_open_inbound'] = self._any_nsg_allows_all_inbound(nsgs)
        features['any_open_ssh'] = self._any_nsg_open_port(nsgs, 22)
        features['any_open_rdp'] = self._any_nsg_open_port(nsgs, 3389)

        # VNet security
        features['any_ddos_disabled'] = self._any_property_missing_or_false(
            vnets, 'properties.enableDdosProtection'
        )

        # Aggregates
        features['all_encryption_enabled'] = 1 if (
            features.get('any_no_encryption', 0) == 0 and
            features.get('any_http_allowed', 0) == 0
        ) else 0

        return features

    def _extract_operational_features(self, resources: List[Dict]) -> Dict[str, Any]:
        """Extract aggregated operational configuration features."""
        features = {}

        storage = self._filter_by_type(resources, 'Microsoft.Storage/storageAccounts')
        vms = self._filter_by_type(resources, 'Microsoft.Compute/virtualMachines')

        # VM operational
        features['any_diagnostics_disabled'] = self._any_property_missing_or_false(
            vms, 'properties.diagnosticsProfile.bootDiagnostics.enabled'
        )
        features['any_no_patching'] = self._any_property_missing(
            vms, 'properties.osProfile.windowsConfiguration.patchSettings.patchMode'
        )
        features['all_auto_patch'] = 1 - features['any_no_patching'] if vms else 1
        features['pct_managed_identity'] = self._percentage_property_exists(
            vms, 'identity.type'
        )
        features['any_no_identity'] = self._any_property_missing(vms, 'identity.type')

        # Storage operational
        features['any_versioning_disabled'] = self._any_property_missing_or_false(
            storage, 'properties.isVersioningEnabled'
        )
        features['any_soft_delete_disabled'] = self._any_property_missing_or_false(
            storage, 'properties.deleteRetentionPolicy.enabled'
        )

        return features

    def _extract_reliability_features(self, resources: List[Dict]) -> Dict[str, Any]:
        """Extract aggregated reliability configuration features."""
        features = {}

        vms = self._filter_by_type(resources, 'Microsoft.Compute/virtualMachines')
        vnets = self._filter_by_type(resources, 'Microsoft.Network/virtualNetworks')

        # VM reliability
        features['any_no_availability_set'] = self._any_property_missing(
            vms, 'properties.availabilitySet'
        )
        features['all_managed_disks'] = self._all_property_exists(
            vms, 'properties.storageProfile.osDisk.managedDisk'
        )
        features['any_unmanaged_disk'] = 1 - features['all_managed_disks'] if vms else 0

        # VNet reliability
        features['any_no_service_endpoints'] = self._any_property_missing(
            vnets, 'properties.subnets'
        ) if vnets else 0
        features['any_broad_address_space'] = self._any_broad_cidr(vnets)

        return features

    def _extract_graph_features(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Extract dependency graph features."""
        features = {}

        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()

        features['num_dependencies'] = num_edges

        if num_nodes > 0:
            degrees = dict(graph.degree())
            features['avg_resource_degree'] = sum(degrees.values()) / num_nodes
            features['max_resource_degree'] = max(degrees.values()) if degrees else 0
            features['has_isolated_resources'] = 1 if any(d == 0 for d in degrees.values()) else 0
        else:
            features['avg_resource_degree'] = 0.0
            features['max_resource_degree'] = 0
            features['has_isolated_resources'] = 0

        # Dependency depth (longest path)
        if num_nodes > 0 and nx.is_directed_acyclic_graph(graph):
            features['max_dependency_depth'] = nx.dag_longest_path_length(graph)
        else:
            features['max_dependency_depth'] = 0

        # Density
        max_edges = num_nodes * (num_nodes - 1)
        features['dependency_density'] = num_edges / max_edges if max_edges > 0 else 0.0

        return features

    def _extract_labels(self, resources: List[Dict]) -> Dict[str, Any]:
        """Extract template-level labels (dependent variables)."""
        labels = {}

        # Evaluate all resources and collect labels
        all_labels_flat: List[str] = []
        for resource in resources:
            resource_labels = self.rules_evaluator.evaluate(resource)
            all_labels_flat.extend(resource_labels)

        # Primary DV
        labels['has_any_misconfiguration'] = 1 if all_labels_flat else 0

        # Counts
        labels['misconfiguration_count'] = len(all_labels_flat)
        labels['unique_rule_count'] = len(set(all_labels_flat))

        # Category counts (inferred from label names)
        labels['security_issue_count'] = sum(
            1 for lbl in all_labels_flat
            if any(x in lbl for x in ['Storage_', 'VM_', 'NSG_', 'VNet_', 'KeyVault_'])
        )
        labels['operational_issue_count'] = sum(
            1 for lbl in all_labels_flat
            if any(x in lbl for x in ['NoVersioning', 'NoSoftDelete', 'NoManagedIdentity', 'NoBootDiagnostics', 'NoAutoPatch'])
        )
        labels['reliability_issue_count'] = sum(
            1 for lbl in all_labels_flat
            if any(x in lbl for x in ['UnmanagedDisk', 'NoAvailability', 'NoDDoS', 'NoServiceEndpoints', 'BroadAddressSpace'])
        )

        # Severity (inferred from mutation metadata on resources)
        has_critical = 0
        has_high = 0
        max_severity = 0

        for resource in resources:
            if '_mutation_applied' in resource:
                for mutation_meta in resource['_mutation_applied']:
                    sev = mutation_meta.get('severity', 'low')
                    if sev == 'critical':
                        has_critical = 1
                        max_severity = max(max_severity, 4)
                    elif sev == 'high':
                        has_high = 1
                        max_severity = max(max_severity, 3)
                    elif sev == 'medium':
                        max_severity = max(max_severity, 2)
                    elif sev == 'low':
                        max_severity = max(max_severity, 1)

        labels['has_critical_issue'] = has_critical
        labels['has_high_issue'] = has_high
        labels['max_severity_level'] = max_severity

        # Compliance percentage (simplified)
        labels['cis_compliance_pct'] = self._calculate_compliance_pct(resources, all_labels_flat)

        return labels

    # --- Helper methods for property extraction ---

    def _filter_by_type(self, resources: List[Dict], resource_type: str) -> List[Dict]:
        """Filter resources by type."""
        return [r for r in resources if r.get('type') == resource_type]

    def _get_nested_property(self, obj: Dict, path: str) -> Any:
        """Get nested property using dot notation."""
        parts = path.split('.')
        current = obj
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
            if current is None:
                return None
        return current

    def _any_property_equals(self, resources: List[Dict], path: str, value: Any) -> int:
        """Check if ANY resource has property equal to value."""
        if not resources:
            return 0
        return 1 if any(
            self._get_nested_property(r, path) == value for r in resources
        ) else 0

    def _all_property_equals(self, resources: List[Dict], path: str, value: Any) -> int:
        """Check if ALL resources have property equal to value."""
        if not resources:
            return 1  # Vacuously true
        return 1 if all(
            self._get_nested_property(r, path) == value for r in resources
        ) else 0

    def _any_property_in(self, resources: List[Dict], path: str, values: List[Any]) -> int:
        """Check if ANY resource has property in list of values."""
        if not resources:
            return 0
        return 1 if any(
            self._get_nested_property(r, path) in values for r in resources
        ) else 0

    def _any_property_missing(self, resources: List[Dict], path: str) -> int:
        """Check if ANY resource is missing property."""
        if not resources:
            return 0
        return 1 if any(
            self._get_nested_property(r, path) is None for r in resources
        ) else 0

    def _any_property_missing_or_false(self, resources: List[Dict], path: str) -> int:
        """Check if ANY resource has property missing or False."""
        if not resources:
            return 0
        return 1 if any(
            self._get_nested_property(r, path) in [None, False] for r in resources
        ) else 0

    def _all_property_exists(self, resources: List[Dict], path: str) -> int:
        """Check if ALL resources have property defined."""
        if not resources:
            return 1
        return 1 if all(
            self._get_nested_property(r, path) is not None for r in resources
        ) else 0

    def _percentage_property_equals(
        self, resources: List[Dict], path: str, value: Any
    ) -> float:
        """Calculate percentage of resources with property equal to value."""
        if not resources:
            return 0.0
        matches = sum(1 for r in resources if self._get_nested_property(r, path) == value)
        return matches / len(resources)

    def _percentage_property_exists(self, resources: List[Dict], path: str) -> float:
        """Calculate percentage of resources with property defined."""
        if not resources:
            return 0.0
        matches = sum(1 for r in resources if self._get_nested_property(r, path) is not None)
        return matches / len(resources)

    def _any_nsg_allows_all_inbound(self, nsgs: List[Dict]) -> int:
        """Check if any NSG has allow-all inbound rule."""
        for nsg in nsgs:
            rules = self._get_nested_property(nsg, 'properties.securityRules') or []
            for rule in rules:
                props = rule.get('properties', {})
                if (props.get('direction') == 'Inbound' and
                    props.get('access') == 'Allow' and
                    props.get('sourceAddressPrefix') in ['*', '0.0.0.0/0', 'Internet']):
                    return 1
        return 0

    def _any_nsg_open_port(self, nsgs: List[Dict], port: int) -> int:
        """Check if any NSG allows access to specific port from internet."""
        for nsg in nsgs:
            rules = self._get_nested_property(nsg, 'properties.securityRules') or []
            for rule in rules:
                props = rule.get('properties', {})
                if (props.get('direction') == 'Inbound' and
                    props.get('access') == 'Allow' and
                    props.get('sourceAddressPrefix') in ['*', '0.0.0.0/0', 'Internet'] and
                    self._port_in_range(props.get('destinationPortRange'), port)):
                    return 1
        return 0

    def _port_in_range(self, port_range: str, port: int) -> bool:
        """Check if port is in port range specification."""
        if port_range is None:
            return False
        if port_range == '*':
            return True
        if '-' in str(port_range):
            start, end = port_range.split('-')
            return int(start) <= port <= int(end)
        return str(port_range) == str(port)

    def _any_broad_cidr(self, vnets: List[Dict]) -> int:
        """Check if any VNet has overly broad address space."""
        for vnet in vnets:
            address_spaces = self._get_nested_property(vnet, 'properties.addressSpace.addressPrefixes') or []
            for cidr in address_spaces:
                if '/' in cidr:
                    prefix_len = int(cidr.split('/')[1])
                    if prefix_len < 16:  # Broader than /16 is suspicious
                        return 1
        return 0

    def _calculate_compliance_pct(
        self, resources: List[Dict], violations: List[str]
    ) -> float:
        """Calculate CIS compliance percentage (simplified)."""
        # Count total applicable rules (simplified approach)
        total_applicable = len(self.rules_evaluator.rules) * len(resources)
        if total_applicable == 0:
            return 100.0
        violations_count = len(set(violations))
        passed = max(0, total_applicable - violations_count)
        return (passed / total_applicable) * 100 if total_applicable > 0 else 100.0

    def _generate_metadata(self, **kwargs) -> Dict[str, Any]:
        """Generate dataset metadata."""
        return {
            'generated_at': datetime.now().isoformat(),
            'generator_version': '1.0.0',
            **kwargs
        }

    def _save_summary_stats(self, df: pd.DataFrame, output_path: Path):
        """Save summary statistics for the dataset."""
        stats = {
            'total_observations': len(df),
            'unique_templates': df['template_id'].nunique(),
            'unique_scenarios': df['scenario_id'].nunique(),
            'dv_positive_rate': float(df['has_any_misconfiguration'].mean()),
            'dv_by_scenario': {k: float(v) for k, v in df.groupby('scenario_id')['has_any_misconfiguration'].mean().to_dict().items()},
            'feature_means': {k: float(v) for k, v in df.select_dtypes(include=['number']).mean().to_dict().items()},
            'feature_stds': {k: float(v) for k, v in df.select_dtypes(include=['number']).std().to_dict().items()}
        }

        with open(output_path / 'summary_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
