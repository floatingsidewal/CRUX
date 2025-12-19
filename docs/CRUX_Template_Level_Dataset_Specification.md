# CRUX Template-Level Dataset Generator Specification

## Overview

This specification defines a new dataset generation mode for CRUX that produces **template-level observations** instead of resource-level observations. The goal is to generate a dataset suitable for logistic regression analysis that:

1. Exceeds 7,000 observations (academic requirement)
2. Avoids the tautological relationship between individual configuration properties and misconfiguration labels
3. Enables a production scoring system that can assess customer ARM templates

## Problem Statement

### Current State (Resource-Level)
- CRUX generates ~9,000 resource-level observations
- Each row = one Azure resource
- **Problem:** The relationship between IVs and DV is deterministic (rule-based), not statistical
- Example: `IF allowBlobPublicAccess == True → has_misconfiguration = 1` (always)

### Target State (Template-Level)
- Generate 8,000-14,000 template-level observations
- Each row = one deployment template in a specific mutation scenario
- **Solution:** At template level, relationships become probabilistic
- Example: `IF any_public_access == True → has_any_misconfiguration = ?` (depends on other factors)

## Mathematical Justification

```
Current templates: ~1,008
Mutation scenarios: 14
Target observations: 1,008 × 14 = 14,112 ✓ (exceeds 7,000)
```

---

## Implementation Requirements

### 1. New File: `crux/dataset/template_level_generator.py`

This is the main implementation file.

#### 1.1 Mutation Scenarios Configuration

```python
"""
Mutation scenarios define which mutations to apply to each template.
Each template × scenario combination produces one observation.
"""

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
            "vm_no_boot_diagnostics",
            "vm_no_auto_patching"
        ]
    },
    "operational_medium": {
        "description": "Medium-severity operational mutations only",
        "category": "operational",
        "severity": "medium",
        "mutations": [
            "storage_no_versioning",
            "storage_no_soft_delete",
            "vm_no_managed_identity"
        ]
    },
    "reliability_high": {
        "description": "High-severity reliability mutations only",
        "category": "reliability",
        "severity": "high",
        "mutations": [
            "vm_no_availability_set",
            "vm_unmanaged_disk"
        ]
    },
    "reliability_medium": {
        "description": "Medium-severity reliability mutations only",
        "category": "reliability",
        "severity": "medium",
        "mutations": [
            "vnet_no_ddos_protection",
            "vnet_no_service_endpoints",
            "vnet_broad_address_space"
        ]
    },
    
    # Combined category scenarios
    "security_all": {
        "description": "All security mutations (high + medium)",
        "category": "security",
        "severity": "all",
        "mutations": ["@security_high", "@security_medium"]  # @ = reference other scenario
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
```

#### 1.2 Feature Schema

The template-level dataset must include these features:

##### Metadata Columns (not used in regression)
| Column | Type | Description |
|--------|------|-------------|
| `template_id` | string | Unique template identifier (path-based) |
| `template_name` | string | Human-readable template name |
| `scenario_id` | string | Mutation scenario applied |
| `scenario_category` | string | Category of scenario (control/security/operational/reliability/combined) |

##### Resource Composition Features (Independent Variables)
| Column | Type | Description |
|--------|------|-------------|
| `num_resources` | int | Total count of resources in template |
| `num_resource_types` | int | Count of distinct resource types |
| `has_storage` | binary (0/1) | Template contains storage account |
| `has_vm` | binary (0/1) | Template contains virtual machine |
| `has_nsg` | binary (0/1) | Template contains network security group |
| `has_vnet` | binary (0/1) | Template contains virtual network |
| `has_keyvault` | binary (0/1) | Template contains key vault |
| `has_sql` | binary (0/1) | Template contains SQL database |
| `has_webapp` | binary (0/1) | Template contains web app |
| `count_storage` | int | Number of storage accounts |
| `count_vm` | int | Number of virtual machines |
| `count_nsg` | int | Number of NSGs |
| `count_vnet` | int | Number of VNets |

##### Security Configuration Features (Independent Variables - Aggregated)
| Column | Type | Description |
|--------|------|-------------|
| `any_public_access` | binary | ANY storage has allowBlobPublicAccess=true |
| `all_https_only` | binary | ALL storage accounts enforce HTTPS only |
| `any_weak_tls` | binary | ANY storage has TLS < 1.2 |
| `any_http_allowed` | binary | ANY storage allows HTTP |
| `any_open_inbound` | binary | ANY NSG has allow-all inbound rule |
| `any_open_ssh` | binary | ANY NSG allows SSH from internet |
| `any_open_rdp` | binary | ANY NSG allows RDP from internet |
| `all_encryption_enabled` | binary | ALL applicable resources have encryption |
| `any_no_encryption` | binary | ANY VM lacks disk encryption |
| `pct_secure_boot` | float | Percentage of VMs with secure boot enabled |
| `pct_vtpm_enabled` | float | Percentage of VMs with vTPM enabled |
| `any_ddos_disabled` | binary | ANY VNet lacks DDoS protection |
| `any_password_auth` | binary | ANY VM allows password authentication |

##### Operational Configuration Features (Independent Variables - Aggregated)
| Column | Type | Description |
|--------|------|-------------|
| `any_diagnostics_disabled` | binary | ANY VM lacks boot diagnostics |
| `all_auto_patch` | binary | ALL VMs have automatic patching |
| `any_no_patching` | binary | ANY VM lacks auto-patching |
| `any_versioning_disabled` | binary | ANY storage lacks blob versioning |
| `any_soft_delete_disabled` | binary | ANY storage lacks soft delete |
| `pct_managed_identity` | float | Percentage of VMs with managed identity |
| `any_no_identity` | binary | ANY VM lacks managed identity |

##### Reliability Configuration Features (Independent Variables - Aggregated)
| Column | Type | Description |
|--------|------|-------------|
| `any_no_availability_set` | binary | ANY VM lacks availability set |
| `all_managed_disks` | binary | ALL VMs use managed disks |
| `any_unmanaged_disk` | binary | ANY VM uses unmanaged disk |
| `any_no_service_endpoints` | binary | ANY VNet lacks service endpoints |
| `any_broad_address_space` | binary | ANY VNet has overly broad CIDR (e.g., /8) |

##### Graph/Dependency Features (Independent Variables)
| Column | Type | Description |
|--------|------|-------------|
| `num_dependencies` | int | Total dependency edges in template |
| `avg_resource_degree` | float | Average connections per resource |
| `max_resource_degree` | int | Maximum connections for any resource |
| `has_isolated_resources` | binary | Any resource with zero dependencies |
| `max_dependency_depth` | int | Longest dependency chain |
| `dependency_density` | float | edges / (nodes * (nodes-1)) |

##### Dependent Variables (Labels)
| Column | Type | Description | Primary DV? |
|--------|------|-------------|-------------|
| `has_any_misconfiguration` | binary | **PRIMARY DV** - Any issue detected | ✓ |
| `misconfiguration_count` | int | Total issues detected | |
| `unique_rule_count` | int | Distinct rules triggered | |
| `security_issue_count` | int | Security-category issues | |
| `operational_issue_count` | int | Operational-category issues | |
| `reliability_issue_count` | int | Reliability-category issues | |
| `has_critical_issue` | binary | Any critical-severity issue | |
| `has_high_issue` | binary | Any high-severity issue | |
| `max_severity_level` | int | 0=none, 1=low, 2=medium, 3=high, 4=critical | |
| `cis_compliance_pct` | float | % of applicable CIS checks passed | |

#### 1.3 Core Classes and Functions

```python
"""
crux/dataset/template_level_generator.py

Template-Level Dataset Generator for CRUX
Generates aggregated template-level observations with multiple mutation scenarios.
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

# Import existing CRUX components (adjust paths as needed)
from crux.compiler import compile_bicep_to_arm
from crux.extractor import extract_resources_from_arm
from crux.graph import build_dependency_graph
from crux.mutations import Mutation, get_all_mutations
from crux.rules import RuleEvaluator, RuleResult


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
        self.mutations = {m.id: m for m in (mutations or get_all_mutations())}
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
        baseline_resources = extract_resources_from_arm(arm_template)
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
        # Use relative path components as ID
        path = Path(template_path)
        # Adjust this logic based on your template directory structure
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
            for resource in mutated:
                if mutation.applies_to(resource):
                    mutation.apply(resource)
        
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
        features['all_auto_patch'] = 1 - features['any_no_patching']
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
        features['any_unmanaged_disk'] = 1 - features['all_managed_disks']
        
        # VNet reliability
        features['any_no_service_endpoints'] = self._any_property_missing(
            vnets, 'properties.subnets[*].serviceEndpoints'
        )
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
            features['avg_resource_degree'] = 0
            features['max_resource_degree'] = 0
            features['has_isolated_resources'] = 0
        
        # Dependency depth (longest path)
        if num_nodes > 0 and nx.is_directed_acyclic_graph(graph):
            features['max_dependency_depth'] = nx.dag_longest_path_length(graph)
        else:
            features['max_dependency_depth'] = 0
        
        # Density
        max_edges = num_nodes * (num_nodes - 1)
        features['dependency_density'] = num_edges / max_edges if max_edges > 0 else 0
        
        return features
    
    def _extract_labels(self, resources: List[Dict]) -> Dict[str, Any]:
        """Extract template-level labels (dependent variables)."""
        labels = {}
        
        # Evaluate all resources
        all_results: List[RuleResult] = []
        for resource in resources:
            results = self.rules_evaluator.evaluate(resource)
            all_results.extend(results)
        
        # Primary DV
        labels['has_any_misconfiguration'] = 1 if all_results else 0
        
        # Counts
        labels['misconfiguration_count'] = len(all_results)
        labels['unique_rule_count'] = len(set(r.rule_id for r in all_results))
        
        # Category counts
        labels['security_issue_count'] = sum(1 for r in all_results if r.category == 'security')
        labels['operational_issue_count'] = sum(1 for r in all_results if r.category == 'operational')
        labels['reliability_issue_count'] = sum(1 for r in all_results if r.category == 'reliability')
        
        # Severity
        severity_map = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        severities = [severity_map.get(r.severity, 0) for r in all_results]
        
        labels['has_critical_issue'] = 1 if 4 in severities else 0
        labels['has_high_issue'] = 1 if 3 in severities or 4 in severities else 0
        labels['max_severity_level'] = max(severities) if severities else 0
        
        # Compliance percentage
        # This requires knowing total applicable rules - implement based on your rule structure
        labels['cis_compliance_pct'] = self._calculate_compliance_pct(resources, all_results)
        
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
        self, resources: List[Dict], violations: List[RuleResult]
    ) -> float:
        """Calculate CIS compliance percentage."""
        # This is a simplified calculation - adjust based on your rule structure
        total_applicable = self.rules_evaluator.count_applicable_rules(resources)
        if total_applicable == 0:
            return 100.0
        violations_count = len(set(r.rule_id for r in violations))
        return ((total_applicable - violations_count) / total_applicable) * 100
    
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
            'dv_positive_rate': df['has_any_misconfiguration'].mean(),
            'dv_by_scenario': df.groupby('scenario_id')['has_any_misconfiguration'].mean().to_dict(),
            'feature_means': df.select_dtypes(include=['number']).mean().to_dict(),
            'feature_stds': df.select_dtypes(include=['number']).std().to_dict()
        }
        
        with open(output_path / 'summary_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
```

### 2. CLI Extension: Add Command to `crux/cli.py`

```python
@app.command()
def generate_template_dataset(
    templates: str = typer.Option(..., help="Path to templates directory"),
    rules: str = typer.Option("rules/", help="Path to rules directory"),
    output: str = typer.Option("dataset/", help="Output directory"),
    name: str = typer.Option(
        f"template-level-{datetime.now().strftime('%Y%m%d')}",
        help="Experiment name"
    ),
    limit: int = typer.Option(None, help="Limit number of templates to process"),
    scenarios: str = typer.Option(
        "all",
        help="Scenario set: 'all', 'minimal' (5 scenarios), or comma-separated list"
    ),
    validate: bool = typer.Option(True, help="Run validation after generation")
):
    """
    Generate template-level dataset for logistic regression analysis.
    
    This produces one observation per template × scenario combination,
    with aggregated features across all resources in each template.
    
    Target: 7,000+ observations for MSDA capstone statistical analysis.
    """
    from crux.dataset.template_level_generator import (
        TemplateLevelDatasetGenerator,
        MUTATION_SCENARIOS
    )
    from crux.dataset.validator import validate_template_dataset
    
    # Discover templates
    template_paths = discover_templates(templates)
    print(f"Found {len(template_paths)} templates")
    
    # Parse scenario selection
    if scenarios == "all":
        scenario_subset = None
    elif scenarios == "minimal":
        scenario_subset = ['baseline', 'security_all', 'operational_all', 
                         'reliability_all', 'all_mutations']
    else:
        scenario_subset = [s.strip() for s in scenarios.split(',')]
    
    # Generate dataset
    generator = TemplateLevelDatasetGenerator(rules_dir=rules)
    
    output_path = generator.generate_dataset(
        template_paths=template_paths,
        output_dir=output,
        experiment_name=name,
        limit=limit,
        scenarios_subset=scenario_subset
    )
    
    print(f"\nDataset generated: {output_path}")
    
    # Validate if requested
    if validate:
        print("\nRunning validation...")
        csv_path = Path(output_path) / 'template_level_data.csv'
        validation_results = validate_template_dataset(str(csv_path))
        
        if validation_results['meets_requirements']:
            print("✓ Dataset meets all requirements")
        else:
            print("⚠ Dataset has issues - see validation report")


def discover_templates(templates_dir: str) -> List[str]:
    """Discover all Bicep/ARM templates in directory."""
    templates_path = Path(templates_dir)
    bicep_files = list(templates_path.rglob("*.bicep"))
    arm_files = list(templates_path.rglob("azuredeploy.json"))
    
    # Prefer Bicep if both exist
    template_dirs = set()
    for f in bicep_files + arm_files:
        template_dirs.add(f.parent)
    
    templates = []
    for d in template_dirs:
        bicep = list(d.glob("main.bicep")) or list(d.glob("*.bicep"))
        if bicep:
            templates.append(str(bicep[0]))
        else:
            arm = list(d.glob("azuredeploy.json"))
            if arm:
                templates.append(str(arm[0]))
    
    return templates
```

### 3. Validation Script: `crux/dataset/validator.py`

```python
"""
Dataset validation for template-level CRUX dataset.
Ensures dataset meets requirements for logistic regression analysis.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def validate_template_dataset(csv_path: str) -> Dict[str, Any]:
    """
    Validate template-level dataset for statistical analysis requirements.
    
    Args:
        csv_path: Path to the template_level_data.csv file
        
    Returns:
        Dictionary containing validation results
    """
    df = pd.read_csv(csv_path)
    results = {
        'meets_requirements': True,
        'checks': {}
    }
    
    # Check 1: Minimum observations
    min_required = 7000
    results['checks']['observation_count'] = {
        'value': len(df),
        'requirement': f'>= {min_required}',
        'passed': len(df) >= min_required
    }
    
    # Check 2: DV variance
    dv_rate = df['has_any_misconfiguration'].mean()
    results['checks']['dv_variance'] = {
        'value': f"{dv_rate:.1%}",
        'requirement': '10% - 90%',
        'passed': 0.10 <= dv_rate <= 0.90
    }
    
    # Check 3: Feature variance (identify low-variance features)
    feature_cols = [c for c in df.columns 
                   if c not in ['template_id', 'template_name', 'scenario_id', 'scenario_category']
                   and not c.endswith('_count') and c != 'has_any_misconfiguration']
    
    low_variance_features = []
    for col in feature_cols:
        if df[col].nunique() <= 2:  # Binary
            pct = df[col].mean()
            if pct < 0.05 or pct > 0.95:
                low_variance_features.append({'column': col, 'positive_rate': pct})
        else:  # Continuous
            if df[col].std() < 0.01:
                low_variance_features.append({'column': col, 'std': df[col].std()})
    
    results['checks']['feature_variance'] = {
        'low_variance_features': low_variance_features,
        'count': len(low_variance_features),
        'passed': len(low_variance_features) < len(feature_cols) * 0.2  # <20% low variance
    }
    
    # Check 4: Multicollinearity
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        high_correlations = []
        for i, c1 in enumerate(numeric_cols):
            for c2 in numeric_cols[i+1:]:
                r = corr_matrix.loc[c1, c2]
                if abs(r) > 0.90:
                    high_correlations.append({
                        'feature_1': c1,
                        'feature_2': c2,
                        'correlation': round(r, 3)
                    })
        
        results['checks']['multicollinearity'] = {
            'high_correlations': high_correlations,
            'count': len(high_correlations),
            'passed': len(high_correlations) == 0
        }
    
    # Check 5: Scenario effectiveness
    scenario_rates = df.groupby('scenario_id')['has_any_misconfiguration'].mean()
    baseline_rate = scenario_rates.get('baseline', 0)
    mutated_rates = scenario_rates.drop('baseline', errors='ignore')
    
    results['checks']['scenario_effectiveness'] = {
        'baseline_rate': round(baseline_rate, 3),
        'mutated_mean_rate': round(mutated_rates.mean(), 3),
        'rate_difference': round(mutated_rates.mean() - baseline_rate, 3),
        'passed': mutated_rates.mean() > baseline_rate
    }
    
    # Check 6: Template coverage
    unique_templates = df['template_id'].nunique()
    unique_scenarios = df['scenario_id'].nunique()
    expected_obs = unique_templates * unique_scenarios
    
    results['checks']['coverage'] = {
        'unique_templates': unique_templates,
        'unique_scenarios': unique_scenarios,
        'expected_observations': expected_obs,
        'actual_observations': len(df),
        'coverage_rate': round(len(df) / expected_obs, 3) if expected_obs > 0 else 0,
        'passed': len(df) >= expected_obs * 0.95  # 95% coverage
    }
    
    # Overall pass/fail
    results['meets_requirements'] = all(
        check.get('passed', True) for check in results['checks'].values()
    )
    
    # Generate report
    _print_validation_report(results)
    
    return results


def _print_validation_report(results: Dict[str, Any]):
    """Print formatted validation report."""
    print("=" * 60)
    print("TEMPLATE-LEVEL DATASET VALIDATION REPORT")
    print("=" * 60)
    
    for check_name, check_result in results['checks'].items():
        status = "✓ PASS" if check_result.get('passed', True) else "✗ FAIL"
        print(f"\n{check_name.upper()}: {status}")
        
        for key, value in check_result.items():
            if key != 'passed':
                if isinstance(value, list) and len(value) > 3:
                    print(f"  {key}: [{len(value)} items]")
                    for item in value[:3]:
                        print(f"    - {item}")
                    if len(value) > 3:
                        print(f"    ... and {len(value) - 3} more")
                else:
                    print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    overall = "✓ MEETS ALL REQUIREMENTS" if results['meets_requirements'] else "✗ ISSUES FOUND"
    print(f"OVERALL: {overall}")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python validator.py <path_to_csv>")
        sys.exit(1)
    validate_template_dataset(sys.argv[1])
```

---

## Testing Requirements

### Unit Tests

Create `tests/test_template_level_generator.py`:

```python
import pytest
from crux.dataset.template_level_generator import (
    TemplateLevelDatasetGenerator,
    MUTATION_SCENARIOS
)

class TestMutationScenarios:
    def test_all_scenarios_have_required_fields(self):
        for scenario_id, config in MUTATION_SCENARIOS.items():
            assert 'description' in config
            assert 'mutations' in config
            assert isinstance(config['mutations'], list)
    
    def test_baseline_has_no_mutations(self):
        assert MUTATION_SCENARIOS['baseline']['mutations'] == []
    
    def test_scenario_references_resolve(self):
        generator = TemplateLevelDatasetGenerator(rules_dir='rules/')
        resolved = generator._resolved_scenarios
        
        # security_all should contain all mutations from security_high and security_medium
        assert 'storage_public_blob_access' in resolved['security_all']
        assert 'storage_weak_tls' in resolved['security_all']


class TestFeatureExtraction:
    def test_composition_features(self):
        generator = TemplateLevelDatasetGenerator(rules_dir='rules/')
        resources = [
            {'type': 'Microsoft.Storage/storageAccounts', 'properties': {}},
            {'type': 'Microsoft.Compute/virtualMachines', 'properties': {}},
        ]
        
        features = generator._extract_composition_features(resources)
        
        assert features['num_resources'] == 2
        assert features['has_storage'] == 1
        assert features['has_vm'] == 1
        assert features['has_nsg'] == 0
        assert features['count_storage'] == 1


class TestIntegration:
    def test_generates_expected_observation_count(self, tmp_path, sample_templates):
        generator = TemplateLevelDatasetGenerator(rules_dir='rules/')
        
        output_path = generator.generate_dataset(
            template_paths=sample_templates,
            output_dir=str(tmp_path),
            experiment_name='test',
            scenarios_subset=['baseline', 'security_all']
        )
        
        df = pd.read_csv(Path(output_path) / 'template_level_data.csv')
        
        # Should have 2 scenarios × number of templates
        expected = len(sample_templates) * 2
        assert len(df) == expected
```

---

## Expected Output

After running `crux generate-template-dataset --templates ./quickstart-templates --name msda-capstone`:

```
dataset/msda-capstone/
├── template_level_data.csv      # Main dataset (14,000+ rows)
├── metadata.json                # Experiment configuration
├── summary_stats.json           # Descriptive statistics
└── failed_templates.json        # Any templates that failed processing
```

### Sample CSV Structure

| template_id | scenario_id | num_resources | has_storage | any_public_access | pct_secure_boot | has_any_misconfiguration | misconfiguration_count |
|-------------|-------------|---------------|-------------|-------------------|-----------------|--------------------------|------------------------|
| vm-simple/main | baseline | 3 | 1 | 0 | 1.0 | 0 | 0 |
| vm-simple/main | security_all | 3 | 1 | 1 | 0.0 | 1 | 5 |
| storage-blob/main | baseline | 2 | 1 | 0 | 0.0 | 0 | 0 |
| storage-blob/main | security_all | 2 | 1 | 1 | 0.0 | 1 | 3 |

---

## Implementation Priority

1. **Phase 1 (Core):** `template_level_generator.py` with basic features
2. **Phase 2 (CLI):** Add `generate-template-dataset` command
3. **Phase 3 (Validation):** Add `validator.py`
4. **Phase 4 (Testing):** Unit and integration tests
5. **Phase 5 (Refinement):** Tune feature extraction based on validation results

---

## Questions for Implementer

Before starting implementation, clarify:

1. What is the current mutation ID naming scheme in CRUX?
2. What is the structure of `RuleResult` returned by the rule evaluator?
3. Are there existing helper functions for property extraction from ARM resources?
4. What is the graph library currently used (NetworkX confirmed)?
5. How are templates organized in the quickstart-templates directory?
