CRUX (Cloud Resource Configuration Analyzer)

**NOTE: This project has pivoted to Option D (Static Template Analysis)**

The original PRD below describes Option A (Azure deployment-based analysis). As of 2025-01-19, CRUX now uses **Option D: Static Template Analysis**, which:
- Analyzes Bicep/ARM templates locally (zero Azure cost)
- Generates datasets from Azure Quickstart Templates (1000+ examples)
- Applies mutations and labels without deploying resources
- See README.md and CLAUDE.md for updated architecture

The Option A codebase is preserved in the `archive/option-a` branch.

---

# Original PRD (Option A - Archived)

1. Problem Statement
Misconfigurations in Azure resource deployments are a leading cause of outages, security incidents, and support tickets. Rule-based policy engines catch obvious violations but often miss subtle, cross-resource issues. CRUX generates a high-quality labeled dataset by deploying small, parameterized Bicep-based environments, harvesting their configuration state, applying controlled misconfigurations (mutations), and harvesting again. These data support research and prototyping of ML models that detect misconfigurations across heterogeneous resources and their relationships.

2. Objectives
- Automate deployment of baseline Azure environments using small Bicep modules.
- Compose multiple resource types (Function App, Storage, Key Vault, NSG, VM, etc.) via simple YAML “scenarios”.
- Implement a configuration fuzzer that applies controlled mutations with deterministic labels.
- Harvest resource state into structured JSON files with clear provenance and metadata.
- Organize datasets into known-good and mutated with ground-truth labels.
- Enable downstream tabular and graph-based ML for misconfiguration detection.

3. Design Overview

3.1 Repository Layout
- `modules/` – Reusable Bicep modules (storage, keyvault, functionapp, nsg).
- `scenarios/` – YAML scenario definitions (compose resources + relations).
- `mutations/` – Fuzzer recipes and constraints (YAML).
- `harness/` – Python CLI orchestrator (deploy → harvest → mutate → cleanup).
- `params/` – Baseline parameter files for known-good deployments.
- `dataset/` – Harvested outputs (git-ignored by default).
- `.github/` – Copilot guidance.
- `prd.md` – This requirements doc.
- `README.md` – Project overview and quick start.

3.2 Scenario Definition (YAML → Bicep)
Example (`scenarios/func_storage_kv.yaml`):

name: func-storage-kv
location: eastus
resources:
  - type: Microsoft.Web/sites@functionapp
    alias: func1
    params:
      httpsOnly: true
  - type: Microsoft.Storage/storageAccounts
    alias: st1
    params:
      sku: Standard_LRS
      allowBlobPublicAccess: false
  - type: Microsoft.KeyVault/vaults
    alias: kv1
    params:
      enablePurgeProtection: true
relations:
  - link: func1 -> st1
  - link: func1 -> kv1

Notes:
- The initial scaffold includes modules and example scenarios; a generator that compiles scenario YAML → a composition Bicep is a planned enhancement.

3.3 Mutation Recipes
Describe how to inject misconfigurations. Each recipe targets a resource type and either sets a property or invokes a resource-specific action.

Example (`mutations/recipes.yaml`):
mutations:
  - id: st_public_access
    target: Microsoft.Storage/storageAccounts
    set:
      properties.allowBlobPublicAccess: true
    labels: [Storage_PublicAccess]
  - id: kv_no_purge_protection
    target: Microsoft.KeyVault/vaults
    set:
      properties.enablePurgeProtection: false
    labels: [KeyVault_NoPurgeProtection]
  - id: nsg_open_ssh
    target: Microsoft.Network/networkSecurityGroups
    action: nsg_rule_update
    args:
      name: "AllowSSH"
      source: "*"
      destPorts: [22]
      access: "Allow"
    labels: [NSG_OpenInbound]

Constraints (e.g., forbidden combos, TTL, cost guards) live in `mutations/constraints.yaml`.

3.4 Harvesting and Dataset Generation
Per experiment:
- Deploy baseline (Bicep). Baseline is known-good.
- Harvest all resources in the RG: `az resource list --resource-group <rg>` → JSON files.
- Apply a mutation (one at a time initially).
- Re-harvest into `mutated/<mutation-id>/` and write `mutation.json` with labels and provenance.
- Cleanup RG or revert.

Output structure:
dataset/exp-YYYYMMDD-HHMMSS[-scenario]/
  metadata.json
  original/
    resources.json
    <resource-n>.json
  mutated/
    <mutation-id>/
      resources.json
      mutation.json
  labels.json

3.5 Dataset Schema
- metadata.json: experiment_id, scenario, resource_group, location, template, parameters, commit_hash, timestamps.
- labels.json: map of resource_id → [labels].
- mutation.json: mutation_id, target, method, args, expected_labels, timestamp.

4. Milestones

- [ ] **Milestone 1 – Foundation Setup**
  - [x] Scaffold repository structure with all directories
  - [x] Create initial Bicep module for storage account
  - [x] Create initial Bicep module for key vault
  - [x] Create initial Bicep module for function app
  - [x] Create initial Bicep module for network security group
  - [x] Create initial Bicep module for virtual machine
  - [x] Create initial Bicep module for virtual machine scale set
  - [x] Create initial Bicep module for virtual network
  - [x] Create initial Bicep module for subnet
  - [x] Create initial Bicep module for public IP address
  - [x] Implement basic Python CLI harness skeleton in crux.py
  - [x] Add argparse CLI with basic commands (deploy, harvest, cleanup)
  - [x] Implement deploy functionality using Azure CLI
  - [x] Implement harvest functionality to collect resource state
  - [x] Implement cleanup functionality to remove resource groups
  - [x] Test end-to-end deploy → harvest → cleanup for a single storage resource

- [ ] **Milestone 1.1 – Development Environment Setup**
  - [x] Add .devcontainer configuration for consistent development environment
  - [x] Add support for .env files to abstract credentials and configuration

- [ ] **Milestone 2 – Scenario Composition**
  - [ ] Design YAML schema for scenario definitions
  - [ ] Create example scenario YAML file (func_storage_kv.yaml)
  - [ ] Implement YAML to Bicep compilation in harness
  - [ ] Add support for resource relations in scenarios
  - [ ] Test compilation and deployment of multi-resource scenario
  - [ ] Validate baseline deployments are known-good

- [ ] **Milestone 3 – Mutation Framework**
  - [ ] Design YAML schema for mutation recipes
  - [ ] Add initial mutation recipes for storage accounts
  - [ ] Add initial mutation recipes for key vaults
  - [ ] Add initial mutation recipes for function apps
  - [ ] Add initial mutation recipes for NSGs
  - [ ] Implement mutation application logic in harness
  - [ ] Add constraints handling from constraints.yaml
  - [ ] Test mutation application on a single resource
  - [ ] Verify labels are correctly applied to mutated resources

- [ ] **Milestone 4 – Dataset Generation**
  - [ ] Implement metadata capture in metadata.json
  - [ ] Structure dataset output directories (original/, mutated/)
  - [ ] Add per-resource JSON dumps
  - [ ] Implement labels.json generation
  - [ ] Add integrity checks for harvested data
  - [ ] Test dataset generation for one scenario with mutations
  - [ ] Organize outputs into known-good and mutated datasets

- [ ] **Milestone 5 – Data Export**
  - [ ] Implement feature export to tabular format (CSV/JSON)
  - [ ] Implement feature export to graph format (nodes/edges)
  - [ ] Add resource relations to graph export
  - [ ] Test tabular export for multiple resources
  - [ ] Test graph export for cross-resource patterns
  - [ ] Validate exported data for ML consumption

- [ ] **Milestone 6 – Model Prototyping**
  - [ ] Prepare dataset for ML (split train/test)
  - [ ] Implement baseline rule-based detection
  - [ ] Implement simple tabular ML model (e.g., XGBoost)
  - [ ] Implement simple graph-based ML model
  - [ ] Compare model performance against rule-based baseline
  - [ ] Document model results and insights

- [ ] **Milestone 7 – Validation & Documentation**
  - [ ] Run experiments across multiple scenarios
  - [ ] Evaluate mutation accuracy and label quality
  - [ ] Assess data quality and coverage metrics
  - [ ] Document findings, limitations, and risks
  - [ ] Update README.md with usage examples
  - [ ] Write final report and lessons learned

5. Risks & Mitigations
- Cost/runaway deployments → enforce TTL, small SKUs, sandbox subscription, tagging.
- Dangerous mutations → restrict to safe properties; avoid broad RBAC; test-only environments.
- Label noise/imbalance → deterministic mutations; policy checks; manual spot checks.
- Template drift → pin module versions; commit scenarios/params.

6. Non-Goals (Initial)
- Production deployment of CRUX; focus is dataset generation and research prototyping.
- Covering every Azure resource type; start with Storage, Key Vault, Function App, NSG.

7. Success Metrics
- Usability: One-command experiment (deploy→harvest) works end-to-end.
- Data quality: Mutations produce expected labels ≥95% of the time.
- Coverage: At least 10 mutation types across 3 scenarios.
- Reproducibility: Experiments reproducible from committed scenarios/params.

