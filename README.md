# CRUX

**CRUX** (Cloud Resource Configuration Analyzer) is a prototype system for generating labeled datasets to train ML models that detect misconfigurations in Azure deployments. CRUX composes small Bicep modules into scenarios, deploys them, harvests running configuration state, applies controlled mutations (a "configuration fuzzer"), then harvests again with ground‑truth labels.

## Why CRUX
- **Deterministic labels**: Every mutation is intentional and recorded → clean ground truth.
- **Schema‑agnostic**: Supports heterogeneous resource types (Function, Storage, Key Vault, NSG, etc.) and their relationships.
- **Graph‑aware**: Designed to export node/edge views for cross‑resource issue detection.

## Repository Layout (initial)
```
crux/
  modules/                 # reusable Bicep modules
  scenarios/               # YAML → Bicep compositions
  mutations/               # fuzzer recipes + constraints
  harness/                 # Python CLI orchestrator
  params/                  # default parameters
  dataset/                 # harvested outputs (git‑ignored by default)
  .github/
    copilot-instructions.md
  prd.md
  README.md
```

## Quick Start (baseline flow)
1) Use a sandbox Azure subscription and login with `az login`.
2) Optionally prepare a Bicep/ARM composition template (see examples in **modules/** and patterns in **prd.md**).
3) Deploy baseline → harvest → mutate → harvest → cleanup using the CLI in `harness/crux.py`.

Examples:

```
# Create RG, (optionally) deploy a template, and harvest baseline
python3 harness/crux.py deploy \
  --rg crux-rg-$RANDOM \
  --location eastus \
  --template out/func_storage_kv.bicep \
  --parameters params/func_storage_kv.json \
  --scenario func-storage-kv

# Apply a raw mutation to a specific resource and harvest again
# (Find resource IDs in dataset/<exp>/original/resources.json)
python3 harness/crux.py mutate-raw \
  --rg <your-rg> \
  --resource-id \
  "/subscriptions/.../resourceGroups/<your-rg>/providers/Microsoft.Storage/storageAccounts/<name>" \
  --set properties.allowBlobPublicAccess=true \
  --out dataset/<your-exp-folder>

# Cleanup the resource group
python3 harness/crux.py cleanup --rg <your-rg>
```

## Dataset Structure (per experiment)
```
crux/dataset/exp-YYYYMMDD-HHMMSS/
  metadata.json         # scenario, commit, timestamps
  original/             # harvested baseline JSONs
  mutated/
    <mutation-id>/      # harvested after a specific mutation
      mutation.json     # what changed + labels
  labels.json           # resource → [labels]
```

## Roadmap (high level)
- Milestone 1: Repo scaffolding, core Bicep modules, deploy + harvest.
- Milestone 2: Fuzzer & labels; mutation recipes and constraints.
- Milestone 3: Dataset ops, feature exporters.
- Milestone 4: Graph export (nodes/edges) for cross‑resource patterns.
- Milestone 5: Modeling baselines (CatBoost/XGBoost + graph models).
- Milestone 6: Validation & capstone write‑up.

See **prd.md** for the full requirements and design.
