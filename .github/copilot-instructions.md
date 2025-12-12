# Copilot Instructions for CRUX

These instructions are intended to guide GitHub Copilot when suggesting code or text within this repository.  Copilot should understand the overall goals of CRUX, follow the coding conventions used, and respect the boundaries of the project.

## Goal

CRUX is a tool to generate and analyze Azure ARM/Bicep configurations.  It deploys baseline resource combinations, mutates them to create controlled misconfigurations, harvests the resource state, and stores the results for machine learning training.

## Coding Conventions
* Language: Use Python 3.10+ for orchestration scripts in harness/.  Use Bicep for infrastructure definitions in modules/ and scenarios/ (compiled from YAML).  Mutation recipes are written in YAML.
* CLI: Expose functionality via an argparse based CLI within harness/crux.py.  Commands should include deploy, harvest, mutate and cleanup with clear parameters.
* Data storage: Harvested data is stored as JSON in structured directories under dataset/.  Each experiment includes a metadata.json file and per‑resource JSON dumps.  Labels are stored in labels.json keyed by resource identifier.
* Modular design: Write small, reusable functions.  Bicep modules should expose only necessary parameters and outputs.  Do not hard‑code resource names or values; read them from parameters or scenario definitions.
* Safety: When generating Azure resources, use inexpensive SKUs, avoid broad RBAC assignments, and deploy to a dedicated test subscription.  The harness should enforce a time‑to‑live on resource groups to prevent cost overruns.
* Documentation: Whenever adding new features or modules, update prd.md and README.md accordingly.  Provide examples in scenario YAML and mutation recipes.

## Suggestions for Copilot
* When editing files in modules/, propose valid Bicep definitions for Azure resources such as Storage Accounts, Key Vaults, Function Apps, Network Security Groups, Virtual Machines, etc.  Include parameters for common properties (SKU, encryption, firewall rules, purge protection, etc.).
* When working in scenarios/, suggest YAML schemas that mirror the examples in prd.md and allow users to specify resources and relationships.  Do not invent resource types that are not supported in the current modules.
* When writing Python in harness/:
* Provide functions to compile scenario YAML into a Bicep composition file.
* Use the Azure CLI via subprocess.run or the Azure SDK (azure-identity, azure-mgmt-resource) for deployments and updates.
* Query resource states with az resource list --resource-group ... -o json or the Azure Resource Graph.
* Write metadata and dataset files to the correct location with descriptive names and timestamps.
* Support verbose logging and graceful error handling.
* When editing mutations/recipes.yaml, suggest realistic misconfigurations such as enabling public access on a storage account, opening inbound NSG rules to 0.0.0.0/0, disabling purge protection on a key vault, or placing secrets directly in app settings.  Each mutation should include a labels field containing one or more concise identifiers for the misconfigurations.
* When adding features that interface with Azure services, remember that this project runs in a test environment; avoid production‑grade features such as high‑end SKUs or scaling settings.  Provide defaults that minimise cost and avoid side effects beyond the resource group.

## Development Workflow
* Always work on one milestone at a time from prd.md.
* Update the checkboxes in prd.md for each subtask once completed.
* Ensure changes are tracked in changelog.md.

## Things to Avoid
* Do not suggest modifications that would permanently alter or destroy data in a live Azure subscription.  All actions should target disposable resource groups in a sandbox subscription.
* Do not commit secrets, credentials or subscription IDs to the repository.  Any necessary authentication should use environment variables or managed identities.
* Do not generate or enforce licensing terms beyond those specified in the LICENSE file.
* Do not add dependencies outside the Python standard library and the Azure SDK without prior discussion.

## Updating These Instructions
If you introduce new languages, frameworks or significant conventions to the project, update this file so Copilot remains helpful and on topic.

## Tracking changes
Always keep the changelog.md updated with a summary of all major changes made to the repository.