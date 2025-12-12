# Change log

## v0.1.3 - 2025-11-22
- Updated README.md documentation
  - Added comprehensive development setup instructions
  - Documented both devcontainer (recommended) and local development options
  - Added prerequisites section (Python 3.10+, Azure CLI, Bicep CLI)
  - Added environment notes for virtual environment usage

## v0.1.2 - 2025-09-24
- Completed Milestone 1.1: Development Environment Setup
  - Added .devcontainer configuration with Python, Azure CLI, and Bicep support
  - Added .env file support using python-dotenv for credential abstraction

## v0.1.1 - 2025-09-24
- Completed Milestone 1: Foundation Setup
  - Scaffolded repository structure
  - Created Bicep modules for storage, key vault, function app, NSG, VM, VMSS, VNet, subnet, public IP
  - Implemented Python CLI harness with deploy, harvest, cleanup commands
  - Tested end-to-end functionality

## v0.1.0 - 2025-09-24
- Initial release of CRUX, a tool for generating and analyzing Azure ARM/Bicep configurations with controlled misconfigurations for machine learning training.
- Features include deployment of baseline resources, mutation to create misconfigurations, harvesting resource states, and structured data storage.