#!/usr/bin/env python3
"""
Generate Maximum Template-Level Dataset for CRUX

This script generates a template-level dataset meeting the specification requirements:
- Target: 7,000+ observations (ideally 8,000-14,000)
- Method: ~1,000 templates × 14 mutation scenarios = ~14,000 observations
- Purpose: Academic logistic regression analysis with probabilistic IV-DV relationships

Usage:
    python3 scripts/generate_maximum_template_dataset.py

Requirements:
    - Azure Quickstart Templates fetched to templates/azure-quickstart-templates/
    - CRUX package installed with dependencies (pip install -e .[dev])
"""

import sys
from pathlib import Path

# Add CRUX to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from crux.dataset.template_level_generator import (
        TemplateLevelDatasetGenerator,
        MUTATION_SCENARIOS
    )
    from crux.dataset.validator import validate_template_dataset
except ImportError as e:
    print(f"Error: Missing dependencies. Please install CRUX with: pip install -e .[dev]")
    print(f"Details: {e}")
    sys.exit(1)


def discover_templates(templates_dir: str, limit: int = None) -> list:
    """
    Discover all ARM/Bicep templates in directory.

    Args:
        templates_dir: Path to templates directory
        limit: Optional limit on number of templates to use

    Returns:
        List of template file paths
    """
    templates_path = Path(templates_dir)

    # Find all azuredeploy.json and main.bicep files
    json_templates = list(templates_path.rglob("azuredeploy.json"))
    bicep_templates = list(templates_path.rglob("main.bicep"))

    all_templates = json_templates + bicep_templates

    print(f"Found {len(json_templates)} ARM templates (azuredeploy.json)")
    print(f"Found {len(bicep_templates)} Bicep templates (main.bicep)")
    print(f"Total: {len(all_templates)} templates")

    if limit:
        all_templates = all_templates[:limit]
        print(f"Limited to: {len(all_templates)} templates")

    return [str(t) for t in all_templates]


def main():
    """Generate maximum template-level dataset."""

    print("=" * 80)
    print("CRUX Maximum Template-Level Dataset Generator")
    print("=" * 80)
    print()

    # Configuration
    templates_dir = "templates/azure-quickstart-templates"
    rules_dir = "rules"
    output_dir = "dataset"
    experiment_name = "maximum-template-level"
    template_limit = 1000  # Per spec: ~1,000 templates

    # Check if templates exist
    if not Path(templates_dir).exists():
        print(f"Error: Templates directory not found: {templates_dir}")
        print()
        print("Please fetch Azure Quickstart Templates first:")
        print("  cd templates")
        print("  git clone https://github.com/Azure/azure-quickstart-templates.git")
        sys.exit(1)

    # Discover templates
    print(f"Discovering templates in: {templates_dir}")
    template_paths = discover_templates(templates_dir, limit=template_limit)

    if not template_paths:
        print("Error: No templates found!")
        sys.exit(1)

    # Calculate expected observations
    num_scenarios = len(MUTATION_SCENARIOS)
    expected_observations = len(template_paths) * num_scenarios

    print()
    print("Dataset Configuration:")
    print(f"  Templates: {len(template_paths)}")
    print(f"  Scenarios: {num_scenarios}")
    print(f"  Expected observations: {expected_observations:,}")
    print(f"  Meets 7,000+ requirement: {'✓ YES' if expected_observations >= 7000 else '✗ NO'}")
    print()

    if expected_observations < 7000:
        print(f"Warning: Expected observations ({expected_observations:,}) is less than 7,000!")
        print(f"Recommendation: Increase template_limit to at least {7000 // num_scenarios} templates")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(1)

    # Initialize generator
    print("Initializing template-level dataset generator...")
    generator = TemplateLevelDatasetGenerator(rules_dir=rules_dir)

    print(f"Loaded {len(MUTATION_SCENARIOS)} mutation scenarios:")
    for scenario_id in MUTATION_SCENARIOS.keys():
        print(f"  - {scenario_id}")
    print()

    # Generate dataset
    print("Generating dataset...")
    print("This may take 10-30 minutes for 1,000 templates...")
    print()

    try:
        output_path = generator.generate_dataset(
            template_paths=template_paths,
            output_dir=output_dir,
            experiment_name=experiment_name,
            limit=None,  # Don't limit further - already limited in discover
            scenarios_subset=None  # Use all scenarios
        )

        print()
        print("=" * 80)
        print("✓ Dataset Generation Complete!")
        print("=" * 80)
        print(f"Output directory: {output_path}")
        print()

        # Validate dataset
        print("Running validation...")
        csv_path = Path(output_path) / 'template_level_data.csv'

        if csv_path.exists():
            validation_results = validate_template_dataset(str(csv_path))

            print()
            if validation_results['meets_requirements']:
                print("✓ Dataset meets all academic requirements!")
            else:
                print("⚠ Dataset has validation issues - see report above")
        else:
            print(f"Error: CSV file not found: {csv_path}")

        print()
        print("Next steps:")
        print("  1. Review dataset: ls -lh", output_path)
        print("  2. Load in pandas: pd.read_csv(f'{csv_path}')")
        print("  3. Run statistical analysis (see README.md)")

    except Exception as e:
        print()
        print("=" * 80)
        print("✗ Dataset Generation Failed!")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
