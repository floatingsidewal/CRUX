#!/usr/bin/env python3
"""
Generate full dataset using all 1,205 ARM JSON templates.
"""

import sys
from pathlib import Path

# Add CRUX root to path (script is in CRUX/scripts/)
script_dir = Path(__file__).parent
crux_root = script_dir.parent
sys.path.insert(0, str(crux_root))

from crux.dataset.generator import DatasetGenerator
from crux.mutations import ALL_MUTATIONS
from crux.rules.evaluator import RuleEvaluator

def main():
    print("=" * 80)
    print("FULL DATASET GENERATION: 1,205 ARM JSON Templates")
    print("=" * 80)

    # Read template list
    with open('/tmp/all_templates.txt') as f:
        template_paths = [Path(line.strip()) for line in f if line.strip()]

    print(f"\nTemplates to process: {len(template_paths)}")
    print(f"Output directory: dataset/full-1205")

    # Get all mutations
    print(f"Mutations available: {len(ALL_MUTATIONS)}")

    # Create dataset generator
    generator = DatasetGenerator(
        mutations=ALL_MUTATIONS,
        rules_dir="rules",
        output_dir="dataset"
    )

    # Generate dataset
    print("\nStarting dataset generation...")
    print("This will take 2-3 hours. Progress will be shown below.\n")

    dataset_path = generator.generate_dataset(
        template_paths=template_paths,
        experiment_name="full-1205",
        include_graphs=True,
        show_progress=True
    )

    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETE!")
    print("=" * 80)
    print(f"Dataset saved to: {dataset_path}")

    # Show metadata
    import json
    metadata_file = dataset_path / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)

        print("\nDataset Statistics:")
        print(f"  Templates processed: {metadata.get('templates_processed', 0)}")
        print(f"  Templates failed: {metadata.get('templates_failed', 0)}")
        print(f"  Baseline resources: {metadata.get('baseline_resources', 0)}")
        print(f"  Mutated resources: {metadata.get('mutated_resources', 0)}")
        print(f"  Mutations applied: {metadata.get('mutations_applied', 0)}")
        print(f"  Labels generated: {metadata.get('labels_generated', 0)}")

    print("\nReady for model training!")

if __name__ == "__main__":
    main()
