"""
Dataset Generator

End-to-end pipeline for generating labeled datasets from Bicep templates.
Pipeline: Fetch → Compile → Extract → Mutate → Label → Export
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from ..mutations.base import Mutation
from ..rules.evaluator import RuleEvaluator
from ..templates.compiler import BicepCompiler
from ..templates.extractor import ResourceExtractor
from ..templates.graph import DependencyGraphBuilder

logger = logging.getLogger(__name__)


class DatasetGenerator:
    """
    Orchestrates the end-to-end dataset generation pipeline.

    Pipeline:
    1. Compile Bicep templates to ARM JSON
    2. Extract resources from ARM templates
    3. Apply mutations to resources
    4. Evaluate security rules to generate labels
    5. Build dependency graphs (optional)
    6. Export structured dataset
    """

    def __init__(
        self,
        mutations: List[Mutation],
        rules_dir: str,
        output_dir: str,
    ):
        """
        Initialize the dataset generator.

        Args:
            mutations: List of Mutation objects to apply
            rules_dir: Directory containing YAML rule files
            output_dir: Base directory for dataset outputs
        """
        self.mutations = mutations
        self.rules_dir = Path(rules_dir)
        self.output_dir = Path(output_dir)

        # Initialize components
        self.compiler = BicepCompiler()
        self.extractor = ResourceExtractor()
        self.rule_evaluator = RuleEvaluator(rules_dir=str(self.rules_dir))
        self.graph_builder = DependencyGraphBuilder()

        logger.info(f"Initialized DatasetGenerator with {len(mutations)} mutations")
        logger.info(f"Loaded {len(self.rule_evaluator.rules)} security rules")

    def generate_dataset(
        self,
        template_paths: List[Path],
        experiment_name: Optional[str] = None,
        include_graphs: bool = True,
        show_progress: bool = True,
    ) -> Path:
        """
        Generate a complete labeled dataset from templates.

        Args:
            template_paths: List of paths to Bicep template files
            experiment_name: Name for this experiment (default: exp-YYYYMMDD-HHMMSS)
            include_graphs: Whether to generate dependency graphs
            show_progress: Whether to show progress bars

        Returns:
            Path to the generated dataset directory
        """
        # Create experiment directory
        if experiment_name is None:
            experiment_name = f"exp-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        exp_dir = self.output_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        baseline_dir = exp_dir / "baseline"
        mutated_dir = exp_dir / "mutated"
        graphs_dir = exp_dir / "graphs"

        baseline_dir.mkdir(exist_ok=True)
        mutated_dir.mkdir(exist_ok=True)
        if include_graphs:
            graphs_dir.mkdir(exist_ok=True)

        logger.info(f"Generating dataset: {experiment_name}")
        logger.info(f"Output directory: {exp_dir}")

        # Track all resources and labels
        baseline_resources = []
        mutated_resources = []
        all_labels: Dict[str, List[str]] = {}

        # Statistics
        stats = {
            "templates_processed": 0,
            "templates_failed": 0,
            "baseline_resources": 0,
            "mutated_resources": 0,
            "mutations_applied": 0,
            "labels_generated": 0,
        }

        # Process each template
        iterator = tqdm(template_paths, desc="Processing templates") if show_progress else template_paths

        for template_path in iterator:
            try:
                # Step 1: Compile Bicep to ARM
                arm_template = self._compile_template(template_path)
                if arm_template is None:
                    stats["templates_failed"] += 1
                    continue

                # Step 2: Extract resources
                resources = self.extractor.extract_resources(arm_template)
                if not resources:
                    logger.debug(f"No resources found in {template_path.name}")
                    stats["templates_failed"] += 1
                    continue

                # Step 3: Process baseline resources
                for resource in resources:
                    # Add metadata
                    resource["_metadata"] = {
                        "source_template": str(template_path),
                        "is_mutated": False,
                    }
                    baseline_resources.append(resource)
                    stats["baseline_resources"] += 1

                    # Evaluate baseline rules (should have no violations)
                    baseline_labels = self.rule_evaluator.evaluate(resource)
                    if baseline_labels:
                        resource_id = resource.get("id", resource.get("name", "unknown"))
                        all_labels[resource_id] = baseline_labels
                        stats["labels_generated"] += len(baseline_labels)

                # Step 4: Apply mutations
                for mutation in self.mutations:
                    for resource in resources:
                        # Check if mutation applies to this resource type
                        if not mutation.applies_to(resource):
                            continue

                        # Apply mutation
                        mutated_resource = mutation.apply(resource.copy())

                        # Add metadata
                        mutated_resource["_metadata"] = {
                            "source_template": str(template_path),
                            "is_mutated": True,
                            "mutation_id": mutation.id,
                            "mutation_description": mutation.description,
                            "mutation_severity": mutation.severity,
                        }
                        mutated_resources.append(mutated_resource)
                        stats["mutated_resources"] += 1
                        stats["mutations_applied"] += 1

                        # Step 5: Evaluate rules and generate labels
                        labels = self.rule_evaluator.evaluate(mutated_resource)
                        if labels:
                            resource_id = mutated_resource.get("id", mutated_resource.get("name", "unknown"))
                            mutation_key = f"{resource_id}#{mutation.id}"
                            all_labels[mutation_key] = labels
                            stats["labels_generated"] += len(labels)

                # Step 6: Build dependency graph (if requested)
                if include_graphs:
                    self._build_and_export_graph(
                        template_path, arm_template, resources, graphs_dir
                    )

                stats["templates_processed"] += 1

            except Exception as e:
                logger.error(f"Error processing {template_path}: {e}")
                stats["templates_failed"] += 1
                continue

        # Export results
        logger.info("Exporting dataset...")
        self._export_resources(baseline_resources, baseline_dir / "resources.json")
        self._export_resources(mutated_resources, mutated_dir / "resources.json")
        self._export_labels(all_labels, exp_dir / "labels.json")
        self._export_metadata(stats, experiment_name, exp_dir / "metadata.json")

        # Print summary
        logger.info("\nDataset generation complete!")
        logger.info(f"  Templates processed: {stats['templates_processed']}")
        logger.info(f"  Templates failed: {stats['templates_failed']}")
        logger.info(f"  Baseline resources: {stats['baseline_resources']}")
        logger.info(f"  Mutated resources: {stats['mutated_resources']}")
        logger.info(f"  Mutations applied: {stats['mutations_applied']}")
        logger.info(f"  Labels generated: {stats['labels_generated']}")

        return exp_dir

    def _compile_template(self, template_path: Path) -> Optional[Dict[str, Any]]:
        """Compile a Bicep template to ARM JSON."""
        try:
            # Check if it's already ARM JSON
            if template_path.suffix == ".json":
                with open(template_path) as f:
                    return json.load(f)

            # Compile Bicep to ARM
            return self.compiler.compile(template_path)
        except Exception as e:
            logger.warning(f"Failed to compile {template_path.name}: {e}")
            return None

    def _build_and_export_graph(
        self,
        template_path: Path,
        arm_template: Dict[str, Any],
        resources: List[Dict[str, Any]],
        graphs_dir: Path,
    ) -> None:
        """Build and export dependency graph for a template."""
        try:
            # Build graph
            graph = self.graph_builder.build_graph(arm_template)

            if graph.number_of_nodes() == 0:
                return

            # Export to GraphML
            graph_file = graphs_dir / f"{template_path.stem}.graphml"
            self.graph_builder.export_to_graphml(graph, graph_file)

            # Also export to JSON for easier inspection
            json_file = graphs_dir / f"{template_path.stem}_graph.json"
            graph_data = {
                "nodes": [
                    {"id": node, **graph.nodes[node]}
                    for node in graph.nodes()
                ],
                "edges": [
                    {"source": u, "target": v, **graph.edges[u, v]}
                    for u, v in graph.edges()
                ],
            }
            with open(json_file, "w") as f:
                json.dump(graph_data, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to build graph for {template_path.name}: {e}")

    def _export_resources(self, resources: List[Dict[str, Any]], output_file: Path) -> None:
        """Export resources to JSON file."""
        with open(output_file, "w") as f:
            json.dump(resources, f, indent=2)
        logger.debug(f"Exported {len(resources)} resources to {output_file}")

    def _export_labels(self, labels: Dict[str, List[str]], output_file: Path) -> None:
        """Export labels to JSON file."""
        with open(output_file, "w") as f:
            json.dump(labels, f, indent=2)
        logger.debug(f"Exported {len(labels)} label entries to {output_file}")

    def _export_metadata(
        self, stats: Dict[str, int], experiment_name: str, output_file: Path
    ) -> None:
        """Export experiment metadata."""
        metadata = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "mutations": [
                {
                    "id": m.id,
                    "target_type": m.target_type,
                    "description": m.description,
                    "severity": m.severity,
                    "labels": m.labels,
                }
                for m in self.mutations
            ],
            "rules": [
                {
                    "id": r.get("id"),
                    "resource_type": r.get("resource_type"),
                    "severity": r.get("severity"),
                }
                for r in self.rule_evaluator.rules
            ],
            "statistics": stats,
        }
        with open(output_file, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.debug(f"Exported metadata to {output_file}")
