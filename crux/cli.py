"""
CRUX CLI

Command-line interface for the CRUX static template analysis system.
"""

import argparse
import logging
import sys
from pathlib import Path

from .templates.fetcher import TemplateFetcher
from .dataset.generator import DatasetGenerator
from .mutations import storage as storage_mutations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_fetch_templates(args: argparse.Namespace) -> None:
    """Fetch Azure Quickstart Templates from GitHub."""
    logger.info("Fetching templates from GitHub...")

    fetcher = TemplateFetcher(output_dir=args.output)
    templates = fetcher.fetch_templates(
        repo_url=args.repo,
        pattern=args.pattern,
        limit=args.limit,
    )

    logger.info(f"Fetched {len(templates)} templates to {args.output}")

    # Print first 10 templates as examples
    if templates:
        logger.info("\nExample templates:")
        for template in templates[:10]:
            logger.info(f"  {template.relative_to(Path(args.output))}")
        if len(templates) > 10:
            logger.info(f"  ... and {len(templates) - 10} more")


def cmd_generate_dataset(args: argparse.Namespace) -> None:
    """Generate a labeled dataset from templates."""
    logger.info("Generating dataset...")

    # Discover templates
    template_dir = Path(args.templates)
    if not template_dir.exists():
        logger.error(f"Template directory not found: {template_dir}")
        sys.exit(1)

    template_files = list(template_dir.glob(args.pattern))
    logger.info(f"Found {len(template_files)} templates")

    if not template_files:
        logger.error("No templates found matching pattern")
        sys.exit(1)

    # Limit if requested
    if args.limit:
        template_files = template_files[: args.limit]
        logger.info(f"Limited to {len(template_files)} templates")

    # Load mutations
    mutations = storage_mutations.ALL_MUTATIONS
    logger.info(f"Loaded {len(mutations)} mutations")

    # Initialize generator
    generator = DatasetGenerator(
        mutations=mutations,
        rules_dir=args.rules,
        output_dir=args.output,
    )

    # Generate dataset
    dataset_dir = generator.generate_dataset(
        template_paths=template_files,
        experiment_name=args.name,
        include_graphs=args.graphs,
        show_progress=not args.no_progress,
    )

    logger.info(f"\nDataset generated successfully: {dataset_dir}")
    logger.info("\nNext steps:")
    logger.info(f"  1. Explore the data: ls -la {dataset_dir}")
    logger.info(f"  2. View labels: cat {dataset_dir}/labels.json")
    logger.info(f"  3. View metadata: cat {dataset_dir}/metadata.json")


def cmd_list_mutations(args: argparse.Namespace) -> None:
    """List available mutations."""
    logger.info("Available mutations:\n")

    mutations = storage_mutations.ALL_MUTATIONS

    # Group by severity
    by_severity = {}
    for mutation in mutations:
        if mutation.severity not in by_severity:
            by_severity[mutation.severity] = []
        by_severity[mutation.severity].append(mutation)

    for severity in ["critical", "high", "medium", "low"]:
        if severity in by_severity:
            print(f"\n{severity.upper()} Severity:")
            for mutation in by_severity[severity]:
                print(f"  [{mutation.id}]")
                print(f"    Description: {mutation.description}")
                print(f"    Target: {mutation.target_type}")
                print(f"    Labels: {', '.join(mutation.labels)}")
                if mutation.cis_references:
                    print(f"    CIS: {', '.join(mutation.cis_references)}")


def cmd_list_rules(args: argparse.Namespace) -> None:
    """List available security rules."""
    from .rules.evaluator import RuleEvaluator

    logger.info(f"Loading rules from {args.rules}...\n")

    evaluator = RuleEvaluator(rules_dir=args.rules)

    if not evaluator.rules:
        logger.warning("No rules found")
        return

    # Group by severity
    by_severity = {}
    for rule in evaluator.rules:
        severity = rule.get("severity", "unknown")
        if severity not in by_severity:
            by_severity[severity] = []
        by_severity[severity].append(rule)

    for severity in ["critical", "high", "medium", "low"]:
        if severity in by_severity:
            print(f"\n{severity.upper()} Severity:")
            for rule in by_severity[severity]:
                print(f"  [{rule.get('id')}]")
                print(f"    Name: {rule.get('name')}")
                print(f"    Resource: {rule.get('resource_type')}")
                print(f"    Labels: {', '.join(rule.get('labels', []))}")
                if rule.get("cis_reference"):
                    print(f"    CIS: {rule.get('cis_reference')}")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="crux",
        description="CRUX: Cloud Resource Configuration Analyzer (Static Analysis)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # fetch-templates command
    fetch_parser = subparsers.add_parser(
        "fetch-templates",
        help="Fetch Azure Quickstart Templates from GitHub",
    )
    fetch_parser.add_argument(
        "--repo",
        default="https://github.com/Azure/azure-quickstart-templates.git",
        help="GitHub repository URL",
    )
    fetch_parser.add_argument(
        "--pattern",
        default="quickstarts/**/*.bicep",
        help="Glob pattern for Bicep files",
    )
    fetch_parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of templates to fetch",
    )
    fetch_parser.add_argument(
        "--output",
        default="templates",
        help="Output directory for templates",
    )
    fetch_parser.set_defaults(func=cmd_fetch_templates)

    # generate-dataset command
    generate_parser = subparsers.add_parser(
        "generate-dataset",
        help="Generate labeled dataset from templates",
    )
    generate_parser.add_argument(
        "--templates",
        required=True,
        help="Directory containing Bicep templates",
    )
    generate_parser.add_argument(
        "--pattern",
        default="**/*.bicep",
        help="Glob pattern for Bicep files",
    )
    generate_parser.add_argument(
        "--rules",
        default="rules",
        help="Directory containing YAML rule files",
    )
    generate_parser.add_argument(
        "--output",
        default="dataset",
        help="Output directory for dataset",
    )
    generate_parser.add_argument(
        "--name",
        help="Experiment name (default: exp-YYYYMMDD-HHMMSS)",
    )
    generate_parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of templates to process",
    )
    generate_parser.add_argument(
        "--graphs",
        action="store_true",
        default=True,
        help="Generate dependency graphs (default: True)",
    )
    generate_parser.add_argument(
        "--no-graphs",
        dest="graphs",
        action="store_false",
        help="Skip generating dependency graphs",
    )
    generate_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )
    generate_parser.set_defaults(func=cmd_generate_dataset)

    # list-mutations command
    list_mutations_parser = subparsers.add_parser(
        "list-mutations",
        help="List available mutations",
    )
    list_mutations_parser.set_defaults(func=cmd_list_mutations)

    # list-rules command
    list_rules_parser = subparsers.add_parser(
        "list-rules",
        help="List available security rules",
    )
    list_rules_parser.add_argument(
        "--rules",
        default="rules",
        help="Directory containing YAML rule files",
    )
    list_rules_parser.set_defaults(func=cmd_list_rules)

    return parser


def main(argv=None):
    """Main entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        args.func(args)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
