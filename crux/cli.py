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
from .mutations import ALL_MUTATIONS

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
    mutations = ALL_MUTATIONS
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

    mutations = ALL_MUTATIONS

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


def cmd_train_model(args: argparse.Namespace) -> None:
    """Train a machine learning model on a dataset."""
    from datetime import datetime
    from .ml.dataset import DatasetLoader
    from .ml.features import FeatureExtractor
    from .ml.models import RandomForestModel, XGBoostModel
    from .ml.evaluation import ModelEvaluator

    logger.info(f"Training {args.model} model on dataset {args.dataset}")

    # Load dataset
    loader = DatasetLoader(args.dataset)
    (
        train_res,
        val_res,
        test_res,
        train_labels,
        val_labels,
        test_labels,
        label_names,
    ) = loader.load_and_prepare(
        include_baseline=False,
        test_size=args.test_size,
        val_size=args.val_size,
    )

    logger.info(f"Loaded {len(train_res)} train, {len(val_res)} val, {len(test_res)} test samples")
    logger.info(f"Training for {len(label_names)} labels: {', '.join(label_names[:5])}...")

    # Extract features
    feature_extractor = FeatureExtractor(max_features=args.max_features)
    X_train, _ = feature_extractor.fit_transform(train_res)
    X_val, _ = feature_extractor.transform(val_res)
    X_test, _ = feature_extractor.transform(test_res)

    logger.info(f"Extracted {X_train.shape[1]} features per sample")

    # Create model
    if args.model == "random-forest":
        model = RandomForestModel(n_estimators=100, random_state=42)
    elif args.model == "xgboost":
        model = XGBoostModel(n_estimators=100, random_state=42)
    else:
        logger.error(f"Unknown model type: {args.model}")
        sys.exit(1)

    # Train model
    model.fit(
        X_train,
        train_labels,
        feature_names=feature_extractor.get_feature_names(),
        label_names=label_names,
    )

    # Evaluate on validation set
    logger.info("\nValidation set evaluation:")
    y_val_pred = model.predict(X_val)
    evaluator = ModelEvaluator(label_names)
    val_metrics = evaluator.evaluate(val_labels, y_val_pred)
    evaluator.print_report(val_metrics)

    # Evaluate on test set
    logger.info("\nTest set evaluation:")
    y_test_pred = model.predict(X_test)
    test_metrics = evaluator.evaluate(test_labels, y_test_pred)
    evaluator.print_report(test_metrics)

    # Save model
    model_name = args.name or f"model-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{model_name}.pkl"
    model.save(str(model_path))

    # Save feature extractor
    import pickle
    feature_path = output_dir / f"{model_name}_features.pkl"
    with open(feature_path, "wb") as f:
        pickle.dump(feature_extractor, f)
    logger.info(f"Feature extractor saved to {feature_path}")

    # Save evaluation metrics
    metrics_path = output_dir / f"{model_name}_metrics.json"
    import json
    with open(metrics_path, "w") as f:
        json.dump({
            "validation": val_metrics,
            "test": test_metrics,
        }, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    logger.info(f"\nModel training complete!")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Test F1 Score (macro): {test_metrics['f1_macro']:.3f}")
    logger.info(f"  Test F1 Score (micro): {test_metrics['f1_micro']:.3f}")


def cmd_evaluate_model(args: argparse.Namespace) -> None:
    """Evaluate a trained model on a dataset."""
    import pickle
    from .ml.dataset import DatasetLoader
    from .ml.features import FeatureExtractor
    from .ml.models import BaselineModel
    from .ml.evaluation import ModelEvaluator

    logger.info(f"Evaluating model {args.model} on dataset {args.dataset}")

    # Load model
    model = BaselineModel("Loaded")
    model.load(args.model)
    logger.info(f"Loaded model with {len(model.label_names)} labels")

    # Load feature extractor
    feature_path = Path(args.model).parent / f"{Path(args.model).stem}_features.pkl"
    if not feature_path.exists():
        logger.error(f"Feature extractor not found: {feature_path}")
        logger.info("Please ensure the feature extractor was saved alongside the model")
        sys.exit(1)

    with open(feature_path, "rb") as f:
        feature_extractor = pickle.load(f)
    logger.info(f"Loaded feature extractor with {len(feature_extractor.feature_names)} features")

    # Load dataset
    loader = DatasetLoader(args.dataset)
    resources = loader.load_resources(include_baseline=False, include_mutated=True)
    labels_dict = loader.load_labels()

    # Prepare data
    labeled_resources, label_matrix, label_names = loader.prepare_training_data(
        resources, labels_dict
    )

    # Extract features
    X, _ = feature_extractor.transform(labeled_resources)

    # Predict
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    # Evaluate
    evaluator = ModelEvaluator(label_names)
    metrics = evaluator.evaluate(label_matrix, y_pred, y_proba)
    evaluator.print_report(metrics)

    # Save report if requested
    if args.output:
        evaluator.save_report(metrics, args.output)

    logger.info(f"\nEvaluation complete!")
    logger.info(f"  F1 Score (macro): {metrics['f1_macro']:.3f}")
    logger.info(f"  F1 Score (micro): {metrics['f1_micro']:.3f}")


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

    # train-model command
    train_parser = subparsers.add_parser(
        "train-model",
        help="Train a model on a dataset",
    )
    train_parser.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset directory (e.g., dataset/exp-20240101-120000)",
    )
    train_parser.add_argument(
        "--model",
        choices=["xgboost", "random-forest"],
        default="random-forest",
        help="Model type to train",
    )
    train_parser.add_argument(
        "--output",
        default="models",
        help="Output directory for trained model",
    )
    train_parser.add_argument(
        "--name",
        help="Model name (default: model-YYYYMMDD-HHMMSS)",
    )
    train_parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data for test set (default: 0.2)",
    )
    train_parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Proportion of training data for validation (default: 0.1)",
    )
    train_parser.add_argument(
        "--max-features",
        type=int,
        default=100,
        help="Maximum number of features to extract (default: 100)",
    )
    train_parser.set_defaults(func=cmd_train_model)

    # evaluate-model command
    evaluate_parser = subparsers.add_parser(
        "evaluate-model",
        help="Evaluate a trained model",
    )
    evaluate_parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model file",
    )
    evaluate_parser.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset directory",
    )
    evaluate_parser.add_argument(
        "--output",
        help="Path to save evaluation report (JSON)",
    )
    evaluate_parser.add_argument(
        "--test-only",
        action="store_true",
        help="Evaluate on test set only (default: all data)",
    )
    evaluate_parser.set_defaults(func=cmd_evaluate_model)

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
