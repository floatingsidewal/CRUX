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
