#!/usr/bin/env python3
"""
Data Validation Script
Validates the loaded clinical trials dataset for completeness and quality
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path


def validate_target_column(df, target_column):
    """Validate the target column"""
    issues = []
    
    if target_column not in df.columns:
        issues.append(f"❌ Target column '{target_column}' not found")
        return issues, {}
    
    target_series = df[target_column]
    target_stats = {
        'total_samples': len(target_series),
        'positive_samples': int(target_series.sum()),
        'negative_samples': int(len(target_series) - target_series.sum()),
        'missing_values': int(target_series.isnull().sum()),
        'success_rate': float(target_series.mean()),
        'unique_values': target_series.unique().tolist()
    }
    
    # Check for issues
    if target_stats['missing_values'] > 0:
        issues.append(f"⚠️ Target has {target_stats['missing_values']} missing values")
    
    if target_stats['success_rate'] < 0.05:
        issues.append(f"⚠️ Very low success rate: {target_stats['success_rate']:.1%}")
    elif target_stats['success_rate'] > 0.95:
        issues.append(f"⚠️ Very high success rate: {target_stats['success_rate']:.1%}")
    
    if len(target_stats['unique_values']) != 2:
        issues.append(f"⚠️ Target should be binary, found values: {target_stats['unique_values']}")
    
    if not issues:
        issues.append(f"✓ Target column valid: {target_stats['positive_samples']}/{target_stats['total_samples']} positive ({target_stats['success_rate']:.1%})")
    
    return issues, target_stats


def validate_required_columns(df):
    """Check for presence of expected columns"""
    
    required_columns = [
        'Trial.ID',
        'reached_phase_3_plus',
        'Trial Status',
        'Disease'
    ]
    
    important_columns = [
        'Study Keywords',
        'MeSH Term',
        'Actual Accrual (% of Target)',
        'Target Accrual',
        'Treatment Duration (Mos.)'
    ]
    
    issues = []
    missing_required = [col for col in required_columns if col not in df.columns]
    missing_important = [col for col in important_columns if col not in df.columns]
    
    if missing_required:
        issues.append(f"❌ Missing required columns: {missing_required}")
    else:
        issues.append(f"✓ All required columns present")
    
    if missing_important:
        issues.append(f"⚠️ Missing important columns: {missing_important}")
    
    return issues, {
        'total_columns': len(df.columns),
        'missing_required': missing_required,
        'missing_important': missing_important,
        'present_columns': df.columns.tolist()
    }


def validate_data_quality(df):
    """Validate overall data quality"""
    
    issues = []
    quality_stats = {}
    
    # Overall completeness
    total_cells = len(df) * len(df.columns)
    missing_cells = df.isnull().sum().sum()
    completeness = (1 - missing_cells / total_cells) * 100
    quality_stats['overall_completeness'] = float(completeness)
    
    if completeness < 70:
        issues.append(f"❌ Low data completeness: {completeness:.1f}%")
    elif completeness < 85:
        issues.append(f"⚠️ Moderate data completeness: {completeness:.1f}%")
    else:
        issues.append(f"✓ Good data completeness: {completeness:.1f}%")
    
    # Check for duplicate trials
    if 'Trial.ID' in df.columns:
        duplicates = df['Trial.ID'].duplicated().sum()
        quality_stats['duplicate_trials'] = int(duplicates)
        if duplicates > 0:
            issues.append(f"⚠️ Found {duplicates} duplicate Trial IDs")
        else:
            issues.append(f"✓ No duplicate trials found")
    
    # Columns with high missing rates
    missing_rates = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    high_missing_cols = missing_rates[missing_rates > 50].head(5)
    quality_stats['high_missing_columns'] = high_missing_cols.to_dict()
    
    if len(high_missing_cols) > 0:
        issues.append(f"⚠️ {len(high_missing_cols)} columns with >50% missing data")
    
    # Completely empty columns
    empty_cols = [col for col in df.columns if df[col].isnull().all()]
    quality_stats['empty_columns'] = empty_cols
    
    if empty_cols:
        issues.append(f"⚠️ {len(empty_cols)} completely empty columns: {empty_cols[:3]}{'...' if len(empty_cols) > 3 else ''}")
    
    return issues, quality_stats


def validate_feature_categories(df):
    """Validate feature categories for modeling"""
    
    issues = []
    category_stats = {}
    
    # Count features by category
    categories = {
        'Sponsor': len([col for col in df.columns if col.lower().startswith('sponsor_')]),
        'Patient': len([col for col in df.columns if any(col.lower().startswith(p) for p in ['patient_', 'includes_', 'gender_', 'age_group_'])]),
        'Enrollment': len([col for col in df.columns if any(k in col.lower() for k in ['enrollment', 'accrual', 'duration', 'study_'])]),
        'Endpoint': len([col for col in df.columns if col.lower().startswith('endpoint_')]),
        'Biomarker': len([col for col in df.columns if col.lower().startswith(('biomarker_', 'has_'))]),
        'Outcome': len([col for col in df.columns if col.lower().startswith('outcome_')]),
        'Text_Features': len([col for col in df.columns if col.lower().startswith(('keyword_', 'mesh_'))]),
    }
    
    category_stats = {k: v for k, v in categories.items() if v > 0}
    total_model_features = sum(category_stats.values())
    
    if total_model_features < 50:
        issues.append(f"⚠️ Only {total_model_features} model-ready features found")
    elif total_model_features > 200:
        issues.append(f"⚠️ High feature count: {total_model_features} (may need reduction)")
    else:
        issues.append(f"✓ Good feature count: {total_model_features} features")
    
    # Check text features for content
    if 'Study Keywords' in df.columns:
        keywords_coverage = (df['Study Keywords'].notna()).mean() * 100
        if keywords_coverage < 30:
            issues.append(f"⚠️ Low keyword coverage: {keywords_coverage:.1f}%")
    
    if 'MeSH Term' in df.columns:
        mesh_coverage = (df['MeSH Term'].notna()).mean() * 100
        if mesh_coverage < 30:
            issues.append(f"⚠️ Low MeSH term coverage: {mesh_coverage:.1f}%")
    
    return issues, category_stats


def validate_numerical_ranges(df):
    """Validate numerical features for reasonable ranges"""
    
    issues = []
    range_stats = {}
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        if df[col].notna().sum() > 0:  # Skip completely empty columns
            col_stats = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'negative_values': int((df[col] < 0).sum()),
                'zero_values': int((df[col] == 0).sum()),
                'outliers': int(np.abs((df[col] - df[col].mean()) / df[col].std()) > 3).sum() if df[col].std() > 0 else 0
            }
            
            # Check for suspicious values
            if 'accrual' in col.lower() and col_stats['max'] > 1000:
                issues.append(f"⚠️ {col}: Very high accrual values (max: {col_stats['max']})")
            
            if 'duration' in col.lower() and col_stats['max'] > 120:
                issues.append(f"⚠️ {col}: Very long duration (max: {col_stats['max']} months)")
            
            if col_stats['negative_values'] > 0 and 'accrual' in col.lower():
                issues.append(f"⚠️ {col}: {col_stats['negative_values']} negative values")
            
            range_stats[col] = col_stats
    
    if not issues:
        issues.append(f"✓ Numerical ranges appear reasonable")
    
    return issues, range_stats


def generate_validation_report(df, config):
    """Generate comprehensive validation report"""
    
    print("🔍 Validating clinical trials dataset...")
    
    all_issues = []
    validation_stats = {}
    
    # Basic dataset info
    basic_info = {
        'shape': df.shape,
        'memory_usage_mb': float(df.memory_usage(deep=True).sum() / 1024 / 1024),
        'dtypes': df.dtypes.value_counts().to_dict()
    }
    validation_stats['basic_info'] = basic_info
    
    print(f"📊 Dataset: {df.shape[0]} trials × {df.shape[1]} columns ({basic_info['memory_usage_mb']:.1f} MB)")
    
    # Run validations
    target_issues, target_stats = validate_target_column(df, config.target_column)
    all_issues.extend(target_issues)
    validation_stats['target'] = target_stats
    
    column_issues, column_stats = validate_required_columns(df)
    all_issues.extend(column_issues)
    validation_stats['columns'] = column_stats
    
    quality_issues, quality_stats = validate_data_quality(df)
    all_issues.extend(quality_issues)
    validation_stats['quality'] = quality_stats
    
    feature_issues, feature_stats = validate_feature_categories(df)
    all_issues.extend(feature_issues)
    validation_stats['features'] = feature_stats
    
    range_issues, range_stats = validate_numerical_ranges(df)
    all_issues.extend(range_issues)
    validation_stats['numerical_ranges'] = range_stats
    
    # Summary
    error_count = len([issue for issue in all_issues if issue.startswith('❌')])
    warning_count = len([issue for issue in all_issues if issue.startswith('⚠️')])
    success_count = len([issue for issue in all_issues if issue.startswith('✓')])
    
    validation_summary = {
        'total_issues': len(all_issues),
        'errors': error_count,
        'warnings': warning_count,
        'successes': success_count,
        'overall_status': 'FAIL' if error_count > 0 else 'WARN' if warning_count > 0 else 'PASS'
    }
    validation_stats['summary'] = validation_summary
    
    return all_issues, validation_stats


def save_validation_report(issues, stats, output_path):
    """Save validation report to file"""
    
    report_lines = [
        "=" * 80,
        "CLINICAL TRIALS DATASET VALIDATION REPORT",
        "=" * 80,
        "",
        f"Generated: {pd.Timestamp.now()}",
        f"Dataset Shape: {stats['basic_info']['shape']}",
        f"Memory Usage: {stats['basic_info']['memory_usage_mb']:.1f} MB",
        "",
        "VALIDATION RESULTS:",
        "-" * 40
    ]
    
    for issue in issues:
        report_lines.append(issue)
    
    report_lines.extend([
        "",
        "SUMMARY:",
        "-" * 40,
        f"Overall Status: {stats['summary']['overall_status']}",
        f"Errors: {stats['summary']['errors']}",
        f"Warnings: {stats['summary']['warnings']}",
        f"Successes: {stats['summary']['successes']}",
        "",
        "FEATURE CATEGORIES:",
        "-" * 40
    ])
    
    for category, count in stats['features'].items():
        report_lines.append(f"{category}: {count} features")
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))


if __name__ == "__main__":
    # Load inputs
    with open(snakemake.input.raw_data, 'rb') as f:
        df = pickle.load(f)
    
    with open(snakemake.input.config, 'rb') as f:
        config = pickle.load(f)
    
    # Run validation
    issues, stats = generate_validation_report(df, config)
    
    # Print results to console
    print("\n" + "="*50)
    print("VALIDATION RESULTS:")
    print("="*50)
    for issue in issues:
        print(issue)
    
    print(f"\n📋 Summary: {stats['summary']['overall_status']} - {stats['summary']['errors']} errors, {stats['summary']['warnings']} warnings")
    
    # Save report
    save_validation_report(issues, stats, snakemake.output.validation_report)
    
    # Exit with error if validation failed
    if stats['summary']['errors'] > 0:
        print(f"\n❌ Validation FAILED - check {snakemake.output.validation_report}")
        exit(1)
    else:
        print(f"✓ Validation report saved to: {snakemake.output.validation_report}")