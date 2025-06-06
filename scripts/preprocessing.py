#!/usr/bin/env python3
"""
Script 3: Data Preprocessing
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif


def smart_feature_selection(X, y, feature_names, config):
    """Apply intelligent feature selection"""
    
    if not config.use_feature_selection:
        return X, feature_names
    
    print(f"Applying feature selection: {X.shape[1]} -> {config.max_features} features")
    
    # Use mutual information for feature selection
    selector = SelectKBest(score_func=mutual_info_classif, k=config.max_features)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]
    
    print(f"Selected {len(selected_features)} most informative features")
    return X_selected, selected_features


def remove_phase_leakage(X_scaled, feature_names, config):
    """Remove trial phase features that leak target information"""
    
    if not config.exclude_phase_features:
        return X_scaled, feature_names
    
    # Identify phase-related features that might leak information
    leakage_patterns = ['trial_phase', 'phase_', '_phase', 'completed', 'terminated']
    
    phase_features = []
    for i, name in enumerate(feature_names):
        name_lower = name.lower()
        if any(pattern in name_lower for pattern in leakage_patterns):
            # Keep outcome features as they are legitimate predictors
            if not name_lower.startswith('outcome_') and not name_lower.startswith('endpoint_'):
                phase_features.append(i)
    
    if phase_features:
        mask = [i for i in range(len(feature_names)) if i not in phase_features]
        X_filtered = X_scaled[:, mask]
        feature_names_filtered = [feature_names[i] for i in mask]
        print(f"Removed {len(phase_features)} potential leakage features")
        return X_filtered, feature_names_filtered
    else:
        return X_scaled, feature_names


def analyze_feature_categories(feature_names):
    """Analyze the distribution of feature categories"""
    
    categories = {
        'Sponsor': len([f for f in feature_names if f.lower().startswith('sponsor_')]),
        'Patient': len([f for f in feature_names if any(f.lower().startswith(p) for p in ['patient_', 'includes_', 'gender_', 'age_group_', 'inclusion_', 'exclusion_', 'population_', 'demographic_'])]),
        'Enrollment': len([f for f in feature_names if any(k in f.lower() for k in ['enrollment', 'accrual', 'duration', 'study_', 'pts/site'])]),
        'Endpoint': len([f for f in feature_names if f.lower().startswith('endpoint_')]),
        'Biomarker': len([f for f in feature_names if f.lower().startswith(('biomarker_', 'has_'))]),
        'Outcome': len([f for f in feature_names if f.lower().startswith('outcome_')]),
        'Keywords': len([f for f in feature_names if f.lower().startswith('keyword_')]),
        'MeSH': len([f for f in feature_names if f.lower().startswith('mesh_')]),
        'Categorical': len([f for f in feature_names if any(cat in f.lower() for cat in ['trial_status', 'disease', 'therapeutic_area', '_complexity', '_specificity', '_selectivity', '_category'])]),
    }
    
    # Remove empty categories and print summary
    active_categories = {k: v for k, v in categories.items() if v > 0}
    print(f"Feature categories: " + " | ".join([f"{cat}: {count}" for cat, count in active_categories.items()]))
    
    return active_categories


def preprocess_comprehensive_data(feature_data, config):
    """Comprehensive preprocessing pipeline"""
    
    # Extract data
    X_combined = feature_data['X_combined']
    feature_names = feature_data['feature_names']
    y = feature_data['target']
    
    # Feature selection: remove low-variance features
    variance_selector = VarianceThreshold(threshold=0.005)
    X_variance_filtered = variance_selector.fit_transform(X_combined.values)
    selected_features = [feature_names[i] for i in range(len(feature_names)) 
                        if i < len(variance_selector.get_support()) and variance_selector.get_support()[i]]
    
    print(f"Features after variance filtering: {len(selected_features)} (removed {len(feature_names) - len(selected_features)})")
    
    # Convert to numpy and standardize
    X_array = X_variance_filtered.astype(np.float32)
    y_array = y.values.astype(np.float32)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_array)
    
    # Apply smart feature selection AFTER scaling
    X_scaled, selected_features = smart_feature_selection(X_scaled, y_array, selected_features, config)
    
    # Create phase groups for stratification
    phase_groups = np.array(['Unknown'] * len(y))  # Simplified for now
    
    return X_scaled, y_array, phase_groups, selected_features, scaler


if __name__ == "__main__":
    # Load inputs
    with open(snakemake.input.features, 'rb') as f:
        feature_data = pickle.load(f)
    
    with open(snakemake.input.config, 'rb') as f:
        config = pickle.load(f)
    
    # Run preprocessing
    print("Running comprehensive preprocessing...")
    X, y, phase_groups, feature_names, scaler = preprocess_comprehensive_data(feature_data, config)
    
    # Remove phase leakage features
    X, feature_names = remove_phase_leakage(X, feature_names, config)
    
    # Analyze feature categories
    feature_categories = analyze_feature_categories(feature_names)
    
    print(f"Final dataset: {X.shape} | Target: {int(y.sum())}/{len(y)} Phase 3+ ({y.mean()*100:.1f}%)")
    
    # Save outputs
    processed_data = {
        'X': X,
        'y': y,
        'phase_groups': phase_groups,
        'feature_categories': feature_categories
    }
    
    with open(snakemake.output.processed_data, 'wb') as f:
        pickle.dump(processed_data, f)
    
    with open(snakemake.output.scaler, 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(snakemake.output.feature_names, 'wb') as f:
        pickle.dump(feature_names, f)
    
    print("✓ Preprocessing complete")