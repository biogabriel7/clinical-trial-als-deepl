#!/usr/bin/env python3
"""
Script 2: Feature Extraction
"""

import pandas as pd
import numpy as np
import pickle
from collections import Counter


def extract_keywords_features(df):
    """Extract individual keywords from Study Keywords column"""
    
    if 'Study Keywords' not in df.columns:
        return pd.DataFrame()
    
    # Extract all individual keywords
    all_keywords = []
    for keywords_str in df['Study Keywords'].dropna():
        keywords = [kw.strip().lower() for kw in str(keywords_str).split(';')]
        all_keywords.extend(keywords)
    
    # Count keyword frequency and create features for common keywords
    keyword_counts = Counter(all_keywords)
    common_keywords = [kw for kw, count in keyword_counts.items() if count >= 5]
    
    # Create keyword features
    keyword_features = pd.DataFrame(index=df.index)
    for keyword in common_keywords:
        feature_name = f"keyword_{keyword.replace(' ', '_').replace('/', '_')}"
        keyword_features[feature_name] = df['Study Keywords'].apply(
            lambda x: 1 if pd.notna(x) and keyword in str(x).lower() else 0
        )
    
    return keyword_features


def extract_mesh_features(df):
    """Extract individual MeSH terms"""
    
    if 'MeSH Term' not in df.columns:
        return pd.DataFrame()
    
    # Extract all individual MeSH terms
    all_mesh = []
    for mesh_str in df['MeSH Term'].dropna():
        terms = [term.strip() for term in str(mesh_str).split(';')]
        all_mesh.extend(terms)
    
    # Count term frequency and create features for common terms
    mesh_counts = Counter(all_mesh)
    common_mesh = [term for term, count in mesh_counts.items() if count >= 5]
    
    # Create MeSH features
    mesh_features = pd.DataFrame(index=df.index)
    for term in common_mesh:
        feature_name = f"mesh_{term.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '').lower()}"
        mesh_features[feature_name] = df['MeSH Term'].apply(
            lambda x: 1 if pd.notna(x) and term in str(x) else 0
        )
    
    return mesh_features


def get_comprehensive_features(df):
    """Extract all features from the comprehensive 120-column dataset"""
    
    feature_sets = []
    feature_names = []
    
    # 1. SPONSOR FEATURES
    sponsor_features = [
        'sponsor_is_industry', 'sponsor_is_academic', 'sponsor_is_government', 'sponsor_is_nonprofit',
        'sponsor_type_complexity', 'sponsor_country_usa', 'sponsor_country_europe', 'sponsor_country_asia',
        'sponsor_country_diversity', 'sponsor_state_california', 'sponsor_state_massachusetts', 
        'sponsor_state_major_biotech', 'sponsor_trial_count', 'sponsor_is_major_pharma',
        'has_multiple_sponsors', 'has_supporting_urls', 'supporting_url_count'
    ]
    
    # 2. PATIENT CHARACTERISTICS
    patient_features = [
        'patient_min_age', 'patient_max_age', 'patient_age_range', 'includes_children',
        'includes_adults', 'includes_elderly', 'age_group_breadth', 'age_group_children',
        'age_group_adults', 'age_group_older_adults', 'gender_both',
        'gender_male_only', 'gender_female_only', 'gender_restricted',
        'inclusion_has_diagnosis', 'inclusion_has_age_requirement', 'inclusion_has_consent',
        'inclusion_has_functional_requirement', 'inclusion_criteria_length',
        'exclusion_has_comorbidity', 'exclusion_has_medication', 'exclusion_has_pregnancy',
        'exclusion_has_cognitive', 'exclusion_criteria_length', 'population_has_specific_type',
        'population_has_severity', 'population_has_duration', 'population_description_length',
        'total_criteria_length', 'demographic_restrictions'
    ]
    
    # 3. ENROLLMENT & TIMELINE
    enrollment_features = [
        'Actual Accrual (% of Target)', 'Actual Accrual (No. of patients)', 'Target Accrual',
        'Treatment Duration (Mos.)', 'Enrollment Duration (Mos.)', 'Pts/Site/Mo',
        'study_start_date', 'study_duration_days'
    ]
    
    # 4. ENDPOINT FEATURES
    endpoint_features = [
        'endpoint_efficacy', 'endpoint_safety', 'endpoint_functional', 'endpoint_biomarker',
        'endpoint_survival', 'endpoint_pharmacokinetic', 'endpoint_total_categories'
    ]

    # 5. BIOMARKER FEATURES
    biomarker_features = [
        'biomarker_diagnostic', 'biomarker_predictive', 'biomarker_prognostic', 
        'biomarker_predisposing', 'biomarker_total_uses', 'biomarker_count',
        'has_sod1', 'has_albumin', 'has_neurofilament', 'has_nfl', 'has_tau', 
        'has_tdp-43', 'has_c9orf72', 'has_creatinine', 'has_inflammatory'
    ]

    # 6. OUTCOME FEATURES  
    outcome_features = [
        'outcome_positive', 'outcome_negative', 'outcome_terminated', 'outcome_completed'
    ]

    # Collect all numeric features
    all_numeric_features = sponsor_features + patient_features + enrollment_features + endpoint_features + biomarker_features + outcome_features
    existing_numeric = [col for col in all_numeric_features if col in df.columns]
    
    if existing_numeric:
        numeric_df = df[existing_numeric].fillna(0)
        feature_sets.append(numeric_df)
        feature_names.extend(existing_numeric)
    
    # 7. CATEGORICAL FEATURES
    categorical_features = [
        'sponsor_experience_level', 'inclusion_criteria_complexity', 
        'exclusion_criteria_complexity', 'population_specificity', 
        'patient_selectivity', 'study_duration_category'
    ]
    
    for col in categorical_features:
        if col in df.columns and df[col].notna().sum() > 0:
            encoded = pd.get_dummies(df[col], prefix=col, dummy_na=False)
            feature_sets.append(encoded)
            feature_names.extend(encoded.columns)
    
    # 8. EXISTING CATEGORICAL FEATURES
    existing_categorical = ['Trial Status', 'Disease', 'endpoint_primary_category']
    for col in existing_categorical:
        if col in df.columns:
            if col == 'Disease':
                value_counts = df[col].value_counts()
                top_categories = value_counts.head(10).index.tolist()
                df_temp = df[col].apply(lambda x: x if x in top_categories else 'Other')
                encoded = pd.get_dummies(df_temp, prefix=col.replace(' ', '_'), dummy_na=True)
            else:
                encoded = pd.get_dummies(df[col], prefix=col.replace(' ', '_'), dummy_na=True)
            
            feature_sets.append(encoded)
            feature_names.extend(encoded.columns)
    
    # 9. TEXT FEATURES
    keyword_features = extract_keywords_features(df)
    if len(keyword_features.columns) > 0:
        feature_sets.append(keyword_features)
        feature_names.extend(keyword_features.columns)
    
    mesh_features = extract_mesh_features(df)
    if len(mesh_features.columns) > 0:
        feature_sets.append(mesh_features)
        feature_names.extend(mesh_features.columns)
    
    if not feature_sets:
        raise ValueError("No features could be extracted from the dataset")
    
    # Combine all features
    X_combined = pd.concat(feature_sets, axis=1)
    
    return X_combined, feature_names


if __name__ == "__main__":
    # Load inputs
    with open(snakemake.input.raw_data, 'rb') as f:
        df = pickle.load(f)
    
    with open(snakemake.input.config, 'rb') as f:
        config = pickle.load(f)
    
    # Extract features
    print("Extracting comprehensive features...")
    X_combined, feature_names = get_comprehensive_features(df)
    print(f"Total features extracted: {len(feature_names)} | Shape: {X_combined.shape}")
    
    # Save outputs
    feature_data = {
        'X_combined': X_combined,
        'feature_names': feature_names,
        'target': df[config.target_column]
    }
    
    with open(snakemake.output.features, 'wb') as f:
        pickle.dump(feature_data, f)
    
    print("✓ Feature extraction complete")