#!/usr/bin/env python3
"""
Script 7: Cross-validation Functions  
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold, train_test_split


def run_comprehensive_cross_validation(X, y, phase_groups, feature_names, config, model_classes, training_functions):
    """Run cross-validation with comprehensive evaluation"""
    
    print(f"Starting {config.splits}-fold cross-validation")
    print(f"Dataset: {len(y)} trials, {X.shape[1]} features, {int(y.sum())}/{len(y)} Phase 3+ ({y.mean()*100:.1f}%)")
    
    # Extract training functions
    train_model = training_functions['train_comprehensive_model']
    evaluate_model = training_functions['evaluate_comprehensive_model']
    lrp_analysis = training_functions['perform_comprehensive_lrp_analysis']
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=config.splits, shuffle=True, random_state=42)
    
    cv_results = []
    best_auc = 0
    best_fold_data = None
    all_training_histories = []
    
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\nFold {fold}/{config.splits}: ", end="")
        
        # Split data
        X_train_val, X_test = X[train_val_idx], X[test_idx]
        y_train_val, y_test = y[train_val_idx], y[test_idx]
        
        # Further split train_val into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
        )
        
        # Train model
        model, epochs, best_val_f1, training_history = train_model(
            X_train, y_train, X_val, y_val, config, model_classes
        )
        all_training_histories.append(training_history)
        
        # Evaluate model
        auc, ap, accuracy, balanced_acc, precision, recall, f1, best_thresh, probs = evaluate_model(
            model, X_test, y_test, config
        )
        
        # Store results
        fold_results = {
            'fold': fold,
            'auc': auc,
            'average_precision': ap,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'optimal_threshold': best_thresh,
            'epochs': epochs,
            'validation_f1': best_val_f1,
            'n_test_samples': len(y_test),
            'test_positives': int(y_test.sum()),
            'model_parameters': model.n_parameters
        }
        cv_results.append(fold_results)
        
        print(f"AUC: {auc:.3f}, F1: {f1:.3f}, Bal.Acc: {balanced_acc:.3f}, Epochs: {epochs}")
        
        # Store best fold for detailed analysis
        if auc > best_auc:
            best_auc = auc
            feature_importance, relevance_scores, _ = lrp_analysis(
                model, X_test, y_test, feature_names, config
            )
            best_fold_data = {
                'model': model,
                'X_test': X_test,
                'y_test': y_test,
                'probs': probs,
                'feature_importance': feature_importance,
                'relevance_scores': relevance_scores,
                'fold_results': fold_results,
                'training_history': training_history
            }
    
    return cv_results, best_fold_data, all_training_histories


def analyze_comprehensive_feature_importance(feature_importance, feature_categories):
    """Analyze feature importance by categories"""
    
    print(f"\n=== FEATURE IMPORTANCE ANALYSIS ===")
    
    # Category-wise importance
    category_importance = {}
    total_importance = feature_importance.sum()
    
    for category, count in feature_categories.items():
        if count > 0:
            # Find features belonging to this category
            cat_features = []
            for feature in feature_importance.index:
                feature_lower = feature.lower()
                if category == 'Sponsor' and feature_lower.startswith('sponsor_'):
                    cat_features.append(feature)
                elif category == 'Patient' and any(feature_lower.startswith(p) for p in ['patient_', 'includes_', 'gender_', 'age_group_', 'inclusion_', 'exclusion_', 'population_', 'demographic_']):
                    cat_features.append(feature)
                elif category == 'Enrollment' and any(k in feature_lower for k in ['enrollment', 'accrual', 'duration', 'study_', 'pts/site']):
                    cat_features.append(feature)
                elif category == 'Endpoint' and feature_lower.startswith('endpoint_'):
                    cat_features.append(feature)
                elif category == 'Biomarker' and feature_lower.startswith(('biomarker_', 'has_')):
                    cat_features.append(feature)
                elif category == 'Outcome' and feature_lower.startswith('outcome_'):
                    cat_features.append(feature)
                elif category == 'Keywords' and feature_lower.startswith('keyword_'):
                    cat_features.append(feature)
                elif category == 'MeSH' and feature_lower.startswith('mesh_'):
                    cat_features.append(feature)
                elif category == 'Categorical' and any(cat in feature_lower for cat in ['trial_status', 'disease', 'therapeutic_area', '_complexity', '_specificity', '_selectivity', '_category']):
                    cat_features.append(feature)
            
            if cat_features:
                cat_importance = feature_importance[cat_features].sum()
                category_importance[category] = cat_importance
                print(f"{category:<12}: {cat_importance/total_importance*100:5.1f}% ({len(cat_features):3d} features)")
    
    # Top features overall
    print(f"\nTOP 15 MOST IMPORTANT FEATURES:")
    for i, (feature, importance) in enumerate(feature_importance.head(15).items(), 1):
        clean_name = feature.replace('endpoint_', '').replace('biomarker_', '').replace('outcome_', '')
        clean_name = clean_name.replace('sponsor_', '').replace('patient_', '').replace('keyword_', '')
        clean_name = clean_name.replace('mesh_', '').replace('has_', '').replace('_', ' ').title()
        print(f"  {i:2d}. {clean_name:<35} - {importance:.4f}")
    
    return category_importance


def print_comprehensive_final_results(cv_results, best_fold_data, feature_categories):
    """Print comprehensive results summary"""
    
    results_df = pd.DataFrame(cv_results)
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE ALS CLINICAL TRIALS MODEL - FINAL RESULTS")
    print(f"{'='*80}")
    
    # Performance metrics
    print(f"\nCROSS-VALIDATION PERFORMANCE (5-Fold):")
    print(f"  AUC:               {results_df['auc'].mean():.4f} ± {results_df['auc'].std():.4f}")
    print(f"  Average Precision: {results_df['average_precision'].mean():.4f} ± {results_df['average_precision'].std():.4f}")
    print(f"  Balanced Accuracy: {results_df['balanced_accuracy'].mean():.4f} ± {results_df['balanced_accuracy'].std():.4f}")
    print(f"  F1-Score:          {results_df['f1_score'].mean():.4f} ± {results_df['f1_score'].std():.4f}")
    print(f"  Precision:         {results_df['precision'].mean():.4f} ± {results_df['precision'].std():.4f}")
    print(f"  Recall:            {results_df['recall'].mean():.4f} ± {results_df['recall'].std():.4f}")
    
    # Performance assessment
    mean_auc = results_df['auc'].mean()
    mean_balanced_acc = results_df['balanced_accuracy'].mean()
    
    print(f"\nPERFORMANCE ASSESSMENT:")
    if mean_auc >= 0.75:
        level = "EXCELLENT"
    elif mean_auc >= 0.65:
        level = "GOOD"
    elif mean_auc >= 0.55:
        level = "MODERATE"
    else:
        level = "LIMITED"
    
    print(f"  Performance Level: {level}")
    print(f"  Mean AUC:          {mean_auc:.3f}")
    print(f"  Balanced Accuracy: {mean_balanced_acc:.3f}")
    
    # Best fold
    best_fold_idx = results_df['auc'].idxmax()
    best_fold = results_df.iloc[best_fold_idx]
    
    print(f"\nBEST FOLD PERFORMANCE (Fold {best_fold['fold']}):")
    print(f"  AUC:               {best_fold['auc']:.4f}")
    print(f"  F1-Score:          {best_fold['f1_score']:.4f}")
    print(f"  Optimal Threshold: {best_fold['optimal_threshold']:.3f}")
    
    # Feature importance analysis
    if best_fold_data and 'feature_importance' in best_fold_data:
        feature_importance = best_fold_data['feature_importance']
        category_importance = analyze_comprehensive_feature_importance(feature_importance, feature_categories)
    
    # Model info
    total_features = sum(feature_categories.values())
    print(f"\nMODEL INFO:")
    print(f"  Total Features:    {total_features}")
    print(f"  Model Parameters:  {best_fold['model_parameters']:,}")
    print(f"  Average Epochs:    {results_df['epochs'].mean():.1f}")

    return results_df, category_importance if 'category_importance' in locals() else {}


if __name__ == "__main__":
    # Load config
    with open(snakemake.input.config, 'rb') as f:
        config = pickle.load(f)
    
    # Save CV functions
    cv_functions = {
        'run_comprehensive_cross_validation': run_comprehensive_cross_validation,
        'analyze_comprehensive_feature_importance': analyze_comprehensive_feature_importance,
        'print_comprehensive_final_results': print_comprehensive_final_results
    }
    
    with open(snakemake.output.cv_functions, 'wb') as f:
        pickle.dump(cv_functions, f)
    
    print("✓ Cross-validation functions defined")