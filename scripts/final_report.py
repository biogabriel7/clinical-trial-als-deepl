#!/usr/bin/env python3
"""
Script 8: Final Report Generation
Creates visualizations, final summary, and exports results
"""

import pickle
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def create_performance_visualizations(cv_results, training_histories, output_path):
    """Create comprehensive performance visualizations"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('ALS Clinical Trials Model - Performance Analysis', fontsize=16, fontweight='bold')
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(cv_results)
    
    # 1. Cross-validation performance metrics
    metrics = ['auc', 'f1_score', 'balanced_accuracy', 'precision', 'recall']
    metric_means = [results_df[metric].mean() for metric in metrics]
    metric_stds = [results_df[metric].std() for metric in metrics]
    
    axes[0, 0].bar(range(len(metrics)), metric_means, yerr=metric_stds, capsize=5, alpha=0.7)
    axes[0, 0].set_xticks(range(len(metrics)))
    axes[0, 0].set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
    axes[0, 0].set_title('Cross-Validation Performance Metrics')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. AUC distribution across folds
    axes[0, 1].boxplot(results_df['auc'], labels=['AUC'])
    axes[0, 1].scatter([1] * len(results_df), results_df['auc'], alpha=0.6, c='red')
    axes[0, 1].set_title('AUC Distribution Across Folds')
    axes[0, 1].set_ylabel('AUC Score')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Training epochs per fold
    axes[0, 2].bar(results_df['fold'], results_df['epochs'], alpha=0.7)
    axes[0, 2].set_title('Training Epochs per Fold')
    axes[0, 2].set_xlabel('Fold')
    axes[0, 2].set_ylabel('Epochs')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Training history for best fold
    if training_histories:
        best_fold_idx = results_df['auc'].idxmax()
        best_history = training_histories[best_fold_idx]
        
        epochs_range = range(1, len(best_history['train_loss']) + 1)
        axes[1, 0].plot(epochs_range, best_history['train_loss'], 'b-', label='Train Loss')
        axes[1, 0].plot(epochs_range, best_history['val_loss'], 'r-', label='Val Loss')
        axes[1, 0].set_title(f'Training History - Best Fold ({best_fold_idx + 1})')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Validation metrics for best fold
        axes[1, 1].plot(epochs_range, best_history['val_f1'], 'g-', label='Val F1')
        axes[1, 1].plot(epochs_range, best_history['val_auc'], 'orange', label='Val AUC')
        axes[1, 1].set_title(f'Validation Metrics - Best Fold ({best_fold_idx + 1})')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Performance vs Model Complexity
    axes[1, 2].scatter(results_df['model_parameters'], results_df['auc'], alpha=0.7)
    axes[1, 2].set_title('Performance vs Model Complexity')
    axes[1, 2].set_xlabel('Model Parameters')
    axes[1, 2].set_ylabel('AUC Score')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Performance visualizations saved to: {output_path}")


def create_feature_importance_analysis(best_fold_data, feature_categories, output_dir):
    """Create detailed feature importance analysis"""
    
    if not best_fold_data or 'feature_importance' not in best_fold_data:
        print("⚠️ No feature importance data available")
        return
    
    feature_importance = best_fold_data['feature_importance']
    
    # Create feature importance CSV
    importance_df = pd.DataFrame({
        'feature': feature_importance.index,
        'importance': feature_importance.values
    })
    
    # Add category information
    def categorize_feature(feature_name):
        feature_lower = feature_name.lower()
        if feature_lower.startswith('sponsor_'):
            return 'Sponsor'
        elif any(feature_lower.startswith(p) for p in ['patient_', 'includes_', 'gender_', 'age_group_', 'inclusion_', 'exclusion_', 'population_', 'demographic_']):
            return 'Patient'
        elif any(k in feature_lower for k in ['enrollment', 'accrual', 'duration', 'study_', 'pts/site']):
            return 'Enrollment'
        elif feature_lower.startswith('endpoint_'):
            return 'Endpoint'
        elif feature_lower.startswith(('biomarker_', 'has_')):
            return 'Biomarker'
        elif feature_lower.startswith('outcome_'):
            return 'Outcome'
        elif feature_lower.startswith('keyword_'):
            return 'Keywords'
        elif feature_lower.startswith('mesh_'):
            return 'MeSH'
        elif any(cat in feature_lower for cat in ['trial_status', 'disease', 'therapeutic_area', '_complexity', '_specificity', '_selectivity', '_category']):
            return 'Categorical'
        else:
            return 'Other'
    
    importance_df['category'] = importance_df['feature'].apply(categorize_feature)
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Save feature importance CSV
    importance_path = output_dir / 'feature_importance.csv'
    importance_df.to_csv(importance_path, index=False)
    
    print(f"✓ Feature importance analysis saved to: {importance_path}")
    
    return importance_df


def generate_performance_metrics_json(cv_results, best_fold_data, feature_categories, output_path):
    """Generate comprehensive performance metrics in JSON format"""
    
    results_df = pd.DataFrame(cv_results)
    
    # Calculate summary statistics
    performance_summary = {
        'cross_validation_summary': {
            'n_folds': len(cv_results),
            'metrics': {
                'auc': {
                    'mean': float(results_df['auc'].mean()),
                    'std': float(results_df['auc'].std()),
                    'min': float(results_df['auc'].min()),
                    'max': float(results_df['auc'].max())
                },
                'f1_score': {
                    'mean': float(results_df['f1_score'].mean()),
                    'std': float(results_df['f1_score'].std()),
                    'min': float(results_df['f1_score'].min()),
                    'max': float(results_df['f1_score'].max())
                },
                'balanced_accuracy': {
                    'mean': float(results_df['balanced_accuracy'].mean()),
                    'std': float(results_df['balanced_accuracy'].std()),
                    'min': float(results_df['balanced_accuracy'].min()),
                    'max': float(results_df['balanced_accuracy'].max())
                },
                'precision': {
                    'mean': float(results_df['precision'].mean()),
                    'std': float(results_df['precision'].std()),
                    'min': float(results_df['precision'].min()),
                    'max': float(results_df['precision'].max())
                },
                'recall': {
                    'mean': float(results_df['recall'].mean()),
                    'std': float(results_df['recall'].std()),
                    'min': float(results_df['recall'].min()),
                    'max': float(results_df['recall'].max())
                }
            }
        },
        'best_fold_performance': {
            'fold_number': int(results_df.loc[results_df['auc'].idxmax(), 'fold']),
            'auc': float(results_df['auc'].max()),
            'f1_score': float(results_df.loc[results_df['auc'].idxmax(), 'f1_score']),
            'balanced_accuracy': float(results_df.loc[results_df['auc'].idxmax(), 'balanced_accuracy']),
            'optimal_threshold': float(results_df.loc[results_df['auc'].idxmax(), 'optimal_threshold'])
        },
        'training_summary': {
            'average_epochs': float(results_df['epochs'].mean()),
            'min_epochs': int(results_df['epochs'].min()),
            'max_epochs': int(results_df['epochs'].max()),
            'average_model_parameters': float(results_df['model_parameters'].mean())
        },
        'feature_summary': {
            'total_features': sum(feature_categories.values()),
            'feature_categories': dict(feature_categories)
        },
        'performance_assessment': {
            'level': 'EXCELLENT' if results_df['auc'].mean() >= 0.75 else 
                    'GOOD' if results_df['auc'].mean() >= 0.65 else 
                    'MODERATE' if results_df['auc'].mean() >= 0.55 else 'LIMITED',
            'clinical_utility': results_df['auc'].mean() >= 0.65,
            'stability': results_df['auc'].std() < 0.05
        }
    }
    
    # Add top features if available
    if best_fold_data and 'feature_importance' in best_fold_data:
        feature_importance = best_fold_data['feature_importance']
        performance_summary['top_features'] = {
            'top_10': [
                {
                    'feature': feature,
                    'importance': float(importance),
                    'rank': i + 1
                }
                for i, (feature, importance) in enumerate(feature_importance.head(10).items())
            ]
        }
    
    # Save JSON
    with open(output_path, 'w') as f:
        json.dump(performance_summary, f, indent=2)
    
    print(f"✓ Performance metrics saved to: {output_path}")
    
    return performance_summary


def print_final_comprehensive_results(cv_results, best_fold_data, feature_categories):
    """Print the final comprehensive results with enhanced formatting"""
    
    # Load CV functions to use the existing print function
    try:
        # Try to use the existing comprehensive results function
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
        print(f"  Balanced Accuracy: {results_df['balanced_accuracy'].mean():.3f}")
        
        # Best fold
        best_fold_idx = results_df['auc'].idxmax()
        best_fold = results_df.iloc[best_fold_idx]
        
        print(f"\nBEST FOLD PERFORMANCE (Fold {best_fold['fold']}):")
        print(f"  AUC:               {best_fold['auc']:.4f}")
        print(f"  F1-Score:          {best_fold['f1_score']:.4f}")
        print(f"  Optimal Threshold: {best_fold['optimal_threshold']:.3f}")
        
        # Feature importance if available
        if best_fold_data and 'feature_importance' in best_fold_data:
            feature_importance = best_fold_data['feature_importance']
            print(f"\n=== FEATURE IMPORTANCE ANALYSIS ===")
            
            # Category-wise importance
            total_importance = feature_importance.sum()
            for category, count in feature_categories.items():
                if count > 0:
                    cat_features = []
                    for feature in feature_importance.index:
                        feature_lower = feature.lower()
                        if ((category == 'Sponsor' and feature_lower.startswith('sponsor_')) or
                            (category == 'Patient' and any(feature_lower.startswith(p) for p in ['patient_', 'includes_', 'gender_', 'age_group_', 'inclusion_', 'exclusion_', 'population_', 'demographic_'])) or
                            (category == 'Enrollment' and any(k in feature_lower for k in ['enrollment', 'accrual', 'duration', 'study_', 'pts/site'])) or
                            (category == 'Endpoint' and feature_lower.startswith('endpoint_')) or
                            (category == 'Biomarker' and feature_lower.startswith(('biomarker_', 'has_'))) or
                            (category == 'Outcome' and feature_lower.startswith('outcome_')) or
                            (category == 'Keywords' and feature_lower.startswith('keyword_')) or
                            (category == 'MeSH' and feature_lower.startswith('mesh_')) or
                            (category == 'Categorical' and any(cat in feature_lower for cat in ['trial_status', 'disease', '_complexity', '_specificity', '_selectivity', '_category']))):
                            cat_features.append(feature)
                    
                    if cat_features:
                        cat_importance = feature_importance[cat_features].sum()
                        print(f"{category:<12}: {cat_importance/total_importance*100:5.1f}% ({len(cat_features):3d} features)")
            
            # Top features
            print(f"\nTOP 15 MOST IMPORTANT FEATURES:")
            for i, (feature, importance) in enumerate(feature_importance.head(15).items(), 1):
                clean_name = feature.replace('endpoint_', '').replace('biomarker_', '').replace('outcome_', '')
                clean_name = clean_name.replace('sponsor_', '').replace('patient_', '').replace('keyword_', '')
                clean_name = clean_name.replace('mesh_', '').replace('has_', '').replace('_', ' ').title()
                print(f"  {i:2d}. {clean_name:<35} - {importance:.4f}")
        
        # Model info
        total_features = sum(feature_categories.values())
        print(f"\nMODEL INFO:")
        print(f"  Total Features:    {total_features}")
        print(f"  Model Parameters:  {best_fold['model_parameters']:,}")
        print(f"  Average Epochs:    {results_df['epochs'].mean():.1f}")
        
    except Exception as e:
        print(f"Error in printing results: {e}")
        # Fallback simple print
        print("\nBasic Results Summary:")
        for i, result in enumerate(cv_results):
            print(f"Fold {i+1}: AUC={result['auc']:.3f}, F1={result['f1_score']:.3f}")


def generate_final_report():
    """Generate comprehensive final report with all outputs"""
    
    # Load all required data
    with open(snakemake.input.results, 'rb') as f:
        cv_results = pickle.load(f)
    
    with open(snakemake.input.best_fold, 'rb') as f:
        best_fold_data = pickle.load(f)
    
    with open(snakemake.input.config, 'rb') as f:
        config = pickle.load(f)
    
    # Create output directory structure
    output_dir = Path(snakemake.output.final_results).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("📊 Generating comprehensive final report...")
    
    # Extract feature categories from best fold data or config
    feature_categories = getattr(config, 'feature_categories', {
        'Sponsor': 8, 'Patient': 26, 'Enrollment': 8, 'Endpoint': 8,
        'Biomarker': 7, 'Outcome': 3, 'Keywords': 10, 'MeSH': 21, 'Categorical': 24
    })
    
    # Print comprehensive results to console
    print_final_comprehensive_results(cv_results, best_fold_data, feature_categories)
    
    # Generate visualizations
    training_histories = []
    if best_fold_data and 'training_history' in best_fold_data:
        training_histories = [best_fold_data['training_history']]
    
    create_performance_visualizations(
        cv_results, training_histories, snakemake.output.plots
    )
    
    # Generate feature importance analysis
    importance_df = create_feature_importance_analysis(
        best_fold_data, feature_categories, output_dir
    )
    
    # Generate performance metrics JSON
    performance_summary = generate_performance_metrics_json(
        cv_results, best_fold_data, feature_categories, snakemake.output.metrics
    )
    
    # Save comprehensive final results
    final_results = {
        'cv_results': cv_results,
        'best_fold_data': best_fold_data,
        'performance_summary': performance_summary,
        'feature_categories': feature_categories,
        'config_summary': {
            'device': config.device,
            'splits': config.splits,
            'batch_size': config.batch_size,
            'lr': config.lr,
            'dropout': config.dropout,
            'early_stopping_patience': config.early_stopping_patience
        }
    }
    
    with open(snakemake.output.final_results, 'wb') as f:
        pickle.dump(final_results, f)
    
    print(f"\n✅ FINAL REPORT GENERATION COMPLETE!")
    print(f"\nGenerated outputs:")
    print(f"  📁 {snakemake.output.final_results} - Complete results")
    print(f"  📊 {snakemake.output.feature_importance} - Feature importance rankings")  
    print(f"  📋 {snakemake.output.metrics} - Model performance metrics")
    print(f"  📈 {snakemake.output.plots} - Training visualizations")


if __name__ == "__main__":
    generate_final_report()