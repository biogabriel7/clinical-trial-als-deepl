#!/usr/bin/env python3
"""
Script 7: Run Analysis
"""

import pickle
import numpy as np


def run_complete_analysis():
    """Run the complete ALS clinical trials analysis pipeline"""
    
    print("=" * 80)
    print("CLINICAL TRIALS ANALYSIS - STARTING COMPLETE EVALUATION")
    print("=" * 80)
    
    # Load all required data and functions
    with open(snakemake.input.processed_data, 'rb') as f:
        processed_data = pickle.load(f)
    
    with open(snakemake.input.feature_names, 'rb') as f:
        feature_names = pickle.load(f)
    
    with open(snakemake.input.config, 'rb') as f:
        config = pickle.load(f)
    
    with open(snakemake.input.model_classes, 'rb') as f:
        model_classes = pickle.load(f)
    
    with open(snakemake.input.training_functions, 'rb') as f:
        training_functions = pickle.load(f)
    
    with open(snakemake.input.cv_functions, 'rb') as f:
        cv_functions = pickle.load(f)
    
    # Extract data
    X = processed_data['X']
    y = processed_data['y']
    phase_groups = processed_data['phase_groups']
    feature_categories = processed_data['feature_categories']
    
    print(f"Loaded data: {X.shape[0]} trials, {X.shape[1]} features")
    print(f"Target distribution: {int(y.sum())}/{len(y)} Phase 3+ trials ({y.mean()*100:.1f}%)")
    
    # Run enhanced cross-validation
    print(f"\nStarting comprehensive cross-validation...")
    
    cv_function = cv_functions['run_comprehensive_cross_validation']
    cv_results, best_fold_data, all_training_histories = cv_function(
        X, y, phase_groups, feature_names, config, model_classes, training_functions
    )
    
    print(f"\n✓ Cross-validation completed")
    print(f"Total folds: {len(cv_results)}")
    print(f"Best fold AUC: {max([r['auc'] for r in cv_results]):.3f}")
    
    # Store results for further analysis
    analysis_results = {
        'cv_results': cv_results,
        'best_fold_data': best_fold_data,
        'training_histories': all_training_histories,
        'feature_categories': feature_categories,
        'config': config,
        'feature_names': feature_names,
        'data_shape': X.shape,
        'target_distribution': {
            'positive': int(y.sum()),
            'total': len(y),
            'success_rate': float(y.mean())
        }
    }
    
    return analysis_results


if __name__ == "__main__":
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run complete analysis
    try:
        results = run_complete_analysis()
        
        # Save results
        with open(snakemake.output.results, 'wb') as f:
            pickle.dump(results['cv_results'], f)
        
        with open(snakemake.output.best_fold, 'wb') as f:
            pickle.dump(results['best_fold_data'], f)
        
        print(f"\n{'='*80}")
        print("ANALYSIS EXECUTION COMPLETE")
        print(f"{'='*80}")
        print(f"✓ Results saved to: {snakemake.output.results}")
        print(f"✓ Best fold data saved to: {snakemake.output.best_fold}")
        print(f"\nSummary:")
        print(f"  - Processed {results['data_shape'][0]} trials with {results['data_shape'][1]} features")
        print(f"  - Target success rate: {results['target_distribution']['success_rate']*100:.1f}%")
        print(f"  - Completed {len(results['cv_results'])} cross-validation folds")
        print(f"  - Best AUC achieved: {max([r['auc'] for r in results['cv_results']]):.3f}")
        
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise