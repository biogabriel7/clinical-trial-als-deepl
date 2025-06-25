"""
ALS Clinical Trial Success Prediction - Quick Usage Script
=========================================================

This script provides a simple interface to use the trained ALS trial success model
for making predictions on new clinical trials.

Usage:
    python predict_trial_success.py --input data.csv --output predictions.csv
"""

import pandas as pd
import numpy as np
import argparse
from als_trial_success_model import ALSTrialSuccessPredictor
import json


def analyze_trial(predictor, trial_data):
    """
    Analyze a single trial and provide detailed insights.
    
    Args:
        predictor: Trained ALSTrialSuccessPredictor
        trial_data: DataFrame row with trial information
        
    Returns:
        Dictionary with analysis results
    """
    # Make prediction
    proba = predictor.predict_proba(trial_data.to_frame().T)[0, 1]
    prediction = (proba >= predictor.optimal_threshold)
    
    # Get feature values for key predictors
    importance_df = predictor.get_feature_importance(top_n=10)
    key_features = {}
    
    for feature in importance_df['Feature']:
        if feature in trial_data.index:
            key_features[feature] = trial_data[feature]
    
    analysis = {
        'success_probability': float(proba),
        'prediction': 'Success' if prediction else 'Failure',
        'confidence': 'High' if proba > 0.7 or proba < 0.3 else 'Medium' if proba > 0.5 or proba < 0.5 else 'Low',
        'key_features': key_features,
        'recommendations': []
    }
    
    # Generate specific recommendations based on features
    if 'Target Accrual' in trial_data.index:
        if trial_data['Target Accrual'] < 50:
            analysis['recommendations'].append("Consider increasing target enrollment to 50-300 patients")
        elif trial_data['Target Accrual'] > 500:
            analysis['recommendations'].append("Very large trial - ensure adequate resources and sites")
    
    if 'endpoint_survival' in trial_data.index and trial_data['endpoint_survival'] == 0:
        analysis['recommendations'].append("Consider adding survival endpoints - they triple success rates")
    
    if 'Enrollment Duration (Mos.)' in trial_data.index and trial_data['Enrollment Duration (Mos.)'] < 12:
        analysis['recommendations'].append("Plan for at least 12-18 months enrollment duration")
    
    if 'sponsor_trial_count' in trial_data.index and trial_data['sponsor_trial_count'] < 3:
        analysis['recommendations'].append("Partner with experienced sponsors or CROs")
    
    return analysis


def batch_predict(predictor, data_file, output_file):
    """
    Make predictions on a batch of trials.
    
    Args:
        predictor: Trained ALSTrialSuccessPredictor
        data_file: Path to input CSV file
        output_file: Path to output CSV file
    """
    # Load data
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    
    # Check if target column exists (for evaluation)
    target_col = 'reached_phase_3_plus'
    has_target = target_col in df.columns
    
    # Prepare features
    print("Preparing features...")
    X, y = predictor.prepare_features(df, target_col) if has_target else (predictor.prepare_features(df.assign(**{target_col: 0}), target_col)[0], None)
    
    # Make predictions
    print("Making predictions...")
    probabilities = predictor.predict_proba(X)[:, 1]
    predictions = predictor.predict(X, use_optimal_threshold=True)
    
    # Create results dataframe
    results = pd.DataFrame({
        'trial_index': df.index,
        'success_probability': probabilities,
        'prediction': ['Success' if p else 'Failure' for p in predictions],
        'confidence': pd.cut(probabilities, bins=[0, 0.3, 0.5, 0.7, 1.0], 
                           labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
    })
    
    # Add trial identifiers if available
    if 'Trial.ID' in df.columns:
        results['trial_id'] = df['Trial.ID'].values
    if 'Trial Title' in df.columns:
        results['trial_title'] = df['Trial Title'].values
    
    # Add actual outcomes if available
    if has_target:
        results['actual_outcome'] = ['Success' if y else 'Failure' for y in y]
        results['correct_prediction'] = results['prediction'] == results['actual_outcome']
        
        # Calculate metrics
        accuracy = results['correct_prediction'].mean()
        precision = (results[results['prediction'] == 'Success']['correct_prediction'].mean() 
                    if (results['prediction'] == 'Success').any() else 0)
        recall = (results[results['actual_outcome'] == 'Success']['correct_prediction'].mean()
                 if (results['actual_outcome'] == 'Success').any() else 0)
        
        print(f"\nPerformance on this dataset:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
    
    # Save results
    results.to_csv(output_file, index=False)
    print(f"\nPredictions saved to {output_file}")
    
    # Summary statistics
    print("\nPrediction Summary:")
    print(f"Total trials: {len(results)}")
    print(f"Predicted successes: {(results['prediction'] == 'Success').sum()} ({(results['prediction'] == 'Success').mean():.1%})")
    print(f"High confidence predictions: {(results['confidence'] == 'High').sum()}")
    
    return results


def interactive_predict(predictor):
    """Interactive mode for analyzing individual trials."""
    print("\nInteractive Trial Analysis")
    print("=" * 50)
    print("Enter trial characteristics (press Enter to skip):")
    
    # Create empty trial data
    trial_data = pd.Series(dtype=float)
    
    # Key features to ask about
    feature_prompts = [
        ('Target Accrual', 'Target patient enrollment (e.g., 100): ', float),
        ('Enrollment Duration (Mos.)', 'Planned enrollment duration in months (e.g., 18): ', float),
        ('Treatment Duration (Mos.)', 'Treatment duration in months (e.g., 12): ', float),
        ('endpoint_survival', 'Does trial include survival endpoints? (1=yes, 0=no): ', int),
        ('endpoint_efficacy', 'Number of efficacy endpoints (e.g., 3): ', int),
        ('endpoint_safety', 'Number of safety endpoints (e.g., 2): ', int),
        ('sponsor_trial_count', 'Number of previous trials by sponsor (e.g., 10): ', int),
        ('sponsor_is_major_pharma', 'Is sponsor a major pharma company? (1=yes, 0=no): ', int),
        ('biomarker_total_uses', 'Number of biomarkers used (e.g., 2): ', int),
        ('population_description_length', 'Length of population description (characters, e.g., 500): ', int),
        ('inclusion_criteria_length', 'Length of inclusion criteria (characters, e.g., 800): ', int),
        ('exclusion_criteria_length', 'Length of exclusion criteria (characters, e.g., 1200): ', int),
        ('supporting_url_count', 'Number of supporting documents/URLs (e.g., 8): ', int)
    ]
    
    for feature, prompt, dtype in feature_prompts:
        value = input(prompt).strip()
        if value:
            try:
                trial_data[feature] = dtype(value)
            except ValueError:
                print(f"Invalid input for {feature}, skipping...")
    
    # Fill missing values with defaults
    for feature in predictor.feature_names:
        if feature not in trial_data.index:
            trial_data[feature] = 0
    
    # Analyze trial
    print("\nAnalyzing trial...")
    analysis = analyze_trial(predictor, trial_data)
    
    # Display results
    print("\n" + "=" * 50)
    print("ANALYSIS RESULTS")
    print("=" * 50)
    print(f"Success Probability: {analysis['success_probability']:.1%}")
    print(f"Prediction: {analysis['prediction']}")
    print(f"Confidence: {analysis['confidence']}")
    
    if analysis['key_features']:
        print("\nKey Feature Values:")
        for feature, value in analysis['key_features'].items():
            print(f"  - {feature}: {value}")
    
    if analysis['recommendations']:
        print("\nRecommendations:")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    print("\nGeneral Best Practices:")
    for rec in predictor.get_recommendations()[:3]:
        print(f"  • {rec.split(': ')[1]}")


def main():
    parser = argparse.ArgumentParser(description='Predict ALS clinical trial success')
    parser.add_argument('--model', default='als_trial_success_model.pkl', 
                       help='Path to trained model file')
    parser.add_argument('--input', help='Input CSV file with trial data')
    parser.add_argument('--output', help='Output CSV file for predictions')
    parser.add_argument('--interactive', action='store_true', 
                       help='Interactive mode for single trial analysis')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    predictor = ALSTrialSuccessPredictor()
    
    try:
        predictor.load_model(args.model)
    except FileNotFoundError:
        print(f"Model file {args.model} not found. Training new model...")
        # Train new model
        df = pd.read_csv('comprehensive_merged_trial_data.csv')
        target_col = 'reached_phase_3_plus'
        X, y = predictor.prepare_features(df, target_col)
        metrics = predictor.train(X, y)
        print(f"Model trained. ROC-AUC: {metrics['roc_auc']:.3f}")
        predictor.save_model(args.model)
    
    # Run appropriate mode
    if args.interactive:
        interactive_predict(predictor)
    elif args.input and args.output:
        batch_predict(predictor, args.input, args.output)
    else:
        print("Please specify either --interactive or both --input and --output")
        parser.print_help()


if __name__ == "__main__":
    main()