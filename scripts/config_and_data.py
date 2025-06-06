#!/usr/bin/env python3
"""
Script 1: Configuration and Data Loading
"""

import torch as tc
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

class ModelConfig:
    """Configuration class for comprehensive dataset model parameters"""
    
    def __init__(self, data_path=None):
        # Data params
        self.data_file = data_path or snakemake.input.data
        self.target_column = 'reached_phase_3_plus'
        self.id_column = 'Trial.ID'
        
        # Feature selection params
        self.use_feature_selection = True
        self.max_features = 100
        
        # Model params - EMERGENCY FIXES
        self.hidden_depth_simple = 2
        self.factor_hidden_nodes = 2.0
        self.use_batch_norm = False
        self.min_hidden_size = 64
        
        self.device = 'cuda' if tc.cuda.is_available() else 'cpu'
        
        # Training params
        self.batch_size = 32
        self.lr = 5e-4
        self.dropout = 0.1
        self.input_dropout = 0.05
        self.weight_decay = 1e-4
        self.splits = 5
        self.lrp_gamma = 0.01
        self.l1_lambda = 1e-6
        self.early_stopping_patience = 20
        self.min_epochs_before_stopping = 30
        self.gradient_clip_norm = 0.5
        
        # Simplified loss
        self.use_focal_loss = False
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0
        self.pos_weight_factor = 1.0
        self.label_smoothing = 0.0
        self.use_cosine_scheduler = False
        
        # Class balancing
        self.use_class_weights = True
        
        # Feature exclusion
        self.exclude_phase_features = True


def load_and_analyze_comprehensive_data(filepath):
    """Load and analyze the comprehensive 120-column dataset"""
    
    df = pd.read_csv(filepath)
    
    print(f"Dataset loaded: {df.shape[0]} trials, {df.shape[1]} columns")
    
    # Target variable analysis
    target_col = 'reached_phase_3_plus'
    target_dist = df[target_col].value_counts()
    success_rate = target_dist.get(1, 0) / len(df) * 100
    print(f"Target: {target_dist.get(1, 0)}/{len(df)} Phase 3+ trials ({success_rate:.1f}% success rate)")
    
    # Data completeness summary
    overall_completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    print(f"Overall data completeness: {overall_completeness:.1f}%")
    
    # Feature categories count
    feature_categories = {
        'Sponsor': len([col for col in df.columns if col.startswith('sponsor_')]),
        'Patient': len([col for col in df.columns if col.startswith(('patient_', 'includes_', 'gender_', 'age_group_'))]),
        'Enrollment': len([col for col in df.columns if any(keyword in col.lower() for keyword in ['enrollment', 'accrual', 'duration', 'study_'])]),
        'Endpoint': len([col for col in df.columns if col.startswith('endpoint_')]),
        'Biomarker': len([col for col in df.columns if col.startswith(('biomarker_', 'has_'))]),
        'Outcome': len([col for col in df.columns if col.startswith('outcome_')]),
    }
    
    print(f"Feature categories: " + " | ".join([f"{cat}: {count}" for cat, count in feature_categories.items() if count > 0]))
    
    return df


if __name__ == "__main__":
    # Set random seeds for reproducibility
    tc.manual_seed(42)
    np.random.seed(42)
    
    # Initialize configuration
    config = ModelConfig()
    print(f"Using device: {config.device}")
    if tc.cuda.is_available():
        print(f"GPU: {tc.cuda.get_device_name()}")
    
    # Load data
    df = load_and_analyze_comprehensive_data(config.data_file)
    
    # Save outputs
    with open(snakemake.output.config, 'wb') as f:
        pickle.dump(config, f)
    
    with open(snakemake.output.raw_data, 'wb') as f:
        pickle.dump(df, f)
    
    print("✓ Configuration and data loading complete")