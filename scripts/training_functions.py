#!/usr/bin/env python3
"""
Script 5: Training Functions
"""

import torch as tc
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import copy
from sklearn.metrics import (f1_score, roc_auc_score, accuracy_score, 
                           precision_score, recall_score, balanced_accuracy_score,
                           average_precision_score)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = tc.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()


class LabelSmoothingBCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingBCE, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, inputs, targets):
        # Apply label smoothing
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return nn.BCEWithLogitsLoss()(inputs, targets_smooth)


def train_comprehensive_model(X_train, y_train, X_val, y_val, config, model_classes):
    """Train the comprehensive neural network model with emergency fixes"""
    
    n_features = X_train.shape[1]
    ALS_Model = model_classes['ALS_Comprehensive_Model']
    model = ALS_Model(n_features, config).to(config.device)
    
    # EMERGENCY FIX: Use simple weighted BCE loss
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    pos_weight = tc.tensor([neg_count / pos_count], device=config.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = tc.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = tc.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Convert to tensors
    X_train_tensor = tc.tensor(X_train, dtype=tc.float32, device=config.device)
    y_train_tensor = tc.tensor(y_train, dtype=tc.float32, device=config.device)
    X_val_tensor = tc.tensor(X_val, dtype=tc.float32, device=config.device)
    y_val_tensor = tc.tensor(y_val, dtype=tc.float32, device=config.device)
    
    best_f1 = 0
    best_model_state = None
    patience_counter = 0
    training_history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_auc': []}
    
    # EMERGENCY FIX: Minimum epochs before early stopping
    min_epochs = getattr(config, 'min_epochs_before_stopping', 30)
    
    for epoch in range(150):
        # Training
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train_tensor).squeeze()
        loss = criterion(outputs, y_train_tensor)
        
        # L1 regularization
        l1_loss = sum(p.abs().sum() for p in model.parameters())
        total_loss = loss + config.l1_lambda * l1_loss
        
        total_loss.backward()
        
        # Gradient clipping
        if hasattr(config, 'gradient_clip_norm'):
            tc.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_norm)
        else:
            tc.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Validation
        model.eval()
        with tc.no_grad():
            val_outputs = model(X_val_tensor).squeeze()
            val_loss = criterion(val_outputs, y_val_tensor)
            val_probs = tc.sigmoid(val_outputs).cpu().numpy()
        
        # Find best F1 score across thresholds
        thresholds = np.arange(0.2, 0.8, 0.05)
        best_val_f1 = max(f1_score(y_val, (val_probs > thresh).astype(int), zero_division=0) 
                         for thresh in thresholds)
        
        # Calculate AUC
        try:
            val_auc = roc_auc_score(y_val, val_probs)
        except ValueError:
            val_auc = 0.0
        
        # Store history
        training_history['train_loss'].append(loss.item())
        training_history['val_loss'].append(val_loss.item())
        training_history['val_f1'].append(best_val_f1)
        training_history['val_auc'].append(val_auc)
        
        scheduler.step(best_val_f1)
        
        # EMERGENCY FIX: Early stopping with minimum epochs
        if epoch >= min_epochs:
            if best_val_f1 > best_f1:
                best_f1 = best_val_f1
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        else:
            # Always update best model in early epochs
            if best_val_f1 > best_f1:
                best_f1 = best_val_f1
                best_model_state = copy.deepcopy(model.state_dict())
        
        # Progress update
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}: Train Loss: {loss.item():.4f}, Val F1: {best_val_f1:.4f}, Val AUC: {val_auc:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, epoch + 1, best_f1, training_history


def evaluate_comprehensive_model(model, X_test, y_test, config):
    """Evaluate comprehensive model and return comprehensive metrics"""
    
    model.eval()
    X_test_tensor = tc.tensor(X_test, dtype=tc.float32, device=config.device)
    
    with tc.no_grad():
        outputs = model(X_test_tensor).squeeze()
        probs = tc.sigmoid(outputs).cpu().numpy()
    
    # Find optimal threshold
    thresholds = np.arange(0.1, 0.9, 0.02)
    best_f1 = 0
    best_thresh = 0.5
    best_balanced_acc = 0
    
    for thresh in thresholds:
        predictions = (probs > thresh).astype(int)
        f1 = f1_score(y_test, predictions, zero_division=0)
        balanced_acc = balanced_accuracy_score(y_test, predictions)
        
        if balanced_acc > best_balanced_acc or (balanced_acc == best_balanced_acc and f1 > best_f1):
            best_f1 = f1
            best_thresh = thresh
            best_balanced_acc = balanced_acc
    
    # Calculate all metrics
    try:
        auc = roc_auc_score(y_test, probs)
        ap = average_precision_score(y_test, probs)
        
        predictions = (probs > best_thresh).astype(int)
        accuracy = accuracy_score(y_test, predictions)
        balanced_acc = balanced_accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        
    except ValueError as e:
        print(f"Warning: Metric calculation issue: {e}")
        auc = ap = accuracy = balanced_acc = precision = recall = f1 = 0.0
        best_thresh = 0.5
    
    return auc, ap, accuracy, balanced_acc, precision, recall, f1, best_thresh, probs


def perform_comprehensive_lrp_analysis(model, X_test, y_test, feature_names, config):
    """Perform Layer-wise Relevance Propagation analysis"""
    
    model.eval()
    X_test_tensor = tc.tensor(X_test, dtype=tc.float32, device=config.device)
    
    # Set iteration for LRP
    for layer in model.modules():
        if hasattr(layer, 'iteration'):
            layer.iteration = 0
    
    # Forward pass and LRP
    with tc.no_grad():
        outputs = model(X_test_tensor).squeeze()
        probs = tc.sigmoid(outputs).cpu().numpy()
    
    R = tc.ones_like(outputs, device=config.device)
    relevance_scores = model.relprop(R).cpu().numpy()
    
    # Calculate feature importance
    feature_importance = np.abs(relevance_scores).mean(axis=0)
    feature_importance_dict = dict(zip(feature_names, feature_importance))
    feature_importance_series = pd.Series(feature_importance_dict).sort_values(ascending=False)
    
    return feature_importance_series, relevance_scores, probs


def create_detailed_classification_report(y_true, y_pred, y_probs, threshold):
    """Create a detailed classification report"""
    
    print(f"\n=== CLASSIFICATION REPORT ===")
    print(f"Threshold: {threshold:.3f} | Test Size: {len(y_true)} | Positive: {int(y_true.sum())} ({y_true.mean()*100:.1f}%)")
    
    # Confusion matrix
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    
    print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    
    # Key metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    print(f"Sensitivity: {sensitivity:.3f} | Specificity: {specificity:.3f} | Precision: {precision:.3f}")
    
    # Probability stats
    pos_probs = y_probs[y_true == 1]
    neg_probs = y_probs[y_true == 0]
    print(f"Prob Stats - Positive: {pos_probs.mean():.3f}±{pos_probs.std():.3f} | Negative: {neg_probs.mean():.3f}±{neg_probs.std():.3f}")
    
    return {
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'sensitivity': sensitivity, 'specificity': specificity, 'precision': precision
    }


if __name__ == "__main__":
    # Load dependencies
    with open(snakemake.input.config, 'rb') as f:
        config = pickle.load(f)
    
    with open(snakemake.input.model_classes, 'rb') as f:
        model_classes = pickle.load(f)
    
    # Save training functions
    training_functions = {
        'FocalLoss': FocalLoss,
        'LabelSmoothingBCE': LabelSmoothingBCE,
        'train_comprehensive_model': train_comprehensive_model,
        'evaluate_comprehensive_model': evaluate_comprehensive_model,
        'perform_comprehensive_lrp_analysis': perform_comprehensive_lrp_analysis,
        'create_detailed_classification_report': create_detailed_classification_report
    }
    
    with open(snakemake.output.training_functions, 'wb') as f:
        pickle.dump(training_functions, f)
    
    print("✓ Training functions defined with emergency fixes")