# Clinical Trial Success Prediction Model

A machine learning model that predicts the probability of clinical trials reaching Phase 3+ based on design-time features, achieving **87.4% ROC-AUC** through advanced feature engineering.

## 🎯 Overview

This model helps predict clinical trial success at the **design stage**, the data was compiled using Citeline. It's specifically designed to:

- ✅ **Avoid data leakage** - Only uses features available at trial design time
- 🔧 **Extract hidden value** - Advanced feature engineering from high-cardinality categorical data
- 📊 **Provide actionable insights** - Identifies key success factors for trial optimization
- 🚀 **Ready for production** - Includes deployment utilities and batch prediction

## 📋 Quick Start

### 1. Installation

```bash
# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn

# Optional: For advanced features
pip install category-encoders scipy
```

### 2. Prepare Your Data

Your CSV file should contain clinical trial data with the following structure:

```python
# Required columns for the model to work
required_columns = [
    'reached_phase_3_plus',  # Target variable (0/1)
    'Target Accrual',        # Planned number of patients
    'Disease',               # Disease area(s)
    'Study Keywords',        # Trial keywords
    'Associated CRO',        # Contract Research Organization
    'Trial Tag/Attribute',   # Trial attributes
    'Patient Segment',       # Patient population focus
    # ... plus other design-time features
]
```

### 3. Run the Model

```python
# Update the data path in the script
df = pd.read_csv('your_data_file.csv')

# Run the complete model
python complete_enhanced_model.py
```

## 📊 Data Structure Requirements

### Target Variable
- **`reached_phase_3_plus`**: Binary (0/1) indicating if trial reached Phase 3 or beyond

### Core Predictive Features

| Feature Category | Example Columns | Description |
|-----------------|----------------|-------------|
| **Trial Design** | `Target Accrual`, `Enrollment Duration (Mos.)` | Basic trial parameters |
| **Endpoints** | `endpoint_survival`, `endpoint_safety`, `endpoint_efficacy` | Endpoint strategy |
| **Sponsor Info** | `sponsor_is_academic`, `sponsor_trial_count` | Sponsor characteristics |
| **Disease** | `Disease` | Disease area(s) - will be processed into categories |
| **Keywords** | `Study Keywords` | Trial keywords - will be processed for high-value terms |
| **CRO** | `Associated CRO` | Contract Research Organization |
| **Attributes** | `Trial Tag/Attribute` | Trial tags and attributes |
| **Patient Focus** | `Patient Segment` | Patient population focus |

### Features to EXCLUDE (Data Leakage)
❌ **Do NOT include these in your dataset:**
- Trial phase information (`Trial Phase_III`, `Trial Phase_IV`, etc.)
- Trial status (`Trial Status_Completed`, `Trial Status_Terminated`, etc.)
- Outcome information (`outcome_positive`, `outcome_negative`, etc.)
- Post-completion dates (`Primary Endpoints Reported Date`, etc.)
- Actual enrollment numbers (`Actual Accrual`, etc.)

## 🔧 Feature Engineering

The model automatically creates enhanced features from your raw data:

### 1. Smart Disease Grouping
```python
# From: Disease = "CNS: Alzheimer's Disease; CNS: Dementia"
# Creates: disease_is_alzheimer_dementia = 1
```

### 2. High-Value Keywords
```python
# From: Study Keywords = "double blind; placebo; efficacy"
# Creates: keyword_double_blind = 1, keyword_placebo = 1, keyword_efficacy = 1
```

### 3. CRO Quality Indicators
```python
# From: Associated CRO = "IQVIA"
# Creates: cro_is_high_performing = 1
```

### 4. Trial Attributes
```python
# From: Trial Tag/Attribute = "Registration; Biomarker/Efficacy"
# Creates: attr_has_registration = 1, attr_has_biomarker = 1
```

### 5. Patient Segment Focus
```python
# From: Patient Segment = "Symptom relief"
# Creates: segment_symptom_relief = 1
```

## 🚀 Usage Examples

### Basic Prediction
```python
# Load and run the model
python complete_enhanced_model.py

# The script will automatically:
# 1. Load your data
# 2. Engineer features
# 3. Train multiple models
# 4. Show performance metrics
# 5. Provide feature importance
# 6. Create visualizations
```

### Predict New Trial
```python
# Example high-potential trial
new_trial = {
    'Target Accrual': 200,           # Larger trial
    'attr_has_registration': 1,      # Registration trial
    'segment_symptom_relief': 1,     # Symptom relief focus
    'keyword_efficacy': 1,           # Efficacy keyword
    'keyword_placebo': 1,            # Placebo keyword
    'cro_is_high_performing': 1,     # High-performing CRO
    'endpoint_survival': 1,          # Survival endpoint
    'sponsor_is_academic': 1         # Academic sponsor
}

# Predict success probability
success_prob = predict_trial_success(new_trial)
print(f"Success probability: {success_prob:.1%}")
# Expected output: ~85-90% success probability
```

### Batch Predictions
```python
# For multiple trials
results = batch_predict_trials(your_trials_df, model_package)
print(results[['Trial_ID', 'predicted_success_probability', 'predicted_success_category']])
```

## 📁 File Structure

```
clinical-trial-model/
├── complete_enhanced_model.py    # Main model implementation
├── model_extensions.py           # Advanced utilities
├── README.md                     # This file
├── requirements.txt              # Python dependencies
└── data/
    └── your_trial_data.csv       # Your clinical trial dataset
```

## 🔍 Model Outputs

### 1. Performance Metrics
- ROC-AUC scores for all models
- Precision, recall, F1-score
- Confusion matrices
- Cross-validation results

### 2. Visualizations
- ROC curves
- Precision-recall curves
- Feature importance plots
- Model comparison charts
- Correlation heatmaps

### 3. Feature Importance
- Top 20 most important features
- Engineered vs original feature breakdown
- Business interpretation of key factors

### 4. Predictions
- Individual trial success probabilities
- Risk categories (Low/Medium/High)
- Confidence intervals

## ⚙️ Advanced Features

### Hyperparameter Optimization
```python
# Optimize model performance
optimized_model = optimize_best_model(X_train, y_train, 'Random Forest')
```

### Feature Interaction Analysis
```python
# Analyze feature correlations
analyze_feature_interactions(model, X_train, y_train)
```

### Model Deployment
```python
# Save model for production
save_complete_model(model, scaler, feature_columns, model_name)

# Load saved model
model_package = load_complete_model('clinical_trial_model.pkl')
```

## 🔧 Troubleshooting

### Common Issues

**1. Missing Columns Error**
```python
# Solution: Ensure your CSV has the required columns
required_cols = ['reached_phase_3_plus', 'Target Accrual', 'Disease', ...]
missing_cols = [col for col in required_cols if col not in df.columns]
print(f"Missing columns: {missing_cols}")
```

**3. Feature Engineering Errors**
```python
# Verify high-cardinality columns exist
high_card_cols = ['Disease', 'Study Keywords', 'Associated CRO', 'Trial Tag/Attribute', 'Patient Segment']
existing_cols = [col for col in high_card_cols if col in df.columns]
print(f"Available for feature engineering: {existing_cols}")
```

## 📞 Support

gabriel.duarte@osumc.edu

### Model Validation Checklist
- [ ] Target variable is binary (0/1)
- [ ] No data leakage features included
- [ ] High-cardinality categorical features present
- [ ] Reasonable success rate (10-50%)
- [ ] Sufficient sample size (>200 trials recommended)

### Getting Help
1. **Check data structure** against requirements above
2. **Verify feature engineering** is working correctly
3. **Review performance metrics** for reasonableness
4. **Examine feature importance** for business logic

## 📄 License

This model is provided under MIT license.

---

**Ready to predict clinical trial success? Start with `complete_enhanced_model.py`!** 

