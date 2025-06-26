# ALS Clinical Trial Success Prediction Model

A machine learning model to predict the success of ALS (Amyotrophic Lateral Sclerosis) clinical trials based on comprehensive trial characteristics. This model helps researchers and pharmaceutical companies optimize trial design and resource allocation.

## 🎯 Overview

This project implements a Gradient Boosting classifier that predicts whether an ALS clinical trial will reach Phase 3 or beyond. The model achieves:

- **ROC-AUC**: 0.849 (excellent discrimination)
- **Recall**: 89% at optimal threshold (catches most successful trials)
- **Precision**: 37% at optimal threshold (worth investigating ~3 trials to find 1 success)

## 📊 Key Features

- **Comprehensive data leakage prevention** - Removes 36+ features that could cause overfitting
- **Advanced feature engineering** - Creates 37 sophisticated features across 6 categories
- **Class balancing with SMOTE** - Handles imbalanced dataset (22.8% positive class)
- **Optimized decision threshold** - Maximizes recall for clinical applications
- **Actionable recommendations** - Provides specific guidance for trial design

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/clinical-trial-als-deepl.git
cd clinical-trial-als-deepl

# Install dependencies
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib
```

### Basic Usage

```python
from als_trial_success_model import ALSTrialSuccessPredictor

# Initialize and train model
predictor = ALSTrialSuccessPredictor(random_state=42)

# Load your data
df = pd.read_csv('comprehensive_merged_trial_data.csv')
X, y = predictor.prepare_features(df, 'reached_phase_3_plus')

# Train model
metrics = predictor.train(X, y)
print(f"ROC-AUC: {metrics['roc_auc']:.3f}")

# Make predictions
probability = predictor.predict_proba(new_trial_data)
prediction = predictor.predict(new_trial_data)
```

## 📁 File Structure

```
clinical-trial-als-deepl/
├── als_trial_success_model.py    # Main model implementation
├── predict_trial_success.py      # CLI for predictions
├── example_usage.ipynb          # Jupyter notebook with examples
├── trying-new-stuff.ipynb       # Original development notebook
├── README_ALS_MODEL.md          # This file
└── README.md                    # General project README
```

## 🔧 Command Line Interface

### Interactive Mode
Analyze a single trial with guided input:
```bash
python predict_trial_success.py --interactive
```

Example session:
```
Interactive Trial Analysis
==================================================
Enter trial characteristics (press Enter to skip):
Target patient enrollment (e.g., 100): 150
Planned enrollment duration in months (e.g., 18): 18
Treatment duration in months (e.g., 12): 12
Does trial include survival endpoints? (1=yes, 0=no): 1
Number of efficacy endpoints (e.g., 3): 3
...

ANALYSIS RESULTS
==================================================
Success Probability: 72.3%
Prediction: Success
Confidence: High
```

### Batch Predictions
Process multiple trials from a CSV file:
```bash
python predict_trial_success.py --input trials.csv --output predictions.csv
```

### Using Pre-trained Model
```bash
python predict_trial_success.py --model saved_model.pkl --input data.csv --output results.csv
```

## 📈 Model Performance

### Top 10 Predictive Features

1. **Target Accrual** (0.173) - Larger trials (>100 patients) have more statistical power
2. **Population Specificity** (0.039) - Specific patient populations improve success
3. **Success Indicator Score** (0.032) - Composite score effectively captures patterns
4. **Survival Endpoints** (0.030) - Trials with survival metrics are 3x more likely to succeed
5. **Trial Quality Index** (0.030) - Overall trial design sophistication
6. **Biomarker Endpoints** (0.029) - Biomarker usage improves patient stratification
7. **Enrollment Duration** (0.023) - Adequate time (12-18 months) for recruitment
8. **Safety Endpoints** (0.023) - Comprehensive safety monitoring
9. **Sponsor Experience** (0.021) - Experienced sponsors have higher success rates
10. **Documentation Quality** (0.017) - Better documentation correlates with success

### Confusion Matrix at Optimal Threshold (0.125)

```
                 Predicted Failed  Predicted Successful
Actually Failed        56%                44%
Actually Successful    11%                89%
```

## 💡 Key Recommendations for Trial Success

1. **PRIORITIZE SURVIVAL ENDPOINTS**: Trials with clear survival metrics are 3x more likely to succeed
2. **PLAN ADEQUATE ENROLLMENT TIME**: Allow at least 12-18 months for patient recruitment
3. **TARGET 100-300 PATIENTS**: This range balances statistical power with feasibility
4. **INVEST IN DOCUMENTATION**: Trials with >10 supporting documents show 2x success rate
5. **PARTNER WITH EXPERIENCED SPONSORS**: First-time sponsors have <10% success rate
6. **DEFINE CLEAR PATIENT POPULATIONS**: Detailed inclusion/exclusion criteria improve outcomes
7. **CONSIDER BIOMARKERS**: Trials with prognostic biomarkers show better stratification

## 🛠️ Advanced Usage

### Custom Feature Engineering

```python
# Add your own features to the model
def create_custom_features(df):
    features = pd.DataFrame(index=df.index)
    features['my_custom_feature'] = df['column1'] * df['column2']
    return features

# Extend the predictor
class CustomALSPredictor(ALSTrialSuccessPredictor):
    def engineer_features(self, df):
        base_features = super().engineer_features(df)
        custom_features = create_custom_features(df)
        return pd.concat([base_features, custom_features], axis=1)
```

### Model Persistence

```python
# Save trained model
predictor.save_model('als_model_v1.pkl')

# Load model later
new_predictor = ALSTrialSuccessPredictor()
new_predictor.load_model('als_model_v1.pkl')
```

### Feature Importance Analysis

```python
# Get detailed feature importance
importance_df = predictor.get_feature_importance(top_n=20)

# Visualize
import matplotlib.pyplot as plt
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance Score')
plt.title('Feature Importance for ALS Trial Success')
plt.show()
```

## 📊 Data Requirements

The model expects a CSV file with the following key columns:

### Required Columns
- `reached_phase_3_plus` (target variable): Binary indicator of trial success
- `Target Accrual`: Planned number of patients
- `Enrollment Duration (Mos.)`: Planned enrollment duration
- `Treatment Duration (Mos.)`: Treatment period length

### Recommended Columns
- Endpoint indicators: `endpoint_survival`, `endpoint_efficacy`, `endpoint_safety`
- Sponsor information: `sponsor_trial_count`, `sponsor_is_major_pharma`
- Biomarker usage: `biomarker_total_uses`, `biomarker_prognostic`
- Trial documentation: `supporting_url_count`, `population_description_length`
- Criteria lengths: `inclusion_criteria_length`, `exclusion_criteria_length`

### Features Automatically Removed (Data Leakage Prevention)
The model automatically removes these features to prevent overfitting:
- Trial phase information (e.g., `Trial Phase_III`)
- Trial status (e.g., `Trial Status_Completed`)
- Outcome features (e.g., `outcome_positive`)
- Post-completion dates
- Actual enrollment numbers

## 🔬 Technical Details

### Feature Engineering Categories

1. **Sponsor Sophistication** (4 features)
   - Experience score, risk profiles

2. **Trial Complexity** (10 features)
   - Design sophistication, endpoint strategies, accrual appropriateness

3. **Biomarker Usage** (6 features)
   - Strategy sophistication, neurological/genetic biomarker counts

4. **Temporal Features** (6 features)
   - Duration appropriateness, enrollment efficiency

5. **Patient Population** (7 features)
   - Age/gender strategies, criteria balance

6. **Composite Indicators** (4 features)
   - Success indicator score, trial quality index, preparedness score

### Model Architecture

- **Algorithm**: Gradient Boosting Classifier
- **Parameters**:
  - n_estimators: 100
  - learning_rate: 0.1
  - max_depth: 4
  - min_samples_split: 30
  - min_samples_leaf: 15
  - subsample: 0.8
- **Class Balancing**: SMOTE with 0.5 sampling strategy
- **Optimal Threshold**: 0.125 (optimized for F2 score)

## 📝 Example: Complete Analysis Workflow

```python
import pandas as pd
from als_trial_success_model import ALSTrialSuccessPredictor

# 1. Load and prepare data
df = pd.read_csv('als_trials.csv')
predictor = ALSTrialSuccessPredictor()
X, y = predictor.prepare_features(df, 'reached_phase_3_plus')

# 2. Train model
metrics = predictor.train(X, y)
print(f"Model Performance: ROC-AUC = {metrics['roc_auc']:.3f}")

# 3. Analyze a new trial
new_trial = pd.DataFrame({
    'Target Accrual': [200],
    'Enrollment Duration (Mos.)': [18],
    'endpoint_survival': [1],
    'sponsor_trial_count': [15],
    # ... other features
})

# Fill missing features
for feature in predictor.feature_names:
    if feature not in new_trial.columns:
        new_trial[feature] = 0

# 4. Get prediction
prob = predictor.predict_proba(new_trial)[0, 1]
pred = predictor.predict(new_trial)[0]

print(f"\nTrial Analysis:")
print(f"Success Probability: {prob:.1%}")
print(f"Recommendation: {'Proceed with trial' if pred else 'Reconsider design'}")

# 5. Get insights
importance = predictor.get_feature_importance()
print(f"\nTop Success Factors:")
for _, row in importance.head(5).iterrows():
    print(f"- {row['Feature']}: {row['Importance']:.3f}")
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📜 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Data source: Comprehensive merged trial data from Citeline database
- Inspired by the urgent need to accelerate ALS drug development
- Thanks to all researchers working to find treatments for ALS

## 📧 Contact

For questions or collaborations, please contact: gabriel.duarte@osumc.edu

---

**Note**: This model is for research purposes only and should not be used as the sole basis for clinical trial decisions. Always consult with clinical experts and regulatory guidelines.