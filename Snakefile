# Snakefile for ALS Clinical Trials Prediction Pipeline

import os

# Configuration
DATA_PATH = "db/comprehensive_merged_trial_data.csv"
OUTPUT_DIR = "results"
SCRIPTS_DIR = "scripts"

# Define all outputs
rule all:
    input:
        f"{OUTPUT_DIR}/final_results.pkl",
        f"{OUTPUT_DIR}/feature_importance.csv",
        f"{OUTPUT_DIR}/performance_metrics.json",
        f"{OUTPUT_DIR}/training_plots.png"

# Rule 1: Setup configuration and load data
rule load_data:
    input:
        data=DATA_PATH
    output:
        config=f"{OUTPUT_DIR}/config.pkl",
        raw_data=f"{OUTPUT_DIR}/raw_data.pkl"
    script:
        f"{SCRIPTS_DIR}/config_and_data.py"

# Rule 2: Validate data
rule validate_data:
    input:
        raw_data=f"{OUTPUT_DIR}/raw_data.pkl",
        config=f"{OUTPUT_DIR}/config.pkl"
    output:
        validation_report=f"{OUTPUT_DIR}/data_validation.txt"
    script:
        f"{SCRIPTS_DIR}/validate_data.py"

# Rule 3: Feature extraction
rule extract_features:
    input:
        raw_data=f"{OUTPUT_DIR}/raw_data.pkl",
        config=f"{OUTPUT_DIR}/config.pkl",
        validation_report=f"{OUTPUT_DIR}/data_validation.txt"  # Add this line
    output:
        features=f"{OUTPUT_DIR}/extracted_features.pkl"
    script:
        f"{SCRIPTS_DIR}/feature_extraction.py"

# Rule 4: Data preprocessing
rule preprocess_data:
    input:
        features=f"{OUTPUT_DIR}/extracted_features.pkl",
        config=f"{OUTPUT_DIR}/config.pkl"
    output:
        processed_data=f"{OUTPUT_DIR}/processed_data.pkl",
        scaler=f"{OUTPUT_DIR}/scaler.pkl",
        feature_names=f"{OUTPUT_DIR}/feature_names.pkl"
    script:
        f"{SCRIPTS_DIR}/preprocessing.py"

# Rule 5: Define model architecture
rule setup_model:
    input:
        config=f"{OUTPUT_DIR}/config.pkl"
    output:
        model_classes=f"{OUTPUT_DIR}/model_classes.pkl"
    script:
        f"{SCRIPTS_DIR}/model_architecture.py"

# Rule 6: Training and evaluation functions
rule setup_training:
    input:
        config=f"{OUTPUT_DIR}/config.pkl",
        model_classes=f"{OUTPUT_DIR}/model_classes.pkl"
    output:
        training_functions=f"{OUTPUT_DIR}/training_functions.pkl"
    script:
        f"{SCRIPTS_DIR}/training_functions.py"

# Rule 7: Cross-validation setup
rule setup_cv:
    input:
        config=f"{OUTPUT_DIR}/config.pkl",
        model_classes=f"{OUTPUT_DIR}/model_classes.pkl"  # Add this
    output:
        cv_functions=f"{OUTPUT_DIR}/cv_functions.pkl"

# Rule 8: Run complete analysis
rule run_analysis:
    input:
        processed_data=f"{OUTPUT_DIR}/processed_data.pkl",
        feature_names=f"{OUTPUT_DIR}/feature_names.pkl",
        config=f"{OUTPUT_DIR}/config.pkl",
        model_classes=f"{OUTPUT_DIR}/model_classes.pkl",
        training_functions=f"{OUTPUT_DIR}/training_functions.pkl",
        cv_functions=f"{OUTPUT_DIR}/cv_functions.pkl"
    output:
        results=f"{OUTPUT_DIR}/cv_results.pkl",
        best_fold=f"{OUTPUT_DIR}/best_fold_data.pkl"
    script:
        f"{SCRIPTS_DIR}/run_analysis.py"

# Rule 9: Generate final report and visualizations
rule generate_report:
    input:
        results=f"{OUTPUT_DIR}/cv_results.pkl",
        best_fold=f"{OUTPUT_DIR}/best_fold_data.pkl",
        config=f"{OUTPUT_DIR}/config.pkl"
    output:
        final_results=f"{OUTPUT_DIR}/final_results.pkl",
        feature_importance=f"{OUTPUT_DIR}/feature_importance.csv",
        metrics=f"{OUTPUT_DIR}/performance_metrics.json",
        plots=f"{OUTPUT_DIR}/training_plots.png"
    script:
        f"{SCRIPTS_DIR}/final_report.py"

# Helper rule to clean outputs (with confirmation)
rule clean:
    shell:
        """
        echo "⚠️  This will delete ALL results in {OUTPUT_DIR}/"
        echo "Continue? (y/N)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            echo "🗑️  Cleaning {OUTPUT_DIR}..."
            rm -rf {OUTPUT_DIR}/*
            echo "✓ Clean complete"
        else
            echo "❌ Clean cancelled"
        fi
        """

# Helper rule to setup directory structure
rule setup:
    shell:
        """
        mkdir -p {OUTPUT_DIR}
        mkdir -p {SCRIPTS_DIR}
        """