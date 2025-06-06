#!/usr/bin/env python3
"""
Main workflow execution script for ALS Clinical Trials Prediction Pipeline
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_directories():
    """Create necessary directories"""
    directories = ['scripts', 'results', 'logs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("✓ Directories created")

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'torch', 'pandas', 'numpy', 'scikit-learn', 
        'matplotlib', 'seaborn', 'snakemake'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✓ All dependencies available")
    return True

def run_snakemake(dry_run=False, cores=1):
    """Run the Snakemake workflow"""
    
    cmd = [
        'snakemake',
        '--cores', str(cores),
        '--printshellcmds'
    ]
    
    if dry_run:
        cmd.append('--dry-run')
        print("🔍 Running dry run...")
    else:
        print(f"🚀 Running workflow with {cores} cores...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Workflow completed successfully")
            if not dry_run:
                print("\n📊 Results saved in 'results/' directory")
            return True
        else:
            print("❌ Workflow failed")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except FileNotFoundError:
        print("❌ Snakemake not found. Install with: conda install -c bioconda snakemake")
        return False

def main():
    """Main execution function"""
    
    print("=" * 60)
    print("ALS CLINICAL TRIALS PREDICTION PIPELINE")
    print("=" * 60)
    
    # Setup
    setup_directories()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Parse arguments
    dry_run = '--dry-run' in sys.argv
    cores = 1
    
    if '--cores' in sys.argv:
        try:
            cores_idx = sys.argv.index('--cores')
            cores = int(sys.argv[cores_idx + 1])
        except (IndexError, ValueError):
            cores = 1
    
    # Run workflow
    success = run_snakemake(dry_run=dry_run, cores=cores)
    
    if success and not dry_run:
        print("\n📈 WORKFLOW COMPLETED SUCCESSFULLY!")
        print("\nGenerated outputs:")
        print("  📁 results/final_results.pkl - Complete results")
        print("  📊 results/feature_importance.csv - Feature importance rankings")  
        print("  📋 results/performance_metrics.json - Model performance metrics")
        print("  📈 results/training_plots.png - Training visualizations")
        print("\n💡 Tip: Use 'python analyze_results.py' to explore results interactively")
    elif success and dry_run:
        print("\n🔍 Dry run completed - workflow looks good!")
        print("Run without --dry-run to execute the pipeline")
    else:
        print("\n❌ Workflow failed - check error messages above")
        sys.exit(1)


if __name__ == "__main__":
    main()