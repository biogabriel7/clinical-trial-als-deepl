#!/bin/bash
#SBATCH --job-name=clinical_trials
#SBATCH --account=PAS2598  # Replace with your project account
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --output=logs/workflow_%j.out
#SBATCH --error=logs/workflow_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=duarte63@osc.edu  # Replace with your email

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules
module load miniconda3/24.1.2-py310
module load cuda/11.8.0

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Activate conda environment
conda activate clinical-trial-new

# Print environment info for debugging
echo "=== JOB INFORMATION ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $SLURM_GPUS_ON_NODE"
echo "Working directory: $(pwd)"
echo "Python path: $(which python)"
echo "CUDA version: $(nvcc --version)"
echo "========================"

# Run the workflow
echo "Starting clinical trials prediction pipeline..."
python run_workflow.py --cores $SLURM_CPUS_PER_TASK

echo "Workflow completed!"