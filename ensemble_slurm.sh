#!/bin/bash

# SLURM job config for ensemble training on the HPC cluster


#SBATCH --job-name=csen240_project_ensemble
#SBATCH --output=csen240_project_ensemble.log
#SBATCH --error=csen240_project_ensemble.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --mail-user=(add email here)
#SBATCH --mail-type=END,FAIL

module load Anaconda3/2024.06-1
module load CUDA/12.2.1

if [ ! -d "$HOME/.conda/envs/cenv" ]; then
        conda create -n cenv python=3.11 -y
        conda install -c conda-forge tensorflow-gpu pandas scikit-learn imbalanced-learn opencv matplotlib seaborn -y
fi
source activate cenv
pip install keras-hub keras-core

echo "Starting job"
python main_ensemble.py
echo "Job completed"

