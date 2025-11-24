#!/bin/bash

# SLURM job config for the HPC cluster


#SBATCH --job-name=csen240_project
#SBATCH --output=csen240_project.log
#SBATCH --error=csen240_project.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --mail-user=(add email here)
#SBATCH --mail-type=END,FAIL

module load Anaconda3/2024.06-1
module load CUDA/12.2.1

if [ ! -d "$HOME/.conda/envs/cenv" ]; then
        conda create -n cenv python=3.11 -y
        conda install -c conda-forge tensorflow-gpu pandas scikit-learn imbalanced-learn opencv matplotlib seaborn -y
fi
source activate cenv

echo "Starting job"
python main.py 2 #TODO Update argument here based on model to run
echo "Job completed"

