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
#SBATCH --mail-user=erscott@scu.edu
#SBATCH --mail-type=END,FAIL

module load Python/3.11.3-GCCcore-12.3.0

python -m  venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt


echo "Starting job"
python main.py 1 #TODO Update argument here based on model to run
echo "Job completed"

