### To Run in HPC Cluster

Doesn't download the modelfile so right now it is a run-and-done workflow. Not sure if we need to download our best model for submission?

1. ssh (your SCU username)@login.wave.scu.edu
2. git clone https://github.com/Evan-R-Scott/CSEN240-Project.git
3. cd CSEN240-Project/
4. sbatch slurm.sh

Check progress using commands like:
  1. squeue
  2. squeue -u (username)
  3. tail -f csen240_project.err
  4. tail -f csen240_project.log -> Best
