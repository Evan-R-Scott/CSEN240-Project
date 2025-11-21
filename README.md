### To Run in HPC Cluster

Doesn't download the modelfile so right now it is a run-and-done workflow. Not sure if we need to download our best model for submission?

ssh <your SCU username>@login.wave.scu.edu
git clone https://github.com/Evan-R-Scott/CSEN240-Project.git
cd CSEN240-Project/
sbatch slurm.sh

Check progress using commands like:
squeue
squeue -u <username>
tail -f csen240_project.err
tail -f csen240_project.log (Best)
