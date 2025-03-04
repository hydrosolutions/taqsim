#!/bin/bash

#SBATCH --job-name=Sim_2       # Name of the job
#SBATCH --output=ZRB_sim_2.out      # Output file
#SBATCH --time=04:00:00            # Maximum runtime (hh:mm:ss)
#SBATCH --ntasks=1                 # Number of tasks (usually 1 for Python)
#SBATCH --cpus-per-task=1          # Number of CPU cores per task
#SBATCH --mem-per-cpu=1024                   # Memory allocation (e.g., 4GB)
#SBATCH --open-mode=truncate

# Run the Python script
source /cluster/home/giannag/myenv/bin/activate
python ZRB_system.py
