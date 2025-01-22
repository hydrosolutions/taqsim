#!/bin/bash

#SBATCH --job-name=ZRB_optuna        # Name of the job
#SBATCH --output=ZRB_optuna.out      # Output file
#SBATCH --time=100:00:00            # Maximum runtime (hh:mm:ss)
#SBATCH --ntasks=1                 # Number of tasks (usually 1 for Python)
#SBATCH --cpus-per-task=4          # Number of CPU cores per task
#SBATCH --mem-per-cpu=1024                   # Memory allocation (e.g., 4GB)
#SBATCH --open-mode=truncate

# Run the Python script
source /cluster/home/giannag/myenv/bin/activate
python ZRB_optuna.py
