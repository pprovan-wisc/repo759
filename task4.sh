#!/bin/bash
#SBATCH --job-name=FirstSlurm        # Job name
#SBATCH --output=FirstSlurm.out      # Std output
#SBATCH --error=FirstSlurm.err       # Std error
#SBATCH --ntasks=1                    # One task
#SBATCH --cpus-per-task=2             # 2 CPU cores
#SBATCH --partition=instruction           # Valid partition

# Print the hostname of the compute node
hostname
