#!/usr/bin/env bash
# submit.sh — SLURM job script for ME759 final project on Euler.
#
# Runs the full pipeline: build, sanity check, benchmark sweep, plot, report.
# Submit from the finalreport/ directory:
#     sbatch submit.sh
# Check status with `squeue -u $USER` and read finalreport.out when done.
#
# If your Euler account uses a different GPU partition or module name, adjust
# the #SBATCH --partition line and the module load line below. The settings
# here follow the general ME759/Euler conventions:
#   --gres=gpu:1   request one GPU of any available type
#   --time=01:00:00 one hour (the sweep takes a few minutes; extra for safety)
#
# After the job finishes, copy these out to your laptop with scp:
#   finalreport.out           - stdout/stderr (build log + sanity check)
#   benchmarks.csv            - raw timings
#   fig_*.png                 - plots
#   final_report.pdf          - rewritten with real numbers

#SBATCH --job-name=me759-final
#SBATCH --output=finalreport.out
#SBATCH --error=finalreport.err
#SBATCH --partition=wacc         # default ME759 partition on Euler; change if needed
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail

# --- Environment -----------------------------------------------------------
# Euler uses Lmod. Check `module avail cuda` on your login node to confirm
# the exact version string; adjust the line below if needed.
module purge
module load nvidia/cuda/12.2.2 || module load cuda || true

echo "== nvcc =="
which nvcc
nvcc --version
echo
echo "== GPU =="
nvidia-smi
echo

# --- Pick a sensible arch based on the GPU actually allocated -------------
# Volta = sm_70, Turing = sm_75, Ampere = sm_80 (A100) / sm_86 (RTX 30xx)
# The default in the Makefile is sm_70, which works on every WMMA-capable
# Euler node. Override here if you know you got an A100 or newer.
ARCH=${ARCH:-sm_70}

echo "== build (arch=$ARCH) =="
make clean
make SM=$ARCH -j 4

echo
echo "== sanity check (N=256) =="
./sanity_check.sh

echo
echo "== benchmark sweep =="
./run_benchmarks.sh

echo
echo "== plots =="
# Euler provides matplotlib/pandas through the python module; if not, ask
# the instructor which Python module has them, or use pip --user.
module load python/miniconda || module load python/3 || true
python3 plot_results.py

echo
echo "== rebuild PDF with real results =="
python3 build_report.py

echo
echo "== done =="
ls -la final_report.pdf benchmarks.csv fig_*.png
