# Monitor with `squeue -u $USER` and read finalreport.out when done.

#SBATCH --job-name=me759-final
#SBATCH --output=finalreport.out
#SBATCH --error=finalreport.err
#SBATCH --partition=instruction
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00

set -euo pipefail

# --- Environment -----------------------------------------------------------
module purge
module load nvidia/cuda || module load cuda || true

echo "== nvcc =="
which nvcc
nvcc --version
echo
echo "== GPU =="
nvidia-smi
echo

# --- Compile ---------------------------------------------------------------
# Target arch: sm_70 (Volta) works on every WMMA-capable Euler node. If you
# land on an Ampere A100, change to sm_80 for newer Tensor Core features.
ARCH=sm_80

echo "== compile cpu_solvers =="
g++ -O3 -std=c++17 -Wall -Wextra cpu_solvers.cpp -o cpu_solvers

echo "== compile jacobi_cuda =="
nvcc -O3 -std=c++17 -arch=$ARCH --expt-relaxed-constexpr \
     -Xcompiler -Wall jacobi_cuda.cu -o jacobi_cuda

echo "== compile lu_cuda =="
nvcc -O3 -std=c++17 -arch=$ARCH --expt-relaxed-constexpr \
     -Xcompiler -Wall lu_cuda.cu -o lu_cuda

echo "== compile lu_wmma =="
nvcc -O3 -std=c++17 -arch=$ARCH --expt-relaxed-constexpr \
     -Xcompiler -Wall lu_wmma.cu -o lu_wmma

# --- Run -------------------------------------------------------------------
echo
echo "== benchmark sweep =="
chmod +x run_benchmarks.sh
./run_benchmarks.sh

echo
echo "== plots =="
module load python/miniconda || module load python/3 || true
python3 plot_results.py

echo
echo "== rebuild PDF with real results =="
python3 build_report.py

echo
echo "== done =="
ls -la final_report.pdf benchmarks.csv fig_*.png
