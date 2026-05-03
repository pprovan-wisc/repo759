# ME759 Final Project: Comparative Acceleration of Direct and Iterative Linear Solvers using CUDA and Mixed-Precision WMMA

**Author:** Porter Provan (`pprovan@wisc.edu`)
**Course:** ME759, Spring 2026
**Repo:** https://github.com/pprovan-wisc/repo759

---

## Overview

Every file for this project lives in this single directory. The code implements and benchmarks two classical linear-system solvers on the GPU and compares them against a serial CPU baseline:

- **Jacobi** (iterative), with a naive kernel and a shared-memory/tiled kernel, using CUB for on-device global error reduction.
- **LU decomposition** (direct, no pivoting), with a rank-1-update naive kernel, a tiled rank-1-update kernel, and a **panel-blocked mixed-precision variant that uses WMMA Tensor Cores** for the trailing matrix update.

All solvers are tested on the same family of diagonally-dominant test matrices so that (a) Jacobi is guaranteed to converge, and (b) LU without pivoting is numerically stable.

## Files in this directory

```
common.h              shared utilities: timing, matrix generation, error checks
cpu_solvers.cpp       serial Jacobi + LU (baseline)
jacobi_cuda.cu        naive + shared-memory Jacobi kernels + CUB reduction
lu_cuda.cu            naive + tiled LU kernels
lu_wmma.cu            panel-blocked mixed-precision LU with WMMA tensor cores
run_benchmarks.sh     sweeps matrix sizes, writes benchmarks.csv
submit.sh             runs all necessary files to create results
plot_results.py       reads benchmarks.csv, writes fig_*.png
build_report.py       reads benchmarks.csv + fig_*.png, writes final_report.pdf
final_report.pdf      the technical report (pre-run; placeholders for figures)
README.md             this file
```

After a full run the directory additionally contains:
```
benchmarks.csv           raw timings
fig_jacobi_time.png
fig_jacobi_speedup.png
fig_lu_time.png
fig_lu_speedup.png
fig_lu_cpu_speedup.png
```
and `final_report.pdf` is rebuilt with the real numbers substituted in.

## Run

Every binary prints one CSV line of the form:
```
solver,implementation,N,time_ms,iters,rel_err,residual
```

Examples:
```bash
./cpu_solvers  1024 jacobi 5000 1e-5
./cpu_solvers  1024 lu
./jacobi_cuda  1024 shared 5000 1e-5
./lu_cuda      1024 tiled
./lu_wmma      1024
```

## Full reproduction

These commands should be ran using Euler.

```bash
sbatch submit.sh
python3 plot_results.py        # writes fig_*.png
python3 build_report.py        # rebuilds final_report.pdf with real numbers
```

## What to expect

| Comparison | Expected result |
|---|---|
| Jacobi shared vs naive GPU | ~2-4x from removing redundant global loads of x |
| LU tiled vs naive GPU | Modest (1.5-2x); the rank-1 update is already memory-bound |
| LU WMMA vs tiled | Largest at big N where the O(N^3) trailing update dominates |
| LU WMMA relative error | Higher than FP32 path (expect ~1e-2 to 1e-3) - this is the FP16-input accuracy cost, and is discussed in Section 4.3 of the report |

All FP32 solvers should produce relative errors around 1e-6 to 1e-7. If any FP32 solver is worse than ~1e-5, something's wrong.

## AI usage disclosure

Consistent with the course policy, generative AI (Claude, Anthropic) was used as a collaborator during this project - for scoping, code scaffolding, and drafting portions of the technical report. All code was reviewed for correctness and is ultimately my responsibility. A more detailed statement appears at the end of the technical report.
