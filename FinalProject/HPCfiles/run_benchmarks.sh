#!/usr/bin/env bash
# run_benchmarks.sh - sweep matrix sizes and record results
#
# Writes benchmarks.csv in the current directory with columns:
#       solver,implementation,N,time_ms,iters,rel_err,residual
#
# Usage:
#   ./run_benchmarks.sh                       # default sizes
#   SIZES_JACOBI="256 512 1024" ./run_benchmarks.sh
#
# Warmup: each configuration runs twice; only the second run is recorded.

set -euo pipefail

OUT="benchmarks.csv"
SIZES_JACOBI=${SIZES_JACOBI:-"256 512 1024 2048 4096"}
SIZES_LU=${SIZES_LU:-"256 512 1024 2048"}          # LU is O(N^3)
SIZES_CPU=${SIZES_CPU:-"256 512 1024"}             # CPU crawls past this
MAX_ITERS=${MAX_ITERS:-5000}
TOL=${TOL:-1e-5}

echo "solver,implementation,N,time_ms,iters,rel_err,residual" > "$OUT"

run() {
    local cmd="$1"
    # warmup
    eval "$cmd" > /dev/null
    # measured
    eval "$cmd" >> "$OUT"
}

echo "== CPU baselines =="
for N in $SIZES_CPU; do
    run "./cpu_solvers $N jacobi $MAX_ITERS $TOL"
    run "./cpu_solvers $N lu"
done

echo "== CUDA Jacobi =="
for N in $SIZES_JACOBI; do
    run "./jacobi_cuda $N naive  $MAX_ITERS $TOL"
    run "./jacobi_cuda $N shared $MAX_ITERS $TOL"
done

echo "== CUDA LU =="
for N in $SIZES_LU; do
    run "./lu_cuda $N naive"
    run "./lu_cuda $N tiled"
done

echo "== CUDA LU + WMMA =="
for N in $SIZES_LU; do
    run "./lu_wmma $N"
done

echo "Done. Results in $OUT"
column -s, -t "$OUT" | head -n 60
