#!/usr/bin/env bash
# sanity_check.sh - quick correctness check on a small problem.
# Runs each solver at N=256 and prints its rel_err/residual.
set -euo pipefail

N=${N:-256}
TOL=1e-5

echo "Running sanity check at N=$N"
echo "Expect rel_err < 1e-3 for FP32 solvers, ~1e-2 for WMMA (FP16 inputs)."
echo "----------------------------------------------------------------"
printf "%-25s %-12s %-12s %-12s\n" "binary" "time_ms" "rel_err" "residual"
echo "----------------------------------------------------------------"

check() {
    local line="$1"
    local label="$2"
    local t rerr res
    t=$(echo "$line"    | cut -d, -f4)
    rerr=$(echo "$line" | cut -d, -f6)
    res=$(echo "$line"  | cut -d, -f7)
    printf "%-25s %-12s %-12s %-12s\n" "$label" "$t" "$rerr" "$res"
}

check "$(./cpu_solvers  $N jacobi 5000 $TOL)"  "cpu/jacobi"
check "$(./cpu_solvers  $N lu)"                 "cpu/lu"
check "$(./jacobi_cuda  $N naive  5000 $TOL)"  "cuda/jacobi/naive"
check "$(./jacobi_cuda  $N shared 5000 $TOL)"  "cuda/jacobi/shared"
check "$(./lu_cuda      $N naive)"              "cuda/lu/naive"
check "$(./lu_cuda      $N tiled)"              "cuda/lu/tiled"
check "$(./lu_wmma      $N)"                    "cuda/lu/wmma"
echo "Sanity check complete."
