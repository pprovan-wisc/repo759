#!/usr/bin/env python3
"""plot_results.py

Reads benchmarks.csv from the current directory (produced by
run_benchmarks.sh) and writes the following figures next to it:

  fig_jacobi_time.png       time-to-solution vs N for Jacobi (CPU/naive/shared)
  fig_jacobi_speedup.png    speedup of shared over naive GPU Jacobi
  fig_lu_time.png           time-to-solution vs N for LU (CPU/naive/tiled/wmma)
  fig_lu_speedup.png        speedup of tiled and wmma over CPU and naive GPU
  fig_lu_cpu_speedup.png    speedup of the three GPU LU variants over CPU

Usage:
  python3 plot_results.py                      # default: reads benchmarks.csv
  python3 plot_results.py <csv>                # custom CSV path
  python3 plot_results.py <csv> <output_dir>   # custom output dir
"""
import sys
import pathlib
import pandas as pd
import matplotlib.pyplot as plt

csv_path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "benchmarks.csv")
out_dir  = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else ".")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)
df["impl"] = df["solver"] + "-" + df["implementation"]

def plot_time(subset, title, fname, impls):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for impl in impls:
        d = subset[subset["impl"] == impl].sort_values("N")
        if d.empty:
            continue
        ax.plot(d["N"], d["time_ms"], marker="o", label=impl)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Matrix size N")
    ax.set_ylabel("Time to solution (ms)")
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=150)
    plt.close(fig)

def plot_speedup(subset, baseline_impl, target_impls, title, fname):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    base = subset[subset["impl"] == baseline_impl].set_index("N")["time_ms"]
    for impl in target_impls:
        d = subset[subset["impl"] == impl].set_index("N")["time_ms"]
        common = base.index.intersection(d.index)
        if len(common) == 0:
            continue
        sp = base.loc[common] / d.loc[common]
        ax.plot(common, sp.values, marker="s",
                label=f"{impl} vs {baseline_impl}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Matrix size N")
    ax.set_ylabel(f"Speedup over {baseline_impl}")
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.axhline(1.0, color="k", lw=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=150)
    plt.close(fig)

# Jacobi figures ------------------------------------------------------------
jacobi = df[df["solver"] == "jacobi"].copy()
plot_time(jacobi, "Jacobi: time to solution",
          "fig_jacobi_time.png",
          ["jacobi-cpu", "jacobi-naive", "jacobi-shared"])
plot_speedup(jacobi, "jacobi-naive", ["jacobi-shared"],
             "Jacobi GPU: shared-memory speedup over naive",
             "fig_jacobi_speedup.png")

# LU figures ---------------------------------------------------------------
lu = df[df["solver"] == "lu"].copy()
plot_time(lu, "LU decomposition: time to solution",
          "fig_lu_time.png",
          ["lu-cpu", "lu-naive", "lu-tiled", "lu-wmma"])
plot_speedup(lu, "lu-naive", ["lu-tiled", "lu-wmma"],
             "LU GPU: tiled and WMMA speedup over naive",
             "fig_lu_speedup.png")
plot_speedup(lu, "lu-cpu", ["lu-naive", "lu-tiled", "lu-wmma"],
             "LU: GPU speedup over CPU baseline",
             "fig_lu_cpu_speedup.png")

# Accuracy summary ---------------------------------------------------------
print("Accuracy summary (max relative error per implementation):")
print(df.groupby("impl")["rel_err"].max().to_string())
print(f"Figures written to {out_dir.resolve()}/")
