#!/usr/bin/env python3
"""
Benchmark Results Analysis Script
Analyzes CSV output from linear solvers benchmark and generates visualizations
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

def load_results(csv_file):
    """Load benchmark results from CSV file"""
    try:
        df = pd.read_csv(csv_file)
        return df
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
        sys.exit(1)

def plot_timing_comparison(df, method='LU'):
    """Plot execution time comparison for a given method"""
    method_data = df[df['Method'] == method].copy()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    variants = method_data['Variant'].unique()
    sizes = sorted(method_data['MatrixSize'].unique())
    
    for variant in variants:
        variant_data = method_data[method_data['Variant'] == variant]
        variant_data = variant_data.sort_values('MatrixSize')
        ax.plot(variant_data['MatrixSize'], variant_data['TimeMs'], 
                marker='o', label=variant, linewidth=2, markersize=8)
    
    ax.set_xlabel('Matrix Size (n×n)', fontsize=12)
    ax.set_ylabel('Execution Time (ms)', fontsize=12)
    ax.set_title(f'{method} Decomposition: Execution Time Comparison', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_speedup_comparison(df, method='LU'):
    """Plot speedup factors relative to CPU"""
    method_data = df[df['Method'] == method].copy()
    
    # Filter out CPU baseline and compute speedups
    gpu_data = method_data[method_data['Variant'] != 'cpu'].copy()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    variants = [v for v in gpu_data['Variant'].unique() if v != 'cpu']
    sizes = sorted(gpu_data['MatrixSize'].unique())
    
    for variant in variants:
        variant_data = gpu_data[gpu_data['Variant'] == variant].copy()
        variant_data = variant_data.sort_values('MatrixSize')
        ax.plot(variant_data['MatrixSize'], variant_data['Speedup'], 
                marker='s', label=variant, linewidth=2, markersize=8)
    
    # Add reference line for 1x speedup
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='No speedup')
    
    ax.set_xlabel('Matrix Size (n×n)', fontsize=12)
    ax.set_ylabel('Speedup Factor', fontsize=12)
    ax.set_title(f'{method} Decomposition: Speedup vs CPU', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_method_comparison(df):
    """Plot comparison between LU and Jacobi methods"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # LU vs Jacobi timing for CPU
    cpu_data = df[df['Variant'] == 'cpu'].copy()
    
    ax = axes[0]
    for method in ['LU', 'Jacobi']:
        method_data = cpu_data[cpu_data['Method'] == method].sort_values('MatrixSize')
        ax.plot(method_data['MatrixSize'], method_data['TimeMs'], 
                marker='o', label=method, linewidth=2, markersize=8)
    
    ax.set_xlabel('Matrix Size (n×n)', fontsize=11)
    ax.set_ylabel('Execution Time (ms)', fontsize=11)
    ax.set_title('LU vs Jacobi (CPU)', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=10)
    
    # GPU speedup comparison
    ax = axes[1]
    gpu_data = df[df['Variant'] != 'cpu'].copy()
    
    for method in ['LU', 'Jacobi']:
        method_data = gpu_data[gpu_data['Method'] == method].copy()
        # Average speedup across variants
        avg_speedup = method_data.groupby('MatrixSize')['Speedup'].mean().sort_index()
        ax.plot(avg_speedup.index, avg_speedup.values, 
                marker='s', label=method, linewidth=2, markersize=8)
    
    ax.set_xlabel('Matrix Size (n×n)', fontsize=11)
    ax.set_ylabel('Average Speedup', fontsize=11)
    ax.set_title('Average GPU Speedup Comparison', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_scaling_analysis(df):
    """Plot computational scaling behavior"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sizes = sorted(df['MatrixSize'].unique())
    
    # Theoretical scaling: O(n^3) for LU, O(n^2) per iteration for Jacobi
    theoretical_lu = np.array(sizes) ** 3
    theoretical_lu = theoretical_lu / theoretical_lu[0]  # Normalize
    
    ax.plot(sizes, theoretical_lu, 'k--', linewidth=2, label='O(n³) Theory', alpha=0.7)
    
    # Plot actual scaling for best GPU variant
    best_variants = {
        'LU': 'gpu_tiled',
        'Jacobi': 'gpu_tiled'
    }
    
    for method in ['LU', 'Jacobi']:
        variant_data = df[(df['Method'] == method) & 
                         (df['Variant'] == best_variants[method])].sort_values('MatrixSize')
        if len(variant_data) > 0:
            times = variant_data['TimeMs'].values
            times = times / times[0]  # Normalize
            ax.plot(variant_data['MatrixSize'], times, 
                    marker='o', label=f'{method} (actual)', linewidth=2, markersize=8)
    
    ax.set_xlabel('Matrix Size (n×n)', fontsize=12)
    ax.set_ylabel('Relative Execution Time', fontsize=12)
    ax.set_title('Computational Scaling Analysis', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    return fig

def print_summary_statistics(df):
    """Print summary statistics"""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY STATISTICS")
    print("="*80 + "\n")
    
    for method in df['Method'].unique():
        print(f"\n{method} Decomposition:")
        print("-" * 40)
        
        method_data = df[df['Method'] == method]
        
        for size in sorted(method_data['MatrixSize'].unique()):
            size_data = method_data[method_data['MatrixSize'] == size]
            cpu_time = size_data[size_data['Variant'] == 'cpu']['TimeMs'].values
            
            if len(cpu_time) > 0:
                cpu_time = cpu_time[0]
                print(f"\n  Matrix Size: {size}×{size}")
                print(f"    CPU Time: {cpu_time:.3f} ms")
                
                gpu_data = size_data[size_data['Variant'] != 'cpu']
                for _, row in gpu_data.iterrows():
                    speedup = row['Speedup']
                    print(f"    {row['Variant']}: {row['TimeMs']:.3f} ms (speedup: {speedup:.2f}x)")
    
    print("\n" + "="*80)

def main():
    # Parse command line arguments
    csv_file = 'benchmark_results.csv'
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    
    print(f"Loading results from: {csv_file}")
    df = load_results(csv_file)
    
    print(f"Loaded {len(df)} benchmark results")
    print(f"Methods: {', '.join(df['Method'].unique())}")
    print(f"Variants: {', '.join(df['Variant'].unique())}")
    print(f"Matrix sizes: {', '.join(map(str, sorted(df['MatrixSize'].unique())))}")
    
    # Print summary statistics
    print_summary_statistics(df)
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    output_dir = Path('benchmark_plots')
    output_dir.mkdir(exist_ok=True)
    
    # LU timing
    fig = plot_timing_comparison(df, 'LU')
    fig.savefig(output_dir / 'lu_timing_comparison.png', dpi=150, bbox_inches='tight')
    print("  Saved: lu_timing_comparison.png")
    plt.close(fig)
    
    # Jacobi timing
    fig = plot_timing_comparison(df, 'Jacobi')
    fig.savefig(output_dir / 'jacobi_timing_comparison.png', dpi=150, bbox_inches='tight')
    print("  Saved: jacobi_timing_comparison.png")
    plt.close(fig)
    
    # LU speedup
    fig = plot_speedup_comparison(df, 'LU')
    fig.savefig(output_dir / 'lu_speedup_comparison.png', dpi=150, bbox_inches='tight')
    print("  Saved: lu_speedup_comparison.png")
    plt.close(fig)
    
    # Jacobi speedup
    fig = plot_speedup_comparison(df, 'Jacobi')
    fig.savefig(output_dir / 'jacobi_speedup_comparison.png', dpi=150, bbox_inches='tight')
    print("  Saved: jacobi_speedup_comparison.png")
    plt.close(fig)
    
    # Method comparison
    fig = plot_method_comparison(df)
    fig.savefig(output_dir / 'method_comparison.png', dpi=150, bbox_inches='tight')
    print("  Saved: method_comparison.png")
    plt.close(fig)
    
    # Scaling analysis
    fig = plot_scaling_analysis(df)
    fig.savefig(output_dir / 'scaling_analysis.png', dpi=150, bbox_inches='tight')
    print("  Saved: scaling_analysis.png")
    plt.close(fig)
    
    print(f"\nAll plots saved to: {output_dir}/")
    print("\nAnalysis complete!")

if __name__ == '__main__':
    main()
