#pragma once

#include <vector>
#include <string>
#include "matrix_utils.h"

struct BenchmarkResult {
    std::string method;
    std::string variant; // "cpu", "gpu_naive", "gpu_tiled", "gpu_wmma"
    int matrix_size;
    float time_ms;
    int iterations;
    float residual;
    float speedup; // relative to CPU baseline
};

class Benchmarker {
public:
    Benchmarker() = default;
    
    // Run benchmark suite for all methods
    void runFullBenchmark(const std::vector<int>& matrix_sizes,
                         const std::string& output_file = "benchmark_results.csv");
    
    // Individual benchmark functions
    BenchmarkResult benchmarkLU_CPU(int size);
    BenchmarkResult benchmarkLU_GPUNaive(int size);
    BenchmarkResult benchmarkLU_GPUTiled(int size);
    BenchmarkResult benchmarkLU_WMMA(int size);
    
    BenchmarkResult benchmarkJacobi_CPU(int size);
    BenchmarkResult benchmarkJacobi_GPUNaive(int size);
    BenchmarkResult benchmarkJacobi_GPUTiled(int size);
    
    // Utility functions
    void saveResults(const std::vector<BenchmarkResult>& results,
                    const std::string& filename);
    void printResults(const std::vector<BenchmarkResult>& results);

private:
    // Generate test matrices
    std::vector<float> generateDiagonallyDominantMatrix(int n);
    std::vector<float> generateRandomMatrix(int n);
    std::vector<float> generateRHS(int n);
    
    // Validation
    bool validateSolution(const std::vector<float>& A,
                         const std::vector<float>& b,
                         const std::vector<float>& x);
};
