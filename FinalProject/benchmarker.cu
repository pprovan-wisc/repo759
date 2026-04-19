#include "benchmarker.h"
#include "lu_solver.h"
#include "jacobi_solver.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <cmath>

std::vector<float> Benchmarker::generateDiagonallyDominantMatrix(int n) {
    std::vector<float> A(n * n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    
    // Fill matrix with random values
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i * n + j] = dis(gen);
        }
    }
    
    // Make diagonally dominant for better convergence in Jacobi
    for (int i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                sum += std::abs(A[i * n + j]);
            }
        }
        A[i * n + i] = sum + 1.0f; // Ensure diagonal dominance
    }
    
    return A;
}

std::vector<float> Benchmarker::generateRandomMatrix(int n) {
    std::vector<float> A(n * n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-10.0, 10.0);
    
    for (int i = 0; i < n * n; ++i) {
        A[i] = dis(gen);
    }
    
    return A;
}

std::vector<float> Benchmarker::generateRHS(int n) {
    std::vector<float> b(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-10.0, 10.0);
    
    for (int i = 0; i < n; ++i) {
        b[i] = dis(gen);
    }
    
    return b;
}

bool Benchmarker::validateSolution(const std::vector<float>& A,
                                  const std::vector<float>& b,
                                  const std::vector<float>& x) {
    int n = b.size();
    float tolerance = 1e-5f;
    
    // Compute Ax
    std::vector<float> Ax(n, 0.0f);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Ax[i] += A[i * n + j] * x[j];
        }
    }
    
    // Compute residual
    float residual_norm = 0.0f, b_norm = 0.0f;
    for (int i = 0; i < n; ++i) {
        float diff = b[i] - Ax[i];
        residual_norm += diff * diff;
        b_norm += b[i] * b[i];
    }
    
    residual_norm = std::sqrt(residual_norm);
    b_norm = std::sqrt(b_norm);
    
    float rel_error = (b_norm > 1e-10) ? residual_norm / b_norm : residual_norm;
    
    std::cout << "Validation: relative error = " << rel_error << std::endl;
    return rel_error < tolerance;
}

BenchmarkResult Benchmarker::benchmarkLU_CPU(int size) {
    std::cout << "\n=== Benchmarking LU (CPU) with size " << size << " ===\n";
    
    auto A = generateRandomMatrix(size);
    auto b = generateRHS(size);
    std::vector<float> x(size);
    
    LUSolver::LUDecompositionCPU solver(size);
    
    Timer timer;
    timer.start();
    solver.solve(A, b, x);
    timer.stop();
    
    float time_ms = timer.getElapsedMilliseconds();
    validateSolution(A, b, x);
    
    BenchmarkResult result;
    result.method = "LU";
    result.variant = "cpu";
    result.matrix_size = size;
    result.time_ms = time_ms;
    result.iterations = 1;
    result.residual = 0.0f;
    result.speedup = 1.0f;
    
    std::cout << "Time: " << time_ms << " ms\n";
    return result;
}

BenchmarkResult Benchmarker::benchmarkLU_GPUNaive(int size) {
    std::cout << "\n=== Benchmarking LU (GPU Naive) with size " << size << " ===\n";
    
    auto A = generateRandomMatrix(size);
    auto b = generateRHS(size);
    
    Matrix A_gpu(size, size, true);
    Matrix b_gpu(size, 1, true);
    Matrix x_gpu(size, 1, true);
    
    A_gpu.copyHostToDevice(A);
    b_gpu.copyHostToDevice(b);
    
    LUSolver::LUDecompositionGPU solver(size);
    
    Timer timer;
    timer.start();
    solver.decomposeNaive(A_gpu);
    timer.stop();
    
    float time_ms = timer.getElapsedMilliseconds();
    
    BenchmarkResult result;
    result.method = "LU";
    result.variant = "gpu_naive";
    result.matrix_size = size;
    result.time_ms = time_ms;
    result.iterations = 1;
    result.residual = 0.0f;
    result.speedup = 0.0f; // Will be computed later
    
    std::cout << "Time: " << time_ms << " ms\n";
    return result;
}

BenchmarkResult Benchmarker::benchmarkLU_GPUTiled(int size) {
    std::cout << "\n=== Benchmarking LU (GPU Tiled) with size " << size << " ===\n";
    
    auto A = generateRandomMatrix(size);
    auto b = generateRHS(size);
    
    Matrix A_gpu(size, size, true);
    Matrix b_gpu(size, 1, true);
    Matrix x_gpu(size, 1, true);
    
    A_gpu.copyHostToDevice(A);
    b_gpu.copyHostToDevice(b);
    
    LUSolver::LUDecompositionGPU solver(size, 32);
    
    Timer timer;
    timer.start();
    solver.decomposeTiled(A_gpu);
    timer.stop();
    
    float time_ms = timer.getElapsedMilliseconds();
    
    BenchmarkResult result;
    result.method = "LU";
    result.variant = "gpu_tiled";
    result.matrix_size = size;
    result.time_ms = time_ms;
    result.iterations = 1;
    result.residual = 0.0f;
    result.speedup = 0.0f;
    
    std::cout << "Time: " << time_ms << " ms\n";
    return result;
}

BenchmarkResult Benchmarker::benchmarkLU_WMMA(int size) {
    std::cout << "\n=== Benchmarking LU (GPU WMMA) with size " << size << " ===\n";
    
    auto A = generateRandomMatrix(size);
    auto b = generateRHS(size);
    
    Matrix A_gpu(size, size, true);
    Matrix b_gpu(size, 1, true);
    Matrix x_gpu(size, 1, true);
    
    A_gpu.copyHostToDevice(A);
    b_gpu.copyHostToDevice(b);
    
    LUSolver::LUDecompositionGPU solver(size);
    
    Timer timer;
    timer.start();
    solver.decomposeWMMA(A_gpu);
    timer.stop();
    
    float time_ms = timer.getElapsedMilliseconds();
    
    BenchmarkResult result;
    result.method = "LU";
    result.variant = "gpu_wmma";
    result.matrix_size = size;
    result.time_ms = time_ms;
    result.iterations = 1;
    result.residual = 0.0f;
    result.speedup = 0.0f;
    
    std::cout << "Time: " << time_ms << " ms\n";
    return result;
}

BenchmarkResult Benchmarker::benchmarkJacobi_CPU(int size) {
    std::cout << "\n=== Benchmarking Jacobi (CPU) with size " << size << " ===\n";
    
    auto A = generateDiagonallyDominantMatrix(size);
    auto b = generateRHS(size);
    std::vector<float> x(size);
    
    JacobiSolver::JacobiMethodCPU solver(size, 1e-6, 10000);
    
    Timer timer;
    timer.start();
    solver.solve(A, b, x);
    timer.stop();
    
    float time_ms = timer.getElapsedMilliseconds();
    validateSolution(A, b, x);
    
    BenchmarkResult result;
    result.method = "Jacobi";
    result.variant = "cpu";
    result.matrix_size = size;
    result.time_ms = time_ms;
    result.iterations = solver.getIterationCount();
    result.residual = solver.getResidual();
    result.speedup = 1.0f;
    
    std::cout << "Time: " << time_ms << " ms, Iterations: " << result.iterations << "\n";
    return result;
}

BenchmarkResult Benchmarker::benchmarkJacobi_GPUNaive(int size) {
    std::cout << "\n=== Benchmarking Jacobi (GPU Naive) with size " << size << " ===\n";
    
    auto A = generateDiagonallyDominantMatrix(size);
    auto b = generateRHS(size);
    
    Matrix A_gpu(size, size, true);
    Matrix b_gpu(size, 1, true);
    Matrix x_gpu(size, 1, true);
    
    A_gpu.copyHostToDevice(A);
    b_gpu.copyHostToDevice(b);
    
    JacobiSolver::JacobiMethodGPU solver(size, 1e-6, 10000);
    
    Timer timer;
    timer.start();
    solver.solveNaive(A_gpu, b_gpu, x_gpu);
    timer.stop();
    
    float time_ms = timer.getElapsedMilliseconds();
    
    BenchmarkResult result;
    result.method = "Jacobi";
    result.variant = "gpu_naive";
    result.matrix_size = size;
    result.time_ms = time_ms;
    result.iterations = solver.getIterationCount();
    result.residual = solver.getResidual();
    result.speedup = 0.0f;
    
    std::cout << "Time: " << time_ms << " ms, Iterations: " << result.iterations << "\n";
    return result;
}

BenchmarkResult Benchmarker::benchmarkJacobi_GPUTiled(int size) {
    std::cout << "\n=== Benchmarking Jacobi (GPU Tiled) with size " << size << " ===\n";
    
    auto A = generateDiagonallyDominantMatrix(size);
    auto b = generateRHS(size);
    
    Matrix A_gpu(size, size, true);
    Matrix b_gpu(size, 1, true);
    Matrix x_gpu(size, 1, true);
    
    A_gpu.copyHostToDevice(A);
    b_gpu.copyHostToDevice(b);
    
    JacobiSolver::JacobiMethodGPU solver(size, 1e-6, 10000, 32);
    
    Timer timer;
    timer.start();
    solver.solveTiled(A_gpu, b_gpu, x_gpu);
    timer.stop();
    
    float time_ms = timer.getElapsedMilliseconds();
    
    BenchmarkResult result;
    result.method = "Jacobi";
    result.variant = "gpu_tiled";
    result.matrix_size = size;
    result.time_ms = time_ms;
    result.iterations = solver.getIterationCount();
    result.residual = solver.getResidual();
    result.speedup = 0.0f;
    
    std::cout << "Time: " << time_ms << " ms, Iterations: " << result.iterations << "\n";
    return result;
}

void Benchmarker::saveResults(const std::vector<BenchmarkResult>& results,
                             const std::string& filename) {
    std::ofstream file(filename);
    
    // CSV header
    file << "Method,Variant,MatrixSize,TimeMs,Iterations,Residual,Speedup\n";
    
    for (const auto& result : results) {
        file << result.method << ","
             << result.variant << ","
             << result.matrix_size << ","
             << std::fixed << std::setprecision(6) << result.time_ms << ","
             << result.iterations << ","
             << result.residual << ","
             << result.speedup << "\n";
    }
    
    file.close();
    std::cout << "\nResults saved to " << filename << "\n";
}

void Benchmarker::printResults(const std::vector<BenchmarkResult>& results) {
    std::cout << "\n" << std::string(100, '=') << "\n";
    std::cout << "BENCHMARK RESULTS\n";
    std::cout << std::string(100, '=') << "\n";
    
    std::cout << std::left
              << std::setw(15) << "Method"
              << std::setw(15) << "Variant"
              << std::setw(15) << "Size"
              << std::setw(15) << "Time (ms)"
              << std::setw(15) << "Iterations"
              << std::setw(15) << "Speedup\n";
    
    std::cout << std::string(100, '-') << "\n";
    
    for (const auto& result : results) {
        std::cout << std::left
                  << std::setw(15) << result.method
                  << std::setw(15) << result.variant
                  << std::setw(15) << result.matrix_size
                  << std::setw(15) << std::fixed << std::setprecision(3) << result.time_ms
                  << std::setw(15) << result.iterations
                  << std::setw(15) << std::setprecision(2) << result.speedup << "\n";
    }
    
    std::cout << std::string(100, '=') << "\n";
}

void Benchmarker::runFullBenchmark(const std::vector<int>& matrix_sizes,
                                  const std::string& output_file) {
    std::vector<BenchmarkResult> results;
    
    for (int size : matrix_sizes) {
        std::cout << "\n" << std::string(80, '#') << "\n";
        std::cout << "Testing with matrix size: " << size << "x" << size << "\n";
        std::cout << std::string(80, '#') << "\n";
        
        // LU benchmarks
        auto lu_cpu = benchmarkLU_CPU(size);
        results.push_back(lu_cpu);
        
        auto lu_gpu_naive = benchmarkLU_GPUNaive(size);
        lu_gpu_naive.speedup = lu_cpu.time_ms / lu_gpu_naive.time_ms;
        results.push_back(lu_gpu_naive);
        
        auto lu_gpu_tiled = benchmarkLU_GPUTiled(size);
        lu_gpu_tiled.speedup = lu_cpu.time_ms / lu_gpu_tiled.time_ms;
        results.push_back(lu_gpu_tiled);
        
        auto lu_gpu_wmma = benchmarkLU_WMMA(size);
        lu_gpu_wmma.speedup = lu_cpu.time_ms / lu_gpu_wmma.time_ms;
        results.push_back(lu_gpu_wmma);
        
        // Jacobi benchmarks
        auto jacobi_cpu = benchmarkJacobi_CPU(size);
        results.push_back(jacobi_cpu);
        
        auto jacobi_gpu_naive = benchmarkJacobi_GPUNaive(size);
        jacobi_gpu_naive.speedup = jacobi_cpu.time_ms / jacobi_gpu_naive.time_ms;
        results.push_back(jacobi_gpu_naive);
        
        auto jacobi_gpu_tiled = benchmarkJacobi_GPUTiled(size);
        jacobi_gpu_tiled.speedup = jacobi_cpu.time_ms / jacobi_gpu_tiled.time_ms;
        results.push_back(jacobi_gpu_tiled);
    }
    
    printResults(results);
    saveResults(results, output_file);
}
