#pragma once

#include "matrix_utils.h"

// Jacobi Method namespace
namespace JacobiSolver {

// CUDA kernel declarations
__global__ void jacobiIterationKernel(const float* A, const float* b,
                                     const float* x_old, float* x_new,
                                     int n);

__global__ void jacobiIterationKernelTiled(const float* A, const float* b,
                                          const float* x_old, float* x_new,
                                          int n, int tile_size);

__global__ void matrixVectorMultiplyKernel(const float* A, const float* x,
                                          float* y, int n);

// Host interface functions
class JacobiMethodCPU {
public:
    JacobiMethodCPU(int n, float tol = 1e-6, int max_iter = 10000)
        : matrix_size(n), tolerance(tol), max_iterations(max_iter) {}
    
    // Solve Ax = b using Jacobi iteration on CPU
    void solve(const std::vector<float>& A, const std::vector<float>& b,
              std::vector<float>& x);
    
    // Single iteration step
    void jacobiIteration(const std::vector<float>& A,
                        const std::vector<float>& b,
                        const std::vector<float>& x_old,
                        std::vector<float>& x_new);
    
    int getIterationCount() const { return last_iteration_count; }
    float getResidual() const { return last_residual; }

private:
    int matrix_size;
    float tolerance;
    int max_iterations;
    int last_iteration_count;
    float last_residual;
    
    float computeResidual(const std::vector<float>& A,
                         const std::vector<float>& b,
                         const std::vector<float>& x);
};

class JacobiMethodGPU {
public:
    JacobiMethodGPU(int n, float tol = 1e-6, int max_iter = 10000,
                   int tile_sz = 32)
        : matrix_size(n), tolerance(tol), max_iterations(max_iter),
          tile_size(tile_sz) {}
    
    // Solve Ax = b using Jacobi iteration on GPU - Naive version
    void solveNaive(const Matrix& A, const Matrix& b, Matrix& x);
    
    // Solve Ax = b using Jacobi iteration on GPU - Tiled version
    void solveTiled(const Matrix& A, const Matrix& b, Matrix& x);
    
    int getIterationCount() const { return last_iteration_count; }
    float getResidual() const { return last_residual; }

private:
    int matrix_size;
    float tolerance;
    int max_iterations;
    int tile_size;
    int last_iteration_count;
    float last_residual;
    
    float computeResidualGPU(const Matrix& A, const Matrix& b,
                            const Matrix& x);
};

} // namespace JacobiSolver

