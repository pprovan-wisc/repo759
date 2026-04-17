#pragma once

#include "matrix_utils.h"

// LU Decomposition namespace
namespace LUSolver {

// CUDA kernel declarations
__global__ void luDecompKernelNaive(float* A, int n);
__global__ void luDecompKernelTiled(float* A, int n, int tile_size);
__global__ void luDecompKernelWMMA(float* A, int n);

// Host interface functions
class LUDecompositionCPU {
public:
    LUDecompositionCPU(int n) : matrix_size(n) {}
    
    // Perform LU decomposition on CPU (reference implementation)
    void decompose(std::vector<float>& A);
    
    // Solve Ax = b using computed LU factors
    void solve(const std::vector<float>& A, const std::vector<float>& b,
              std::vector<float>& x);

private:
    int matrix_size;
    void forward_substitution(const std::vector<float>& L, 
                             const std::vector<float>& b,
                             std::vector<float>& y);
    void backward_substitution(const std::vector<float>& U,
                              const std::vector<float>& y,
                              std::vector<float>& x);
};

class LUDecompositionGPU {
public:
    LUDecompositionGPU(int n, int tile_sz = 32)
        : matrix_size(n), tile_size(tile_sz) {}
    
    // Perform LU decomposition on GPU - Naive version
    void decomposeNaive(Matrix& A);
    
    // Perform LU decomposition on GPU - Tiled version
    void decomposeTiled(Matrix& A);
    
    // Perform LU decomposition on GPU - WMMA accelerated version
    void decomposeWMMA(Matrix& A);
    
    // Solve Ax = b using computed LU factors on GPU
    void solveGPU(const Matrix& A, const Matrix& b, Matrix& x);

private:
    int matrix_size;
    int tile_size;
};

} // namespace LUSolver
