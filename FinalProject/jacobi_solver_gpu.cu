#include "jacobi_solver.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

namespace JacobiSolver {

// ============================================================================
// NAIVE KERNEL - Simple thread-per-element approach
// ============================================================================
__global__ void jacobiIterationKernel(const float* A, const float* b,
                                     const float* x_old, float* x_new,
                                     int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        float sum = 0.0f;
        
        // Compute sum of A[i][j] * x_old[j] for j != i
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                sum += A[i * n + j] * x_old[j];
            }
        }
        
        float diag = A[i * n + i];
        x_new[i] = (b[i] - sum) / diag;
    }
}

// ============================================================================
// TILED KERNEL - Optimized with shared memory
// ============================================================================
__global__ void jacobiIterationKernelTiled(const float* A, const float* b,
                                          const float* x_old, float* x_new,
                                          int n, int tile_size) {
    extern __shared__ float shared_mem[];
    
    float* x_tile = shared_mem;
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    
    if (i < n) {
        float sum = 0.0f;
        
        // Process the matrix in tiles
        for (int tile_start = 0; tile_start < n; tile_start += tile_size) {
            // Load tile of x_old into shared memory
            for (int k = tx; k < tile_size && tile_start + k < n; k += blockDim.x) {
                x_tile[k] = x_old[tile_start + k];
            }
            __syncthreads();
            
            // Compute partial sum using tile
            for (int j = 0; j < tile_size && tile_start + j < n; ++j) {
                int col = tile_start + j;
                if (i != col) {
                    sum += A[i * n + col] * x_tile[j];
                }
            }
            __syncthreads();
        }
        
        float diag = A[i * n + i];
        x_new[i] = (b[i] - sum) / diag;
    }
}

// ============================================================================
// MATRIX-VECTOR MULTIPLICATION KERNEL
// ============================================================================
__global__ void matrixVectorMultiplyKernel(const float* A, const float* x,
                                          float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        float sum = 0.0f;
        for (int j = 0; j < n; ++j) {
            sum += A[i * n + j] * x[j];
        }
        y[i] = sum;
    }
}

// ============================================================================
// GPU CLASS IMPLEMENTATIONS
// ============================================================================

void JacobiMethodGPU::solveNaive(const Matrix& A, const Matrix& b, Matrix& x) {
    Timer timer;
    timer.start();
    
    thrust::device_vector<float> x_old(matrix_size, 0.0f);
    thrust::device_vector<float> x_new(matrix_size, 0.0f);
    thrust::device_vector<float> residual_vec(matrix_size);
    
    last_iteration_count = 0;
    
    // Thread configuration
    int threads_per_block = 256;
    int blocks = (matrix_size + threads_per_block - 1) / threads_per_block;
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Launch Jacobi kernel
        jacobiIterationKernel<<<blocks, threads_per_block>>>(
            A.data,
            b.data,
            thrust::raw_pointer_cast(x_old.data()),
            thrust::raw_pointer_cast(x_new.data()),
            matrix_size
        );
        
        checkCuda(cudaGetLastError(), "Jacobi kernel failed");
        checkCuda(cudaDeviceSynchronize(), "Jacobi kernel synchronization failed");
        
        // Compute residual (simplified - just use differences)
        // In full implementation: compute ||b - Ax|| / ||b||
        
        // Compute norm of difference
        thrust::transform(x_old.begin(), x_old.end(), x_new.begin(),
                         residual_vec.begin(), 
                         thrust::minus<float>());
        
        float diff_norm = std::sqrt(thrust::reduce(
            residual_vec.begin(), residual_vec.end(), 0.0f,
            thrust::plus<float>()
        ));
        
        last_residual = diff_norm;
        last_iteration_count++;
        
        if (diff_norm < tolerance) {
            std::cout << "Jacobi (GPU Naive) converged in " << last_iteration_count
                      << " iterations with residual " << last_residual << std::endl;
            break;
        }
        
        // Swap vectors
        std::swap(x_old, x_new);
    }
    
    // Copy result to output
    thrust::copy(x_old.begin(), x_old.end(),
                thrust::device_pointer_cast(x.data));
    
    timer.stop();
    std::cout << "Jacobi GPU Naive solve: " << timer.getElapsedMilliseconds()
              << " ms\n";
}

void JacobiMethodGPU::solveTiled(const Matrix& A, const Matrix& b, Matrix& x) {
    Timer timer;
    timer.start();
    
    thrust::device_vector<float> x_old(matrix_size, 0.0f);
    thrust::device_vector<float> x_new(matrix_size, 0.0f);
    
    last_iteration_count = 0;
    
    int threads_per_block = 256;
    int blocks = (matrix_size + threads_per_block - 1) / threads_per_block;
    size_t shared_mem = tile_size * sizeof(float);
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Launch tiled Jacobi kernel
        jacobiIterationKernelTiled<<<blocks, threads_per_block, shared_mem>>>(
            A.data,
            b.data,
            thrust::raw_pointer_cast(x_old.data()),
            thrust::raw_pointer_cast(x_new.data()),
            matrix_size,
            tile_size
        );
        
        checkCuda(cudaGetLastError(), "Jacobi tiled kernel failed");
        checkCuda(cudaDeviceSynchronize(), "Jacobi tiled kernel synchronization failed");
        
        // Simplified convergence check
        float diff_norm = 0.0f;
        for (int i = 0; i < matrix_size; ++i) {
            float diff = x_old[i] - x_new[i];
            diff_norm += diff * diff;
        }
        diff_norm = std::sqrt(diff_norm);
        
        last_residual = diff_norm;
        last_iteration_count++;
        
        if (diff_norm < tolerance) {
            std::cout << "Jacobi (GPU Tiled) converged in " << last_iteration_count
                      << " iterations\n";
            break;
        }
        
        std::swap(x_old, x_new);
    }
    
    thrust::copy(x_old.begin(), x_old.end(),
                thrust::device_pointer_cast(x.data));
    
    timer.stop();
    std::cout << "Jacobi GPU Tiled solve: " << timer.getElapsedMilliseconds()
              << " ms\n";
}

} // namespace JacobiSolver
