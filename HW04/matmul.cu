#include "matmul.cuh"
#include <cuda_runtime.h>

// Each thread computes one element of the output matrix C
__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n) {
    // Determine the global thread ID
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the thread is within the bounds of the nxn matrix
    if (tid < n * n) {
        // Calculate row and column for this thread (row-major)
        unsigned int row = tid / n;
        unsigned int col = tid % n;

        float sum = 0.0f;
        for (size_t k = 0; k < n; ++k) {
            // A[row][k] * B[k][col]
            sum += A[row * n + k] * B[k * n + col];
        }
        C[tid] = sum;
    }
}

void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block) {
    // Total number of elements in the square matrix
    size_t total_elements = n * n;

    // Calculate the number of blocks needed for a 1D configuration
    unsigned int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    matmul_kernel<<<blocks_per_grid, threads_per_block>>>(A, B, C, n);

    // Note: The assignment mentions using CUDA events for timing in task1.cu, 
    // which serves as an implicit synchronization point. [cite: 22]
}
