#include "matmul.cuh"
#include <cuda_runtime.h>

template <typename T>
__global__ void matmul_kernel(const T* A, const T* B, T* C, unsigned int n) {
    // Shared memory for tiles
    extern __shared__ char shared_mem[];
    T* sA = reinterpret_cast<T*>(shared_mem);
    T* sB = sA + blockDim.x * blockDim.y;

    unsigned int bx = blockIdx.x, by = blockIdx.y;
    unsigned int tx = threadIdx.x, ty = threadIdx.y;

    unsigned int row = by * blockDim.y + ty;
    unsigned int col = bx * blockDim.x + tx;

    T sum = 0;

    for (unsigned int m = 0; m < (n + blockDim.x - 1) / blockDim.x; ++m) {
        // Load tiles to shared memory with boundary checks
        if (row < n && (m * blockDim.x + tx) < n)
            sA[ty * blockDim.x + tx] = A[row * n + m * blockDim.x + tx];
        else
            sA[ty * blockDim.x + tx] = 0;

        if (col < n && (m * blockDim.y + ty) < n)
            sB[ty * blockDim.x + tx] = B[(m * blockDim.y + ty) * n + col];
        else
            sB[ty * blockDim.x + tx] = 0;

        __syncthreads();

        for (unsigned int k = 0; k < blockDim.x; ++k)
            sum += sA[ty * blockDim.x + k] * sB[k * blockDim.x + tx];

        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}

template <typename T>
void matmul_wrapper(const T* A, const T* B, T* C, unsigned int n, unsigned int block_dim) {
    dim3 threads(block_dim, block_dim);
    dim3 blocks((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    size_t shared_size = 2 * block_dim * block_dim * sizeof(T);

    matmul_kernel<T><<<blocks, threads, shared_size>>>(A, B, C, n);
    cudaDeviceSynchronize();
}

void matmul_1(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim) {
    matmul_wrapper(A, B, C, n, block_dim);
}
void matmul_2(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim) {
    matmul_wrapper(A, B, C, n, block_dim);
}
void matmul_3(const double *A, const double *B, double *C, unsigned int n, unsigned int block_dim) {
    matmul_wrapper(A, B, C, n, block_dim);
}
