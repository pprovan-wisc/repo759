// lu_cuda.cu
// CUDA implementation of right-looking LU decomposition (no pivoting).
//
// The outer loop steps over pivot rows k = 0..N-1. At each step:
//   (a) scale_kernel      : computes A[i,k] /= A[k,k] for i > k   (the L factor)
//   (b) update_kernel     : rank-1 update  A[i,j] -= A[i,k] * A[k,j]  for i,j>k
//
// Two variants of the rank-1 update are provided:
//   * update_naive_kernel  - plain 2D kernel, reads each operand from global mem
//   * update_tiled_kernel  - caches the k-th row and k-th column slices into
//                            shared memory so each thread block reuses them.
//
// Forward/back substitution is done on the host since it is O(N^2) and
// negligible compared to the O(N^3) factorization.
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_70 -Iinclude src/lu_cuda.cu -o lu_cuda
// Usage:
//   ./lu_cuda <N> <naive|tiled>
// Prints CSV: lu,variant,N,time_ms,iters,rel_err,residual

#include "common.h"
#include <iostream>
#include <string>

constexpr int BLK = 16;   // block tile for the rank-1 update (BLK x BLK)

// Scale column k below the pivot: A[i,k] /= A[k,k] for i = k+1..N-1
__global__ void scale_kernel(float* A, int N, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + (k + 1);
    if (i >= N) return;
    A[i * N + k] /= A[k * N + k];
}

// Naive rank-1 update: A[i,j] -= A[i,k] * A[k,j] for i,j in (k, N).
__global__ void update_naive_kernel(float* A, int N, int k) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + (k + 1);
    int j = blockIdx.x * blockDim.x + threadIdx.x + (k + 1);
    if (i >= N || j >= N) return;
    A[i * N + j] -= A[i * N + k] * A[k * N + j];
}

// Tiled rank-1 update: each BLK x BLK block first caches the column slice
// A[i, k] (for its rows) and the row slice A[k, j] (for its columns) into
// shared memory, then performs the multiply-subtract.
__global__ void update_tiled_kernel(float* A, int N, int k) {
    __shared__ float col_k[BLK]; // A[i, k] for rows in this block
    __shared__ float row_k[BLK]; // A[k, j] for cols in this block

    int i = blockIdx.y * BLK + threadIdx.y + (k + 1);
    int j = blockIdx.x * BLK + threadIdx.x + (k + 1);

    // Cooperative load. The first column of threads fetches col_k, the first
    // row fetches row_k.
    if (threadIdx.x == 0 && i < N) col_k[threadIdx.y] = A[i * N + k];
    if (threadIdx.y == 0 && j < N) row_k[threadIdx.x] = A[k * N + j];
    __syncthreads();

    if (i < N && j < N) {
        A[i * N + j] -= col_k[threadIdx.y] * row_k[threadIdx.x];
    }
}

void lu_forward_back_host(const float* LU, const float* b, float* x, int N) {
    std::vector<float> y(N);
    for (int i = 0; i < N; ++i) {
        float s = b[i];
        for (int j = 0; j < i; ++j) s -= LU[i * N + j] * y[j];
        y[i] = s;
    }
    for (int i = N - 1; i >= 0; --i) {
        float s = y[i];
        for (int j = i + 1; j < N; ++j) s -= LU[i * N + j] * x[j];
        x[i] = s / LU[i * N + i];
    }
}

double run_lu(const std::string& variant, const float* h_A, const float* h_b,
              float* h_x, int N) {
    size_t bytesNN = (size_t)N * N * sizeof(float);
    float* d_A;
    CUDA_CHECK(cudaMalloc(&d_A, bytesNN));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytesNN, cudaMemcpyHostToDevice));

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));

    for (int k = 0; k < N - 1; ++k) {
        int rem = N - k - 1;
        int s_block = 128;
        int s_grid  = (rem + s_block - 1) / s_block;
        scale_kernel<<<s_grid, s_block>>>(d_A, N, k);

        if (variant == "naive") {
            dim3 block(16, 16);
            dim3 grid((rem + 15) / 16, (rem + 15) / 16);
            update_naive_kernel<<<grid, block>>>(d_A, N, k);
        } else {
            dim3 block(BLK, BLK);
            dim3 grid((rem + BLK - 1) / BLK, (rem + BLK - 1) / BLK);
            update_tiled_kernel<<<grid, block>>>(d_A, N, k);
        }
    }
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));

    std::vector<float> h_LU(N * N);
    CUDA_CHECK(cudaMemcpy(h_LU.data(), d_A, bytesNN, cudaMemcpyDeviceToHost));
    cudaFree(d_A);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);

    lu_forward_back_host(h_LU.data(), h_b, h_x, N);
    return ms;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s <N> <naive|tiled>\n", argv[0]);
        return 1;
    }
    int N = std::atoi(argv[1]);
    std::string variant = argv[2];

    std::vector<float> A(N * N), b(N), x(N, 0.0f), x_true(N);
    generate_diag_dominant(A.data(), N);
    build_rhs(A.data(), b.data(), x_true.data(), N);

    double ms = run_lu(variant, A.data(), b.data(), x.data(), N);
    double rerr  = rel_error(x.data(), x_true.data(), N);
    double rnorm = residual_norm(A.data(), x.data(), b.data(), N);

    std::printf("lu,%s,%d,%.4f,%d,%.3e,%.3e\n",
                variant.c_str(), N, ms, 1, rerr, rnorm);
    return 0;
}
