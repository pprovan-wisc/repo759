// jacobi_cuda.cu
// CUDA implementation of the Jacobi iterative solver.
//
// Two kernels are provided:
//   (1) jacobi_naive_kernel   - one thread per row, reads A from global memory.
//   (2) jacobi_shared_kernel  - tiled: cooperatively load x into shared memory
//                               in blocks of TILE columns, reuse across rows.
// After each iteration, CUB's DeviceReduce computes ||x_new - x||_2 so that
// convergence can be tested on-device with a single device->host copy.
//
// Build (requires CUB, ships with modern CUDA Toolkit):
//   nvcc -O3 -std=c++17 -arch=sm_70 -Iinclude \
//        src/jacobi_cuda.cu -o jacobi_cuda
//
// Usage:
//   ./jacobi_cuda <N> <variant: naive|shared> [max_iters=10000] [tol=1e-5]
// Prints CSV: jacobi,variant,N,time_ms,iters,rel_err,residual

#include "common.h"
#include <cub/cub.cuh>
#include <iostream>
#include <string>

// Tile width for the shared-memory variant. 128 fits comfortably on most GPUs
// and gives enough work per block to amortize the shared-memory load.
constexpr int TILE = 128;

// ---------------------------------------------------------------------------
// Naive Jacobi: every thread owns one row. Reads full row of A from global
// memory plus the entire x vector. Simple but memory bound.
// ---------------------------------------------------------------------------
__global__ void jacobi_naive_kernel(const float* __restrict__ A,
                                    const float* __restrict__ b,
                                    const float* __restrict__ x,
                                    float* __restrict__ x_new,
                                    float* __restrict__ diff_sq,
                                    int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float sigma = 0.0f;
    #pragma unroll 4
    for (int j = 0; j < N; ++j) sigma += A[i * N + j] * x[j];
    sigma -= A[i * N + i] * x[i]; // subtract diagonal term
    float xi_new = (b[i] - sigma) / A[i * N + i];
    x_new[i] = xi_new;

    float d = xi_new - x[i];
    diff_sq[i] = d * d;
}

// ---------------------------------------------------------------------------
// Shared-memory Jacobi: threads in a block cooperatively tile x through
// shared memory. Each thread still computes one row, but the x vector is
// loaded once per tile and reused by all threads in the block. Dramatically
// reduces redundant global loads of x.
// ---------------------------------------------------------------------------
__global__ void jacobi_shared_kernel(const float* __restrict__ A,
                                     const float* __restrict__ b,
                                     const float* __restrict__ x,
                                     float* __restrict__ x_new,
                                     float* __restrict__ diff_sq,
                                     int N) {
    __shared__ float x_tile[TILE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float sigma = 0.0f;
    float diag = 0.0f;

    for (int tile_start = 0; tile_start < N; tile_start += TILE) {
        // Cooperative load: each thread in the block loads one (or more) entries
        // of x for this tile. Using threadIdx.x directly means threads beyond
        // the tile width skip — but blockDim.x is chosen >= TILE.
        int tile_end = min(tile_start + TILE, N);
        int tile_len = tile_end - tile_start;
        if (threadIdx.x < tile_len) {
            x_tile[threadIdx.x] = x[tile_start + threadIdx.x];
        }
        __syncthreads();

        if (i < N) {
            const float* Arow = A + i * N + tile_start;
            #pragma unroll 8
            for (int k = 0; k < tile_len; ++k) {
                int j = tile_start + k;
                float aij = Arow[k];
                if (j == i) {
                    diag = aij;                 // save diagonal for update
                } else {
                    sigma += aij * x_tile[k];
                }
            }
        }
        __syncthreads();
    }

    if (i < N) {
        float xi_new = (b[i] - sigma) / diag;
        x_new[i] = xi_new;
        float d = xi_new - x[i];
        diff_sq[i] = d * d;
    }
}

// Host driver that runs either variant and returns iteration count.
int run_jacobi(const std::string& variant, const float* h_A, const float* h_b,
               float* h_x, int N, int max_iters, float tol, double& out_ms) {
    float *d_A, *d_b, *d_x, *d_x_new, *d_diff;
    size_t bytesN  = N * sizeof(float);
    size_t bytesNN = (size_t)N * N * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_A,     bytesNN));
    CUDA_CHECK(cudaMalloc(&d_b,     bytesN));
    CUDA_CHECK(cudaMalloc(&d_x,     bytesN));
    CUDA_CHECK(cudaMalloc(&d_x_new, bytesN));
    CUDA_CHECK(cudaMalloc(&d_diff,  bytesN));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytesNN, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytesN,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_x,     0, bytesN));
    CUDA_CHECK(cudaMemset(d_x_new, 0, bytesN));

    // CUB reduction scratch
    float* d_diff_out;
    CUDA_CHECK(cudaMalloc(&d_diff_out, sizeof(float)));
    void*  d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceReduce::Sum(d_temp, temp_bytes, d_diff, d_diff_out, N);
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));

    int block = TILE;                    // match TILE so cooperative load is exact
    int grid  = (N + block - 1) / block;

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));

    int it = 0;
    float h_diff_sum = 0.0f;
    for (; it < max_iters; ++it) {
        if (variant == "naive") {
            jacobi_naive_kernel<<<grid, block>>>(d_A, d_b, d_x, d_x_new,
                                                 d_diff, N);
        } else {
            jacobi_shared_kernel<<<grid, block>>>(d_A, d_b, d_x, d_x_new,
                                                  d_diff, N);
        }
        cub::DeviceReduce::Sum(d_temp, temp_bytes, d_diff, d_diff_out, N);

        // Check convergence every 8 iterations to avoid the host sync each step.
        if ((it & 7) == 7) {
            CUDA_CHECK(cudaMemcpy(&h_diff_sum, d_diff_out, sizeof(float),
                                  cudaMemcpyDeviceToHost));
            if (std::sqrt(h_diff_sum) < tol) { ++it; std::swap(d_x, d_x_new); break; }
        }
        std::swap(d_x, d_x_new);
    }
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    out_ms = ms;

    CUDA_CHECK(cudaMemcpy(h_x, d_x, bytesN, cudaMemcpyDeviceToHost));

    cudaFree(d_A); cudaFree(d_b); cudaFree(d_x); cudaFree(d_x_new);
    cudaFree(d_diff); cudaFree(d_diff_out); cudaFree(d_temp);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    return it;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "Usage: %s <N> <naive|shared> [max_iters=10000] [tol=1e-5]\n",
            argv[0]);
        return 1;
    }
    int N = std::atoi(argv[1]);
    std::string variant = argv[2];
    int max_iters = (argc > 3) ? std::atoi(argv[3]) : 10000;
    float tol = (argc > 4) ? (float)std::atof(argv[4]) : 1e-5f;

    std::vector<float> A(N * N), b(N), x(N, 0.0f), x_true(N);
    generate_diag_dominant(A.data(), N);
    build_rhs(A.data(), b.data(), x_true.data(), N);

    double ms = 0.0;
    int iters = run_jacobi(variant, A.data(), b.data(), x.data(), N,
                           max_iters, tol, ms);
    double rerr  = rel_error(x.data(), x_true.data(), N);
    double rnorm = residual_norm(A.data(), x.data(), b.data(), N);

    std::printf("jacobi,%s,%d,%.4f,%d,%.3e,%.3e\n",
                variant.c_str(), N, ms, iters, rerr, rnorm);
    return 0;
}
