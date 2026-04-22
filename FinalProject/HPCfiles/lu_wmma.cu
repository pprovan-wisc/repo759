// lu_wmma.cu
// Mixed-precision blocked LU decomposition that uses WMMA (Tensor Cores)
// for the trailing matrix-matrix update.
//
// Standard right-looking LU does a rank-1 update at each step. By panel-
// blocking the algorithm we can instead perform a rank-B update once per
// panel, which is a dense matrix multiply — exactly what Tensor Cores
// accelerate. We:
//   1. Factor a B-wide panel on the "naive" path (small B, so overhead is
//      negligible).
//   2. Triangular-solve to form U12 = L11^{-1} * A12.
//   3. Use a WMMA kernel to compute A22 -= L21 * U12 in FP16 x FP16 -> FP32.
//
// The FP32 accumulation keeps accuracy close to the pure-FP32 baseline while
// the multiply is done at FP16 on Tensor Cores — the "mixed precision" win.
//
// Build (requires sm_70+ for WMMA):
//   nvcc -O3 -std=c++17 -arch=sm_70 -Iinclude src/lu_wmma.cu -o lu_wmma
// Usage:
//   ./lu_wmma <N>         (N should be a multiple of 16; we round up)
// Prints CSV: lu,wmma,N,time_ms,iters,rel_err,residual

#include "common.h"
#include <cuda_fp16.h>
#include <mma.h>
#include <iostream>
#include <string>

using namespace nvcuda::wmma;

constexpr int WMMA_M   = 16;
constexpr int WMMA_N   = 16;
constexpr int WMMA_K   = 16;
constexpr int PANEL_B  = 32;   // panel width; divisible by WMMA_K

// ---------------------------------------------------------------------------
// Panel factorization (right-looking within a single B-wide panel). Handles
// both the scaling (L column) and the rank-1 updates inside the panel.
// Runs with a single block for simplicity since the panel is tiny.
// ---------------------------------------------------------------------------
__global__ void factor_panel_kernel(float* A, int N, int k0, int B) {
    // Threads cooperate over rows i > k within this panel.
    int tid = threadIdx.x;
    for (int kk = 0; kk < B; ++kk) {
        int k = k0 + kk;
        float pivot = A[k * N + k];
        // scale column: A[i,k] /= pivot  for i in (k, N)
        for (int i = k + 1 + tid; i < N; i += blockDim.x) {
            A[i * N + k] /= pivot;
        }
        __syncthreads();
        // rank-1 update inside the remaining (B-kk-1) panel columns
        for (int i = k + 1 + tid; i < N; i += blockDim.x) {
            float lik = A[i * N + k];
            for (int j = k + 1; j < k0 + B && j < N; ++j) {
                A[i * N + j] -= lik * A[k * N + j];
            }
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Triangular solve: U12 = L11^{-1} * A12   (rows k0..k0+B-1, cols k0+B..N-1)
// L11 is the B x B unit lower triangle stored in A[k0..k0+B, k0..k0+B].
// We overwrite A12 in place with U12. Row-by-row forward substitution.
// ---------------------------------------------------------------------------
__global__ void trsm_kernel(float* A, int N, int k0, int B) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + (k0 + B);
    if (j >= N) return;
    for (int ii = 0; ii < B; ++ii) {
        int i = k0 + ii;
        float s = A[i * N + j];
        for (int kk = 0; kk < ii; ++kk) {
            s -= A[i * N + (k0 + kk)] * A[(k0 + kk) * N + j];
        }
        // L11 has unit diagonal so no division needed.
        A[i * N + j] = s;
    }
}

// ---------------------------------------------------------------------------
// Convert an FP32 slab to FP16 for the Tensor-Core multiply inputs.
// Copies A[row_off + i, col_off + j] (FP32) into Hbuf[i * ld + j] (FP16).
// ---------------------------------------------------------------------------
__global__ void f32_to_f16_block(const float* __restrict__ A, int N,
                                 int row_off, int col_off, int rows, int cols,
                                 __half* __restrict__ Hbuf, int ld) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows || j >= cols) return;
    int r = row_off + i;
    int c = col_off + j;
    float v = (r < N && c < N) ? A[r * N + c] : 0.0f;
    Hbuf[i * ld + j] = __float2half(v);
}

// ---------------------------------------------------------------------------
// Trailing update using WMMA:  A22 -= L21 * U12
//   L21 is (M x B),   U12 is (B x N_tail),   A22 is (M x N_tail).
// Each warp computes one 16x16 output tile. FP16 inputs, FP32 accumulator.
// ---------------------------------------------------------------------------
__global__ void wmma_trailing_update(const __half* __restrict__ L21,
                                     const __half* __restrict__ U12,
                                     float* __restrict__ A,
                                     int N, int row_off, int col_off,
                                     int M_rows, int N_cols, int K_dim,
                                     int ldL, int ldU) {
    // One warp per block: blockDim.x == 32, blockDim.y == 1.
    // Block (bx, by) computes the 16x16 output tile at (by*16, bx*16).
    int tile_row = blockIdx.y * WMMA_M;
    int tile_col = blockIdx.x * WMMA_N;
    if (tile_row >= M_rows || tile_col >= N_cols) return;

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, row_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    fill_fragment(acc, 0.0f);

    for (int kk = 0; kk < K_dim; kk += WMMA_K) {
        const __half* Aptr = L21 + tile_row * ldL + kk;
        const __half* Bptr = U12 + kk * ldU + tile_col;
        load_matrix_sync(a_frag, Aptr, ldL);
        load_matrix_sync(b_frag, Bptr, ldU);
        mma_sync(acc, a_frag, b_frag, acc);
    }

    // Store the 16x16 result into shared memory, then cooperatively
    // subtract it from the trailing submatrix of A (in FP32).
    __shared__ float tile_buf[WMMA_M * WMMA_N];
    store_matrix_sync(tile_buf, acc, WMMA_N, mem_row_major);
    __syncwarp();

    int lane = threadIdx.x;
    for (int idx = lane; idx < WMMA_M * WMMA_N; idx += 32) {
        int li = idx / WMMA_N;
        int lj = idx % WMMA_N;
        int gi = row_off + tile_row + li;
        int gj = col_off + tile_col + lj;
        if (gi < N && gj < N) {
            A[gi * N + gj] -= tile_buf[idx];
        }
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

double run_wmma_lu(const float* h_A, const float* h_b, float* h_x, int N) {
    size_t bytesNN = (size_t)N * N * sizeof(float);
    float* d_A;
    CUDA_CHECK(cudaMalloc(&d_A, bytesNN));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytesNN, cudaMemcpyHostToDevice));

    // Reusable FP16 panel buffers sized to the largest possible trailing
    // submatrix (N x PANEL_B for L21, PANEL_B x N for U12), padded to a
    // multiple of 16.
    auto pad16 = [](int x) { return ((x + 15) / 16) * 16; };
    int Npad = pad16(N);
    __half *d_L21, *d_U12;
    CUDA_CHECK(cudaMalloc(&d_L21, (size_t)Npad * PANEL_B * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_U12, (size_t)PANEL_B * Npad * sizeof(__half)));

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));

    for (int k0 = 0; k0 < N; k0 += PANEL_B) {
        int B = std::min(PANEL_B, N - k0);

        // Panel factorization (single block, 128 threads).
        factor_panel_kernel<<<1, 128>>>(d_A, N, k0, B);

        int tail_cols = N - (k0 + B);
        int tail_rows = N - (k0 + B);
        if (tail_cols > 0) {
            // TRSM: form U12 in place
            int block = 128;
            int grid  = (tail_cols + block - 1) / block;
            trsm_kernel<<<grid, block>>>(d_A, N, k0, B);
        }
        if (tail_rows > 0 && tail_cols > 0) {
            // Fill FP16 copies of L21 and U12, padded to multiples of 16.
            int M_pad = pad16(tail_rows);
            int N_pad = pad16(tail_cols);
            // Reuse d_L21 / d_U12 with leading dim = PANEL_B / N_pad
            {
                dim3 bk(16, 16);
                dim3 gr((PANEL_B + 15) / 16, (M_pad + 15) / 16);
                f32_to_f16_block<<<gr, bk>>>(d_A, N, k0 + B, k0, M_pad,
                                             PANEL_B, d_L21, PANEL_B);
            }
            {
                dim3 bk(16, 16);
                dim3 gr((N_pad + 15) / 16, (PANEL_B + 15) / 16);
                f32_to_f16_block<<<gr, bk>>>(d_A, N, k0, k0 + B, PANEL_B,
                                             N_pad, d_U12, N_pad);
            }
            // WMMA trailing update: one warp per block, one 16x16 output
            // tile per warp. Grid dimensions cover the padded trailing sub-
            // matrix; tiles fully inside the padding are still launched but
            // their stores are masked by gi/gj < N.
            int warps_y = M_pad / WMMA_M;
            int warps_x = N_pad / WMMA_N;
            dim3 block(32, 1);
            dim3 grid(warps_x, warps_y);
            wmma_trailing_update<<<grid, block>>>(d_L21, d_U12, d_A, N,
                                                  k0 + B, k0 + B,
                                                  M_pad, N_pad, PANEL_B,
                                                  PANEL_B, N_pad);
        }
    }
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));

    std::vector<float> h_LU(N * N);
    CUDA_CHECK(cudaMemcpy(h_LU.data(), d_A, bytesNN, cudaMemcpyDeviceToHost));
    cudaFree(d_A); cudaFree(d_L21); cudaFree(d_U12);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);

    lu_forward_back_host(h_LU.data(), h_b, h_x, N);
    return ms;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <N>\n", argv[0]);
        return 1;
    }
    int N = std::atoi(argv[1]);

    std::vector<float> A(N * N), b(N), x(N, 0.0f), x_true(N);
    generate_diag_dominant(A.data(), N);
    build_rhs(A.data(), b.data(), x_true.data(), N);

    double ms = run_wmma_lu(A.data(), b.data(), x.data(), N);
    double rerr  = rel_error(x.data(), x_true.data(), N);
    double rnorm = residual_norm(A.data(), x.data(), b.data(), N);

    std::printf("lu,wmma,%d,%.4f,%d,%.3e,%.3e\n", N, ms, 1, rerr, rnorm);
    return 0;
}
