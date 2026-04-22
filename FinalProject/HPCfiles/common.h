// common.h
// Shared utilities for CUDA linear solver benchmarks.
// ME759 Spring 2026 - Final Project - Porter Provan
#ifndef COMMON_H
#define COMMON_H

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <random>
#include <vector>

#ifdef __CUDACC__
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err__));                           \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)
#endif

// Simple host-side timer returning elapsed milliseconds.
class HostTimer {
public:
    void start() { t0_ = std::chrono::high_resolution_clock::now(); }
    double stop_ms() {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0_).count();
    }
private:
    std::chrono::high_resolution_clock::time_point t0_;
};

// Generate a diagonally dominant, well-conditioned N x N matrix.
// Diagonal dominance guarantees Jacobi convergence and a stable LU without
// pivoting, which keeps the CUDA implementations clean and comparable.
inline void generate_diag_dominant(float* A, int N, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < N; ++i) {
        float row_sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            if (i != j) {
                float v = dist(rng);
                A[i * N + j] = v;
                row_sum += std::fabs(v);
            }
        }
        // Make diagonal dominate row by a healthy margin.
        A[i * N + i] = row_sum + 1.0f + std::fabs(dist(rng));
    }
}

// Produce b = A * x_true for a chosen x_true (here, x_true = 1, 2, ..., N
// normalized) so we have a ground-truth solution to check against.
inline void build_rhs(const float* A, float* b, float* x_true, int N,
                      unsigned seed = 7) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.5f, 1.5f);
    for (int i = 0; i < N; ++i) x_true[i] = dist(rng);
    for (int i = 0; i < N; ++i) {
        float s = 0.0f;
        for (int j = 0; j < N; ++j) s += A[i * N + j] * x_true[j];
        b[i] = s;
    }
}

// Relative L2 error ||x - x_true|| / ||x_true||.
inline double rel_error(const float* x, const float* x_true, int N) {
    double num = 0.0, den = 0.0;
    for (int i = 0; i < N; ++i) {
        double d = (double)x[i] - (double)x_true[i];
        num += d * d;
        den += (double)x_true[i] * (double)x_true[i];
    }
    return std::sqrt(num / (den > 0.0 ? den : 1.0));
}

// Residual ||Ax - b||_2.
inline double residual_norm(const float* A, const float* x, const float* b,
                            int N) {
    double s = 0.0;
    for (int i = 0; i < N; ++i) {
        double r = 0.0;
        for (int j = 0; j < N; ++j) r += (double)A[i * N + j] * (double)x[j];
        r -= b[i];
        s += r * r;
    }
    return std::sqrt(s);
}

#endif // COMMON_H
