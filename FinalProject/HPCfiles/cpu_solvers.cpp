// cpu_solvers.cpp
// Serial CPU implementations of Jacobi iteration and LU decomposition (Doolittle)
// used as the baseline against which the CUDA kernels are compared.
//
// Build:
//   g++ -O3 -std=c++17 -Iinclude src/cpu_solvers.cpp -o cpu_solvers
//
// Usage:
//   ./cpu_solvers <N> <solver: jacobi|lu> [max_iters=10000] [tol=1e-5]
//
// Prints a single CSV line: solver,N,time_ms,iters,rel_err,residual

#include "common.h"

#include <iostream>
#include <string>

// ---------------------------------------------------------------------------
// Jacobi: iterate x_{k+1}[i] = (b[i] - sum_{j!=i} A[i,j] x_k[j]) / A[i,i]
// ---------------------------------------------------------------------------
int jacobi_cpu(const float* A, const float* b, float* x, int N,
               int max_iters, float tol) {
    std::vector<float> x_new(N, 0.0f);
    for (int i = 0; i < N; ++i) x[i] = 0.0f;

    int it = 0;
    for (; it < max_iters; ++it) {
        double diff2 = 0.0, norm2 = 0.0;
        for (int i = 0; i < N; ++i) {
            float sigma = 0.0f;
            for (int j = 0; j < N; ++j)
                if (j != i) sigma += A[i * N + j] * x[j];
            x_new[i] = (b[i] - sigma) / A[i * N + i];
            double d = (double)x_new[i] - (double)x[i];
            diff2 += d * d;
            norm2 += (double)x_new[i] * (double)x_new[i];
        }
        for (int i = 0; i < N; ++i) x[i] = x_new[i];
        if (std::sqrt(diff2) < tol * std::sqrt(norm2 + 1e-30)) {
            ++it;
            break;
        }
    }
    return it;
}

// ---------------------------------------------------------------------------
// LU (Doolittle, no pivoting — safe because the matrix is diagonally dominant).
// Overwrites A in place: lower triangle (unit diagonal implicit) + upper.
// Then performs forward/back substitution to solve.
// ---------------------------------------------------------------------------
void lu_decompose(float* A, int N) {
    for (int k = 0; k < N; ++k) {
        float pivot = A[k * N + k];
        for (int i = k + 1; i < N; ++i) {
            float factor = A[i * N + k] / pivot;
            A[i * N + k] = factor;
            for (int j = k + 1; j < N; ++j) {
                A[i * N + j] -= factor * A[k * N + j];
            }
        }
    }
}

void lu_solve(const float* LU, const float* b, float* x, int N) {
    std::vector<float> y(N);
    // Forward: L y = b  (L is unit lower triangular; diagonal is implicit 1)
    for (int i = 0; i < N; ++i) {
        float s = b[i];
        for (int j = 0; j < i; ++j) s -= LU[i * N + j] * y[j];
        y[i] = s;
    }
    // Back: U x = y
    for (int i = N - 1; i >= 0; --i) {
        float s = y[i];
        for (int j = i + 1; j < N; ++j) s -= LU[i * N + j] * x[j];
        x[i] = s / LU[i * N + i];
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "Usage: %s <N> <jacobi|lu> [max_iters=10000] [tol=1e-5]\n", argv[0]);
        return 1;
    }
    int N = std::atoi(argv[1]);
    std::string solver = argv[2];
    int max_iters = (argc > 3) ? std::atoi(argv[3]) : 10000;
    float tol = (argc > 4) ? (float)std::atof(argv[4]) : 1e-5f;

    std::vector<float> A(N * N), b(N), x(N, 0.0f), x_true(N);
    generate_diag_dominant(A.data(), N);
    build_rhs(A.data(), b.data(), x_true.data(), N);

    HostTimer t;
    int iters = 0;
    double time_ms = 0.0;

    if (solver == "jacobi") {
        t.start();
        iters = jacobi_cpu(A.data(), b.data(), x.data(), N, max_iters, tol);
        time_ms = t.stop_ms();
    } else if (solver == "lu") {
        std::vector<float> LU = A; // LU decomposition is destructive
        t.start();
        lu_decompose(LU.data(), N);
        lu_solve(LU.data(), b.data(), x.data(), N);
        time_ms = t.stop_ms();
        iters = 1;
    } else {
        std::fprintf(stderr, "Unknown solver: %s\n", solver.c_str());
        return 1;
    }

    double rerr = rel_error(x.data(), x_true.data(), N);
    double rnorm = residual_norm(A.data(), x.data(), b.data(), N);

    // CSV: solver,implementation,N,time_ms,iters,rel_err,residual
    std::printf("%s,cpu,%d,%.4f,%d,%.3e,%.3e\n",
                solver.c_str(), N, time_ms, iters, rerr, rnorm);
    return 0;
}
