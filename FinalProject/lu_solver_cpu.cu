#include "lu_solver.h"
#include <cmath>
#include <algorithm>

namespace LUSolver {

void LUDecompositionCPU::decompose(std::vector<float>& A) {
    // In-place LU decomposition without pivoting
    // A is stored in row-major order
    
    for (int k = 0; k < matrix_size; ++k) {
        // Check for zero pivot
        if (std::abs(A[k * matrix_size + k]) < 1e-10) {
            std::cerr << "Warning: Small pivot at position " << k << std::endl;
        }
        
        // Scale column k (find multipliers for L)
        for (int i = k + 1; i < matrix_size; ++i) {
            A[i * matrix_size + k] /= A[k * matrix_size + k];
        }
        
        // Update the rest of the matrix
        for (int i = k + 1; i < matrix_size; ++i) {
            for (int j = k + 1; j < matrix_size; ++j) {
                A[i * matrix_size + j] -= A[i * matrix_size + k] * A[k * matrix_size + j];
            }
        }
    }
}

void LUDecompositionCPU::forward_substitution(const std::vector<float>& L,
                                             const std::vector<float>& b,
                                             std::vector<float>& y) {
    // Solve Ly = b
    for (int i = 0; i < matrix_size; ++i) {
        y[i] = b[i];
        for (int j = 0; j < i; ++j) {
            y[i] -= L[i * matrix_size + j] * y[j];
        }
        // L is unit lower triangular (diagonal is 1)
    }
}

void LUDecompositionCPU::backward_substitution(const std::vector<float>& U,
                                              const std::vector<float>& y,
                                              std::vector<float>& x) {
    // Solve Ux = y
    for (int i = matrix_size - 1; i >= 0; --i) {
        x[i] = y[i];
        for (int j = i + 1; j < matrix_size; ++j) {
            x[i] -= U[i * matrix_size + j] * x[j];
        }
        x[i] /= U[i * matrix_size + i];
    }
}

void LUDecompositionCPU::solve(const std::vector<float>& A,
                              const std::vector<float>& b,
                              std::vector<float>& x) {
    // Copy A since decompose modifies it in place
    std::vector<float> LU = A;
    
    // Perform decomposition
    decompose(LU);
    
    // Extract L and U from combined matrix
    std::vector<float> L(matrix_size * matrix_size, 0.0f);
    std::vector<float> U(matrix_size * matrix_size, 0.0f);
    
    for (int i = 0; i < matrix_size; ++i) {
        L[i * matrix_size + i] = 1.0f; // Unit lower triangular
        for (int j = 0; j < i; ++j) {
            L[i * matrix_size + j] = LU[i * matrix_size + j];
        }
        for (int j = i; j < matrix_size; ++j) {
            U[i * matrix_size + j] = LU[i * matrix_size + j];
        }
    }
    
    // Solve Ly = b
    std::vector<float> y(matrix_size);
    forward_substitution(L, b, y);
    
    // Solve Ux = y
    backward_substitution(U, y, x);
}

} // namespace LUSolver
