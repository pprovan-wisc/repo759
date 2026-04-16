#include "jacobi_solver.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace JacobiSolver {

float JacobiMethodCPU::computeResidual(const std::vector<float>& A,
                                       const std::vector<float>& b,
                                       const std::vector<float>& x) {
    // Compute ||b - Ax||_2 / ||b||_2
    std::vector<float> residual(matrix_size, 0.0f);
    
    for (int i = 0; i < matrix_size; ++i) {
        residual[i] = b[i];
        for (int j = 0; j < matrix_size; ++j) {
            residual[i] -= A[i * matrix_size + j] * x[j];
        }
    }
    
    // Compute L2 norm of residual and RHS
    float norm_residual = 0.0f, norm_b = 0.0f;
    for (int i = 0; i < matrix_size; ++i) {
        norm_residual += residual[i] * residual[i];
        norm_b += b[i] * b[i];
    }
    
    norm_residual = std::sqrt(norm_residual);
    norm_b = std::sqrt(norm_b);
    
    return (norm_b > 1e-10) ? norm_residual / norm_b : norm_residual;
}

void JacobiMethodCPU::jacobiIteration(const std::vector<float>& A,
                                     const std::vector<float>& b,
                                     const std::vector<float>& x_old,
                                     std::vector<float>& x_new) {
    // x_new[i] = (b[i] - sum(A[i][j] * x_old[j] for j != i)) / A[i][i]
    
    for (int i = 0; i < matrix_size; ++i) {
        float sum = 0.0f;
        
        for (int j = 0; j < matrix_size; ++j) {
            if (i != j) {
                sum += A[i * matrix_size + j] * x_old[j];
            }
        }
        
        float diag = A[i * matrix_size + i];
        if (std::abs(diag) < 1e-10) {
            std::cerr << "Warning: Small diagonal element at position " << i << std::endl;
        }
        
        x_new[i] = (b[i] - sum) / diag;
    }
}

void JacobiMethodCPU::solve(const std::vector<float>& A,
                           const std::vector<float>& b,
                           std::vector<float>& x) {
    // Initialize x to zero
    std::fill(x.begin(), x.end(), 0.0f);
    
    std::vector<float> x_new(matrix_size, 0.0f);
    last_iteration_count = 0;
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Perform Jacobi iteration
        jacobiIteration(A, b, x, x_new);
        
        // Compute residual and check convergence
        last_residual = computeResidual(A, b, x_new);
        last_iteration_count++;
        
        if (last_residual < tolerance) {
            std::cout << "Jacobi converged in " << last_iteration_count
                      << " iterations with residual " << last_residual << std::endl;
            x = x_new;
            return;
        }
        
        // Swap for next iteration
        std::swap(x, x_new);
    }
    
    std::cout << "Jacobi did not converge after " << max_iterations
              << " iterations. Final residual: " << last_residual << std::endl;
    x = x_new;
}

} // namespace JacobiSolver

