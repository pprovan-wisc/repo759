#include <iostream>
#include <vector>
#include <cmath>
#include "../include/matrix_utils.h"
#include "../include/lu_solver.h"
#include "../include/jacobi_solver.h"

// Test framework macros
#define ASSERT_TRUE(condition, message) \
    if (!(condition)) { \
        std::cerr << "FAILED: " << message << std::endl; \
        return false; \
    }

#define ASSERT_NEAR(actual, expected, tolerance, message) \
    if (std::abs(actual - expected) > tolerance) { \
        std::cerr << "FAILED: " << message \
                  << " (got " << actual << ", expected " << expected << ")" << std::endl; \
        return false; \
    }

bool testTimerBasics() {
    std::cout << "Testing Timer basics... ";
    
    Timer timer;
    timer.start();
    cudaDeviceSynchronize();
    timer.stop();
    
    float elapsed = timer.getElapsedMilliseconds();
    ASSERT_TRUE(elapsed >= 0, "Timer elapsed time should be non-negative");
    
    std::cout << "PASSED\n";
    return true;
}

bool testMatrixUtilities() {
    std::cout << "Testing matrix utilities... ";
    
    std::vector<float> data(9);
    fillMatrixIdentity(data, 9);
    
    // Check identity matrix
    ASSERT_NEAR(data[0], 1.0f, 1e-6, "Identity[0][0] should be 1");
    ASSERT_NEAR(data[4], 1.0f, 1e-6, "Identity[1][1] should be 1");
    ASSERT_NEAR(data[8], 1.0f, 1e-6, "Identity[2][2] should be 1");
    ASSERT_NEAR(data[1], 0.0f, 1e-6, "Identity[0][1] should be 0");
    
    std::cout << "PASSED\n";
    return true;
}

bool testLUDecompositionCPU() {
    std::cout << "Testing LU Decomposition (CPU)... ";
    
    // Create a simple test matrix (3x3)
    std::vector<float> A = {
        4.0f,  3.0f,  2.0f,
        1.0f,  2.0f,  3.0f,
        2.0f,  1.0f,  4.0f
    };
    
    std::vector<float> b = {9.0f, 6.0f, 7.0f};
    std::vector<float> x(3);
    
    LUSolver::LUDecompositionCPU solver(3);
    solver.solve(A, b, x);
    
    // Verify solution: Ax = b
    std::vector<float> Ax(3, 0.0f);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Ax[i] += A[i * 3 + j] * x[j];
        }
    }
    
    ASSERT_NEAR(Ax[0], b[0], 1e-3, "Solution check: Ax[0] = b[0]");
    ASSERT_NEAR(Ax[1], b[1], 1e-3, "Solution check: Ax[1] = b[1]");
    ASSERT_NEAR(Ax[2], b[2], 1e-3, "Solution check: Ax[2] = b[2]");
    
    std::cout << "PASSED\n";
    return true;
}

bool testJacobiMethodCPU() {
    std::cout << "Testing Jacobi Method (CPU)... ";
    
    // Create a diagonally dominant system
    std::vector<float> A = {
        10.0f, 1.0f,  2.0f,
        1.0f,  8.0f,  1.0f,
        2.0f,  1.0f,  10.0f
    };
    
    std::vector<float> b = {13.0f, 10.0f, 13.0f};
    std::vector<float> x(3);
    
    JacobiSolver::JacobiMethodCPU solver(3, 1e-6, 1000);
    solver.solve(A, b, x);
    
    // Verify solution
    std::vector<float> Ax(3, 0.0f);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Ax[i] += A[i * 3 + j] * x[j];
        }
    }
    
    ASSERT_NEAR(Ax[0], b[0], 1e-3, "Jacobi solution check: Ax[0] ≈ b[0]");
    ASSERT_NEAR(Ax[1], b[1], 1e-3, "Jacobi solution check: Ax[1] ≈ b[1]");
    ASSERT_NEAR(Ax[2], b[2], 1e-3, "Jacobi solution check: Ax[2] ≈ b[2]");
    
    std::cout << "PASSED (iterations: " << solver.getIterationCount() << ")\n";
    return true;
}

bool testMatrixMemoryTransfer() {
    std::cout << "Testing GPU memory transfer... ";
    
    std::vector<float> host_data = {1.0f, 2.0f, 3.0f, 4.0f};
    
    Matrix gpu_matrix(2, 2, true);
    gpu_matrix.copyHostToDevice(host_data);
    
    std::vector<float> retrieved = gpu_matrix.copyDeviceToHost();
    
    ASSERT_NEAR(retrieved[0], 1.0f, 1e-6, "Transfer test: element [0]");
    ASSERT_NEAR(retrieved[1], 2.0f, 1e-6, "Transfer test: element [1]");
    ASSERT_NEAR(retrieved[2], 3.0f, 1e-6, "Transfer test: element [2]");
    ASSERT_NEAR(retrieved[3], 4.0f, 1e-6, "Transfer test: element [3]");
    
    std::cout << "PASSED\n";
    return true;
}

int main(int argc, char* argv[]) {
    std::cout << "\n=== Running Unit Tests ===\n\n";
    
    int tests_passed = 0;
    int tests_failed = 0;
    
    // Run tests
    if (testTimerBasics()) tests_passed++; else tests_failed++;
    if (testMatrixUtilities()) tests_passed++; else tests_failed++;
    if (testMatrixMemoryTransfer()) tests_passed++; else tests_failed++;
    if (testLUDecompositionCPU()) tests_passed++; else tests_failed++;
    if (testJacobiMethodCPU()) tests_passed++; else tests_failed++;
    
    std::cout << "\n=== Test Results ===\n";
    std::cout << "Passed: " << tests_passed << "\n";
    std::cout << "Failed: " << tests_failed << "\n";
    
    return (tests_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
