#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "mmul.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./task1 <n> <n_tests>\n";
        return 1;
    }

    int n = std::atoi(argv[1]);
    int n_tests = std::atoi(argv[2]);

    size_t bytes = n * n * sizeof(float);
    float *A, *B, *C;

    cudaMallocManaged(&A, bytes);
    cudaMallocManaged(&B, bytes);
    cudaMallocManaged(&C, bytes);

    // Fill with random floats in range [-1, 1]
    for (int i = 0; i < n * n; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
        B[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
        C[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    }

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < n_tests; ++i) {
        mmul(handle, A, B, C, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start, stop);

    std::cout << total_ms / n_tests << std::endl;

    cublasDestroy(handle);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
