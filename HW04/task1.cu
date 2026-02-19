#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include "matmul.cuh"

int main(int argc, char** argv) {
    // 1. Parse command line arguments [cite: 24, 25]
    if (argc != 3) {
        std::cerr << "Usage: ./task1 n threads_per_block" << std::endl;
        return 1;
    }

    size_t n = std::stoul(argv[1]);
    unsigned int threads_per_block = std::stoull(argv[2]);

    // 2. Create matrices A and B on the host (row-major) 
    size_t size = n * n;
    std::vector<float> h_A(size);
    std::vector<float> h_B(size);
    std::vector<float> h_C(size);

    // 3. Fill matrices with random numbers in range [-1, 1] [cite: 18]
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < size; ++i) {
        h_A[i] = dis(gen);
        h_B[i] = dis(gen);
    }

    // 4. Prepare arrays allocated as device memory [cite: 19]
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, size * sizeof(float));
    cudaMalloc(&d_C, size * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    // 5. Setup CUDA events for timing [cite: 22]
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 6. Call matmul function and time it [cite: 20, 22]
    cudaEventRecord(start);
    matmul(d_A, d_B, d_C, n, threads_per_block);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 7. Copy result back to host to print the last element [cite: 21]
    cudaMemcpy(h_C.data(), d_C, size * sizeof(float), cudaMemcpyDeviceToHost);

    // 8. Output results [cite: 21, 22, 29]
    std::cout << h_C[size - 1] << std::endl;
    std::cout << milliseconds << std::endl;

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
