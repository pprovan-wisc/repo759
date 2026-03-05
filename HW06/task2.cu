#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include "scan.cuh"

int main(int argc, char** argv) {
    if (argc != 3) {
        return 1;
    }

    unsigned int n = std::atoi(argv[1]);
    unsigned int threads_per_block = std::atoi(argv[2]);

    size_t bytes = n * sizeof(float);
    float *input, *output;

    cudaMallocManaged(&input, bytes);
    cudaMallocManaged(&output, bytes);

    for (unsigned int i = 0; i < n; ++i) {
        input[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    scan(input, output, n, threads_per_block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start, stop);

    // Print last element of inclusive scan array followed by scan duration in ms
    std::cout << output[n - 1] << std::endl;
    std::cout << total_ms << std::endl;

    cudaFree(input);
    cudaFree(output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
