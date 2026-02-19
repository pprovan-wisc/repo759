#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include "stencil.cuh"

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: ./task2 n R threads_per_block" << std::endl;
        return 1;
    }

    unsigned int n = std::stoul(argv[1]);
    unsigned int R = std::stoul(argv[2]);
    unsigned int threads_per_block = std::stoul(argv[3]);

    // Allocation on host
    std::vector<float> h_image(n);
    std::vector<float> h_mask(2 * R + 1);
    std::vector<float> h_output(n);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (unsigned int i = 0; i < n; ++i) h_image[i] = dis(gen);
    for (unsigned int i = 0; i < 2 * R + 1; ++i) h_mask[i] = dis(gen);

    // Allocation on device
    float *d_image, *d_mask, *d_output;
    cudaMalloc(&d_image, n * sizeof(float));
    cudaMalloc(&d_mask, (2 * R + 1) * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));

    cudaMemcpy(d_image, h_image.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask.data(), (2 * R + 1) * sizeof(float), cudaMemcpyHostToDevice);

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    stencil(d_image, d_mask, d_output, n, R, threads_per_block);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_output.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Output per requirements
    std::cout << h_output[n - 1] << std::endl;
    std::cout << ms << std::endl;

    // Cleanup
    cudaFree(d_image);
    cudaFree(d_mask);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
