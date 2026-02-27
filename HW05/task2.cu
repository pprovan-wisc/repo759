#include <iostream>
#include <vector>
#include <random>
#include "reduce.cuh"

int main(int argc, char** argv) {
    unsigned int N = std::stoul(argv[1]);
    unsigned int threads_per_block = std::stoul(argv[2]);

    std::vector<float> h_in(N);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0, 1.0);
    for (auto &val : h_in) val = dis(gen);

    float *d_in, *d_out;
    unsigned int initial_blocks = (N + (threads_per_block * 2) - 1) / (threads_per_block * 2);
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, initial_blocks * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    reduce(&d_in, &d_out, N, threads_per_block);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    float result;
    cudaMemcpy(&result, d_in, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << result << "\n" << ms << std::endl;

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
