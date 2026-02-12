#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>

// CUDA kernel
__global__ void computeKernel(int* dA, int a) {
    int x = threadIdx.x;   // 0..7
    int y = blockIdx.x;    // 0..1

    int globalIndex = y * blockDim.x + x;  // unique position 0..15

    dA[globalIndex] = a * x + y;
}

int main() {
    const int N = 16;
    const int threadsPerBlock = 8;
    const int numBlocks = 2;

    int hA[N];
    int* dA = nullptr;

    // Generate random a
    std::srand(std::time(nullptr));
    int a = std::rand() % 20 + 1;  // random integer between 1 and 20

    // Allocate device memory
    cudaMalloc((void**)&dA, N * sizeof(int));

    // Launch kernel
    computeKernel<<<numBlocks, threadsPerBlock>>>(dA, a);

    cudaDeviceSynchronize();

    // Copy back to host
    cudaMemcpy(hA, dA, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print 16 sequential values separated by single space
    for (int i = 0; i < N; i++) {
        std::cout << hA[i];
        if (i < N - 1)
            std::cout << " ";
    }
    std::cout << std::endl;

    cudaFree(dA);

    return 0;
}
