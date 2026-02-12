#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel
__global__ void factorialKernel(int* dA) {
    int a = threadIdx.x;  // thread index (0 to 7)

    int value = 1;
    int n = a + 1;        // compute (a+1)!

    for (int i = 1; i <= n; i++) {
        value *= i;
    }

    dA[a] = value;        // store result in device array
}

int main() {
    const int N = 8;

    int hA[N];        // host array
    int* dA = nullptr;

    // Allocate device memory
    cudaMalloc((void**)&dA, N * sizeof(int));

    // Launch kernel: 1 block, 8 threads
    factorialKernel<<<1, N>>>(dA);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(hA, dA, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print results (one per line)
    for (int i = 0; i < N; i++) {
        std::cout << hA[i] << std::endl;
    }

    // Free device memory
    cudaFree(dA);

    return 0;
}
