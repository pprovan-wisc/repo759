#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include "vscale.cuh"

// Error checking macro for CUDA calls
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main(int argc, char** argv) {

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <n>" << std::endl;
        return 1;
    }

    int n = std::atoi(argv[1]);
    size_t bytes = n * sizeof(float);

    // 1. Allocate Host memory
    std::vector<float> h_a(n);
    std::vector<float> h_b(n);

    // 2. Initialize with random numbers
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> dist_a(-10.0f, 10.0f);
    std::uniform_real_distribution<float> dist_b(0.0f, 1.0f);

    for (int i = 0; i < n; i++) {
        h_a[i] = dist_a(gen);
        h_b[i] = dist_b(gen);
    }

    // 3. Allocate Device memory
    float *d_a, *d_b;
    cudaCheckError(cudaMalloc(&d_a, bytes));
    cudaCheckError(cudaMalloc(&d_b, bytes));

    // 4. Copy Host -> Device
    cudaCheckError(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    // 5. Setup Execution Configuration
    int threadsPerBlock = 512;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // 6. Setup Timing (CUDA Events)
    cudaEvent_t start, stop;
    cudaCheckError(cudaEventCreate(&start));
    cudaCheckError(cudaEventCreate(&stop));

    // 🔥 Warmup launch (NOT timed)
    vscale<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, n);
    cudaCheckError(cudaDeviceSynchronize());

    // 7. Execute Kernel with Timing
    cudaCheckError(cudaEventRecord(start));

    vscale<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, n);

    cudaCheckError(cudaEventRecord(stop));
    cudaCheckError(cudaEventSynchronize(stop));

    float milliseconds = 0;
    cudaCheckError(cudaEventElapsedTime(&milliseconds, start, stop));

    // 8. Copy Device -> Host
    cudaCheckError(cudaMemcpy(h_b.data(), d_b, bytes, cudaMemcpyDeviceToHost));

    // 9. Print Output
    std::cout << milliseconds << std::endl;
    std::cout << h_b[0] << std::endl;
    std::cout << h_b[n-1] << std::endl;

    // 10. Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
