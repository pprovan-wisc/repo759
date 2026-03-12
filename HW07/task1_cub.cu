#include <iostream>
#include <cstdlib>
#include <cub/cub.cuh>

int main(int argc, char* argv[]) {
    if (argc != 2) return -1;
    int n = std::atoi(argv[1]);

    // Host allocation and initialization
    float* h_in = new float[n];
    for (int i = 0; i < n; ++i) {
        h_in[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }

    // Device allocation
    float* d_in = nullptr;
    float* d_out = nullptr;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);

    // Determine temporary storage requirements
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, n);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Setup timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // Run reduction
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy result back
    float h_out;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    // Print output
    std::cout << h_out << "\n";
    std::cout << ms << "\n";

    // Cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_temp_storage);
    delete[] h_in;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
