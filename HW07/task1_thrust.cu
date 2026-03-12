#include <iostream>
#include <cstdlib>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/generate.h>

// Generator for random floats between -1.0 and 1.0
struct RandomFloat {
    __host__ float operator()() const {
        return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }
};

int main(int argc, char* argv[]) {
    if (argc != 2) return -1;
    int n = std::atoi(argv[1]);

    // Create and fill host vector
    thrust::host_vector<float> h_vec(n);
    thrust::generate(h_vec.begin(), h_vec.end(), RandomFloat());

    // Copy to device
    thrust::device_vector<float> d_vec = h_vec;

    // Setup timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // Perform reduction
    float result = thrust::reduce(d_vec.begin(), d_vec.end(), 0.0f, thrust::plus<float>());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // Print expected output
    std::cout << result << "\n";
    std::cout << ms << "\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
