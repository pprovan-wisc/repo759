#include <iostream>
#include <cstdlib>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include "count.cuh"

struct RandomInt {
    __host__ int operator()() const {
        return rand() % 501; // Range [0, 500]
    }
};

int main(int argc, char* argv[]) {
    if (argc != 2) return -1;
    int n = std::atoi(argv[1]);

    // Host vector filled with random numbers [0, 500]
    thrust::host_vector<int> h_vec(n);
    thrust::generate(h_vec.begin(), h_vec.end(), RandomInt());

    // Device vector copy
    thrust::device_vector<int> d_in = h_vec;
    thrust::device_vector<int> values;
    thrust::device_vector<int> counts;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // Execute count
    count(d_in, values, counts);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // Print output
    std::cout << values.back() << "\n";
    std::cout << counts.back() << "\n";
    std::cout << ms << "\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
