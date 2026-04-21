#include <iostream>
#include <vector>
#include "matrix_utils.h"
#include "lu_solver.h"
#include "jacobi_solver.h"
#include "benchmarker.h"

int main(int argc, char* argv[]) {
    std::cout << "========================================\n";
    std::cout << "Linear Solvers: CUDA Optimization Study\n";
    std::cout << "========================================\n\n";
    
    // Print CUDA device information
    int device_count;
    cudaGetDeviceCount(&device_count);
    std::cout << "Number of CUDA devices: " << device_count << "\n\n";
    
    if (device_count > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "Device 0: " << prop.name << "\n";
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << "\n";
        std::cout << "  Shared Memory per Block: " << prop.sharedMemPerBlock << " bytes\n";
        std::cout << "  Memory Clock Rate: " << prop.memoryClockRate / 1000 << " MHz\n";
        std::cout << "  Memory Bus Width: " << prop.memBusWidth << " bits\n\n";
    }
    
    // Set matrix sizes for benchmarking
    std::vector<int> matrix_sizes = {256, 512, 1024, 2048};
    
    std::cout << "Running benchmark suite with matrix sizes: ";
    for (int size : matrix_sizes) {
        std::cout << size << " ";
    }
    std::cout << "\n\n";
    
    // Create benchmarker and run full suite
    Benchmarker benchmarker;
    benchmarker.runFullBenchmark(matrix_sizes, "benchmark_results.csv");
    
    std::cout << "\n========================================\n";
    std::cout << "Benchmark complete!\n";
    std::cout << "Results saved to: benchmark_results.csv\n";
    std::cout << "========================================\n";
    
    return EXIT_SUCCESS;
}
