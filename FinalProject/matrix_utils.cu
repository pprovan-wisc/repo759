#include "matrix_utils.h"
#include <cstring>
#include <cmath>
#include <random>

// Error checking utility
void checkCuda(cudaError_t result, const char* message) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << message << " - "
                  << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Print matrix utility
void printMatrix(const std::vector<float>& data, int rows, int cols, const char* name) {
    std::cout << "\n" << name << " (" << rows << "x" << cols << "):\n";
    for (int i = 0; i < std::min(rows, 5); ++i) {
        for (int j = 0; j < std::min(cols, 5); ++j) {
            std::cout << data[i * cols + j] << "\t";
        }
        if (cols > 5) std::cout << "...";
        std::cout << "\n";
    }
    if (rows > 5) std::cout << "...\n";
}

// Fill matrix with random values
void fillMatrixRandom(std::vector<float>& data, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-10.0, 10.0);
    
    for (int i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
}

// Fill matrix with identity
void fillMatrixIdentity(std::vector<float>& data, int size) {
    std::fill(data.begin(), data.end(), 0.0f);
    int n = static_cast<int>(std::sqrt(size));
    for (int i = 0; i < n; ++i) {
        data[i * n + i] = 1.0f;
    }
}

// Timer implementation
Timer::Timer() {
    checkCuda(cudaEventCreate(&start_event), "Failed to create start event");
    checkCuda(cudaEventCreate(&stop_event), "Failed to create stop event");
}

Timer::~Timer() {
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
}

void Timer::start() {
    checkCuda(cudaEventRecord(start_event, 0), "Failed to record start event");
}

void Timer::stop() {
    checkCuda(cudaEventRecord(stop_event, 0), "Failed to record stop event");
    checkCuda(cudaEventSynchronize(stop_event), "Failed to synchronize stop event");
}

float Timer::getElapsedMilliseconds() {
    float ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms, start_event, stop_event),
             "Failed to compute elapsed time");
    return ms;
}

void Timer::reset() {
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    checkCuda(cudaEventCreate(&start_event), "Failed to create start event");
    checkCuda(cudaEventCreate(&stop_event), "Failed to create stop event");
}
