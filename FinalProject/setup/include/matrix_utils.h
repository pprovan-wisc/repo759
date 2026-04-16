#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <iostream>

// Matrix data structure
struct Matrix {
    float* data;
    int rows;
    int cols;
    bool gpu_allocated;

    Matrix(int r, int c, bool on_gpu = true)
        : rows(r), cols(c), gpu_allocated(on_gpu) {
        if (on_gpu) {
            cudaMalloc(&data, rows * cols * sizeof(float));
        } else {
            data = new float[rows * cols];
        }
    }

    ~Matrix() {
        if (gpu_allocated) {
            cudaFree(data);
        } else {
            delete[] data;
        }
    }

    // Copy host to device
    void copyHostToDevice(const std::vector<float>& host_data) {
        if (gpu_allocated) {
            cudaMemcpy(data, host_data.data(), rows * cols * sizeof(float),
                      cudaMemcpyHostToDevice);
        } else {
            std::copy(host_data.begin(), host_data.end(), data);
        }
    }

    // Copy device to host
    std::vector<float> copyDeviceToHost() {
        std::vector<float> result(rows * cols);
        if (gpu_allocated) {
            cudaMemcpy(result.data(), data, rows * cols * sizeof(float),
                      cudaMemcpyDeviceToHost);
        } else {
            std::copy(data, data + rows * cols, result.begin());
        }
        return result;
    }

    float* operator[](int row) {
        return data + row * cols;
    }
};

// Utility functions
void checkCuda(cudaError_t result, const char* message);
void printMatrix(const std::vector<float>& data, int rows, int cols, const char* name = "Matrix");
void fillMatrixRandom(std::vector<float>& data, int size);
void fillMatrixIdentity(std::vector<float>& data, int size);

// Timing utilities
class Timer {
public:
    Timer();
    ~Timer();
    void start();
    void stop();
    float getElapsedMilliseconds();
    void reset();

private:
    cudaEvent_t start_event, stop_event;
};

