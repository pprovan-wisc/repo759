#include <iostream>
#include <vector>
#include "matmul.cuh"

template <typename T>
void run_test(unsigned int n, unsigned int block_dim, int type_idx) {
    size_t size = n * n * sizeof(T);
    T *h_A = (T*)malloc(size), *h_B = (T*)malloc(size), *h_C = (T*)malloc(size);
    for (unsigned int i = 0; i < n * n; ++i) { h_A[i] = 1; h_B[i] = 1; }

    T *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size); cudaMalloc(&d_B, size); cudaMalloc(&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    if (type_idx == 1) matmul_1((int*)d_A, (int*)d_B, (int*)d_C, n, block_dim);
    else if (type_idx == 2) matmul_2((float*)d_A, (float*)d_B, (float*)d_C, n, block_dim);
    else matmul_3((double*)d_A, (double*)d_B, (double*)d_C, n, block_dim);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    std::cout << h_C[0] << "\n" << h_C[n * n - 1] << "\n" << ms << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
}

int main(int argc, char** argv) {
    unsigned int n = std::stoi(argv[1]);
    unsigned int block_dim = std::stoi(argv[2]);
    run_test<int>(n, block_dim, 1);
    run_test<float>(n, block_dim, 2);
    run_test<double>(n, block_dim, 3);
    return 0;
}
