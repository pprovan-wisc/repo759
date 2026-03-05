#include "mmul.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

void mmul(cublasHandle_t handle, const float* A, const float* B, float* C, int n) {
    const float alpha = 1.0f;
    const float beta = 1.0f;
    
    // cuBLAS uses column-major order natively.
    // Operation: C = alpha * op(A) * op(B) + beta * C
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                n, n, n, 
                &alpha, 
                A, n, 
                B, n, 
                &beta, 
                C, n);
                
    cudaDeviceSynchronize();
}
