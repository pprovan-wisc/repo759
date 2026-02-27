#include "reduce.cuh"

__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // First add during load
    sdata[tid] = (i < n ? g_idata[i] : 0) + (i + blockDim.x < n ? g_idata[i + blockDim.x] : 0);
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

void reduce(float **input, float **output, unsigned int N, unsigned int threads_per_block) {
    unsigned int n = N;
    float *d_in = *input;
    float *d_out = *output;

    while (n > 1) {
        unsigned int blocks = (n + (threads_per_block * 2) - 1) / (threads_per_block * 2);
        reduce_kernel<<<blocks, threads_per_block, threads_per_block * sizeof(float)>>>(d_in, d_out, n);
        
        n = blocks;
        d_in = d_out; // Output of this stage is input for next
    }
    cudaDeviceSynchronize();
    // Copy result back to first element of original input as required
    cudaMemcpy(*input, d_in, sizeof(float), cudaMemcpyDeviceToDevice);
}
