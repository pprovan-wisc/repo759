#include <cuda_runtime.h>
#include "scan.cuh"

__global__ void hillis_steele(const float* input, float* output, float* block_sums, unsigned int n) {
    extern __shared__ float temp[];

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + tid;

    if (gid < n)
        temp[tid] = input[gid];
    else
        temp[tid] = 0.0f;

    __syncthreads();

    for (unsigned int offset = 1; offset < blockDim.x; offset *= 2) {
        float val = 0.0f;

        if (tid >= offset)
            val = temp[tid - offset];

        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }

    if (gid < n)
        output[gid] = temp[tid];

    if (tid == blockDim.x - 1)
        block_sums[blockIdx.x] = temp[tid];
}

__global__ void add_offsets(float* output, const float* offsets, unsigned int n) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < n && blockIdx.x > 0) {
        output[gid] += offsets[blockIdx.x - 1];
    }
}

void scan(const float* input, float* output, unsigned int n, unsigned int threads_per_block) {

    unsigned int blocks = (n + threads_per_block - 1) / threads_per_block;

    float* block_sums;
    float* block_offsets;

    cudaMallocManaged(&block_sums, blocks * sizeof(float));
    cudaMallocManaged(&block_offsets, blocks * sizeof(float));

    size_t shared_mem = threads_per_block * sizeof(float);

    hillis_steele<<<blocks, threads_per_block, shared_mem>>>(input, output, block_sums, n);
    cudaDeviceSynchronize();

    if (blocks > 1) {

        hillis_steele<<<1, blocks, blocks * sizeof(float)>>>(block_sums, block_offsets, block_offsets, blocks);
        cudaDeviceSynchronize();

        add_offsets<<<blocks, threads_per_block>>>(output, block_offsets, n);
        cudaDeviceSynchronize();
    }

    cudaFree(block_sums);
    cudaFree(block_offsets);
}
