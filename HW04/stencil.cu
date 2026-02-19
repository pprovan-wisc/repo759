#include "stencil.cuh"

__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R) {
    // Dynamic shared memory allocation
    // Shared memory layout: [mask] [image_tile] [output_tile]
    extern __shared__ float shared_mem[];
    
    float* s_mask = shared_mem;
    float* s_image = &shared_mem[2 * R + 1];
    float* s_output = &shared_mem[2 * R + 1 + (blockDim.x + 2 * R)];

    unsigned int g_tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int l_tid = threadIdx.x;

    // 1. Load mask into shared memory
    if (l_tid < 2 * R + 1) {
        s_mask[l_tid] = mask[l_tid];
    }

    // 2. Load image into shared memory (with halos)
    // Load center
    if (g_tid < n) {
        s_image[l_tid + R] = image[g_tid];
    } else {
        s_image[l_tid + R] = 0.0f; // Padding for out of bounds
    }

    // Load left halo
    if (l_tid < R) {
        if (g_tid >= R) {
            s_image[l_tid] = image[g_tid - R];
        } else {
            s_image[l_tid] = 0.0f;
        }
    }

    // Load right halo
    if (l_tid >= blockDim.x - R) {
        if (g_tid + R < n) {
            s_image[l_tid + 2 * R] = image[g_tid + R];
        } else {
            s_image[l_tid + 2 * R] = 0.0f;
        }
    }

    __syncthreads();

    // 3. Compute stencil
    if (g_tid < n) {
        float result = 0.0f;
        for (int j = - (int)R; j <= (int)R; j++) {
            result += s_image[l_tid + R + j] * s_mask[j + R];
        }
        s_output[l_tid] = result;
    }

    __syncthreads();

    // 4. Write back to global memory
    if (g_tid < n) {
        output[g_tid] = s_output[l_tid];
    }
}

void stencil(const float* image, const float* mask, float* output, unsigned int n, unsigned int R, unsigned int threads_per_block) {
    unsigned int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    // Shared memory size calculation:
    // mask: (2*R + 1)
    // image tile: (threads_per_block + 2*R)
    // output tile: threads_per_block
    size_t shared_size = (2 * R + 1 + (threads_per_block + 2 * R) + threads_per_block) * sizeof(float);

    stencil_kernel<<<blocks, threads_per_block, shared_size>>>(image, mask, output, n, R);
}
