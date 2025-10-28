#include <cuda_runtime.h>
#include <stdio.h>

// ================================================
// GPU kernel: elementwise vector addition
// ================================================
__global__ void vector_add_kernel(const float *a,
                                  const float *b,
                                  float *c,
                                  int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// ================================================
// Host launcher (C-callable wrapper)
//   h_a, h_b, h_c : host pointers (float*)
//   n             : number of elements
// Returns 0 on success, -1 on failure.
// ================================================
extern "C"
int launch_vector_add(const float *h_a,
                      const float *h_b,
                      float *h_c,
                      int n)
{
    float *d_a = NULL;
    float *d_b = NULL;
    float *d_c = NULL;
    cudaError_t err;
    size_t bytes = (size_t)n * sizeof(float);

    // --- Allocate device memory ---
    err = cudaMalloc((void**)&d_a, bytes);
    if (err != cudaSuccess) return -1;

    err = cudaMalloc((void**)&d_b, bytes);
    if (err != cudaSuccess) {
        cudaFree(d_a);
        return -1;
    }

    err = cudaMalloc((void**)&d_c, bytes);
    if (err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        return -1;
    }

    // --- Copy input data to device ---
    err = cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return -1;
    }

    err = cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return -1;
    }

    // --- Launch kernel ---
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    vector_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return -1;
    }

    // --- Synchronize ---
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return -1;
    }

    // --- Copy result back to host ---
    err = cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return -1;
    }

    // --- Clean up ---
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
