#include <cuda_runtime.h>
#include <stdio.h>

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

// helper to print CUDA errors
static void report_cuda_error(const char *msg, cudaError_t err) {
    fprintf(stderr, "[launch_vector_add] %s: %s\n",
            msg,
            cudaGetErrorString(err));
}

extern "C"
int launch_vector_add(const float *h_a,
                      const float *h_b,
                      float *h_c,
                      int n)
{
    cudaSetDevice(0);
    
    if (n <= 0) {
        fprintf(stderr, "[launch_vector_add] invalid n=%d\n", n);
        return -1;
    }

    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    fprintf(stderr,
        "[launch_vector_add] GPU mem free = %zu / %zu bytes (%.2f MB / %.2f MB)\n",
        free_bytes, total_bytes,
        free_bytes / (1024.0 * 1024.0),
        total_bytes / (1024.0 * 1024.0));

    float *d_a = NULL;
    float *d_b = NULL;
    float *d_c = NULL;
    cudaError_t err;
    size_t bytes = (size_t)n * sizeof(float);

    // --- Allocate device memory ---
    err = cudaMalloc((void**)&d_a, bytes);
    if (err != cudaSuccess) {
        report_cuda_error("cudaMalloc d_a failed", err);
        return -1;
    }

    err = cudaMalloc((void**)&d_b, bytes);
    if (err != cudaSuccess) {
        report_cuda_error("cudaMalloc d_b failed", err);
        cudaFree(d_a);
        return -1;
    }

    err = cudaMalloc((void**)&d_c, bytes);
    if (err != cudaSuccess) {
        report_cuda_error("cudaMalloc d_c failed", err);
        cudaFree(d_a);
        cudaFree(d_b);
        return -1;
    }

    // --- Copy input data to device ---
    err = cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        report_cuda_error("cudaMemcpy h_a->d_a failed", err);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return -1;
    }

    err = cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        report_cuda_error("cudaMemcpy h_b->d_b failed", err);
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
        report_cuda_error("kernel launch failed", err);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return -1;
    }

    // --- Synchronize ---
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        report_cuda_error("cudaDeviceSynchronize failed", err);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return -1;
    }

    // --- Copy result back to host ---
    err = cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        report_cuda_error("cudaMemcpy d_c->h_c failed", err);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return -1;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
