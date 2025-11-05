#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <complex.h>

// CUDA kernel
__global__ void simple_kernel() {
    printf("Hello from CUDA thread %d!\n", threadIdx.x);
}

// helper to print CUDA errors
static void report_cuda_error(const char *msg, cudaError_t err) {
    fprintf(stderr, "[kernel] %s: %s\n",
            msg,
            cudaGetErrorString(err));
}

// Function callable from C
extern "C" 
int launch_kernel(int n) {

    cudaError_t err;

    int threadsPerBlock = 1; // 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    simple_kernel<<<blocksPerGrid, threadsPerBlock>>>();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        report_cuda_error("kernel launch failed", err);
        return -1;
    }

    // --- Synchronize ---
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        report_cuda_error("cudaDeviceSynchronize failed", err);
        return -1;
    }

    return 0;
}