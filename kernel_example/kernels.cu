#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel
__global__ void myKernel() {
    printf("Hello from CUDA thread %d!\n", threadIdx.x);
}

// Function callable from C
extern "C" void launch_kernel(void) {
    myKernel<<<1, 5>>>();
    cudaDeviceSynchronize();
}