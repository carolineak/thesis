#include <stdio.h>

// Declare the CUDA function defined in kernels.cu
void launch_kernel(void);

int main(void) {
    printf("Launching CUDA kernel from C...\n");
    launch_kernel();
    return 0;
}
