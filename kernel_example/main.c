// main.c
#include <stdio.h>
#include <stdlib.h>
#include "kernels.h"

int main(void)
{
    int n = 10;
    float a[10];
    float b[10];
    float c[10];

    // Fill a and b with some test data
    for (int i = 0; i < n; ++i) {
        a[i] = (float)i;
        b[i] = 100.0f + (float)i;
    }

    // Call the CUDA launcher
    int status = launch_vector_add(a, b, c, n);
    if (status != 0) {
        fprintf(stderr, "launch_vector_add failed\n");
        return 1;
    }

    // Print result
    printf("Result c = a + b:\n");
    for (int i = 0; i < n; ++i) {
        printf("c[%d] = %f\n", i, c[i]);
    }

    return 0;
}
