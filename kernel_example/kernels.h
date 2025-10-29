#ifndef KERNELS_H
#define KERNELS_H

#include <cuComplex.h>  // cuFloatComplex

// Vector add: elementwise c = a + b
int launch_vector_add(const cuFloatComplex *h_a,
                      const cuFloatComplex *h_b,
                      cuFloatComplex *h_c,
                      int n);

// Matrix-vector multiply using cuBLAS cgemv:
// Computes y := alpha * A * x + beta * y
// A is m x n, column-major, length m*n
// x is length n
// y is length m (in/out)
// alpha, beta are scalars
int launch_matvec_cgemv(const cuFloatComplex *h_A,
                        const cuFloatComplex *h_x,
                        cuFloatComplex *h_y,
                        int m,
                        int n,
                        cuFloatComplex alpha,
                        cuFloatComplex beta);

#endif