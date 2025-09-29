#ifndef DENSE_FUNCTIONS_H
#define DENSE_FUNCTIONS_H

#include <stddef.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <string.h>
#include "block_sparse_format.h"

void dense_matvec(int M, 
                  int N,
                  const float complex *A,
                  const float complex *X, 
                  float complex *Y,    
                  float complex alpha, 
                  float complex beta,
                  const enum CBLAS_ORDER order);

int dense_lu(float complex *A, int n, lapack_int *ipiv);

void dense_print_lu(const float complex *A, int n);

void dense_print_matrix(const float complex *A, int n);

void dense_identity_test(int n, float complex *A, float complex *B, int *piv, int lu_factorise_dense);

#endif