#ifndef CHECK_CORRECTNESS_H
#define CHECK_CORRECTNESS_H

#include <stddef.h>
#include <stdlib.h>
#include <complex.h>
#include "block_sparse_format.h"
#include "dense_functions.h"

double relative_error(const float complex *y_bsf, const float complex *y_dense, int n);

void all_factors_identity_test(int n, const block_sparse_format *bsf, float complex *A, 
                                      float complex *B, int *piv, int lu_factorise_dense);

void all_factors_vector_test(int n, const block_sparse_format *bsf, float complex *dense, 
                                    float complex *vec_in, complex float *vec_out, int *piv, 
                                    int lu_factorise_dense);                                      

#endif