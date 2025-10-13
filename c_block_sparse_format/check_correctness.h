#ifndef CHECK_CORRECTNESS_H
#define CHECK_CORRECTNESS_H

#include <stddef.h>
#include <stdlib.h>
#include <complex.h>
#include "block_sparse_format.h"
#include "dense_functions.h"

double relative_error(const float complex *y_bsf, const float complex *y_dense, int n);

void sparse_dense_identity_test(int n, const block_sparse_format *bsf, int A_n, float complex *A, 
                                      float complex *B, int *piv, int lu_factorise_dense);

void sparse_dense_trimul(int n, const block_sparse_format *bsf, int dense_size, float complex *dense, 
                                    float complex *vec_in, complex float *vec_out, int *piv, 
                                    int lu_factorise_dense); 
                                    
void sparse_large_dense_identity_test(int n, const block_sparse_format *bsf, float complex *A, 
                                      float complex *B, int *piv, int lu_factorise_dense);                                     

void sparse_large_dense_trimul(int n, const block_sparse_format *bsf, float complex *dense, 
                                    float complex *vec_in, complex float *vec_out, int *piv, 
                                    int lu_factorise_dense);  
void sparse_dense_trisolve(int n, const block_sparse_format *bsf, int dense_size, float complex *dense, 
                           float complex *vec_in, complex float *vec_out, int *piv, int lu_factorise_dense);

#endif