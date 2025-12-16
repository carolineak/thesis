#ifndef CHECK_CORRECTNESS_H
#define CHECK_CORRECTNESS_H

#include <stddef.h>
#include <stdlib.h>
#include <complex.h>
#include "block_sparse_format.h"
#include "dense_functions.h"

// ==================================================================
// Compute relative error between two vectors: ||y_bsf - y_dense||_2 / ||y_dense||_2
// 
// Arguments
//   y_bsf   : Result vector from block sparse format matvec
//   y_dense : Result vector from dense matvec
//   n       : Length of the vectors
// ==================================================================
double relative_error(const float complex *y_bsf, 
                      const float complex *y_dense, 
                      int n);

// ==================================================================
// Test that sparse * dense identity holds: A_s * A_d == A
//
// Arguments
//   n                : Size of the overall matrix size
//   bsf              : Pointer to block sparse format structure
//   A_n              : Size of the dense matrix (A_d is A_n x A_n)
//   A                : Pointer to dense matrix A_d
//   B                : Pointer to output matrix to store result of A_s * A_d
//   piv              : Pivot vector for dense matrix LU factorisation
//   lu_factorise_dense : Flag indicating whether dense part is LU factorised
// ==================================================================
void sparse_dense_identity_test(int n, 
                                const block_sparse_format *bsf, 
                                int A_n, 
                                float complex *A, 
                                float complex *B, 
                                int *piv, 
                                int lu_factorise_dense);

// ==================================================================
// Perform sparse * dense triangular multiplication: v := L_s * L_d * v or v := U_s * U_d * v
//
// Arguments
//   n                : Size of the overall matrix size
//   bsf              : Pointer to block sparse format structure
//   dense_size      : Size of the dense matrix (dense is dense_size x dense_size
//   dense           : Pointer to dense matrix
//   vec_in          : Input vector (size n)
//   vec_out         : Output vector (size n)
//   piv             : Pivot vector for dense matrix LU factorisation
//   lu_factorise_dense : Flag indicating whether dense part is LU factorised
// ==================================================================
void sparse_dense_trimul(int n, 
                         const block_sparse_format *bsf, 
                         int dense_size, 
                         float complex *dense, 
                         float complex *vec_in, 
                         complex float *vec_out, 
                         int *piv, 
                         int lu_factorise_dense); 
                                    
// ==================================================================   
// Perform sparse + large dense identity test: A_s * A_d == A
//
// Arguments
//   n                : Size of the overall matrix size
//   bsf              : Pointer to block sparse format structure
//   A                : Pointer to dense matrix A_d
//   B                : Pointer to output matrix to store result of A_s * A_d
//   piv              : Pivot vector for dense matrix LU factorisation
//   lu_factorise_dense : Flag indicating whether dense part is LU factorised
// ==================================================================                                 
void sparse_large_dense_identity_test(int n, 
                                      const block_sparse_format *bsf, 
                                      float complex *A, 
                                      float complex *B, 
                                      int *piv, 
                                      int lu_factorise_dense);                                     

// ==================================================================
// Perform sparse + large dense triangular multiplication: v := L_s * L_d * v or v := U_s * U_d * v
//
// Arguments
//   n               : Size of the overall matrix size
//   bsf             : Pointer to block sparse format structure
//   dense           : Pointer to dense matrix
//   vec_in          : Input vector (size n)
//   vec_out         : Output vector (size n)
//   piv             : Pivot vector for dense matrix LU factorisation
//   lu_factorise_dense : Flag indicating whether dense part is LU factorised
// ==================================================================
void sparse_large_dense_trimul(int n, 
                               const block_sparse_format *bsf, 
                               float complex *dense, 
                               float complex *vec_in, 
                               complex float *vec_out, 
                               int *piv, 
                               int lu_factorise_dense);  

// ==================================================================
// Perform sparse + large dense triangular solve: v := L_s^{-1} * L_d^{-1} * v or v := U_s^{-1} * U_d^{-1} * v
//
// Arguments
//   n               : Size of the overall matrix size
//   bsf             : Pointer to block sparse format structure
//   dense_size      : Size of the dense matrix (dense is dense_size x dense_size
//   dense           : Pointer to dense matrix
//   vec_in          : Input vector (size n)
//   vec_out         : Output vector (size n)
//   piv             : Pivot vector for dense matrix LU factorisation
//   lu_factorise_dense : Flag indicating whether dense part is LU factorised
// ==================================================================
void sparse_dense_trisolve(int n, 
                           const block_sparse_format *bsf, 
                           int dense_size, 
                           float complex *dense, 
                           float complex *vec_in, 
                           complex float *vec_out, 
                           int *piv, 
                           int lu_factorise_dense);

#endif