#ifndef DENSE_FUNCTIONS_H
#define DENSE_FUNCTIONS_H

#include <stddef.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <string.h>
#include "block_sparse_format.h"

// ===========================================================================
// Compute matvec on dense matrix
// Y = alpha * A * X + beta * Y
// 
// Arguments
//   M        : Number of rows of A
//   N        : Number of columns of A
//   A        : Pointer to dense matrix A (row-major, len M*N)
//   X        : Pointer to input vector X (len N)
//   Y        : Pointer to output vector Y (len M)
//   alpha    : Scalar multiplier for A * X
//   beta     : Scalar multiplier for Y
//   order    : CBLAS_ORDER enum indicating row-major or column-major storage
// ===========================================================================
void dense_matvec(int M, 
                  int N,
                  const float complex *A,
                  const float complex *X, 
                  float complex *Y,    
                  float complex alpha, 
                  float complex beta,
                  const enum CBLAS_ORDER order);

// ===========================================================================
// Factor A (col-major, n x n) in place: A = P * L * U
// 
// Arguments
//   A        : Pointer to dense matrix A (col-major, n x n)
//   n        : Size of the matrix (n x n)
//   ipiv     : Pointer to pivot vector (len n)
// ===========================================================================
int dense_lu(float complex *A, 
             int n, 
             lapack_int *ipiv);

// ===========================================================================
// Compute the relative error of two complex vectors
//
// Arguments
//   y_bsf   : Result vector from block sparse format matvec
//   y_dense : Result vector from dense matvec
//   n       : Length of the vectors
void dense_print_lu(const float complex *A, 
                    int n);

// ===========================================================================
// Print a dense matrix A (n x n)
//
// Arguments
//   A       : Pointer to dense matrix A (col-major, n x n)
//   n       : Size of the matrix (n x n)
// ===========================================================================
void dense_print_matrix(const float complex *A, 
                        int n);

// ===========================================================================
// Computes B = A * I, where A is dense (n x n) and I is the identity matrix
// 
// Arguments
//   n                : Size of the overall matrix size
//   A                : Pointer to dense matrix A (col-major, n x n)
//   B                : Pointer to output matrix to store result of A * I (col-major, n x n)
//   piv              : Pivot vector for dense matrix LU factorisation
//   lu_factorise_dense : Flag indicating whether dense part is LU factorised
// ===========================================================================
void dense_identity_test(int n, 
                         int A_n, 
                         float complex *A, 
                         float complex *B, 
                         int *piv, 
                         int lu_factorise_dense);

// ===========================================================================
// Perform dense + large dense identity test: A_d * I = A_d
//
// Arguments
//   n                : Size of the overall matrix size
//   A                : Pointer to dense matrix A_d (col-major, n x n)
//   B                : Pointer to output matrix to store result of A_d * I (col-major, n x n)
//   piv              : Pivot vector for dense matrix LU factorisation
//   lu_factorise_dense : Flag indicating whether dense part is LU factorised
// ===========================================================================
void dense_large_identity_test(int n, 
                               float complex *A, 
                               float complex *B,
                               int *piv,
                               int lu_factorise_dense);

#endif