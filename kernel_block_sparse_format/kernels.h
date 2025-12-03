#ifndef KERNELS_H
#define KERNELS_H

#include <cuComplex.h> 

// ==================================================================
// Initialise and finalise global GPU libraries (cuBLAS + cuSOLVER)
// ==================================================================
int gpu_init(void);
int gpu_finalise(void);

// ==================================================================
// Compute a matrix-vector product for a block-sparse format
//
// Arguments
//   d_flat_data  : Device pointer to flattened blocks (as cuFloatComplex)
//   num_blocks    : Number of blocks
//   h_row_start    : Device pointer to per-block row start indices
//   h_M           : Device pointer to per-block row sizes
//   h_col_start    : Device pointer to per-block column start indices
//   h_N           : Device pointer to per-block column sizes
//   h_offsets      : Device pointer to per-block offsets
//   d_x           : Device input vector (length = n)
//   d_y           : Device output vector (length = m)
//
// Returns 0 on success, <0 on error.
// ==================================================================
int matvec_cu(const cuFloatComplex* d_flat_data,
               int num_blocks,
               const int* h_row_start,
               const int* h_M,
               const int* h_col_start,
               const int* h_N,
               const int* h_offsets,
               const cuFloatComplex* d_x,
               cuFloatComplex* d_y);

// ==================================================================
// Perform triangular solve on device using cuBLAS
//
// Arguments
//   side         : 'L' or 'R'
//   uplo         : 'L' or 'U'
//   trans        : 'N' or 'T' or 'C'
//   diag         : 'U' or 'N'
//   m            : Number of rows of B
//   n            : Number of columns of B
//   alpha        : Scalar multiplier
//   d_A          : Device pointer to triangular matrix A
//   lda          : Leading dimension of A
//   d_B          : Device pointer to right-hand side matrix B (overwritten with solution)
//   ldb          : Leading dimension of B
//
// Returns 0 on success, <0 on error.
// ==================================================================
int trisolve_cu(char side,
                 char uplo,
                 char trans,
                 char diag,
                 int m,
                 int n,
                 const cuFloatComplex* alpha,
                 const cuFloatComplex* d_A,
                 int lda,
                 cuFloatComplex* d_B,
                 int ldb);

// ==================================================================
// Apply pivot array to device block A in-place
//
// Arguments
//   d_A         : Device pointer to block stored column-major with leading dimension lda
//   lda         : Leading dimension of A
//   num_cols    : Number of columns in the block
//   h_ipiv      : Host pointer to pivot array (1-based indices), length = m
//
// Returns 0 on success, <0 on error.
// ==================================================================
int apply_pivots_cu(cuFloatComplex* d_A, 
                    int lda, 
                    int num_cols, 
                    const int* h_ipiv, 
                    int m);

// ==================================================================
// Update Schur complement on device: C := C - A * B
//
// Arguments
//   d_C         : Device pointer to C (m x n), lda = m
//   d_A         : Device pointer to A (m x k), lda = m
//   d_B         : Device pointer to B (k x n), lda = k
//   m           : Number of rows of C
//   n           : Number of columns of C
//   k           : Inner dimension
//
// Returns 0 on success, <0 on error.
// ==================================================================
int block_schur_update_cu(cuFloatComplex* d_C,
                           const cuFloatComplex* d_A,
                           const cuFloatComplex* d_B,
                           int m, 
                           int n, 
                           int k);

// ==================================================================
// Perform LU factorization on device for a single block A (in-place on d_A)
//
// Arguments
//   d_A         : Device pointer to n x n matrix (column-major), lda >= n
//   n           : Size of the block
//   lda         : Leading dimension of A
//   h_ipiv      : Host pointer to int array length n (output, 1-based pivots)
//   info        : Host pointer to int (output, LAPACK-style info)
//
// Returns 0 on success, <0 on error.
// ==================================================================
int block_getrf_cu(cuFloatComplex* d_A, 
                   int n, 
                   int lda, 
                   int* h_ipiv, 
                   int* info);

#endif