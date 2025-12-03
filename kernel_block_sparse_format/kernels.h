#ifndef KERNELS_H
#define KERNELS_H

#include <cuComplex.h> 

// ==================================================================
// Initialise / finalise global GPU libraries (cuBLAS + cuSOLVER)
// ==================================================================
int gpu_init(void);
int gpu_finalise(void);

// ==================================================================
// Matrix-vector product for block-sparse format
//
// d_flat_data: device pointer to flattened blocks (as cuFloatComplex)
// num_blocks: number of blocks
// d_row_start,d_M,d_col_start,d_N,d_offsets: per-block metadata (device pointers)
// d_x: device input vector (length = n)
// d_y: device output vector (length = m) (accumulated)
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
// Triangular solve on device using cuBLAS
//
// side: 'L' or 'R'
// uplo: 'L' or 'U'
// diag: 'U' (unit) or 'N' (non-unit)
// m: number of rows of B
// n: number of columns of B
// alpha: scalar multiplier
// d_A: device pointer to triangular matrix A
// lda: leading dimension of A
// d_B: device pointer to right-hand side matrix B (overwritten with solution)
// ldb: leading dimension of B
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
// Apply pivot array (host ipiv of length m) to device block A in-place
//
// d_A: device pointer to block stored column-major with leading dimension lda
// lda: leading dimension of A
// num_cols: number of columns in the block
// h_ipiv: host pointer to pivot array (1-based indices), length = m
// ==================================================================
int apply_pivots_cu(cuFloatComplex* d_A, 
                    int lda, 
                    int num_cols, 
                    const int* h_ipiv, 
                    int m);

// ==================================================================
// Schur complement update on device: C := C - A * B
//
// d_C: device pointer to C (m x n), lda = m
// d_A: device pointer to A (m x k), lda = m
// d_B: device pointer to B (k x n), lda = k
// m: number of rows of C
// n: number of columns of C
// k: inner dimension
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
// d_A: device pointer to n x n matrix (column-major), lda >= n
// n: size of the block
// lda: leading dimension of A
// h_ipiv: host pointer to int array length n (output, 1-based pivots)
// info: host pointer to int (output, LAPACK-style info)
// ==================================================================
int block_getrf_cu(cuFloatComplex* d_A, 
                   int n, 
                   int lda, 
                   int* h_ipiv, 
                   int* info);

#endif