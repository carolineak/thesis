#ifndef TEST_MATRIX_H
#define TEST_MATRIX_H

#include <stddef.h>
#include <stdlib.h>
#include <complex.h>
#include "block_sparse_format.h"

// ==================================================================
// Create a test matrix in block sparse format
//
// Arguments
//   n                : Size of the dense matrix (n x n).
//   b                : Block size for the block sparse format.
//   block_structure   : Structure of the blocks in the sparse matrix.
//   dense            : Pointer to the dense matrix representation (output).
//   bsf              : Pointer to the block sparse format structure (output).
//
// Returns 0 on success, <0 on error.
// ==================================================================
int create_test_matrix(int n, 
                       int b, 
                       int block_structure, 
                       float complex *dense, 
                       block_sparse_format *bsf);

// ==================================================================
// Print information about the test matrix
//
// Arguments
//   n                : Size of the dense matrix (n x n).
//   dense            : Pointer to the dense matrix representation.
//   bsf              : Pointer to the block sparse format structure.
//
// This function outputs relevant information about the test matrix,
// including its dimensions and block structure.
// ==================================================================
void print_test_matrix_information(int n, 
                                   float complex *dense, 
                                   block_sparse_format *bsf);

#endif