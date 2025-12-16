#ifndef LOAD_BINARY_DATA_H
#define LOAD_BINARY_DATA_H

// ==================================================================
// Load a block sparse matrix from a binary file
//
// Arguments
//   path : Path to the binary file containing the block sparse matrix
//   bsf  : Pointer to a block_sparse_format structure where the loaded
//          matrix will be stored
//
// Returns 0 on success, <0 on error.
// ==================================================================
int load_block_sparse_from_bin(const char *path, block_sparse_format *bsf);

// ==================================================================
// Check the validity of a block sparse format structure
//
// Arguments
//   bsf : Pointer to a block_sparse_format structure to be checked
//

// ==================================================================
void check_block_sparse_format(const block_sparse_format *bsf);

// ==================================================================
// Debug print the contents of a binary input file
//
// Arguments
//   path : Path to the binary file to be printed for debugging
//
// ==================================================================
void debug_print_input_bin(const char *path);

#endif