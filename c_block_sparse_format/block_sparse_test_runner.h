#ifndef BLOCK_SPARSE_TEST_RUNNER_H
#define BLOCK_SPARSE_TEST_RUNNER_H

// ==================================================================
// Run matrix-vector multiplication tests for synthetic test matrices
//
// Arguments
//   n                : Matrix dimension
//   b                : Block size
//   block_structure  : Block structure pattern
//   print            : Flag to enable/disable output printing
//   tolerance        : Numerical tolerance for comparison
//   passed           : Pointer to store test result (1 if passed, 0 if failed)
//
// ==================================================================
void run_matvec_test(int n, 
                     int b, 
                     int block_structure, 
                     int print, 
                     double tolerance, 
                     int *passed);

// ==================================================================
// Run LU triangular multiplication tests for synthetic test matrices
//
// Arguments
//   n                : Matrix dimension
//   b                : Block size
//   block_structure  : Block structure pattern
//   print            : Flag to enable/disable output printing
//   tolerance        : Numerical tolerance for comparison
//   passed           : Pointer to store test result (1 if passed, 0 if failed)
//
// ==================================================================
void run_lu_trimul_test(int n, 
                        int b, 
                        int block_structure, 
                        int print, 
                        double tolerance, 
                        int *passed);
                        
// ==================================================================
// Run LU identity tests for synthetic test matrices
//
// Arguments
//   n                : Matrix dimension
//   b                : Block size
//   block_structure  : Block structure pattern
//
// ==================================================================
void run_lu_identity_test(int n, 
                          int b,    
                          int block_structure);

#endif