#ifndef BLOCK_SPARSE_TEST_RUNNER_H
#define BLOCK_SPARSE_TEST_RUNNER_H

void run_matvec_test(int n, int b, int block_structure, int print, double tolerance, int *passed);

void run_lu_trimul_test(int n, int b, int block_structure, int print, double tolerance, int *passed);

void run_lu_identity_test(int n, int b, int block_structure);

#endif