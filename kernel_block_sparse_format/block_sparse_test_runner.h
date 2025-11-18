#ifndef BLOCK_SPARSE_TEST_RUNNER_H
#define BLOCK_SPARSE_TEST_RUNNER_H

void run_data_structure_test(int n, int b, int block_structure);

void run_matvec_test(int n, int b, int block_structure, int print, double tolerance, int *passed);

void run_lu_trimul_test(int n, int b, int block_structure, int print, double tolerance, int *passed);

void run_lu_trimul_test_on_bin_data(int print, double tolerance, int *passed, char *data);

void run_lu_identity_test(int n, int b, int block_structure);

void run_lu_identity_test_with_bin_data(char *data);

#endif