#ifndef BLOCK_SPARSE_TEST_RUNNER_H
#define BLOCK_SPARSE_TEST_RUNNER_H

void run_data_structure_test(int n, int b, int block_structure);

void run_matvec_test(int n, int b, int block_structure, int print, double tolerance, int *passed);

#endif