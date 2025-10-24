#ifndef TEST_MATRIX_H
#define TEST_MATRIX_H

#include <stddef.h>
#include <stdlib.h>
#include <complex.h>
#include "block_sparse_format.h"

int create_test_matrix(int n, int b, int block_structure, float complex *dense, block_sparse_format *bsf);

void print_test_matrix_information(int n, float complex *dense, block_sparse_format *bsf);

#endif