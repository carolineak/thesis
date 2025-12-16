#include <stdio.h>
#include <stdlib.h>
// #include "block_sparse_format.cu.h"
#include "test_matrix.cu.h"

int main(void) {

    printf("Testing Block Sparse Format\n");

    int b = 2;
    int n = 4*b;
    int block_structure = 1;

    block_sparse_format bsf;
    cuFloatComplex *dense = (cuFloatComplex*)malloc((size_t)n * (size_t)n * sizeof(cuFloatComplex));
    if (!dense) {
        fprintf(stderr, "Allocation failed for dense matrix\n");
        return 1;
    }

    create_test_matrix(n, b, block_structure, dense, &bsf);
    print_test_matrix_information(n, dense, &bsf);

    cuFloatComplex *x = (cuFloatComplex*)malloc((size_t)n * sizeof(cuFloatComplex));
    cuFloatComplex *y_bsf = (cuFloatComplex*)malloc((size_t)n * sizeof(cuFloatComplex));

    for (int i = 0; i < n; i++) {
        x[i] = make_cuFloatComplex(1.0f, 0.0f);
        y_bsf[i] = make_cuFloatComplex(0.0f, 0.0f);
    }

    sparse_matvec(&bsf, x, n, y_bsf, n);
    // TODO: compare with dense result

    //print y_bsf
    printf("y_bsf:\n");
    for (int i = 0; i < n; i++) {
        printf("(%5.2f,%5.2f) ", cuCrealf(y_bsf[i]), cuCimagf(y_bsf[i]));
    }
    printf("\n");

    free(dense);
    free(x);
    free(y_bsf);
    
    return 0;
}