#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <complex.h>
#include <cblas.h>
#include <lapacke.h>
#include <math.h>
#include <string.h>
#include "block_sparse_format.h"
#include "block_sparse_test_runner.h"
#include "load_binary_data.h"

int main(void) {
    // Test parameters
    const int print           = 2;              // 0=silent, 1=results, 2=show data, 3=show LU
    const double tolerance    = 1e-3;
    const int block_sizes[]   = {40};
    const int structures[]    = {0,1,2,3};      // 0=no fill-ins, 1=fill-ins 2,3=fill-ins+varying block sizes
    
    int passed = 0, total = 0;
    
    size_t num_block_sizes = sizeof(block_sizes)/sizeof(block_sizes[0]);
    size_t num_structures = sizeof(structures)/sizeof(structures[0]);
    printf("Running block-sparse tests (%zu sizes × %zu structures)\n\n",
           num_block_sizes,
           num_structures);

    for (size_t ib = 0; ib < num_block_sizes; ib++) {
        int b = block_sizes[ib];
        int n = 4 * b;

        for (size_t is = 0; is < num_structures; is++) {
            int s = structures[is];
            
            printf("→ Test: matrix of size %d x %d using block structure no. %d\n", n, n, s);

            if ((s == 2 || s == 3) && (b % 2 != 0)) {
                printf("TEST SKIPPED: b=%d (must be even) for structure %d\n\n", b, s);
                continue;
            }

            // run_data_structure_test(n, b, s);
            // run_matvec_test(n, b, s, print, tolerance, &passed); total++;
            run_lu_trimul_test(n, b, s, print, tolerance, &passed); total++;
            // run_lu_identity_test(n, b, s); // Test for debugging

            printf("\n");
        }
    }

    printf("\nAll tests completed. Passed %d out of %d tests.\n", passed, total);

    return 0;
}
