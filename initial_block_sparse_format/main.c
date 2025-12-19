#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <complex.h>
#include <lapacke.h>
#include <cblas.h>

#include "block_sparse_format.h"
#include "test_matrix.h"
#include "dense_functions.h"
#include "check_correctness.h"
#include "block_sparse_test_runner.h"

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [b] [structure] [print]\n"
        "  b          : base block size (default 16)\n"
        "  structure  : 0=no fill-ins, 1=fill-ins, 2/3=varying (default 1)\n"
        "  print      : 0=silent, 1=summary, 2=verbose (default 1)\n",
        prog);
}

int main(int argc, char **argv) {
    // Default demo configuration (can be overridden via command-line arguments)
    int b         = 16;
    int structure = 1;
    int print     = 1;
    double tol    = 1e-5;

    // Optional command-line overrides for b, structure, and print
    if (argc > 1) {
        b = atoi(argv[1]);
        if (b <= 0) {
            fprintf(stderr, "Invalid b=%d\n", b);
            usage(argv[0]);
            return 1;
        }
    }

    if (argc > 2) {
        structure = atoi(argv[2]);
    }

    if (argc > 3) {
        print = atoi(argv[3]);
    }

    // Enforce the same structural restriction as in the test suite:
    // for structure 2 and 3, the block size must be even
    if (structure == 2 || structure == 3) {
        if (b % 2 != 0) {
            fprintf(stderr, "For structure %d, b must be even (got %d)\n",
                    structure, b);
            return 1;
        }
    }

    // Use the same relationship n = 4 * b as in the tests
    int n = 4 * b;
    printf("Demo: n = %d, b = %d, structure = %d, print = %d\n",
           n, b, structure, print);

    int passed = 0;
    int total  = 0;

    clock_t t0 = clock();

    // Single matvec correctness check
    run_matvec_test(n, b, structure, print, tol, &passed);
    total++;

    // Single LU + triangular multiply correctness check
    run_lu_trimul_test(n, b, structure, print, tol, &passed);
    total++;

    clock_t t1 = clock();
    double elapsed = (double)(t1 - t0) / CLOCKS_PER_SEC;

    printf("\nDemo completed. Passed %d/%d checks. Elapsed time: %.3f s\n",
           passed, total, elapsed);

    return (passed == total) ? 0 : 1;
}
