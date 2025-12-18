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
    const int print           = 1;      // 0=silent, 1=results, 2=show data, 3=show LU
    const double tolerance    = 1e-3;
    int on_server             = 1;      // Set to 1 when running on TICRA's server to enable those tests
    
    int passed = 0, total = 0;

    // Local tests
    printf("Test on xs binary data\n");
    char *data = "../experiments/data/sparse_data_xs.bin";
    run_lu_trimul_test_on_bin_data(print, tolerance, &passed, data); total++;
    // debug_print_input_bin(data);

    printf("Test on s binary data\n");
    data = "../experiments/data/sparse_data_s.bin";
    run_lu_trimul_test_on_bin_data(print, tolerance, &passed, data); total++;


    // Tests on TICRA's server
    if (on_server) {
        printf("Test on patch array with 5x5 patches\n");
        data = "/x/users/mhg/til_ck/patch_array/patch_array_5x5.bin";
        run_lu_trimul_test_on_bin_data(print, tolerance, &passed, data); total++;

        printf("\nTest on patch array with 8x8 patches\n");
        data = "/x/users/mhg/til_ck/patch_array/patch_array_8x8.bin";
        run_lu_trimul_test_on_bin_data(print, tolerance, &passed, data); total++;

        printf("\nTest on patch array with 10x10 patches\n");
        data = "/x/users/mhg/til_ck/patch_array/patch_array_10x10.bin";
        run_lu_trimul_test_on_bin_data(print, tolerance, &passed, data); total++;

        printf("\nTest on patch array with 12x12 patches\n");
        data = "/x/users/mhg/til_ck/patch_array/patch_array_12x12.bin";
        run_lu_trimul_test_on_bin_data(print, tolerance, &passed, data); total++;

        printf("\nTest on patch array with 15x15 patches\n");
        data = "/x/users/mhg/til_ck/patch_array/patch_array_15x15.bin";
        run_lu_trimul_test_on_bin_data(print, tolerance, &passed, data); total++;

        printf("\nTest on reflector with struts 25GHz\n");
        data = "/x/users/mhg/til_ck/reflector_with_struts/case_25GHz.bin";
        run_lu_trimul_test_on_bin_data(print, tolerance, &passed, data); total++;

        printf("\nTest on reflector with struts 30GHz\n");
        data = "/x/users/mhg/til_ck/reflector_with_struts/case_30GHz.bin";
        run_lu_trimul_test_on_bin_data(print, tolerance, &passed, data); total++;
    }

    printf("\nAll tests completed. Passed %d out of %d tests.\n", passed, total);

    return 0;
}
