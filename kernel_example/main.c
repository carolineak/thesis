#include <stdio.h>
#include <cuComplex.h>
#include "kernels.h"

static void print_vec(const char *name, const cuFloatComplex *v, int n) {
    printf("%s:\n", name);
    for (int i = 0; i < n; ++i)
        printf("  %s[%d] = %8.3f + %8.3fi\n",
               name, i, cuCrealf(v[i]), cuCimagf(v[i]));
}

int main(void)
{
    int n = 5;
    cuFloatComplex a[5], b[5], c[5];

    for (int i = 0; i < n; ++i) {
        a[i] = make_cuFloatComplex((float)i, (float)(i));
        b[i] = make_cuFloatComplex((float)i, (float)(i));
    }

    print_vec("a", a, n);
    print_vec("b", b, n);

    if (launch_vector_add(a, b, c, n) != 0)
        return fprintf(stderr, "vector_add failed\n"), 1;

    print_vec("c = a + b", c, n);

    int m = 3;
    cuFloatComplex A[6], x[2], y[3];

    A[0] = make_cuFloatComplex(1,1); A[1] = make_cuFloatComplex(2,2); A[2] = make_cuFloatComplex(3,3);
    A[3] = make_cuFloatComplex(4,4); A[4] = make_cuFloatComplex(5,5); A[5] = make_cuFloatComplex(6,6);

    x[0] = make_cuFloatComplex(1,0);
    x[1] = make_cuFloatComplex(1,0);
    for (int i = 0; i < m; ++i) y[i] = make_cuFloatComplex(0,0);

    cuFloatComplex alpha = make_cuFloatComplex(1,0);
    cuFloatComplex beta  = make_cuFloatComplex(0,0);

    if (launch_matvec_cgemv(A, x, y, m, 2, alpha, beta) != 0)
        return fprintf(stderr, "matvec_cgemv failed\n"), 1;

    print_vec("y = A*x", y, m);

    return 0;
}
