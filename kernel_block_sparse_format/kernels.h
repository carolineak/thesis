#ifndef KERNELS_H
#define KERNELS_H

#include <cuComplex.h> 

int launch_kernel(int n);

int trisolve_cu(char side,
                char uplo,
                char trans,
                char diag,
                int m,
                int n,
                const float complex* alpha_host,
                const float complex* A_host,
                int lda,
                float complex* B_host,
                int ldb);

#endif