#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <complex.h>

// CUDA kernel
__global__ void simple_kernel() {
    printf("Hello from CUDA thread %d!\n", threadIdx.x);
}

// helper to print CUDA errors
static void report_cuda_error(const char *msg, cudaError_t err) {
    fprintf(stderr, "[kernel] %s: %s\n",
            msg,
            cudaGetErrorString(err));
}

// Function callable from C
extern "C" 
int launch_kernel(int n) {

    cudaError_t err;

    int threadsPerBlock = 1; // 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    simple_kernel<<<blocksPerGrid, threadsPerBlock>>>();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        report_cuda_error("kernel launch failed", err);
        return -1;
    }

    // --- Synchronize ---
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        report_cuda_error("cudaDeviceSynchronize failed", err);
        return -1;
    }

    return 0;
}

// Helper: map cublasStatus_t to printable string (basic)
static const char* cublasGetStatusString(cublasStatus_t st) {
    switch(st) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
#if defined(CUBLAS_STATUS_NOT_SUPPORTED)
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if defined(CUBLAS_STATUS_LICENSE_ERROR)
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
        default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

// Helper: check CUDA error
static int checkCuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error %s : %s\n", msg, cudaGetErrorString(e));
        return -1;
    }
    return 0;
}

// Helper: check cuBLAS error
static int checkCublas(cublasStatus_t s, const char* msg) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error %s : %s\n", msg, cublasGetStatusString(s));
        return -1;
    }
    return 0;
}

/*
 * trisolve_cu:
 *   Perform B := alpha * op(A)^{-1} * B   or   B := alpha * B * op(A)^{-1}
 *   using cuBLAS cublasCtrsm.
 *
 * Parameters (C-callable):
 *   side  - 'L' or 'R'  (Left or Right)
 *   uplo  - 'U' or 'L'  (Upper or Lower triangular A)
 *   trans - 'N', 'T', or 'C'  (NoTrans, Transpose, ConjTrans)
 *   diag  - 'U' or 'N'  (Unit or Non-unit diagonal)
 *   m     - number of rows of B
 *   n     - number of columns of B
 *   alpha_host - pointer to cuComplex scalar (host memory)
 *   A_host - pointer to host matrix A (column-major, complex single)
 *   lda   - leading dimension of A
 *   B_host - pointer to host matrix B (column-major)
 *   ldb   - leading dimension of B
 *
 * Notes:
 *   - A is triangular; layout expected column-major (Fortran-style).
 *   - Copies A and B to device, runs cublasCtrsm, copies B result back to host.
 *
 * Returns 0 on success, non-zero on failure.
 */
extern "C"
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
                int ldb)
{
    if (m < 0 || n < 0 || lda < 1 || ldb < 1) {
        fprintf(stderr, "trisolve_cu: invalid matrix dimensions\n");
        return -1;
    }

    // --- Map C chars to cuBLAS enums
    cublasSideMode_t sideMode = (side == 'L' || side == 'l') ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
    cublasFillMode_t fillMode = (uplo == 'U' || uplo == 'u') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
    cublasOperation_t opA;
    if (trans == 'N' || trans == 'n') opA = CUBLAS_OP_N;
    else if (trans == 'T' || trans == 't') opA = CUBLAS_OP_T;
    else opA = CUBLAS_OP_C;
    cublasDiagType_t diagType = (diag == 'U' || diag == 'u') ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;

    // --- Derived sizes
    int k = (sideMode == CUBLAS_SIDE_LEFT) ? m : n;
    size_t sizeA = (size_t)lda * k;
    size_t sizeB = (size_t)ldb * n;

    // --- Convert A, B, alpha from float complex -> cuComplex
    cuComplex *A_cu = (cuComplex*)malloc(sizeA * sizeof(cuComplex));
    cuComplex *B_cu = (cuComplex*)malloc(sizeB * sizeof(cuComplex));
    if (!A_cu || !B_cu) {
        fprintf(stderr, "trisolve_cu: malloc failed\n");
        free(A_cu); free(B_cu);
        return -1;
    }

    for (size_t i = 0; i < sizeA; ++i) {
        A_cu[i].x = crealf(A_host[i]);
        A_cu[i].y = cimagf(A_host[i]);
    }
    for (size_t i = 0; i < sizeB; ++i) {
        B_cu[i].x = crealf(B_host[i]);
        B_cu[i].y = cimagf(B_host[i]);
    }

    cuComplex alpha_cu = make_cuComplex(crealf(*alpha_host), cimagf(*alpha_host));

    // --- Device memory
    cuComplex *d_A = nullptr, *d_B = nullptr;
    if (checkCuda(cudaMalloc((void**)&d_A, sizeA * sizeof(cuComplex)), "cudaMalloc d_A") != 0) goto error;
    if (checkCuda(cudaMalloc((void**)&d_B, sizeB * sizeof(cuComplex)), "cudaMalloc d_B") != 0) goto error;

    if (checkCuda(cudaMemcpy(d_A, A_cu, sizeA * sizeof(cuComplex), cudaMemcpyHostToDevice), "Memcpy A") != 0) goto error;
    if (checkCuda(cudaMemcpy(d_B, B_cu, sizeB * sizeof(cuComplex), cudaMemcpyHostToDevice), "Memcpy B") != 0) goto error;

    cublasHandle_t handle;
    if (checkCublas(cublasCreate(&handle), "cublasCreate") != 0) goto error;
    if (checkCublas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode") != 0) {
        cublasDestroy(handle);
        goto error;
    }

    cublasStatus_t s = cublasCtrsm(handle,
                                   sideMode, fillMode,
                                   opA, diagType,
                                   m, n,
                                   &alpha_cu,
                                   d_A, lda,
                                   d_B, ldb);
    if (checkCublas(s, "cublasCtrsm") != 0) {
        cublasDestroy(handle);
        goto error;
    }

    // --- Copy result back
    if (checkCuda(cudaMemcpy(B_cu, d_B, sizeB * sizeof(cuComplex), cudaMemcpyDeviceToHost), "Memcpy result") != 0) {
        cublasDestroy(handle);
        goto error;
    }

    // --- Convert back to float complex
    for (size_t i = 0; i < sizeB; ++i)
        B_host[i] = B_cu[i].x + B_cu[i].y * I;

    // --- Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    free(A_cu);
    free(B_cu);
    return 0;

error:
    if (d_A) cudaFree(d_A);
    if (d_B) cudaFree(d_B);
    if (A_cu) free(A_cu);
    if (B_cu) free(B_cu);
    return -1;
}

