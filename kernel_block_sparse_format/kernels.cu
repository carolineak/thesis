#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <complex.h>

// Helper to print CUDA errors
static void report_cuda_error(const char *msg, cudaError_t err) {
    fprintf(stderr, "[kernel] %s: %s\n",
            msg,
            cudaGetErrorString(err));
}

// Global persistent handles for cuBLAS and cuSOLVER
static cublasHandle_t g_cublas = NULL;
static cusolverDnHandle_t g_cusolver = NULL;

// Initialize GPU libraries
extern "C" int bsf_gpu_init(void) {
    cublasStatus_t cst = cublasCreate(&g_cublas);
    if (cst != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[bsf_gpu_init] cublasCreate failed (%d)\n", (int)cst);
        g_cublas = NULL;
        return -1;
    }
    cusolverStatus_t cus = cusolverDnCreate(&g_cusolver);
    if (cus != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "[bsf_gpu_init] cusolverDnCreate failed (%d)\n", (int)cus);
        cublasDestroy(g_cublas);
        g_cublas = NULL;
        g_cusolver = NULL;
        return -2;
    }
    return 0;
}

// Finalize GPU libraries
extern "C" int bsf_gpu_finalize(void) {
    int rc = 0;
    if (g_cublas) {
        cublasStatus_t cst = cublasDestroy(g_cublas);
        if (cst != CUBLAS_STATUS_SUCCESS) rc = -1;
        g_cublas = NULL;
    }
    if (g_cusolver) {
        cusolverStatus_t cus = cusolverDnDestroy(g_cusolver);
        if (cus != CUSOLVER_STATUS_SUCCESS) rc = -2;
        g_cusolver = NULL;
    }
    return rc;
}

// Use cuBLAS to perform a per-block GEMV on the device. Metadata arrays are host-side.
extern "C"
int matvec_cu(const cuFloatComplex* d_flat_data,
                   int num_blocks,
                   const int* h_row_start,
                   const int* h_M,
                   const int* h_col_start,
                   const int* h_N,
                   const int* h_offsets,
                   const cuFloatComplex* d_x,
                   cuFloatComplex* d_y)
{
    if (!d_flat_data || !h_row_start || !h_M || !h_col_start || !h_N || !h_offsets || !d_x || !d_y) return -1;
    
    if (!g_cublas) {
        fprintf(stderr, "[matvec_cu] cuBLAS not initialized\n");
        return -2;
    }

    cuFloatComplex alpha = make_cuFloatComplex(1.0f, 0.0f);
    cuFloatComplex beta  = make_cuFloatComplex(1.0f, 0.0f);

    for (int k = 0; k < num_blocks; ++k) {
        int M = h_M[k];
        int N = h_N[k];
        int row_start = h_row_start[k];
        int col_start = h_col_start[k];
        int offset = h_offsets[k];

        if (M <= 0 || N <= 0) continue;

        const cuFloatComplex* A = d_flat_data + offset; // device pointer to block (col-major, lda=M)
        const cuFloatComplex* x = d_x + col_start;
        cuFloatComplex* y = d_y + row_start;

        // cublasCgemv(handle, trans, m, n, &alpha, A, lda, x, incx, &beta, y, incy)
        cublasStatus_t stat = cublasCgemv(g_cublas, CUBLAS_OP_N, M, N, (const cuComplex*)&alpha, (const cuComplex*)A, M, (const cuComplex*)x, 1, (const cuComplex*)&beta, (cuComplex*)y, 1);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "[matvec_cu] cublasCgemv failed for block %d (M=%d,N=%d) status=%d\n", k, M, N, (int)stat);
            return -3;
        }
    }
    return 0;
}

// Triangular solve using cuBLAS: wraps cublasCtrsm
extern "C"
int trisolve_cu(char side,
                char uplo,
                char trans,
                char diag,
                int m,
                int n,
                const cuFloatComplex* alpha_host,
                const cuFloatComplex* A_host,
                int lda,
                cuFloatComplex* B_host,
                int ldb)
{
    if (!alpha_host || !A_host || !B_host) return -1;

    // Map char args to cuBLAS enums
    cublasSideMode_t sideMode = (side == 'L' || side == 'l') ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
    cublasFillMode_t uploMode = (uplo == 'L' || uplo == 'l') ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    cublasOperation_t transMode = (trans == 'N' || trans == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasDiagType_t diagMode = (diag == 'U' || diag == 'u') ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;

    int M = m;
    int N = n;

    // Treat A_host and B_host as device pointers (caller must ensure data is on device)
    const cuFloatComplex* d_A = (const cuFloatComplex*)A_host;
    cuFloatComplex* d_B = (cuFloatComplex*)B_host;

    if (!g_cublas) {
        fprintf(stderr, "[trisolve_cu] cuBLAS not initialized\n");
        return -2;
    }

    cuFloatComplex alpha = *alpha_host;

    // Call cuBLAS trsm with device pointers using global handle
    cublasStatus_t stat = cublasCtrsm(g_cublas, sideMode, uploMode, transMode, diagMode,
                       M, N,
                       (const cuComplex*)&alpha,
                       (const cuComplex*)d_A, lda,
                       (cuComplex*)d_B, ldb);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[trisolve_cu] cublasCtrsm failed status=%d\n", (int)stat);
        return -3;
    }
    return 0;
}

// Kernel to apply pivots to a device block in-place.
__global__ void apply_pivots_kernel(cuFloatComplex* d_A, int lda, int ncols, const int* d_ipiv, int m)
{
    int i = blockIdx.x;
    if (i >= m) return;
    int piv = d_ipiv[i] - 1; // ipiv is 1-based
    if (piv == i || piv < 0) return;
    // Only perform swap when pivot index > i to avoid double-swapping
    if (piv > i) {
        for (int c = threadIdx.x; c < ncols; c += blockDim.x) {
            int idx1 = i + c * lda;
            int idx2 = piv + c * lda;
            cuFloatComplex tmp = d_A[idx1];
            d_A[idx1] = d_A[idx2];
            d_A[idx2] = tmp;
        }
    }
}

// Host wrapper: copy ipiv to device and launch pivot kernel
extern "C" int apply_pivots_cu(cuFloatComplex* d_A, int lda, int ncols, const int* h_ipiv, int m)
{
    if (!d_A || !h_ipiv || m <= 0) return -1;
    int *d_ipiv = NULL;
    size_t bytes = (size_t)m * sizeof(int);
    cudaError_t cerr = cudaMalloc((void**)&d_ipiv, bytes);
    if (cerr != cudaSuccess) {
        report_cuda_error("apply_pivots_cu cudaMalloc failed", cerr);
        return -2;
    }
    cerr = cudaMemcpy(d_ipiv, (const void*)h_ipiv, bytes, cudaMemcpyHostToDevice);
    if (cerr != cudaSuccess) {
        report_cuda_error("apply_pivots_cu cudaMemcpy H2D failed", cerr);
        cudaFree(d_ipiv);
        return -3;
    }

    int threads = 128;
    dim3 grid(m);
    dim3 block(threads);
    apply_pivots_kernel<<<grid, block>>>(d_A, lda, ncols, d_ipiv, m);
    cerr = cudaGetLastError();
    if (cerr != cudaSuccess) {
        report_cuda_error("apply_pivots_kernel launch failed", cerr);
        cudaFree(d_ipiv);
        return -4;
    }
    cerr = cudaDeviceSynchronize();
    if (cerr != cudaSuccess) {
        report_cuda_error("apply_pivots_kernel sync failed", cerr);
        cudaFree(d_ipiv);
        return -5;
    }

    cudaFree(d_ipiv);
    return 0;
}

// Device Schur update: C := C - A * B using cuBLAS
extern "C"
int block_schur_update_cu(cuFloatComplex* d_C,
                              const cuFloatComplex* d_A,
                              const cuFloatComplex* d_B,
                              int m, int n, int k)
{
    if (!d_C || !d_A || !d_B) return -1;

    if (!g_cublas) {
        fprintf(stderr, "[block_schur_update_cu] cuBLAS not initialized\n");
        return -2;
    }

    cuFloatComplex alpha = make_cuFloatComplex(-1.0f, 0.0f);
    cuFloatComplex beta  = make_cuFloatComplex(1.0f, 0.0f);

    // cublasCgemm(handle, transA, transB, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc)
    cublasStatus_t stat = cublasCgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                       m, n, k,
                       (const cuComplex*)&alpha,
                       (const cuComplex*)d_A, m,
                       (const cuComplex*)d_B, k,
                       (const cuComplex*)&beta,
                       (cuComplex*)d_C, m);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[block_schur_update_cu] cublasCgemm failed status=%d\n", (int)stat);
        return -3;
    }
    return 0;
}

// LU on device for a single square block using cuSOLVER (in-place)
extern "C"
int block_getrf_cu(cuFloatComplex* d_A, int n, int lda, int* h_ipiv, int* info)
{
    if (!d_A || n <= 0 || !h_ipiv || !info) return -1;
    if (!g_cusolver) {
        fprintf(stderr, "[block_getrf_cu] cuSOLVER not initialized\n");
        return -2;
    }

    int lwork = 0;
    cusolverStatus_t cs = cusolverDnCgetrf_bufferSize(g_cusolver, n, n, (cuComplex*)d_A, lda, &lwork);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "[block_getrf_cu] bufferSize failed (%d)\n", (int)cs);
        return -3;
    }

    cuFloatComplex *d_work = NULL;
    if (lwork > 0) {
        cudaError_t cerr = cudaMalloc((void**)&d_work, (size_t)lwork * sizeof(cuFloatComplex));
        if (cerr != cudaSuccess) {
            report_cuda_error("block_getrf_cu cudaMalloc workspace failed", cerr);
            return -4;
        }
    }

    int *d_ipiv = NULL;
    int *d_info = NULL;
    cudaError_t cerr;
    cerr = cudaMalloc((void**)&d_ipiv, n * sizeof(int));
    if (cerr != cudaSuccess) { if (d_work) cudaFree(d_work); return -5; }
    cerr = cudaMalloc((void**)&d_info, sizeof(int));
    if (cerr != cudaSuccess) { cudaFree(d_ipiv); if (d_work) cudaFree(d_work); return -6; }

    cs = cusolverDnCgetrf(g_cusolver, n, n, (cuComplex*)d_A, lda, (cuComplex*)d_work, d_ipiv, d_info);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "[block_getrf_cu] cusolverDnCgetrf failed (%d)\n", (int)cs);
        cudaFree(d_ipiv); cudaFree(d_info); if (d_work) cudaFree(d_work); return -7;
    }

    // copy ipiv and info back to host
    cerr = cudaMemcpy(h_ipiv, d_ipiv, n * sizeof(int), cudaMemcpyDeviceToHost);
    if (cerr != cudaSuccess) {
        report_cuda_error("block_getrf_cu cudaMemcpy ipiv D2H failed", cerr);
        cudaFree(d_ipiv); cudaFree(d_info); if (d_work) cudaFree(d_work); return -8;
    }
    int info_dev = 0;
    cerr = cudaMemcpy(&info_dev, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    if (cerr != cudaSuccess) {
        report_cuda_error("block_getrf_cu cudaMemcpy info D2H failed", cerr);
        cudaFree(d_ipiv); cudaFree(d_info); if (d_work) cudaFree(d_work); return -9;
    }
    *info = info_dev;

    // free temps
    cudaFree(d_ipiv);
    cudaFree(d_info);
    if (d_work) cudaFree(d_work);

    return 0;
}
