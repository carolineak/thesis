#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <complex.h>

// ==================================================================
// Global persistent handles for cuBLAS and cuSOLVER
// ==================================================================
static cublasHandle_t cublas_handle = NULL;
static cusolverDnHandle_t cusolver_handle = NULL;

// ==================================================================
// Initialise CUDA handles
// ==================================================================
extern "C" 
int gpu_init(void) {
    cublasStatus_t cublas_status = cublasCreate(&cublas_handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[gpu_init] cublasCreate failed (%d)\n", (int)cublas_status);
        cublas_handle = NULL;
        return -1;
    }
    cusolverStatus_t cusolver_status = cusolverDnCreate(&cusolver_handle);
    if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "[gpu_init] cusolverDnCreate failed (%d)\n", (int)cusolver_status);
        cublasDestroy(cublas_handle);
        cublas_handle = NULL;
        cusolver_handle = NULL;
        return -1;
    }
    return 0;
}

// ==================================================================
// Finalise CUDA handles
// ==================================================================
extern "C" 
int gpu_finalise(void) {
    int rc = 0;
    if (cublas_handle) {
        cublasStatus_t cublas_status = cublasDestroy(cublas_handle);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) rc = -1;
        cublas_handle = NULL;
    }
    if (cusolver_handle) {
        cusolverStatus_t cusolver_status = cusolverDnDestroy(cusolver_handle);
        if (cusolver_status != CUSOLVER_STATUS_SUCCESS) rc = -1;
        cusolver_handle = NULL;
    }
    return rc;
}

// ==================================================================
// Matrix-vector multiplication for block sparse format using cuBLAS
// ==================================================================
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
    
    if (!cublas_handle) {
        fprintf(stderr, "[matvec_cu] cuBLAS not initialised\n");
        return -1;
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

        // Pointers to current block and vectors
        const cuFloatComplex* A = d_flat_data + offset;
        const cuFloatComplex* x = d_x + col_start;
        cuFloatComplex* y = d_y + row_start;

        // Call cuBLAS gemv for this block
        cublasStatus_t stat = cublasCgemv(cublas_handle, 
                                          CUBLAS_OP_N, 
                                          M, N, 
                                          (const cuComplex*)&alpha, 
                                          (const cuComplex*)A, M, 
                                          (const cuComplex*)x, 1, 
                                          (const cuComplex*)&beta, 
                                          (cuComplex*)y, 1);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "[matvec_cu] cublasCgemv failed for block %d (M=%d,N=%d) status=%d\n", 
                             k, M, N, (int)stat);
            return -1;
        }
    }
    return 0;
}
// ==================================================================
// Triangular solve for block sparse format using cuBLAS
// ==================================================================
extern "C" 
int trisolve_cu(char side,
                char uplo,
                char trans,
                char diag,
                int m,
                int n,
                const cuFloatComplex* alpha,
                const cuFloatComplex* d_A,
                int lda,
                cuFloatComplex* d_B,
                int ldb)
{
    if (!alpha || !d_A || !d_B) return -1;

    // Map the char arguments to cuBLAS enums
    cublasSideMode_t sideMode = (side == 'L' || side == 'l') ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
    cublasFillMode_t uploMode = (uplo == 'L' || uplo == 'l') ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    cublasOperation_t transMode = (trans == 'N' || trans == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasDiagType_t diagMode = (diag == 'U' || diag == 'u') ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;

    if (!cublas_handle) {
        fprintf(stderr, "[trisolve_cu] cuBLAS not initialised\n");
        return -1;
    }

    // Call cuBLAS trsm with device pointers using global handle
    cublasStatus_t stat = cublasCtrsm(cublas_handle, sideMode, uploMode, transMode, diagMode,
                       m, n,
                       (const cuComplex*)alpha,
                       (const cuComplex*)d_A, lda,
                       (cuComplex*)d_B, ldb);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[trisolve_cu] cublasCtrsm failed status=%d\n", (int)stat);
        return -1;
    }
    return 0;
}

// ==================================================================
// Kernel to apply pivots to a device block in-place
// ==================================================================
__global__ void apply_pivots_kernel(cuFloatComplex* d_A, int lda, int num_cols, const int* d_ipiv, int m)
{
    int i = blockIdx.x;
    if (i >= m) return;

    int piv = d_ipiv[i] - 1; // ipiv is 1-based
    if (piv == i || piv < 0) return;
    if (piv >= m) return;

    // Only perform swap when pivot index > i to avoid double-swapping
    if (piv > i) {
        for (int c = threadIdx.x; c < num_cols; c += blockDim.x) {
            int idx1 = i + c * lda;
            int idx2 = piv + c * lda;
            cuFloatComplex tmp = d_A[idx1];
            d_A[idx1] = d_A[idx2];
            d_A[idx2] = tmp;
        }
    }
}

// ==================================================================
// Wrapper to apply pivots to a device block in-place
// ==================================================================
extern "C" 
int apply_pivots_cu(cuFloatComplex* d_A, int lda, int num_cols, const int* h_ipiv, int m)
{
    if (!d_A || !h_ipiv || m <= 0) return -1;

    // Copy pivot array to device
    int *d_ipiv = NULL;
    size_t bytes = (size_t)m * sizeof(int);
    cudaError_t err = cudaMalloc((void**)&d_ipiv, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[apply_pivots_cu] cudaMalloc failed\n");
        return -1;
    }
    err = cudaMemcpy(d_ipiv, (const void*)h_ipiv, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[apply_pivots_cu] cudaMemcpy H2D failed\n");
        cudaFree(d_ipiv);
        return -1;
    }

    // Launch kernel to apply pivots
    int threads = 128; // Number of threads per block, can be tuned, not above 1024
    dim3 grid(m);
    dim3 block(threads);
    apply_pivots_kernel<<<grid, block>>>(d_A, lda, num_cols, d_ipiv, m);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[apply_pivots_cu] apply_pivots_kernel launch failed\n");
        cudaFree(d_ipiv);
        return -1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[apply_pivots_cu] apply_pivots_kernel sync failed\n");
        cudaFree(d_ipiv);
        return -1;
    }

    cudaFree(d_ipiv);
    return 0;
}

// ==================================================================
// Schur update for blocks using cuBLAS
// ==================================================================
extern "C"
int block_schur_update_cu(cuFloatComplex* d_C,
                              const cuFloatComplex* d_A,
                              const cuFloatComplex* d_B,
                              int m, int n, int k)
{
    if (!d_C || !d_A || !d_B) return -1;

    if (!cublas_handle) {
        fprintf(stderr, "[block_schur_update_cu] cuBLAS not initialised\n");
        return -1;
    }

    // Set alpha and beta for C = C - A*B
    cuFloatComplex alpha = make_cuFloatComplex(-1.0f, 0.0f);
    cuFloatComplex beta  = make_cuFloatComplex(1.0f, 0.0f);

   // Call cuBLAS gemm with device pointers using global handle
    cublasStatus_t stat = cublasCgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                       m, n, k,
                       (const cuComplex*)&alpha,
                       (const cuComplex*)d_A, m,
                       (const cuComplex*)d_B, k,
                       (const cuComplex*)&beta,
                       (cuComplex*)d_C, m);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[block_schur_update_cu] cublasCgemm failed status=%d\n", (int)stat);
        return -1;
    }
    return 0;
}

// ==================================================================
// LU factorisation for a single square block using cuSOLVER 
// ==================================================================
extern "C"
int block_getrf_cu(cuFloatComplex* d_A, int n, int lda, int* h_ipiv, int* info)
{
    if (!d_A || n <= 0 || !h_ipiv || !info) return -1;
    if (!cusolver_handle) {
        fprintf(stderr, "[block_getrf_cu] cuSOLVER not initialised\n");
        return -1;
    }

    // Figure out workspace size
    int lwork = 0;
    cusolverStatus_t cs = cusolverDnCgetrf_bufferSize(cusolver_handle, n, n, (cuComplex*)d_A, lda, &lwork);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "[block_getrf_cu] bufferSize failed (%d)\n", (int)cs);
        return -1;
    }

    // Allocate workspace on device
    cuFloatComplex *d_work = NULL;
    if (lwork > 0) {
        cudaError_t err = cudaMalloc((void**)&d_work, (size_t)lwork * sizeof(cuFloatComplex));
        if (err != cudaSuccess) {
            fprintf(stderr, "[block_getrf_cu] cudaMalloc workspace failed\n");
            return -1;
        }
    }

    // Allocate device memory for pivot array and info
    int *d_ipiv = NULL;
    int *d_info = NULL;
    cudaError_t err;
    err = cudaMalloc((void**)&d_ipiv, n * sizeof(int));
    if (err != cudaSuccess) { if (d_work) cudaFree(d_work); return -1; }
    err = cudaMalloc((void**)&d_info, sizeof(int));
    if (err != cudaSuccess) { cudaFree(d_ipiv); if (d_work) cudaFree(d_work); return -1; }

    // Perform LU factorisation
    cs = cusolverDnCgetrf(cusolver_handle, 
                          n, n, 
                          (cuComplex*)d_A, lda, 
                          (cuComplex*)d_work, 
                          d_ipiv, 
                          d_info);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "[block_getrf_cu] cusolverDnCgetrf failed (%d)\n", (int)cs);
        cudaFree(d_ipiv); cudaFree(d_info); if (d_work) cudaFree(d_work); return -1;
    }

    // copy ipiv and info back to host
    err = cudaMemcpy(h_ipiv, d_ipiv, n * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "[block_getrf_cu] cudaMemcpy ipiv D2H failed\n");
        cudaFree(d_ipiv); cudaFree(d_info); if (d_work) cudaFree(d_work); return -1;
    }
    int info_dev = 0;
    err = cudaMemcpy(&info_dev, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "[block_getrf_cu] cudaMemcpy info D2H failed\n");
        cudaFree(d_ipiv); cudaFree(d_info); if (d_work) cudaFree(d_work); return -1;
    }
    *info = info_dev;

    // Free device memory
    cudaFree(d_ipiv);
    cudaFree(d_info);
    if (d_work) cudaFree(d_work);

    return 0;
}
