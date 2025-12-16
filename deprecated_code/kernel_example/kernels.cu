#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <stdio.h>

__global__ void vector_add_kernel(const cuFloatComplex *a,
                                  const cuFloatComplex *b,
                                  cuFloatComplex *c,
                                  int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // c[idx] = a[idx] + b[idx] (complex add)
        c[idx] = cuCaddf(a[idx], b[idx]);
    }
}

// helper to print CUDA errors
static void report_cuda_error(const char *msg, cudaError_t err) {
    fprintf(stderr, "[launch_vector_add] %s: %s\n",
            msg,
            cudaGetErrorString(err));
}

extern "C"
int launch_vector_add(const cuFloatComplex *h_a,
                      const cuFloatComplex *h_b,
                      cuFloatComplex *h_c,
                      int n)
{
    cudaSetDevice(0);

    if (n <= 0) {
        fprintf(stderr, "[launch_vector_add] invalid n=%d\n", n);
        return -1;
    }

    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    fprintf(stderr,
        "[launch_vector_add] GPU mem free = %zu / %zu bytes (%.2f MB / %.2f MB)\n",
        free_bytes, total_bytes,
        free_bytes / (1024.0 * 1024.0),
        total_bytes / (1024.0 * 1024.0));

    cuFloatComplex *d_a = NULL;
    cuFloatComplex *d_b = NULL;
    cuFloatComplex *d_c = NULL;
    cudaError_t err;
    size_t bytes = (size_t)n * sizeof(cuFloatComplex);

    // --- Allocate device memory ---
    err = cudaMalloc((void**)&d_a, bytes);
    if (err != cudaSuccess) {
        report_cuda_error("cudaMalloc d_a failed", err);
        return -1;
    }

    err = cudaMalloc((void**)&d_b, bytes);
    if (err != cudaSuccess) {
        report_cuda_error("cudaMalloc d_b failed", err);
        cudaFree(d_a);
        return -1;
    }

    err = cudaMalloc((void**)&d_c, bytes);
    if (err != cudaSuccess) {
        report_cuda_error("cudaMalloc d_c failed", err);
        cudaFree(d_a);
        cudaFree(d_b);
        return -1;
    }

    // --- Copy input data to device ---
    err = cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        report_cuda_error("cudaMemcpy h_a->d_a failed", err);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return -1;
    }

    err = cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        report_cuda_error("cudaMemcpy h_b->d_b failed", err);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return -1;
    }

    // --- Launch kernel ---
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    vector_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        report_cuda_error("kernel launch failed", err);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return -1;
    }

    // --- Synchronize ---
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        report_cuda_error("cudaDeviceSynchronize failed", err);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return -1;
    }

    // --- Copy result back to host ---
    err = cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        report_cuda_error("cudaMemcpy d_c->h_c failed", err);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return -1;
    }

    // --- Cleanup ---
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

// helper to print cuBLAS errors
static void report_cublas_error(const char *msg, cublasStatus_t stat) {
    const char *errstr = "unknown cublas error";
    switch (stat) {
        case CUBLAS_STATUS_SUCCESS:
            errstr = "CUBLAS_STATUS_SUCCESS";
            break;
        case CUBLAS_STATUS_NOT_INITIALIZED:
            errstr = "CUBLAS_STATUS_NOT_INITIALIZED";
            break;
        case CUBLAS_STATUS_ALLOC_FAILED:
            errstr = "CUBLAS_STATUS_ALLOC_FAILED";
            break;
        case CUBLAS_STATUS_INVALID_VALUE:
            errstr = "CUBLAS_STATUS_INVALID_VALUE";
            break;
        case CUBLAS_STATUS_ARCH_MISMATCH:
            errstr = "CUBLAS_STATUS_ARCH_MISMATCH";
            break;
        case CUBLAS_STATUS_MAPPING_ERROR:
            errstr = "CUBLAS_STATUS_MAPPING_ERROR";
            break;
        case CUBLAS_STATUS_EXECUTION_FAILED:
            errstr = "CUBLAS_STATUS_EXECUTION_FAILED";
            break;
        case CUBLAS_STATUS_INTERNAL_ERROR:
            errstr = "CUBLAS_STATUS_INTERNAL_ERROR";
            break;
        default:
            break;
    }
    fprintf(stderr, "[launch_matvec_cgemv] %s: %s\n", msg, errstr);
}

// C-callable launcher for complex matrix-vector multiply using cuBLAS.
// Computes y := alpha * A * x + beta * y
//
// A: m x n complex matrix (column-major layout)
// x: length n complex vector
// y: length m complex vector (in/out)
// alpha, beta: complex scalars
//
// On success returns 0, on failure returns -1.
// All pointers are HOST pointers.
extern "C"
int launch_matvec_cgemv(const cuFloatComplex *h_A,
                        const cuFloatComplex *h_x,
                        cuFloatComplex *h_y,
                        int m,
                        int n,
                        cuFloatComplex alpha,
                        cuFloatComplex beta)
{
    cudaSetDevice(0);

    if (m <= 0 || n <= 0) {
        fprintf(stderr, "[launch_matvec_cgemv] invalid dims m=%d n=%d\n", m, n);
        return -1;
    }

    // GPU memory pointers
    cuFloatComplex *d_A = NULL;
    cuFloatComplex *d_x = NULL;
    cuFloatComplex *d_y = NULL;

    cudaError_t cerr;
    cublasStatus_t stat;
    cublasHandle_t handle = NULL;

    size_t bytes_A = (size_t)m * (size_t)n * sizeof(cuFloatComplex);
    size_t bytes_x = (size_t)n * sizeof(cuFloatComplex);
    size_t bytes_y = (size_t)m * sizeof(cuFloatComplex);

    // Allocate device memory
    cerr = cudaMalloc((void**)&d_A, bytes_A);
    if (cerr != cudaSuccess) {
        report_cuda_error("cudaMalloc d_A failed", cerr);
        return -1;
    }

    cerr = cudaMalloc((void**)&d_x, bytes_x);
    if (cerr != cudaSuccess) {
        report_cuda_error("cudaMalloc d_x failed", cerr);
        cudaFree(d_A);
        return -1;
    }

    cerr = cudaMalloc((void**)&d_y, bytes_y);
    if (cerr != cudaSuccess) {
        report_cuda_error("cudaMalloc d_y failed", cerr);
        cudaFree(d_A);
        cudaFree(d_x);
        return -1;
    }

    // Copy host -> device
    cerr = cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    if (cerr != cudaSuccess) {
        report_cuda_error("cudaMemcpy A failed", cerr);
        cudaFree(d_A);
        cudaFree(d_x);
        cudaFree(d_y);
        return -1;
    }

    cerr = cudaMemcpy(d_x, h_x, bytes_x, cudaMemcpyHostToDevice);
    if (cerr != cudaSuccess) {
        report_cuda_error("cudaMemcpy x failed", cerr);
        cudaFree(d_A);
        cudaFree(d_x);
        cudaFree(d_y);
        return -1;
    }

    cerr = cudaMemcpy(d_y, h_y, bytes_y, cudaMemcpyHostToDevice);
    if (cerr != cudaSuccess) {
        report_cuda_error("cudaMemcpy y failed", cerr);
        cudaFree(d_A);
        cudaFree(d_x);
        cudaFree(d_y);
        return -1;
    }

    // cuBLAS init
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        report_cublas_error("cublasCreate failed", stat);
        cudaFree(d_A);
        cudaFree(d_x);
        cudaFree(d_y);
        return -1;
    }

    // We want y := alpha*A*x + beta*y
    //
    // cuBLAS is column-major by default, so we call cublasCgemv with:
    //   op    = CUBLAS_OP_N  (no transpose)
    //   m,n   = m,n
    //   lda   = m  (leading dimension of A is the #rows in column-major)
    //
    // Strides/incx/incy are 1 for contiguous vectors.
    stat = cublasCgemv(
        handle,
        CUBLAS_OP_N,              // op(A) = A
        m,                        // rows of A
        n,                        // cols of A
        &alpha,                   // alpha
        d_A,                      // A on device
        m,                        // lda (leading dimension of A)
        d_x,                      // x on device
        1,                        // incx
        &beta,                    // beta
        d_y,                      // y on device (in/out)
        1                         // incy
    );

    if (stat != CUBLAS_STATUS_SUCCESS) {
        report_cublas_error("cublasCgemv failed", stat);
        cublasDestroy(handle);
        cudaFree(d_A);
        cudaFree(d_x);
        cudaFree(d_y);
        return -1;
    }

    // Sync to ensure computation is done
    cerr = cudaDeviceSynchronize();
    if (cerr != cudaSuccess) {
        report_cuda_error("cudaDeviceSynchronize failed", cerr);
        cublasDestroy(handle);
        cudaFree(d_A);
        cudaFree(d_x);
        cudaFree(d_y);
        return -1;
    }

    // Copy result y back to host
    cerr = cudaMemcpy(h_y, d_y, bytes_y, cudaMemcpyDeviceToHost);
    if (cerr != cudaSuccess) {
        report_cuda_error("cudaMemcpy y->host failed", cerr);
        cublasDestroy(handle);
        cudaFree(d_A);
        cudaFree(d_x);
        cudaFree(d_y);
        return -1;
    }

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
