////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    lu_cuda.cpp
/// @brief   The implementation of solving linear system by QR using CUDA
///
/// @author  William Liao
///

#include <cuda_runtime.h>
#include <cusolverSp.h>

void qr_host(
    int m,
    int nnz,
    const double *A_val,
    const int *A_row,
    const int *A_col,
    const double *b,
    double *x
) {
    cusolverSpHandle_t sp_handle;
    double tol = 1e-12;
    int reorder = 1, singularity;
    cusolverSpCreate(&sp_handle);

    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA); 
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT);

    cusolverSpDcsrlsvqrHost(sp_handle, m, nnz, descrA, A_val, A_row, A_col, b, tol, reorder, x, &singularity);

    cusparseDestroyMatDescr(descrA);
    cusolverSpDestroy(sp_handle);
}

void qr_dev(
    int m,
    int nnz,
    const double *A_val,
    const int *A_row,
    const int *A_col,
    const double *b,
    double *x
) {
    cusolverSpHandle_t sp_handle;
    double tol = 1e-12;
    double *db = nullptr, *dx = nullptr;
    int *dA_row = nullptr, *dA_col = nullptr;
    double *dA_val = nullptr;
    int reorder = 1, singularity;
    cusolverSpCreate(&sp_handle);

    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT);

    cudaMalloc(&db, m*sizeof(double));
    cudaMalloc(&dx, m*sizeof(double));
    cudaMalloc(&dA_row, (m+1)*sizeof(int));
    cudaMalloc(&dA_col, nnz*sizeof(int));
    cudaMalloc(&dA_val, nnz*sizeof(double));

    cudaMemcpy(db, b, m*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dA_row, A_row, (m+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dA_col, A_col, nnz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dA_val, A_val, nnz*sizeof(double), cudaMemcpyHostToDevice);

    cusolverSpDcsrlsvqr(sp_handle, m, nnz, descrA, dA_val, dA_row, dA_col, db, tol, reorder, dx, &singularity);

    cudaMemcpy(x, dx, m*sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(db);
    cudaFree(dx);
    cudaFree(dA_row);
    cudaFree(dA_col);
    cudaFree(dA_val);
    cusparseDestroyMatDescr(descrA);
    cusolverSpDestroy(sp_handle);
}