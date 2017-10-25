////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    lu_cuda.cpp
/// @brief   The implementation of solving linear system by LU using CUDA
///
/// @author  William Liao
///

#include <cuda_runtime.h>
#include <cusolverSp.h>

void lu_host(
    int m,
    int nnz,
    const double *A_val,
    const int *A_row,
    const int *A_col,
    const double *b,
    double *x,
    double tol
) {
    cusolverSpHandle_t sp_handle;
    int reorder = 1, singularity;
    cusolverSpCreate(&sp_handle);

    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA); 
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT);

    cusolverSpDcsrlsvluHost(sp_handle, m, nnz, descrA, A_val, A_row, A_col, b, tol, reorder, x, &singularity);

    cusparseDestroyMatDescr(descrA);
    cusolverSpDestroy(sp_handle);
}
