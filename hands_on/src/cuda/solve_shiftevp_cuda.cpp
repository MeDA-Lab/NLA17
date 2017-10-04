////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    solve_shiftevp_cuda.cpp
/// @brief   The implementation of solving eigenvalueproblem by shift-method by CUDA
///
/// @author  Yuhsiang Mike Tsai
///

#include <iostream>
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include "sgp.hpp"
using namespace std;

void solveShiftEVPHost(
    int m,
    int nnz,
    const double *A_val,
    const int *A_row,
    const int *A_col,
    const double mu0,
    double *mu,
    double *x
) {
    cusolverSpHandle_t sp_handle;
    double *x0 = nullptr, tol = 1e-12;
    int maxite = 1000;
    cusolverSpCreate(&sp_handle);

    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA); 
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    //cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT);
    x0 = new double[m];
    for (int i = 0; i < m; i++) {
        x0[i] = 0;
    }
    x0[0] = 1.0;
    cusolverSpDcsreigvsiHost(sp_handle, m, nnz, descrA, A_val, A_row, A_col,
                             mu0, x0, maxite, tol, mu, x);
    cusparseDestroyMatDescr(descrA);
    cusolverSpDestroy(sp_handle);
    delete x0;
}

void solveShiftEVP(
    int m,
    int nnz,
    const double *A_val,
    const int *A_row,
    const int *A_col,
    const double mu0,
    double *mu,
    double *x
) {
    cusolverSpHandle_t sp_handle;
    double *x0 = nullptr, tol = 1e-12;
    double *dx0 = nullptr, *dmu = nullptr, *dx = nullptr;
    int *dA_row = nullptr, *dA_col = nullptr;
    double *dA_val = nullptr;
    int maxite = 1000;
    cusolverSpCreate(&sp_handle);

    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT);
    x0 = new double[m];
    for (int i = 0; i < m; i++) {
        x0[i] = 0;
    }
    x0[0] = 1.0;

    cudaMalloc(&dx0, m*sizeof(double));
    cudaMalloc(&dmu, sizeof(double));
    cudaMalloc(&dx, m*sizeof(double));
    cudaMalloc(&dA_row, (m+1)*sizeof(int));
    cudaMalloc(&dA_col, nnz*sizeof(int));
    cudaMalloc(&dA_val, nnz*sizeof(double));

    cudaMemcpy(dx0, x0, m*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dA_row, A_row, (m+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dA_col, A_col, nnz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dA_val, A_val, nnz*sizeof(double), cudaMemcpyHostToDevice);

    cusolverSpDcsreigvsi(sp_handle, m, nnz, descrA, dA_val, dA_row, dA_col,
                         mu0, dx0, maxite, tol, dmu, dx);

    cudaMemcpy(mu, dmu, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(x, dx, m*sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(dx0);
    cudaFree(dmu);
    cudaFree(dx);
    cudaFree(dA_row);
    cudaFree(dA_col);
    cudaFree(dA_val);
    cusparseDestroyMatDescr(descrA);
    cusolverSpDestroy(sp_handle);
    delete x0;
}
