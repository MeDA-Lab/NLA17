////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    solve_ls_cuda.cpp
/// @brief   The implementation of solving linear system by a chosen solver using CUDA
///
/// @author  William Liao
/// @author  Yuhsiang M. Tsai

#include <iostream>
#include "cuda_wrapper.hpp"
#include "sgp.hpp"

void solvels(
    LS ls,
    int m,
    int nnz,
    const double *A_val,
    const int *A_row,
    const int *A_col,
    const double *b,
    double *x,
    double tol
) {
    switch (ls) {
        case LS::HOST_QR :
            qr_host(m, nnz, A_val, A_row, A_col, b, x, tol);
            break;
        case LS::HOST_CHOL :
            chol_host(m, nnz, A_val, A_row, A_col, b, x, tol);
            break;
        case LS::HOST_LU :
            lu_host(m, nnz, A_val, A_row, A_col, b, x, tol);
            break;
        case LS::DEVICE_QR :
            qr_dev(m, nnz, A_val, A_row, A_col, b, x, tol);
            break;
        case LS::DEVICE_CHOL :
            chol_dev(m, nnz, A_val, A_row, A_col, b, x, tol);
            break;
    }
}