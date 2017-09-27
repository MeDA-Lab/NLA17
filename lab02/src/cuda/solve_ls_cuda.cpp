////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    solve_ls_cuda.cpp
/// @brief   The implementation of solving linear system by a chosen solver using CUDA
///
/// @author  William Liao
///

#include <iostream>
#include "cuda_wrapper.hpp"

void solvelsHost(
    int m,
    int nnz,
    const double *A_val,
    const int *A_row,
    const int *A_col,
    const double *b,
    double *x,
    int solver
) {
    if ( solver == 0 )
    {
        lu_host(m, nnz, A_val, A_row, A_col, b, x);
    }else if ( solver == 1 )
    {
        chol_host(m, nnz, A_val, A_row, A_col, b, x);
    }else if ( solver == 2 )
    {
        qr_host(m, nnz, A_val, A_row, A_col, b, x);
    }else{
        std::cout << "Unknown option!" << std::endl;
    }
}

void solvels(
    int m,
    int nnz,
    const double *A_val,
    const int *A_row,
    const int *A_col,
    const double *b,
    double *x,
    int solver
) {
    if ( solver == 0 )
    {
        // CUDA lu solver currently host only!
        /*
        lu_dev(m, nnz, A_val, A_row, A_col, b, x);
        */
        std::cout << "Option currently not yet supported!" << std::endl;
    }else if ( solver == 1 )
    {
        chol_dev(m, nnz, A_val, A_row, A_col, b, x);
    }else if ( solver == 2 )
    {
        qr_dev(m, nnz, A_val, A_row, A_col, b, x);
    }else{
        std::cout << "Unknown option!" << std::endl;
    }
}
