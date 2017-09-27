////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    generate_RHS.cpp
/// @brief   Generate the RHS for the linear system
///
/// @author  William Liao
///

#include <vector>
#include "mkl.h"

void genRHS(double *b, int n, int nnz, double *A_val, int *A_row, int *A_col){
	std::vector<double> v(n, 1.0);
	double *tmp_v = v.data();
	char transa = 'N';

	mkl_cspblas_dcsrgemv(&transa, &n, A_val, A_row, A_col, tmp_v, b);
}