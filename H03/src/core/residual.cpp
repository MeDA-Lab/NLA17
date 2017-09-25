////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    residual.cpp
/// @brief   Compute the error of the solution to the linear system
///
/// @author  William Liao
///

#include "mkl.h"

double residual(int n, int nnz, double *A_val, int *A_row, int *A_col, double *b, double *x){
	double alpha = 1.0, beta = -1.0, r;
	char transa = 'N';
	char *matdescra;

	matdescra = new char[6];

	matdescra[0] = 'G';
	matdescra[3] = 'C';

	mkl_dcsrmv(&transa, &n, &n, &alpha, matdescra, A_val, A_col, A_row, A_row+1, x, &beta, b);
	r = cblas_dnrm2(n, b, 1);

	return r;
}