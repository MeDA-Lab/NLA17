////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    graph_adjacency.cpp
/// @brief   Construct adjacency matrix of graph
///
/// @author  William Liao
///

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <cstring>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include "cusparse.h"
#include <mkl.h>

using namespace std;

int GraphAdjacency(int *E, int E_size,
	int *nnz, int **cooRowIndA,
	int **cooColIndA, double **cooValA, int *n, char flag){
	int pos1, pos2, info;
	int *d_cooRowIndA, *d_cooColIndA;
	double  *d_val, *d_val_sorted;
	double *tmp_array, beta = 1.0;
	vector<double> v1 (E_size , 1.0);
	cusparseHandle_t handle;
	cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;
	cusparseStatus_t stat;
	size_t pBufferSizeInBytes = 0;
	void *pBuffer = NULL;
	int *P = NULL;
	char trans = 'T';
	int request = 1, sort;

	tmp_array = new double[2*E_size];
	copy(E, E+2*E_size, tmp_array);

	pos1 = cblas_idamax(E_size, tmp_array, 1);
	pos2 = cblas_idamax(E_size, tmp_array+E_size, 1);
	*n   = max(E[pos1] , E[pos2+E_size])+1;
	//cout << "n = " << *n << endl;

	if( flag == 'S' )
	{
		int  *job, *csrRowInd, *csrColInd, *cooRowInd, *cooColInd;
		double  *csrVal, *cooVal;

		cooRowInd = new int[E_size];
		cooColInd = new int[E_size];
		cooVal    = new double[E_size];

		job = new int[6];
		*nnz = E_size;
		copy(E , E+E_size , cooRowInd);
		copy(E+E_size, E+2*E_size, cooColInd);
		copy(v1.begin(), v1.end(), cooVal);

		// A+trans(A)
		csrVal    = new double[*nnz];
		csrRowInd = new int[*n+1];
		csrColInd = new int[*nnz];

		job[0] = 2;
	  	job[1] = 1;
	  	job[2] = 0;
	  	job[4] = (*n)*(*n);
	  	job[5] = 0;
		mkl_dcsrcoo(job, n, csrVal, csrColInd, csrRowInd, nnz, cooVal, cooRowInd, cooColInd, &info);
		delete cooVal;
		delete cooRowInd;
		delete cooColInd;
		cooRowInd = new int[*n+1];
		int nzmax = (*n)*(*n);
		mkl_dcsradd(&trans, &request, &sort, n, n, csrVal, csrColInd, csrRowInd, &beta, csrVal, csrColInd, csrRowInd, cooVal, cooColInd, cooRowInd, &nzmax, &info);
		assert( info == 0 );
		*nnz = cooRowInd[*n]-1;
		request = 2;
		cooVal    = new double[*nnz];
		cooColInd = new int[*nnz];
		mkl_dcsradd(&trans, &request, &sort, n, n, csrVal, csrColInd, csrRowInd, &beta, csrVal, csrColInd, csrRowInd, cooVal, cooColInd, cooRowInd, &nzmax, &info);
		assert( info == 0 );

		job[0] = 0;
		job[4] = *nnz;
		job[5] = 3;
		*cooValA    = new double[*nnz];
		*cooRowIndA = new int[*nnz];
		*cooColIndA = new int[*nnz];
		mkl_dcsrcoo(job, n, cooVal, cooColInd, cooRowInd, nnz, *cooValA, *cooRowIndA, *cooColIndA, &info);
		//cout << "info = " << info << endl;
		assert( info == 0 );
	}else if( flag == 'W' ){
		//cout << "debug W" << endl;
		*nnz = E_size;
		*cooValA    = new double[*nnz];
		*cooRowIndA = new int[*nnz];
		*cooColIndA = new int[*nnz];
		copy(E , E+E_size , *cooRowIndA);
		copy(E+E_size, E+2*E_size, *cooColIndA);
		copy(E+2*E_size, E+3*E_size, *cooValA);
	}else if( flag == 'D' ){
		//cout << "debug D" << endl;
		*nnz = E_size;
		*cooValA    = new double[*nnz];
		*cooRowIndA = new int[*nnz];
		*cooColIndA = new int[*nnz];
		copy(E , E+E_size , *cooRowIndA);
		copy(E+E_size, E+2*E_size, *cooColIndA);
		copy(v1.begin(), v1.end(), *cooValA);
	}

	stat = cusparseCreate(&handle);
	assert( stat == CUSPARSE_STATUS_SUCCESS );

	cudaMalloc( &d_cooColIndA, (*nnz)*sizeof(int) );
	cudaMalloc( &d_cooRowIndA, (*nnz)*sizeof(int) );
	cudaMalloc( &d_val, (*nnz)*sizeof(double) );
	cudaMalloc( &d_val_sorted, (*nnz)*sizeof(double) );

	cudaMemcpy(d_cooColIndA, *cooColIndA, (*nnz)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cooRowIndA, *cooRowIndA, (*nnz)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_val, *cooValA, (*nnz)*sizeof(double), cudaMemcpyHostToDevice);

	cusparseXcoosort_bufferSizeExt(handle, *n, *n, *nnz, d_cooRowIndA, d_cooColIndA, &pBufferSizeInBytes);
	cudaMalloc( &pBuffer, sizeof(char)* pBufferSizeInBytes);

	cudaMalloc( (void**)&P, sizeof(int)*(*nnz));
	cusparseCreateIdentityPermutation(handle, *nnz, P);

	cusparseXcoosortByRow(handle, *n, *n, *nnz, d_cooRowIndA, d_cooColIndA, P, pBuffer);

	cusparseDgthr(handle, *nnz, d_val, d_val_sorted, P, CUSPARSE_INDEX_BASE_ZERO);

	cudaMemcpy(*cooRowIndA, d_cooRowIndA, (*nnz)*sizeof(int),  cudaMemcpyDeviceToHost);
	cudaMemcpy(*cooColIndA, d_cooColIndA, (*nnz)*sizeof(int),  cudaMemcpyDeviceToHost);
	cudaMemcpy(*cooValA, d_val_sorted, (*nnz)*sizeof(double),  cudaMemcpyDeviceToHost);

	cudaFree(d_val);
	cudaFree(d_val_sorted);
	cudaFree(d_cooColIndA);
	cudaFree(d_cooRowIndA);
	cudaFree(pBuffer);
	cudaFree(P);
	stat = cusparseDestroy(handle);
	assert( stat == CUSPARSE_STATUS_SUCCESS );

	return 0;
}