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
#include "mkl.h"

using namespace std;

int GraphAdjacency(int *E, int E_size,
	int *nnz, int **cooRowIndA,
	int **cooColIndA, double **cooValA, int *n, char flag){
	int pos1, pos2, info;
	int *d_cooRowIndA, *d_cooColIndA;
	double  *d_val, *d_val_sorted;
	double *tmp_array, beta = 1.0;
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
		vector<double> v1 (2*E_size , 1.0);

		*cooRowIndA = new int[2*E_size];
		*cooColIndA = new int[2*E_size];
		*cooValA    = new double[2*E_size];

		// A+trans(A)
		*nnz = 2*E_size;
		copy(E , E+2*E_size , *cooRowIndA);
		copy(E+E_size, E+2*E_size, *cooColIndA);
		copy(E, E+E_size, (*cooColIndA)+E_size);
		copy(v1.begin(), v1.end(), *cooValA);
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
		vector<double> v1 (E_size , 1.0);
		//cout << "debug D" << endl;
		*nnz = E_size;
		*cooValA    = new double[*nnz];
		*cooRowIndA = new int[*nnz];
		*cooColIndA = new int[*nnz];
		copy(E , E+E_size , *cooRowIndA);
		copy(E+E_size, E+2*E_size, *cooColIndA);
		copy(v1.begin(), v1.end(), *cooValA);
	}else if( flag == 'U' ){
		*cooRowIndA = new int[2*E_size];
		*cooColIndA = new int[2*E_size];
		*cooValA    = new double[2*E_size];

		// A+trans(A)
		*nnz = 2*E_size;
		copy(E , E+2*E_size , *cooRowIndA);
		copy(E+E_size, E+2*E_size, *cooColIndA);
		copy(E, E+E_size, (*cooColIndA)+E_size);
		copy(E+2*E_size, E+3*E_size, *cooValA);
		copy(E+2*E_size, E+3*E_size, (*cooValA)+E_size);
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