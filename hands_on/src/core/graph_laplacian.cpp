////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    graph_laplacian.cpp
/// @brief   The implementation of Laplacian construction.
///
/// @author  William Liao
///

#include <harmonic.hpp>
#include <iostream>
#include <cmath>
#include <cassert>
#include <mkl_spblas.h>
#include "sgp.hpp"
using namespace std;

void GraphLaplacian(int *nnz, int *cooRowIndA,
  int *cooColIndA, double *cooValA, int n, int **csrRowIndA,
  int **csrColIndA, double **csrValA, double shift_sigma){
  double *rowsum, *acsr, *dcsr, tmp=0, beta=-1.0, *cooValD;
  int *sumInd, *ja, *ia, *jd, *id, *tmp_RInd, info, k=0, tmp1;
  int *job;
  char trans = 'N';
  int request = 1;
  int sort, nzmax=n*n;

  rowsum = new double[n];
  acsr   = new double[*nnz];
  ja     = new int[*nnz];
  ia     = new int[n+1];
  job    = new int[6];

  // Compute sum of each row of A
  for (int i = 0; i < n; i++)
  {
    rowsum[i] = 0;
  }

  for (int i = 0; i < *nnz; i++)
  {
    if (i>0 && cooRowIndA[i]!=cooRowIndA[i-1])
    {
      tmp1 = cooRowIndA[i]-cooRowIndA[i-1];
      if( tmp1 == 1 ){
          rowsum[k] = tmp+shift_sigma;
          tmp = 0;
          k++;
      }else{
          rowsum[k] = tmp+shift_sigma;
          tmp = 0;
          for(int m = k+1; m < k+tmp1; m++){
              rowsum[m] = shift_sigma;
          }
          k = k + tmp1;
      }
    }
    tmp = tmp + cooValA[i];
    if (i==*nnz-1)
    {
      rowsum[k] = tmp+shift_sigma;
    }
  }

  tmp1 = 0;
  for (int i = 0; i < n; i++)
  {
    if ( rowsum[i] != 0 )
    {
      tmp1++;
    }
  }

  sumInd  = new int[tmp1];
  cooValD = new double[tmp1];
  dcsr    = new double[tmp1];
  jd      = new int[tmp1];
  id      = new int[n+1];

  k = 0;
  for (int i = 0; i < n; i++)
  {
    if ( rowsum[i] != 0 )
    {
      sumInd[k]  = k;
      cooValD[k] = rowsum[i];
      k++;
    }
  }

  //L = D - A
  job[0] = 2;
  job[1] = 1;
  job[2] = 0;
  job[5] = 0;
  mkl_dcsrcoo(job, &n, acsr, ja, ia, nnz, cooValA, cooRowIndA, cooColIndA, &info);
  mkl_dcsrcoo(job, &n, dcsr, jd, id, &tmp1, cooValD, sumInd, sumInd, &info);
  *csrRowIndA = new int[n+1];
  tmp_RInd    = new int[n+1];
  mkl_dcsradd(&trans, &request, &sort, &n, &n, dcsr, jd, id, &beta, acsr, ja, ia, *csrValA, *csrColIndA, tmp_RInd, &nzmax, &info);
  assert( info == 0 );
  request = 2;
  k = tmp_RInd[n]-1;
  //cout << "k = " << k << endl;
  *csrColIndA   = new int[k];
  *csrValA      = new double[k];
  mkl_dcsradd(&trans, &request, &sort, &n, &n, dcsr, jd, id, &beta, acsr, ja, ia, *csrValA, *csrColIndA, tmp_RInd, &nzmax, &info);
  assert( info == 0 );
  copy(tmp_RInd, tmp_RInd+(n+1), *csrRowIndA);
  *nnz = k;

  delete rowsum;
  delete sumInd;
  delete cooValD;
  delete job;
  delete tmp_RInd;
  delete ja;
  delete jd;
  delete ia;
  delete id;
  delete acsr;
  delete dcsr;
}