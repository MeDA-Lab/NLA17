////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    solve_harmonic_sparse_magma.cpp
/// @brief   The implementation of harmonic problem solving using MAGMA.
///
/// @author  Yuhsiang Mike Tsai
///

#include <harmonic.hpp>
#include <iostream>
#include "magma_v2.h"
#include "magmasparse.h"
#include "magma_lapack.h"
using namespace std;

void solveHarmonic(
    const int nv,
    const int nb,
    double *L,
    double *U
) {
  // Liiui=Libub
  magma_init();
  magma_queue_t queue;
  magma_queue_create(0, &queue);
  double *dL = NULL, *dU = NULL;
  magma_malloc((void **)&dL, nv*nv*sizeof(double));
  magma_malloc((void **)&dU, nv*2*sizeof(double));
  magma_setvector(nv*nv, sizeof(double), L, 1, dL, 1, queue);
  magma_setvector(nv*2, sizeof(double), U, 1, dU, 1, queue);
  magmablas_dgemm(MagmaNoTrans, MagmaNoTrans, nv-nb, 2, nb, -1, dL+nb, nv, dU, nv, 0, dU+nb, nv, queue);
  //
  int *ipiv=new int [nv-nb], info = 0;
  magma_dgesv_gpu(nv-nb, 2, dL+nv*nb+nb, nv, ipiv, dU+nb, nv, &info);
  if (info != 0){
    cerr<<info<<" Magma Solve Error\n";
  }
  magma_getvector(nv*2, sizeof(double), dU, 1, U, 1, queue);
  magma_free(dL);
  magma_free(dU);
  magma_finalize();
}
