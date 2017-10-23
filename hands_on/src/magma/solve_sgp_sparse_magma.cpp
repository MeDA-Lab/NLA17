////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    solve_sgp_sparse_magma.cpp
/// @brief   The implementation of spectral graph partitioning solving using MAGMA.
///
/// @author  William Liao
/// @author  Yuhsiang Mike Tsai
///

#include <iostream>
#include <cstring>
#include <string>
#include "magma_v2.h"
#include "magmasparse.h"
#include "magma_lapack.h"
#include "tool.hpp"
using namespace std;



void solveGraph(
    string solver_settings,
    int m,
    int nnz,
    const double *A_val,
    const int *A_row,
    const int *A_col,
    const double *b,
    double *x
) {
  string magma_settings = "./solver "+solver_settings+" A.mtx";
  int argc;
  char **argv;
  string2arg(magma_settings, &argc, &argv);
  magma_init();
  magma_queue_t queue;
  magma_queue_create(0, &queue);
  double *dA_val, *db;
  int *dA_row, *dA_col;
  magma_malloc((void**) &dA_val, nnz*sizeof(double));
  magma_malloc((void**) &dA_col, nnz*sizeof(int));
  magma_malloc((void**) &dA_row, (m+1)*sizeof(int));
  magma_malloc((void**) &db, m*sizeof(double));
  magma_setvector(m, sizeof(double), b, 1, db, 1, queue);
  magma_setvector(nnz, sizeof(int), A_col, 1, dA_col, 1, queue);
  magma_setvector(m+1, sizeof(int), A_row, 1, dA_row, 1, queue);
  magma_setvector(nnz, sizeof(double), A_val, 1, dA_val, 1, queue);
  magma_d_matrix dA;
  magma_d_matrix dx, drhs;
  magma_dvinit(&dx, Magma_DEV, m, 1, 0, queue);
  magma_dvset_dev(m, 1, db, &drhs, queue);

  magma_dcsrset_gpu(m, m, dA_row, dA_col, dA_val, &dA, queue);

  magma_dopts dopts;
  int k = 1;

  // Init
  magma_dparse_opts(argc, argv, &dopts, &k, queue);
  magma_dsolverinfo_init(&dopts.solver_par, &dopts.precond_par, queue);
  magma_d_precondsetup(dA, drhs,
    &dopts.solver_par, &dopts.precond_par, queue);
  // Solve
  magma_d_solver(dA, drhs, &dx, &dopts, queue);
  // Get Info
  magma_dsolverinfo(&dopts.solver_par, &dopts.precond_par, queue);
  magma_getvector(m, sizeof(double), dx.dval, 1, x, 1, queue);
  // Free Info
  magma_dsolverinfo_free(&dopts.solver_par, &dopts.precond_par, queue);
  magma_dmfree(&dA, queue);
  magma_dmfree(&dx, queue);
  magma_dmfree(&drhs, queue);
  magma_finalize();
}