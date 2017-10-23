////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    solve_SMEVP.cpp
/// @brief   The implementation of lobpcg
///
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

void solveSMEVP(
    const int ev_num,
    const int m,
    const int nnz,
    const double *A_val,
    const int *A_row,
    const int *A_col,
    double *eig_vals,
    double *eig_vecs
) {
  string ev_num_str = to_string(ev_num);
  string magma_settings = "./solver --solver LOBPCG --ev "+ev_num_str+" A.mtx";
  int argc;
  char **argv;
  string2arg(magma_settings, &argc, &argv);
  magma_init();
  magma_queue_t queue;
  magma_queue_create(0, &queue);
  double *dA_val;
  int *dA_row, *dA_col;
  magma_malloc((void**) &dA_val, nnz*sizeof(double));
  magma_malloc((void**) &dA_col, nnz*sizeof(int));
  magma_malloc((void**) &dA_row, (m+1)*sizeof(int));
  magma_setvector(nnz, sizeof(int), A_col, 1, dA_col, 1, queue);
  magma_setvector(m+1, sizeof(int), A_row, 1, dA_row, 1, queue);
  magma_setvector(nnz, sizeof(double), A_val, 1, dA_val, 1, queue);

  magma_d_matrix dA;
  magma_d_matrix dx, drhs;
  magma_dcsrset_gpu(m, m, dA_row, dA_col, dA_val, &dA, queue);
  magma_dvinit(&dx, Magma_DEV, m, 1, 0, queue);
  magma_dvinit(&drhs, Magma_DEV, m, 1, 1, queue);
  magma_dopts dopts;
  int k = 1;
  magmaDoubleComplex *eigenvectors = new magmaDoubleComplex[ev_num*dA.num_rows];
  // Init
  magma_dparse_opts(argc, argv, &dopts, &k, queue);
  magma_dsolverinfo_init(&dopts.solver_par, &dopts.precond_par, queue);
  dopts.solver_par.ev_length = dA.num_cols;
  magma_deigensolverinfo_init(&dopts.solver_par, queue);
  magma_d_precondsetup(dA, drhs,
    &dopts.solver_par, &dopts.precond_par, queue);
  // Solve
  magma_d_solver(dA, drhs, &dx, &dopts, queue);
  // magma_dlobpcg(dA, &dopts.solver_par, &dopts.precond_par, queue);
  // Get Info
  magma_dsolverinfo(&dopts.solver_par, &dopts.precond_par, queue);
  magma_getvector(ev_num * dA.num_rows, sizeof(magmaDoubleComplex),
    dopts.solver_par.eigenvectors, 1, eigenvectors, 1, queue);
  for (int i = 0; i < ev_num; i++) {
      eig_vals[i] = dopts.solver_par.eigenvalues[i];
  }
  for (int i = 0; i < ev_num*dA.num_rows; i++) {
      eig_vecs[i] = MAGMA_Z_REAL(eigenvectors[i]);
  }
  // Free Info
  magma_dsolverinfo_free(&dopts.solver_par, &dopts.precond_par, queue);
  delete [] eigenvectors;
  magma_dmfree(&dA, queue);
  magma_dmfree(&dx, queue);
  magma_dmfree(&drhs, queue);
  magma_finalize();
}