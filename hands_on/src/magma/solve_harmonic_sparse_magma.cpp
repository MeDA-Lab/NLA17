////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    solve_harmonic_sparse_magma.cpp
/// @brief   The implementation of harmonic problem solving using MAGMA.
///
/// @author  Yuhsiang Mike Tsai
///

#include <harmonic.hpp>
#include <iostream>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "magma_v2.h"
#include "magmasparse.h"
#include "magma_lapack.h"
#include "tool.hpp"
using namespace std;

void writeRes(magma_d_solver_par *solver_par, string res_filename){
  int i;
  FILE *resvec;
  resvec=fopen(res_filename.c_str(),"w");
  if ( solver_par->verbose > 0)
  {
    int m = solver_par->verbose;
    for (i = 0; i < (solver_par->numiter)/m+1; i++)
    {
      fprintf(resvec, "%d %.16e\n", i*m, solver_par->res_vec[i]);
    }
  }
  fclose(resvec);
}

void solveHarmonicSparse(
  string solver_settings,
  int res_flag,
  string res_filename,
  const int nv,
  const int nb,
  const double *Lii_val,
  const int *Lii_row,
  const int *Lii_col,
  const double *Lib_val,
  const int *Lib_row,
  const int *Lib_col,
  double *U
) {
  string magma_settings = "./solver "+solver_settings+" A.mtx";
  int argc = 0;
  char **argv;
  string2arg(magma_settings, &argc, &argv);
  magma_init();
  magma_queue_t queue;
  magma_queue_create(0, &queue);
  int ni = nv-nb;
  double *dLii_val, *dLib_val;
  int *dLii_row, *dLii_col, *dLib_row, *dLib_col;
  magma_malloc((void**) &dLii_val, Lii_row[ni]*sizeof(double));
  magma_malloc((void**) &dLii_col, Lii_row[ni]*sizeof(int));
  magma_malloc((void**) &dLii_row, (ni+1)*sizeof(int));
  magma_malloc((void**) &dLib_val, Lib_row[ni]*sizeof(double));
  magma_malloc((void**) &dLib_col, Lib_row[ni]*sizeof(int));
  magma_malloc((void**) &dLib_row, (ni+1)*sizeof(int));
  magma_setvector(Lii_row[ni], sizeof(double), Lii_val, 1, dLii_val, 1, queue);
  magma_setvector(Lii_row[ni], sizeof(int), Lii_col, 1, dLii_col, 1, queue);
  magma_setvector(ni+1, sizeof(int), Lii_row, 1, dLii_row, 1, queue);
  magma_setvector(Lib_row[ni], sizeof(double), Lib_val, 1, dLib_val, 1, queue);
  magma_setvector(Lib_row[ni], sizeof(int), Lib_col, 1, dLib_col, 1, queue);
  magma_setvector(ni+1, sizeof(int), Lib_row, 1, dLib_row, 1, queue);
  magma_d_matrix dLii, dLib;
  magma_d_matrix dx, du, drhs;
  magma_dvinit(&du, Magma_DEV, nb, 1, 0, queue);
  magma_dvinit(&dx, Magma_DEV, ni, 1, 0, queue);
  magma_dvinit(&drhs, Magma_DEV, ni, 1, 0, queue);

  magma_dcsrset_gpu(ni, nb, dLib_row, dLib_col, dLib_val, &dLib, queue);
  magma_dcsrset_gpu(ni, ni, dLii_row, dLii_col, dLii_val, &dLii, queue);

  magma_dopts dopts;
  int k = 1;
  // Solver Settings
  // argc : length of argv
  // argv : {"first item", ..., "last item"}.
  //        First item and last item are unused.
  for (int i = 0; i < 2; i++) {
    magma_setvector(nb, sizeof(double), U+i*nv, 1, du.dval, 1, queue);
    magma_d_spmv(-1, dLib, du, 0, drhs, queue);
    // Init
    magma_dparse_opts(argc, argv, &dopts, &k, queue);
    magma_dsolverinfo_init(&dopts.solver_par, &dopts.precond_par, queue);
    magma_d_precondsetup(dLii, drhs,
      &dopts.solver_par, &dopts.precond_par, queue);
    // Solve
    magma_d_solver(dLii, drhs, &dx, &dopts, queue);
    // Get Info
    magma_dsolverinfo(&dopts.solver_par, &dopts.precond_par, queue);
    magma_getvector(ni, sizeof(double), dx.dval, 1, U+i*nv+nb, 1, queue);
    if ( dopts.solver_par.verbose>0 && res_flag == 1 )
    {
      writeRes(&dopts.solver_par, res_filename);
    }
    // Free Info
    magma_dsolverinfo_free(&dopts.solver_par, &dopts.precond_par, queue);
  }
  magma_dmfree(&dLii, queue);
  magma_dmfree(&dLib, queue);
  magma_dmfree(&dx, queue);
  magma_dmfree(&du, queue);
  magma_dmfree(&drhs, queue);
  magma_finalize();
}

void solveGraphLS(
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
  //magma_dsolverinfo(&dopts.solver_par, &dopts.precond_par, queue);
  magma_getvector(m, sizeof(double), dx.dval, 1, x, 1, queue);
  
  // Free Info
  magma_dsolverinfo_free(&dopts.solver_par, &dopts.precond_par, queue);
  magma_dmfree(&dA, queue);
  magma_dmfree(&dx, queue);
  magma_dmfree(&drhs, queue);
  magma_finalize();
}