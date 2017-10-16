////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    solve_sgp_sparse_magma.cpp
/// @brief   The implementation of spectral graph partitioning solving using MAGMA.
///
/// @author  William Liao
///

#include <iostream>
#include "magma_v2.h"
#include "magmasparse.h"
#include "magma_lapack.h"
using namespace std;

magma_int_t
magma_dcsrset_gpu(
    magma_int_t m,
    magma_int_t n,
    magmaIndex_ptr row,
    magmaIndex_ptr col,
    magmaDouble_ptr val,
    magma_d_matrix *A,
    magma_queue_t queue )
{
    A->num_rows = m;
    A->num_cols = n;
    magma_index_t nnz;
    magma_index_getvector( 1, row+m, 1, &nnz, 1, queue );
    A->nnz = (magma_int_t) nnz;
    A->storage_type = Magma_CSR;
    A->memory_location = Magma_DEV;
    A->dval = val;
    A->dcol = col;
    A->drow = row;

    return MAGMA_SUCCESS;
}

void solveGraph(
    int m,
    int nnz,
    const double *A_val,
    const int *A_row,
    const int *A_col,
    const double *b,
    double *x
) {
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
  // Solver Settings
  // argc : length of argv
  // argv : {"first item", ..., "last item"}.
  //        First item and last item are unused.
  int argc = 4;
  char *argv[]={"./solver", "--solver", "CG", "A.mtx"};
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

void solveGraphCust(
    int m,
    int nnz,
    const double *A_val,
    const int *A_row,
    const int *A_col,
    const double *b,
    double *x,
    const char *solver, 
    std::string atol,
    std::string rtol, 
    std::string maxiter,
    std::string precond,
    std::string restart
) {
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
  // Solver Settings
  // argc : length of argv
  // argv : {"first item", ..., "last item"}.
  //        First item and last item are unused.
  int argc = 14;
  const string str[] = {"./solver", "--solver", solver, "--atol", atol, "--rtol", rtol, "--maxiter", maxiter, "--precond", precond, "restart", restart, "A.mtx"};
  char **argv = new char *[argc];
  int i, len;
  for(i=0; i<argc; i++){
    len = str[i].length();
    argv[i] = new char[len];
    strcpy(argv[i], str[i].c_str());
  }

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