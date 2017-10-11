////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    sgp.hpp
/// @brief   The main header for spectral graph partitioning.
///
/// @author  William Liao
///

#ifndef SCSC_SGP_HPP
#define SCSC_SGP_HPP

#include <cassert>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  Reads the graph file.
///
/// @param[in]   input   the path to the object file.
///
/// @param[out]  E_size_r      number of data in each row of the edge list; pointer.
///
/// @param[out]   E_size_c  number of data pair in edge lists.
/// @note  The arrays are allocated by this routine (using new).
///
int readGraph(char *input, int **E, int *E_size_r, int *E_size_c);
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  Sets the graph type.
///
/// @param[in]   E_size_c   number of data pair in edge lists.
///
/// @param[out]  type  number of data pair in edge lists.
///
int setgraphtype(int E_size_c);
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  Sets the graph type.
///
/// @param[in]   input      user defined type id from the command line.
///
/// @param[in]   E_size_c   number of data pair in edge lists.
///
/// @param[out]  type  number of data pair in edge lists.
///
int setgraphtype(char *input, int E_size_c);
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  Construct adjacency matrix of graph.
///
/// @param[in]   E       the edge list; pointer.
///
/// @param[in]   E_size  number of data pair in edge lists.
///
/// @param[out]  nnz     number of nonzero elements in the matrix.
///
/// @param[out]  csrRowPtrA     CSR row pointer; pointer.
///
/// @param[out]  csrColIndA     CSR column index; pointer.
///
/// @param[out]  csrValA  nonzero values of the matrix; pointer.
///
/// @param[out]  n        size of the matrix;
///
/// @note  The output arrays are allocated by this routine (using new).
///
int GraphAdjacency(int *E, int E_size,
	int *nnz, int **cooRowIndA,
	int **cooColIndA, double **cooValA, int *n, char flag);
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  Construct adjacency matrix of graph.
///
/// @param[in/out]  nnz     number of nonzero elements in the matrix.
///
/// @param[in/out]  csrRowPtrA     CSR row pointer; pointer.
///
/// @param[in/out]  csrColIndA     CSR column index; pointer.
///
/// @param[in/out]  csrValA  nonzero values of the matrix; pointer.
///
/// @param[in]  n        size of the matrix;
///
/// @param[in]  flag     graph type indicator;
///
/// @note  The output arrays are allocated by this routine (using new).
///
void GraphLaplacian(int *nnz, int *cooRowPtrA,
  int *cooColIndA, double *cooValA, int n, int **csrRowIndA,
  int **csrColIndA, double **csrValA, double shift_sigma);
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  Solve eigenvalue near mu0 on host.
///
/// @param[in]  mu0     initial guess of eigenvalue.
///
/// @param[in]  nnz     number of nonzero elements in the matrix.
///
/// @param[in/out]  A_row     CSR row pointer; pointer.
///
/// @param[in/out]  A_col     CSR column index; pointer.
///
/// @param[in/out]  A_val  nonzero values of the matrix; pointer.
///
/// @param[in]  m        size of the matrix;
///
/// @param[out] mu       estimated eigenvalue;
///
/// @param[out] x        estimated eigenvector w.r.t. mu; pointer.
///
/// @note  All inputs should be stored on host.
///
void solveShiftEVPHost(
    int m,
    int nnz,
    const double *A_val,
    const int *A_row,
    const int *A_col,
    const double mu0,
    double *mu,
    double *x
);
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  Solve eigenvalue near mu0 on device.
///
/// @param[in]  mu0     initial guess of eigenvalue.
///
/// @param[in]  nnz     number of nonzero elements in the matrix.
///
/// @param[in/out]  A_row     CSR row pointer; pointer.
///
/// @param[in/out]  A_col     CSR column index; pointer.
///
/// @param[in/out]  A_val  nonzero values of the matrix; pointer.
///
/// @param[in]  m        size of the matrix;
///
/// @param[out] mu       estimated eigenvalue;
///
/// @param[out] x        estimated eigenvector w.r.t. mu; pointer.
///
/// @note  All inputs should be stored on host.
///
void solveShiftEVP(
    int m,
    int nnz,
    const double *A_val,
    const int *A_row,
    const int *A_col,
    const double mu0,
    double *mu,
    double *x
);
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  Solve linear system Ax = b with requested solver on host.
///
/// @param[in]  solver    type of solver; Possible options are; 0:LU; 1:Cholesky; 2:QR
///
/// @param[in]  nnz     number of nonzero elements in the matrix.
///
/// @param[in/out]  A_row     CSR row pointer; pointer.
///
/// @param[in/out]  A_col     CSR column index; pointer.
///
/// @param[in/out]  A_val  nonzero values of the matrix; pointer.
///
/// @param[in]  m        size of the matrix.
///
/// @param[in]  b        RHS of the linear system; pointer.
///
/// @param[out] x        estimated solution; pointer.
///
/// @note  All inputs should be stored on host.
///
void solvelsHost(
    int m,
    int nnz,
    const double *A_val,
    const int *A_row,
    const int *A_col,
    const double *b,
    double *x,
    int solver
);
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  Solve linear system Ax = b with requested solver on device.
///
/// @param[in]  solver    type of solver; Possible options are; 1:Cholesky; 2:QR
///
/// @param[in]  nnz     number of nonzero elements in the matrix.
///
/// @param[in/out]  A_row     CSR row pointer; pointer.
///
/// @param[in/out]  A_col     CSR column index; pointer.
///
/// @param[in/out]  A_val  nonzero values of the matrix; pointer.
///
/// @param[in]  m        size of the matrix.
///
/// @param[in]  b        RHS of the linear system; pointer.
///
/// @param[out] x        estimated eigenvector w.r.t. mu; pointer.
///
/// @note  All inputs should be stored on host.
///
void solvels(
    int m,
    int nnz,
    const double *A_val,
    const int *A_row,
    const int *A_col,
    const double *b,
    double *x,
    int solver
);
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  Generate RHS b of the linear system Ax = b.
///
/// @param[in/out]  b         RHS of the linear system; pointer.
///
/// @param[in]  n       size of the matrix;
///
/// @param[in]  nnz     number of nonzero elements in the matrix.
///
/// @param[in/out]  A_row     CSR row pointer; pointer.
///
/// @param[in/out]  A_col     CSR column index; pointer.
///
/// @param[in/out]  A_val  nonzero values of the matrix; pointer.
///
/// @note  All inputs should be stored on host.
///
void genRHS(double *b, int n, int nnz, double *A_val, int *A_row, int *A_col);
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  Compute the error ||Ax - b||.
///
/// @param[in]  n       size of the matrix.
///
/// @param[in]  nnz     number of nonzero elements in the matrix.
///
/// @param[in]  A_row   CSR row pointer; pointer.
///
/// @param[in]  A_col   CSR column index; pointer.
///
/// @param[in]  A_val   nonzero values of the matrix; pointer.
///
/// @param[in]  b       RHS of the linear system. at the end of the routine it is overwritten; pointer.
///
/// @param[in]  x       estimated solution to the linear system; pointer.
///
/// @note  All inputs should be stored on host.
///
double residual(int n, int nnz, double *A_val, int *A_row, int *A_col, double *b, double *x);
#endif  // SCSC_SGP_HPP