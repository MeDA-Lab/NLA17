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
/// @brief  The enumeration of Laplacian construction methods.
///
enum class Method {
  SIMPLE = 0,    ///< Simple graph.
  DIRECTED = 1,  ///< Directed (multi) graph
  WEIGHTED = 2,  ///< Directed weighted graph
  UW = 3,        ///< Undirected weighted graph
  COUNT,         ///< Used for counting number of methods.
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  The enumeration of eigenvalue problem class.
///
enum class EVP {
    NONE = 0,   ///< Do not calculate EVP
    HOST = 1,   ///< Use host function to calculate EVP
    DEVICE = 2, ///< Use device function to calculate EVP
    COUNT,      ///< Used for counting number of methods.
  };

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  The enumeration of linear system problem class.
///
enum class LS {
    NONE = 0,      ///< Do not calculate LS
    HOST = 1,      ///< Use host direct solver to calculate LS
    DEVICE = 2,    ///< Use device direct solver to calculate LS
    ITERATIVE = 3, ///< Use iterative solver to calculate LS
    COUNT,         ///< Used for counting number of methods.
  };

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  The enumeration of linear solver type.
///
enum class LSOLVER {
    LU = 0,        ///< LU factorization
    CHOL = 1,      ///< Cholesky factorization
    QR = 2,        ///< QR factorization
    ITERATIVE = 3, ///< Iterative solver
    COUNT,         ///< Used for counting number of methods.
  };

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  Reads the arguments.
///
/// @param[in]   argc    The number of input arguments.
/// @param[in]   argv    The input arguments.
///
/// @param[out]  input   The input file.
/// @param[out]  para    The parameter setting file.
/// @param[out]  method  The method (type of graph).
/// @param[out]  evp     Eigenvalue problem indicator.
/// @param[out]  ls      Linear system problem indicator.
/// @param[out]  tflag   Method flag.
/// @param[out]  pflag   Parameter setting file flag.
///
void readArgs( int argc, char** argv, const char *&input, const char *&para, Method &method, EVP &evp, LS &ls, int &tflag,
  int &pflag);
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
int readGraph(const char *input, int **E, int *E_size_r, int *E_size_c);
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  Reads the parameter setting file for direct linear solver and eigensolver.
///
/// @param[in]      parafile    the path to the setting file.
///
/// @param[in/out]  shift_sigma shift for Laplacian matrix.
///
/// @param[in/out]  mu0         initial guess of eigenvalue.
/// @param[in/out]  eigtol      tolerance for eigensolver.
/// @param[in/out]  eigmaxite   max iteration number for eigensolver.
/// @param[in/out]  solflag     linear solver flag.
/// @param[in/out]  solver      type of linear solver.
/// @param[in/out]  tol         tolerance for direct linear solver.
/// @note  The arrays are allocated by this routine (using new).
///
void readParaDEVP(const char *parafile,
    double &shift_sigma,
    double &mu0,
    double &eigtol,
    int &eigmaxite,
    LSOLVER &solflag,
    const char *&solver,
    double &tol);
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  Reads the parameter setting file for iterative linear solver and eigensolver.
///
/// @param[in]      parafile    the path to the setting file.
///
/// @param[in/out]  shift_sigma shift for Laplacian matrix.
///
/// @param[in/out]  mu0         initial guess of eigenvalue.
/// @param[in/out]  eigtol      tolerance for eigensolver.
/// @param[in/out]  eigmaxite   max iteration number for eigensolver.
/// @param[in/out]  solflag     linear solver flag.
/// @param[in/out]  solver      type of linear solver.
/// @param[in/out]  atol        absolute residual.
/// @param[in/out]  rtol        relative residual.
/// @param[in/out]  maxiter     max iteration number for iterative linear solver.
/// @param[in/out]  precond     type of preconditioner.
/// @param[in/out]  restart     Only take effects for GMRES and IDR.
/// @note  The arrays are allocated by this routine (using new).
///
void readParaIEVP(const char *parafile,
    double &shift_sigma,
    double &mu0,
    double &eigtol,
    int &eigmaxite,
    LSOLVER &solflag,
    const char *&solver,
    std::string &atol,
    std::string &rtol,
    std::string &maxiter,
    std::string &precond,
    std::string &restart);
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  Reads the parameter setting file for iterative linear solver.
///
/// @param[in]      parafile    the path to the setting file.
///
/// @param[in/out]  solflag     linear solver flag.
/// @param[in/out]  solver      type of linear solver.
/// @param[in/out]  atol        absolute residual.
/// @param[in/out]  rtol        relative residual.
/// @param[in/out]  maxiter     max iteration number for iterative linear solver.
/// @param[in/out]  precond     type of preconditioner.
/// @param[in/out]  restart     Only take effects for GMRES and IDR.
/// @note  The arrays are allocated by this routine (using new).
///
void readParaILS(const char *parafile,
    double &shift_sigma,
    LSOLVER &solflag,
    const char *&solver,
    std::string &atol,
    std::string &rtol,
    std::string &maxiter,
    std::string &precond,
    std::string &restart);
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  Reads the parameter setting file for direct linear solver.
///
/// @param[in]      parafile    the path to the setting file.
///
/// @param[in/out]  shift_sigma shift for Laplacian matrix.
///
/// @param[in/out]  solflag     linear solver flag.
/// @param[in/out]  solver      type of linear solver.
/// @param[in/out]  tol         tolerance for direct linear solver.
/// @note  The arrays are allocated by this routine (using new).
///
void readParaDLS(const char *parafile,
    double &shift_sigma,
    LSOLVER &solflag,
    const char *&solver,
    double &tol);
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  Reads the parameter setting file for eigensolver.
///
/// @param[in]      parafile    the path to the setting file.
///
/// @param[in/out]  shift_sigma shift for Laplacian matrix.
///
/// @param[in/out]  mu0         initial guess of eigenvalue.
/// @param[in/out]  eigtol      tolerance for eigensolver.
/// @param[in/out]  eigmaxite   max iteration number for eigensolver.
/// @note  The arrays are allocated by this routine (using new).
///
void readParaEVP(const char *parafile,
    double &shift_sigma,
    double &mu0,
    double &eigtol,
    int &eigmaxite);
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
/// @param[in]   E_size_c   number of data pair in edge lists.
///
/// @param[out]  method     graph type.
///
void setgraphtype(Method &method, int E_size_c);
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  Sets the graph type.
///
/// @param[in]   input      user defined type id from the command line.
///
/// @param[in]   E_size_c   number of data pair in edge lists.
///
/// @param[out]  type       graph type.
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
/// @param[in]  m        size of the matrix.
///
/// @param[out] mu       estimated eigenvalue.
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
/// @param[in]  m        size of the matrix.
///
/// @param[in]  tol      tolerance.
///
/// @param[in]  maxite   upper limit for the iteration count.
///
/// @param[out] mu       estimated eigenvalue.
///
/// @param[out] x        estimated eigenvector w.r.t. mu; pointer.
///
/// @note  All inputs should be stored on host.
///
void solveShiftEVPHostCust(
    int m,
    int nnz,
    const double *A_val,
    const int *A_row,
    const int *A_col,
    const double mu0,
    double *mu,
    double *x,
    double tol, 
    int maxite
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
/// @param[in]  m        size of the matrix.
///
/// @param[out] mu       estimated eigenvalue.
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
/// @param[in]  m        size of the matrix.
///
/// @param[in]  tol      tolerance.
///
/// @param[in]  maxite   upper limit for the iteration count.
///
/// @param[out] mu       estimated eigenvalue.
///
/// @param[out] x        estimated eigenvector w.r.t. mu; pointer.
///
/// @note  All inputs should be stored on host.
///
void solveShiftEVPCust(
    int m,
    int nnz,
    const double *A_val,
    const int *A_row,
    const int *A_col,
    const double mu0,
    double *mu,
    double *x,
    double tol, 
    int maxite
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
/// @brief  Solve linear system Ax = b with requested direct solver on host.
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
/// @param[out] tol      tolerance.
///
/// @note  All inputs should be stored on host.
///
void solvelsHostCust(
    int m,
    int nnz,
    const double *A_val,
    const int *A_row,
    const int *A_col,
    const double *b,
    double *x,
    int solver,
    double tol
);
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  Solve linear system Ax = b with requested direct solver on device.
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
/// @param[out] x        estimated solution; pointer.
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
/// @brief  Solve linear system Ax = b with requested direct solver on device.
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
/// @param[out] x        estimated solution; pointer.
///
/// @param[out] tol      tolerance.
///
/// @note  All inputs should be stored on host.
///
void solvelsCust(
    int m,
    int nnz,
    const double *A_val,
    const int *A_row,
    const int *A_col,
    const double *b,
    double *x,
    int solver,
    double tol
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
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  Print CUDA linear system solver info.
///
/// @param[in]  flag    indicates solver on CPU or GPU.
///
/// @param[in]  solver  indicates type of solver.
///
int cudasolverinfo(int flag, int solver);
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  MAGMA iterative linear system solver wrapper.
///
/// @param[in]  m       size of the Laplacian matrix.
///
/// @param[in]  nnz     number of nonzeros of the Laplacian matrix.
///
/// @param[in]  A_val   value of the Laplacian matrix.
///
/// @param[in]  A_row   row pointer of the Laplacian matrix.
///
/// @param[in]  A_col   column index of the Laplacian matrix.
///
/// @param[in]  b       the RHS of AX = b.
///
/// @param[out] x       the estimated solution.
///
void solveGraph(
    int m,
    int nnz,
    const double *A_val,
    const int *A_row,
    const int *A_col,
    const double *b,
    double *x
);
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  MAGMA iterative linear system solver wrapper.
///
/// @param[in]  m       size of the Laplacian matrix.
///
/// @param[in]  nnz     number of nonzeros of the Laplacian matrix.
///
/// @param[in]  A_val   value of the Laplacian matrix.
///
/// @param[in]  A_row   row pointer of the Laplacian matrix.
///
/// @param[in]  A_col   column index of the Laplacian matrix.
///
/// @param[in]  b       the RHS of AX = b.
///
/// @param[in]  solver  type of iterative solver.
///
/// @param[in]  atol    absolute residual.
///
/// @param[in]  rtol    relative residual.
///
/// @param[in]  maxiter upper limit for the iteration count.
///
/// @param[in]  precond type of preconditioner.
///
/// @param[in]  restart Only take effects for GMRES and IDR.
///
/// @param[out] x       the estimated solution.
///
void solveGraphCust(
    int m,
    int nnz,
    const double *A_val,
    const int *A_row,
    const int *A_col,
    const double *b,
    double *x,
    const char *&solver, 
    std::string atol,
    std::string rtol, 
    std::string maxiter,
    std::string precond,
    std::string restart
);
#endif  // SCSC_SGP_HPP