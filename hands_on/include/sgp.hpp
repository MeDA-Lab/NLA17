////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    sgp.hpp
/// @brief   The main header for spectral graph partitioning.
///
/// @author  William Liao
/// @author  Yuhsiang Mike Tsai
#ifndef SCSC_SGP_HPP
#define SCSC_SGP_HPP

#include <string>
#include <cassert>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  The enumeration of target 
///
enum class Target {
    LOBPCG   = 0,  ///< solve m smallest eigenvectors
    SIPM     = 1,  ///< use shift inverse power method to solve
    LS       = 2,  ///< solve A+sigmaI linear system
    LANCZOS  = 3,  ///< use inverse Lanczos to solve
    COUNT,         ///< Used for counting number of methods.
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  The enumeration of network format.
///
enum class Network {
  UNDEFINED  = 0,  ///< undefined network format in file
  UNDIRECTED = 1,  ///< undirected networks.
  DIRECTED   = 2,  ///< directed network
  BIPARTITE  = 3,  ///< Bipartite networks
  COUNT,           ///< Used for counting number of methods.
};
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  The enumeration of edge weight
///
enum class Edge {
    UNDEFINED   = 0,  ///< undefined edge weight in file
    UNWEIGHTED  = 1,  ///< unweighted edge
    MULTIPLE    = 2,  ///< multiple edge
    POSITIVE    = 3,  ///< positive weighted edge
    SIGNED      = 4,  ///< signed weighted edge
    MULT_SIGNED = 5,  ///< multiple signed weighted edge
    RATING      = 6,  ///< rating networks
    MULT_RATING = 7,  ///< multiple ratings networks
    DYNAMIC     = 8,  ///< Dynamic network
    COUNT,            ///< Used for counting number of methods.
};
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  The enumeration of eigenvalue problem class.
///
enum class SIPM {
    HOST = 0,    ///< Use host function to calculate EVP
    DEVICE = 1,  ///< Use device function to calculate EVP
    COUNT,       ///< Used for counting number of methods.
  };

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  The enumeration of linear system problem class.
///
enum class LS {
    MAGMA       = 0,  ///< MAGMA Iterative solver
    HOST_QR     = 1,  ///< HOST QR
    HOST_CHOL   = 2,  ///< HOST CHOLESKY
    HOST_LU     = 3,  ///< HOST LU
    DEVICE_QR   = 4,  ///< DEVICE QR
    DEVICE_CHOL = 5,  ///< DEVICE LU
    COUNT,            ///< Used for counting number of methods.
  };

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  The structure wrapper of parameters for Lanczos.
///
typedef struct
{
    double tol;
    int    maxit;
    int    Nwant;
    int    Nstep;
} LSEV_INFO;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  The structure wrapper of all possible input arguments.
///
typedef struct {
    Target target;
    SIPM sipm;
    LS ls;
    std::string solver_settings, file, output, res_filename;
    double sigma, tol;
    int eig_maxiter;
    int res_flag;
    LSEV_INFO LSEV_info;
} args;
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
void readArgs( int argc, char** argv, args *setting);
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
void readGraph(const std::string input, int *E_size_r, int *E_size_c, int **E,
    double **W, Network *network_type, Edge *edge_type);


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
void GraphAdjacency(int E_size, int *E, double *W,
    int *n, int *nnz, double **cooValA, int **cooRowIndA, int **cooColIndA);
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
    const int maxite,
    const double tol,
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
    const int maxite,
    const double tol,
    double *mu,
    double *x
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
    LS ls,
    int m,
    int nnz,
    const double *A_val,
    const int *A_row,
    const int *A_col,
    const double *b,
    double *x,
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
/// @param[in]  solver_settings       MAGMA solver settings.
///
/// @param[in]  res_flag       write residual vector to file if equals 1.
///
/// @param[in]  res_filename   destination file to write the residual vector if res_flag=1.
///
/// @param[out] x       the estimated solution.
///
void solveGraph(
    std::string solver_settings,
    int res_flag,
    std::string res_filename,
    int m,
    int nnz,
    const double *A_val,
    const int *A_row,
    const int *A_col,
    const double *b,
    double *x
);
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  Display informations of the input graph data.
///
/// @param[in]  network       specify the type of graph.
///
/// @param[in]  edge_type     specify the edge property of graph.
///
void printKonectHeader(Network network, Edge edge_type);
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  Solve spectral graph partitioning problem by LOBPCG.
///
void solveSMEVP(
    std::string solver_settings,
    const int m,
    const int nnz,
    const double *A_val,
    const int *A_row,
    const int *A_col,
    int *eig_num,
    double **eig_vals,
    double **eig_vecs
);
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  Write the result to file.
///
void writePartition(
    const int nv,
    const int E_size_r,
    const int *E,
    const int ev_num,
    const double *eig_vals,
    const double *eig_vecs,
    const std::string filename
);
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  Inverse Lanczos eigensolver.
///
int invLanczos_gpu(int            m,
                   int            nnz,
                   double         *csrValA,
                   int            *csrRowIndA,
                   int            *csrColIndA,
                   LSEV_INFO      LSEV_info, 
                   double         *egval,
                   std::string         solver_settings);
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  Internal Linear Solver for Inverse Lanczos eigensolver.
///
void solveGraphLS(
    std::string solver_settings,
    int m,
    int nnz,
    const double *A_val,
    const int *A_row,
    const int *A_col,
    const double *b,
    double *x
)
#endif  // SCSC_SGP_HPP