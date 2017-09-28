////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    main_3Dface_evp.cpp
/// @brief   The main function. (sparse version)
///
/// @author  Mu Yang <<emfomy@gmail.com>>
/// @author  William Liao
///

#include <iostream>
#include <harmonic.hpp>
#include <timer.hpp>
#include "mkl.h"
using namespace std;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  Main function
///
int main( int argc, char** argv ) {

  const char *input  = "input.obj";
  const char *output = "output.obj";
  Method method  = Method::KIRCHHOFF;

  int nv, nf, nb, *F = nullptr, *idx_b, *Lii_row = nullptr, *Lii_col = nullptr, *Lib_row = nullptr, *Lib_col = nullptr;
  double timer, *V = nullptr, *C = nullptr, *Lii_val = nullptr, *Lib_val = nullptr, *U;


  // Read arguments
  readArgs(argc, argv, input, output, method);

  // Read object
  readObject(input, &nv, &nf, &V, &C, &F);

  cout << endl;

  // Verify boundary
  idx_b = new int[nv];
  cout << "Verifying boundary ....................." << flush;
  tic(&timer);
  verifyBoundarySparse(nv, nf, F, &nb, idx_b); cout << " Done.  ";
  toc(&timer);

  // Reorder vertices
  cout << "Reordering vertices ...................." << flush;
  tic(&timer);
  reorderVertex(nv, nb, nf, V, C, F, idx_b); cout << " Done.  ";
  toc(&timer);

  // Construct Laplacian
  cout << "Constructing Laplacian ................." << flush;
  tic(&timer);
  constructLaplacianSparse(method, nv, nb, nf, V, F, &Lii_val, &Lii_row, &Lii_col, &Lib_val, &Lib_row, &Lib_col);
  cout << " Done.  ";
  toc(&timer);

  // Map boundary
  U = new double[2 * nv];
  cout << "Mapping Boundary ......................." << flush;
  tic(&timer);
  mapBoundary(nv, nb, V, U); cout << " Done.  ";
  toc(&timer);

  // Generate RHS
  double *b;
  int nnz = Lii_row[nv-nb];
  b = new double[nv-nb];
  genRHS(b, nv-nb, nnz, Lib_val, Lib_row, Lib_col);
  cblas_dscal(nv-nb, -1.0, b, 1);

  // Solve LS
  cout << "Solving Linear System .................." << flush;
  double *x;
  x = new double[nv-nb];
  char flag = 'H';     // Modify flag to choose solver on GPU
                       // or CPU. Possible options are
                       // 'H': solver on host    (CPU)
                       // 'D': solver on device  (GPU)
  
  int solver = 0;      // Modify solver to switch between
                       // different linear solvers. Possible
                       // options are
                       // 0: LU
                       // 1: Cholesky
                       // 2: QR

  switch (flag){
    case 'H':
      tic(&timer);
      solvelsHost(nv-nb, nnz, Lii_val, Lii_row, Lii_col, b, x, solver); cout << " Done.  ";
      toc(&timer);
      break;
    case 'D':
      tic(&timer);
      solvels(nv-nb, nnz, Lii_val, Lii_row, Lii_col, b, x, solver);
      cout << " Done.  ";
      toc(&timer);
      break;
  }
  cout << "n = " << nv-nb << endl;
  cout << "nnz = " << nnz << endl;
  cout << endl;

  // Compute redsidual
  double res;
  res = residual(nv-nb, nnz, Lii_val, Lii_row, Lii_col, b, x);

  cout << "||Ax - b|| =  "  << res << endl;

  cout << endl;

  // Free memory
  delete[] V;
  delete[] C;
  delete[] F;
  delete[] Lii_val;
  delete[] Lii_row;
  delete[] Lii_col;
  delete[] Lib_val;
  delete[] Lib_row;
  delete[] Lib_col;
  delete[] U;
  delete[] idx_b;

  return 0;
}
