////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    surfpara.cpp
/// @brief   The main function. (sparse version)
///
/// @author  Mu Yang <<emfomy@gmail.com>>
/// @author  William Liao
/// @author  Yuhsiang Mike Tsai
///

#include <iostream>
#include <iomanip>
#include <harmonic.hpp>
#include <timer.hpp>
using namespace std;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  Main function
///
int main( int argc, char** argv ) {
  args setting;
  setting.target = Target::LS;
  setting.sipm = SIPM::HOST;
  setting.ls = LS::MAGMA;
  setting.solver_settings = "--solver CG";
  setting.method = Method::KIRCHHOFF;
  setting.file = "input.obj";
  setting.output = "output.obj";
  setting.sigma = 1.5;
  setting.tol = 1e-12;
  setting.eig_maxiter = 1000;
  int nv, nf, nb, *F = nullptr, *idx_b = nullptr, *Lii_row = nullptr,
    *Lii_col = nullptr, *Lib_row = nullptr, *Lib_col = nullptr;
  double timer, *V = nullptr, *C = nullptr, *Lii_val = nullptr,
    *Lib_val = nullptr, *U = nullptr;


  // Read arguments
  readArgs(argc, argv, &setting);

  // Read object
  readObject(setting.file, &nv, &nf, &V, &C, &F);

  cout << endl;

  // Verify boundary
  idx_b = new int[nv];
  cout << "Verifying boundary ....................." << flush;
  tic(&timer);
  verifyBoundarySparse(nv, nf, F, &nb, idx_b);
  cout << " Done.  ";
  toc(&timer);

  // Reorder vertices
  cout << "Reordering vertices ...................." << flush;
  tic(&timer);
  reorderVertex(nv, nb, nf, V, C, F, idx_b);
  cout << " Done.  ";
  toc(&timer);

  // Construct Laplacian
  cout << "Constructing Laplacian ................." << flush;
  tic(&timer);
  constructLaplacianSparse(setting.method, nv, nb, nf, V, F,
    &Lii_val, &Lii_row, &Lii_col,
    &Lib_val, &Lib_row, &Lib_col);
  cout << " Done.  ";
  toc(&timer);

  // Map boundary
  U = new double[2 * nv];
  cout << "Mapping Boundary ......................." << flush;
  tic(&timer);
  mapBoundary(nv, nb, V, U);
  cout << " Done.  ";
  toc(&timer);
  switch (setting.target) {
    case Target::LS : {
      // Solve harmonic
      cout << "Solving Harmonic ......................." << flush;
      tic(&timer);
      solveHarmonicSparse(setting.solver_settings,
        nv, nb, Lii_val, Lii_row, Lii_col,
        Lib_val, Lib_row, Lib_col, U);
        cout << " Done.  ";
      toc(&timer);
      cout << endl;
      // Write object
      writeObject(setting.output, nv, nf, U, C, F);
      break;
    }
    case Target::SIPM : {
      cout << "Solving Eigenvalue Problem ............." << flush;
      double mu = 0;
      double *x;
      x = new double[nv-nb];
      int nnz = Lii_row[nv-nb];
      tic(&timer);
      switch (setting.sipm) {
        case SIPM::HOST :
          solveShiftEVPHost(nv-nb, nnz, Lii_val, Lii_row, Lii_col,
            setting.sigma, setting.eig_maxiter, setting.tol, &mu, x);
          break;
        case SIPM::DEVICE :
          solveShiftEVP(nv-nb, nnz, Lii_val, Lii_row, Lii_col,
            setting.sigma, setting.eig_maxiter, setting.tol, &mu, x);
          break;
      }
      cout << " Done.  ";
      toc(&timer);

      cout << endl;
      cout << "n = " << nv-nb << endl;
      cout << "nnz = " << nnz << endl;
      cout << "The estimated eigenvalue near "  << setting.sigma << " = ";
      cout << fixed << setprecision(13) << mu << endl;

      cout << endl;
      delete x;
    }
  }
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
