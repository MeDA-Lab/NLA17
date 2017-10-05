////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    main_3Dface_evp.cpp
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

  const char *input  = "input.obj";
  const char *output = "output.obj";
  Method method  = Method::KIRCHHOFF;
  EVP evp = EVP::NONE;
  int nv, nf, nb, *F = nullptr, *idx_b, *Lii_row = nullptr, *Lii_col = nullptr, *Lib_row = nullptr, *Lib_col = nullptr;
  double timer, *V = nullptr, *C = nullptr, *Lii_val = nullptr, *Lib_val = nullptr, *U;


  // Read arguments
  readArgs(argc, argv, input, output, method, evp);

  // Read object
  readObject(input, &nv, &nf, &V, &C, &F);

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
  constructLaplacianSparse(method, nv, nb, nf, V, F,
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

  // Solve harmonic
  cout << "Solving Harmonic ......................." << flush;
  tic(&timer);
  solveHarmonicSparse(nv, nb, Lii_val, Lii_row, Lii_col,
    Lib_val, Lib_row, Lib_col, U);
    cout << " Done.  ";
  toc(&timer);
  cout << endl;
  // Write object
  writeObject(output, nv, nf, U, C, F);

  if (evp != EVP::NONE) {
    cout << "Solving Eigenvalue Problem ............." << flush;
    double mu0 = 1.5, mu;  // Modify mu0 to change the initial
                           // guess of eigenvalue
    double *x;
    x = new double[nv-nb];
    char flag = 'D';      // Modify flag to choose solver on GPU
                          // or CPU. Possible options are
                          // 'H': solver on host    (CPU)
                          // 'D': solver on device  (GPU)
    int nnz = Lii_row[nv-nb];

    switch (evp) {
      case EVP::HOST:
        tic(&timer);
        solveShiftEVPHost(nv-nb, nnz, Lii_val, Lii_row, Lii_col,
          mu0, &mu, x);
          cout << " Done.  ";
        toc(&timer);
        break;
      case EVP::DEVICE:
        tic(&timer);
        solveShiftEVP(nv-nb, nnz, Lii_val, Lii_row, Lii_col, mu0, &mu, x);
        cout << " Done.  ";
        toc(&timer);
        break;
    }

    cout << endl;
    cout << "n = " << nv-nb << endl;
    cout << "nnz = " << nnz << endl;
    cout << "The estimated eigenvalue near "  << mu0 << " = ";
    cout << fixed << setprecision(13) << mu << endl;

    cout << endl;
    delete x;
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
