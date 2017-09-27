////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    write_object.cpp
/// @brief   The implementation of object writing.
///
/// @author  Yuhsiang Mike Tsai
///
#include <harmonic.hpp>
#include <iostream>
#include <fstream>
using namespace std;

void writeObject(
    const char *input,
    const int nv,
    const int nf,
    double *U,
    double *C,
    int *F
) {
  cout << "Stores in \"" << input << "\"." << endl;
  ofstream fout(input, ofstream::out);
  if ( fout.good() == 0 ) {
    cerr<<"Can not write the file "<<input<<"\n";
    exit(1);
  }

  fout<<"# "<<nv<<" vertex\n";
  if ( C[0] == -1 ) {
    for (int i=0; i<nv; i++){
      fout<<"v "<<U[i]<<" "<<U[nv+i]<<" 0\n";
    }
  }
  else {
    for (int i=0; i<nv; i++){
      fout<<"v "<<U[i]<<" "<<U[nv+i]<<" 0 "<<C[i]<<" "<<C[nv+i]<<" "<<C[2*nv+i]<<"\n";
    }
  }

  fout<<"# "<<nf<<" faces\n";
  for (int i=0; i<nf; i++) {
    fout<<"f "<<F[i]<<" "<<F[nf+i]<<" "<<F[2*nf+i]<<"\n";
  }
  fout.close();
}
