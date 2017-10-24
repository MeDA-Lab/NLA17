////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    write_partition.cpp
/// @brief   write the object file of partition
///
/// @author  Yuhsiang Mike Tsai

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <cstring>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include "cusparse.h"
#include "mkl.h"

using namespace std;

void writePartition(
    const int nv,
    const int E_size_r,
    const double *E,
    const int ev_num,
    const double *eig_vals,
    const double *eig_vecs,
    const char *filename
) {
    double aszero = 1e-4;
    int zero_index = 0;
    double color[2][3] = {{1, 0, 0}, {0, 0, 1}};
    for (zero_index = 0; zero_index < ev_num; zero_index++) {
        if (eig_vals[zero_index] > aszeros) {
            break;
        }
    }
    if (ev_num-zero_index < 2) {
        cerr << "Error\n";
        exit(1);
    }
    cout << "Stores in \"" << filename << "\"." << endl;
    ofstream fout(filename, ofstream::out);
    if ( fout.good() == 0 ) {
      cerr << "Can not write the file " << input << "\n";
      exit(1);
    }
    fout << "# " << E_size_r + nv << " vertex\n";
    int posi = 0;
    for (int i = 0; i < nv; i++) {
        fout << "v " << eig_vecs[zero_index*nv+i]
             << " "  << eig_vecs[(zero_index+1)*nv+i]
             << " 0";
        posi = eigvecs[zero_index*nv+i] > 0 ? 1 : 0;
        for (int j = 0; j < 3; j++) {
            fout << " " << color[posi][j];
        }
        fout << endl;
    }
    for (int i = 0; i < E_size_r; i++) {
        
    }
    fout << "# " << E_size_r << " faces\n";
    for (int i=0; i<nf; i++) {
      fout<<"f "<<F[i]<<" "<<F[nf+i]<<" "<<F[2*nf+i]<<"\n";
    }
    fout.close();
}