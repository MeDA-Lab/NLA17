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
#include "sgp.hpp"
using namespace std;

void writePartition(
    const int nv,
    const int E_size_r,
    const int *E,
    const int ev_num,
    const double *eig_vals,
    const double *eig_vecs,
    const string filename
) {
    double aszero = 1e-4;
    int zero_index = 0;
    double color[2][3] = {{1, 0, 0}, {0, 0, 1}};
    for (zero_index = 0; zero_index < ev_num; zero_index++) {
        if (eig_vals[zero_index] > aszero) {
            break;
        }
    }
    if (ev_num-zero_index < 2) {
        cerr << "The number of non-trivial eigenvectors is not enough\n";
        exit(1);
    }
    cout << "Stores in \"" << filename << "\"." << endl;
    ofstream fout(filename, ofstream::out);
    if ( fout.good() == 0 ) {
      cerr << "Can not write the file " << filename << "\n";
      exit(1);
    }
    fout << "# " << E_size_r + nv << " vertex\n";
    int *color_i = new int[nv];
    for (int i = 0; i < nv; i++) {
        fout << "v " << eig_vecs[zero_index*nv+i]
             << " "  << eig_vecs[(zero_index+1)*nv+i]
             << " 0";
        color_i[i] = eig_vecs[zero_index*nv+i] > 0 ? 1 : 0;
        for (int j = 0; j < 3; j++) {
            fout << " " << color[color_i[i]][j];
        }
        fout << endl;
    }
    double *tp = new double[3], *tc = new double[3];
    int s, e;
    for (int i = 0; i < E_size_r; i++) {
        s = E[i];
        e = E[E_size_r+i];
        for (int j = 0; j < 3; j++) {
            tp[j] = (eig_vecs[(zero_index+j)*nv+s]
                    +eig_vecs[(zero_index+j)*nv+e])/2;
            tc[j] = (color[color_i[s]][j] + color[color_i[e]][j])/2;
        }
        tp[2] = 0;
        fout << "v " << tp[0] << " " << tp[1] << " " << tp[2]
             << " "  << 1 << " " << 1 << " " << 1 << endl;
    }
    fout << "# " << E_size_r << " faces\n";
    for (int i = 0; i < E_size_r; i++) {
        fout << "f " << E[i]+1 << " " << i+nv+1 << " " << E[E_size_r+i]+1 << endl;
    }
    fout.close();
    delete [] tp;
    delete [] tc;
    delete [] color_i;
}