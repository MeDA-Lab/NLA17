////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    reorder_vertex.cpp
/// @brief   The implementation of vertex reordering.
///
/// @author  Yuhsiang Mike Tsai
///

#include <harmonic.hpp>
#include <iostream>
#include <algorithm>
using namespace std;

void reorderVertex(
    const int nv,
    const int nb,
    const int nf,
    double *V,
    double *C,
    int *F,
    const int *idx_b
) {
  double *V_cp = new double [nv*3], *C_cp = new double [nv*3];
  int *used = new int [nv];
  for (int i=0; i<nv; i++){
    used[i]=-1;
  }
  copy(V, V+nv*3, V_cp);
  copy(C, C+nv*3, C_cp);
  for (int i=0; i<nb; i++){
    V[i]=V_cp[idx_b[i]-1];
    V[nv+i]=V_cp[nv+idx_b[i]-1];
    V[2*nv+i]=V_cp[2*nv+idx_b[i]-1];
    C[i]=C_cp[idx_b[i]-1];
    C[nv+i]=C_cp[nv+idx_b[i]-1];
    C[2*nv+i]=C_cp[2*nv+idx_b[i]-1];
    used[idx_b[i]-1]=i;
    // if (idx_b[i]-1>=nv){
      // cout<<idx_b[i]<<"Q\n";
    // }
  }
  int index=nb;
  // cout << nb<<"T\n";
  for (int i=0; i<nv; i++){
    if (used[i]==-1){
      V[index]=V_cp[i];
      V[nv+index]=V_cp[nv+i];
      V[2*nv+index]=V_cp[2*nv+i];
      C[index]=C_cp[i];
      C[nv+index]=C_cp[nv+i];
      C[2*nv+index]=C_cp[2*nv+i];
      used[i]=index;
      index++;
    }
  }
  for (int i=0; i<nf*3; i++){
    F[i]=used[F[i]-1]+1;
  }
  if (index!=nv){
    cerr<<index<<" Reorder Error"<<nv<<"\n";
  }
  delete [] V_cp;
  delete [] C_cp;
  delete [] used;
  return;
}
