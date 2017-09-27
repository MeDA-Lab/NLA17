////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    verify_boundary.cpp
/// @brief   The implementation of boundary verification.
///
/// @author  Mu Yang <<emfomy@gmail.com>>
/// @author  Yuhsiang Mike Tsai
///

#include <algorithm>
#include <harmonic.hpp>
using namespace std;

void verifyBoundary(
    const int nv,
    const int nf,
    const int *F,
    int *ptr_nb,
    int *idx_b
) {

  int *Gb = new int[nv*nv];
  int &nb = *ptr_nb;
  int p[3];

  for ( auto ij = 0; ij < nv*nv; ++ij ) {
    Gb[ij] = 0;
  }

  // Generate graph
  for ( int i = 0; i < nf; ++i ) {
    p[0] = F[i]-1;
    p[1] = F[nf+i]-1;
    p[2] = F[2*nf+i]-1;

    Gb[p[1]*nv + p[0]]++;
    Gb[p[2]*nv + p[1]]++;
    Gb[p[0]*nv + p[2]]++;

    Gb[p[0]*nv + p[1]]--;
    Gb[p[1]*nv + p[2]]--;
    Gb[p[2]*nv + p[0]]--;
  }

  // List boundary
  int idx0 = (find(Gb, Gb+nv*nv, 1)-Gb) / nv;
  int idx = idx0;
  for ( nb = 0; nb < nv; ) {
    idx_b[nb] = idx+1;
    idx = find(Gb+idx*nv, Gb+(idx+1)*nv, 1) - (Gb+idx*nv);
    ++nb;
    if ( idx == idx0 ) {
      break;
    }
  }
}
