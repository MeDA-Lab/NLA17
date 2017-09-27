////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    map_boundary.cpp
/// @brief   The implementation of boundary mapping.
///
/// @author  Yuhsiang Mike Tsai
///

#define _USE_MATH_DEFINES
#include <cmath>
#include <harmonic.hpp>

double diffnorm(const double *x, const double *y, int n, int inc){
  double ans=0;
  double temp=0;
  for (int i=0; i<n; i++){
    temp=x[i*inc]-y[i*inc];
    ans+=temp*temp;
  }
  return sqrt(ans);
}
void mapBoundary(
    const int nv,
    const int nb,
    const double *V,
    double *U
) {
  double prefix_sum=0, temp=0;
  for (int i=0; i<nb; i++){
    if (i<nb-1){
      temp=diffnorm(V+i, V+i+1, 3, nv);
    }
    else{
      temp=diffnorm(V+nb-1, V, 3, nv);
    }
    prefix_sum+=temp;
    U[i]=prefix_sum;
  }
  double total=U[nb-1];
  for (int i=0; i<nb; i++){
    temp=U[i]/total;
    U[i]=cos(2*M_PI*temp);
    U[nv+i]=sin(2*M_PI*temp);
  }
}
