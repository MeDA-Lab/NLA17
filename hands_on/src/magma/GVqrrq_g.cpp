////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    GVqrrq_g.cpp
/// @brief   The implementation of Shifted QR for T.
///
/// @author  William Liao
///

// n is the size of v

#include "sgp.hpp"
#include <cstdio>
#include <cstdlib>

int GVqrrq_g(double *v, double *u, double *c, double *s, double shift, int n, 
             double *u_temp, double *uu){
    
    int i,j;
    double vv[2], R[4];
    
    memcpy( u_temp, u, (n-1)*sizeof(double) );

    // shift
    for (j=0; j<n; j++) { v[j] -= shift; }
    
    /* Q^t T */
    for (j=0; j<n-1; j++){
        
        vv[0]     = v[j]; 
        vv[1]     = u[j];
        v[j]      = sqrt( vv[0]*vv[0] + vv[1]*vv[1] );
        u[j]      = 0.0;
        c[j]      =  vv[0] / v[j];
        s[j]      = -vv[1] / v[j];
        
        vv[0]     = u_temp[j];
        vv[1]     = v[j+1];
        u_temp[j] = c[j]*vv[0] - s[j]*vv[1];
        v[j+1]    = s[j]*vv[0] + c[j]*vv[1];

        if ( j < n-2 ){
            uu[j]       = -s[j]*u_temp[j+1];
            u_temp[j+1] =  c[j]*u_temp[j+1];
        }

    }
    /* T Q */
    for (j=0; j<n-1; j++){

        //u_temp[j-1] = u[j-1];
        R[0]   = v[j];
        R[1]   = u_temp[j];
        R[2]   = u[j];
        R[3]   = v[j+1];
        v[j]   = R[0]*c[j] - R[1]*s[j] + shift;
        u[j]   = R[2]*c[j] - R[3]*s[j];
        v[j+1] = R[2]*s[j] + R[3]*c[j];
    }
    v[n-1] += shift;
    // for (j=0; j<n; j++) { v[j] += shift ;}

    return 0;

}
