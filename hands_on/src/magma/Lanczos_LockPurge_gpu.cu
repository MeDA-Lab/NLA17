////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    Lanczos_LockPurge_gpu.cu
/// @brief   The implementation of Implicit Restart: Lock and Purge.
///
/// @author  William Liao
///

#include "sgp.hpp"
#include "Lanczos.cuh"
#include <cublas_v2.h>

int Lanczos_LockPurge_gpu( double           *Talpha,
                           double           *Tbeta,
                           double           *U,
                           double           *T_d,
                           LSEV_INFO        LSEV_info, 
                           const int        Asize, 
                           cublasHandle_t   cublas_handle)
{
    int      j, jj;
    int      Nwant = LSEV_info.Nwant;
    int      Nstep = LSEV_info.Nstep;
    double   cublas_scale;
    double  *c      = new double[Nstep-1];
    double  *s      = new double[Nstep-1];
    double  *u_temp = new double[Nstep-1];
    double  *uu     = new double[Nstep-2];
    double  *Rval   = T_d + Nwant;
    cublasStatus_t cublasErr;

    for (j=0; j<Nstep-Nwant; j++){

        /* Shifted QR for T */
        GVqrrq_g( Talpha, Tbeta, c, s, Rval[j], Nstep-j, u_temp, uu );

        /* Update: U(:,0:Nstep-1-j) * Q */
        for (jj=0; jj<Nstep-1-j; jj++){
            cublasErr = cublasDrot( cublas_handle, Asize, U+Asize*(jj+1), 1, U+Asize*jj, 1, c+jj, s+jj );
            assert( cublasErr == CUBLAS_STATUS_SUCCESS );
        }
        /* Update U(:,Nstep-1-j), Note jj = Nstep-1-j now */
        s[jj-1] *= Tbeta[jj];
        cublasErr = cublasDrot( cublas_handle, Asize, U+Asize*(jj+1), 1, U+Asize*jj, 1, Tbeta+jj-1, s+jj-1 );
        assert( cublasErr == CUBLAS_STATUS_SUCCESS );

        /* Normalize U(:,Nstep-1-j) */
        cublasErr = cublasDnrm2(cublas_handle, Asize, U+Asize*jj, 1, Tbeta+jj-1 );
        assert( cublasErr == CUBLAS_STATUS_SUCCESS );
        
        cublas_scale = 1.0 / Tbeta[jj-1];
        cublasErr = cublasDscal(cublas_handle, Asize, &(cublas_scale), U+Asize*jj, 1 );
        assert( cublasErr == CUBLAS_STATUS_SUCCESS );

        /* Purge Talpha(Nstep-1-j) and Tbeta(Nstep-1-j) */
        Talpha[jj] = 0;
        Tbeta[jj]  = 0;

        jj++;
        /* Purge U(:,Nstep-j) */
        CCE( cudaMemset( U+Asize*jj, 0, Asize*sizeof(double) ) );
    
    } // end of j
    
    return 0;

}
