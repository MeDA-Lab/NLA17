////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    invLanczos_magma_gpu.cu
/// @brief   The implementation of inverse Lanczos eigensolver.
///
/// @author  William Liao
///

#include "sgp.hpp"
#include <string>
#include "Lanczos.cuh"
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <magma_v2.h>
#include <magma_lapack.h>
#include <magma_operators.h>
#include "cublas_v2.h"
#include <cuComplex.h>

using namespace std;

static __host__ void zfill_matrix(
    magma_int_t m, magma_int_t n, double *A, magma_int_t lda );

int invLanczos_gpu(int            m,
                   int            nnz,
                   double         *csrValA,
                   int            *csrRowIndA,
                   int            *csrColIndA,
                   LSEV_INFO      LSEV_info, 
                   double         *egval,
                   string         solver_settings)
{
    int     iter, i, tmpIdx, errFlag, flag;
    double  TOL;
    int    MAXIT, Nwant, Nstep, Asize, conv;
    
    TOL     = LSEV_info.tol;
    MAXIT   = LSEV_info.maxit;
    Nwant   = LSEV_info.Nwant;
    Nstep   = LSEV_info.Nstep;
    Asize   = m;

    /* Variables for MAGMA*/
    magma_int_t  n, ldz;
    n           = (int) Nstep;
    ldz         = n;
    magma_int_t liwork = 3 + 5*n;
    magma_int_t * iwork = NULL;
    double vl = 0;
    double vu = 0;
    magma_int_t il = 0;
    magma_int_t iu = 0;
    magma_int_t *info = NULL;
    magma_int_t lrwork = 1 + 4*n + 2*n*n;
    double * rwork = NULL;
    magma_int_t ngpu = 1;

    double *Talpha, *Tbeta, *T_d, *T_e, *z, *U;
    Talpha = new double[Nstep];
    Tbeta  = new double[Nstep];
    T_d    = new double[Nstep];
    T_e    = new double[Nstep-1];
    z      = new double[Nstep*Nstep];
    cudaMalloc(&U, 2*m*(Nstep+1)*sizeof(double));

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    magma_imalloc_cpu(&iwork, liwork);
    magma_dmalloc_cpu(&rwork, lrwork);
    magma_imalloc_cpu(&info, 1);
    
    /* Initial Decomposition */
    errFlag = Lanczos_decomp_gpu(m, nnz, csrValA, csrRowIndA, csrColIndA, LSEV_info, U, Talpha, Tbeta, 1, solver_settings, cublas_handle );
    assert( errFlag == 0 );

    /* Begin Lanczos iteration */
    magma_range_t range = MagmaRangeAll;
    zfill_matrix(n, n, z, ldz);

    for (iter=1; iter<=MAXIT; iter++){
        memcpy( T_d, Talpha,  Nstep   * sizeof(double) );
        memcpy( T_e, Tbeta,  (Nstep-1)* sizeof(double) );

        /* Get the Ritz values */
        magma_dstedx_m( ngpu, range, n, vl, vu, il, iu, T_d, T_e, z,
        ldz, rwork, lrwork, iwork, liwork, info );

        /* Note that T_d will stored in descending order */
        thrust::stable_sort(T_d, T_d + n, thrust::greater<double>());

        /* Check convergence, T_e will store the residules */
        conv = 0;
        for (i=0; i<Nwant; i++){
            tmpIdx = Nstep-1+Nstep*(Nstep-i-1);
            T_e[i] =  abs(Tbeta[Nstep-1]*z[tmpIdx]*z[tmpIdx]);
            if ( T_e[i] < TOL ){
                conv++;
            }else{
                break;
            }
        } // end of i

        /* Converged!! */
        if ( conv == Nwant ){
            flag = iter;
            break;
        }

        /* MAXIT iteration */
        if ( iter == MAXIT ){
            flag = MAXIT + 1;
            break;
        }

        /* Implicit Restart: Lock and Purge */
        errFlag = Lanczos_LockPurge_gpu( Talpha, Tbeta, U, T_d, LSEV_info, Asize, cublas_handle );
        assert( errFlag == 0 );

        /* Restart */
        errFlag = Lanczos_decomp_gpu( m, nnz, csrValA, csrRowIndA, csrColIndA, LSEV_info, U, Talpha, Tbeta, 0, solver_settings, cublas_handle );
        assert( errFlag == 0 );
    } // end of iter

    for (i=0; i<Nwant; i++) {   egval[i] = 1.0/T_d[i];  }

    return flag;
}

static __host__ void zfill_matrix(
    magma_int_t m, magma_int_t n, double *A, magma_int_t lda )
{
    #define A(i_, j_) A[ (i_) + (j_)*lda ]
    
    magma_int_t i, j;
    for( j=0; j < n; ++j ) {
        for( i=0; i < m; ++i ) {
            if(i == j) {
                A(i,j) = 1.0;
            }
            else{
                A(i,j) = 0.0;
            }
        }
    }
    
    #undef A
}