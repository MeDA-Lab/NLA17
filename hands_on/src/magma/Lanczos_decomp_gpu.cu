////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    Lanczos_decomp_gpu.cu
/// @brief   The implementation of Lanczos tridiagonalization.
///
/// @author  William Liao
///

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include "sgp.hpp"
#include <string>
#include <cublas_v2.h>
#include <helper_cuda.h>

using namespace std;

#define BLOCK_SIZE 256

static __global__ void Lanczos_init_kernel(double *U, const int Asize);

int Lanczos_decomp_gpu(int         m,
                       int         nnz,
                       double      *csrValA,
                       int         *csrRowIndA,
                       int         *csrColIndA,
                       LSEV_INFO   LSEV_info,
                       double      *U,
                       double      *Talpha,
                       double      *Tbeta,
                       int         isInit,
                       string      solver_settings,
                       cublasHandle_t cublas_handle)
{
    int i, j, loopStart;
    int Nwant, Nstep, Asize;
    double cublas_scale;
    double cublas_zcale, alpha_tmp, Loss;
    cublasStatus_t cublasErr;

    Nwant = LSEV_info.Nwant;
    Nstep = LSEV_info.Nstep;
    Asize = m;
    
    switch (isInit){

        case 1: // The case of initial decomposition
        { 
            /* The initial vector */
            dim3 DimBlock( BLOCK_SIZE, 1, 1);
            dim3 DimGrid( (Asize-1)/BLOCK_SIZE +1, 1, 1);
            Lanczos_init_kernel<<<DimGrid, DimBlock>>>(U, Asize);
            getLastCudaError("Lanczos_init_kernel");

            solveGraphLS(solver_settings, m, nnz, csrValA, csrRowIndA, csrColIndA, U, U+Asize);
        
            cublasErr = cublasDdot(cublas_handle, Asize, U+Asize, 1, U, 1, &alpha_tmp );
            assert( cublasErr == CUBLAS_STATUS_SUCCESS );
            Talpha[0] = alpha_tmp;

            cublas_zcale = -1.0*Talpha[0];
            cublasErr = cublasDaxpy(cublas_handle, Asize, &cublas_zcale, U, 1, U+Asize, 1);
            assert( cublasErr == CUBLAS_STATUS_SUCCESS );

            cublasErr = cublasDnrm2(cublas_handle, Asize, U+Asize, 1, Tbeta );
            assert( cublasErr == CUBLAS_STATUS_SUCCESS );
            cublas_scale = 1.0 / Tbeta[0];
            cublasErr = cublasDscal( cublas_handle, Asize, &(cublas_scale), U+Asize, 1 );
            assert( cublasErr == CUBLAS_STATUS_SUCCESS );

            loopStart = 1;
            break;
        }
        case 0: // The restarted decomposition
            
            loopStart = Nwant - 1;
            break;

        default:
            std::cout << "ERROR: wrong use of isInit is Lanczos_decomp" << std::endl;
            return -1;

    } // end of switch

    for (j=loopStart; j<Nstep; j++){

        solveGraphLS(solver_settings, m, nnz, csrValA, csrRowIndA, csrColIndA, U+Asize*j, U+Asize*(j+1));

        cublas_zcale = -1.0*Tbeta[j-1];
        cublasErr = cublasDaxpy(cublas_handle, Asize, &cublas_zcale, U+Asize*(j-1), 1, U+Asize*(j+1), 1);
        assert( cublasErr == CUBLAS_STATUS_SUCCESS );
      
        cublasErr = cublasDdot(cublas_handle, Asize, U+Asize*(j+1), 1, U+Asize*j, 1, &alpha_tmp );
        assert( cublasErr == CUBLAS_STATUS_SUCCESS );

        cublas_zcale = -1.0*alpha_tmp;
        cublasErr = cublasDaxpy(cublas_handle, Asize, &cublas_zcale, U+Asize*j, 1, U+Asize*(j+1), 1);
        assert( cublasErr == CUBLAS_STATUS_SUCCESS );

        /* Full Reorthogonalization */
        for (i=0; i<=j; i++){
            
            cublasErr = cublasDdot(cublas_handle, Asize, U+Asize*i, 1, U+Asize*(j+1), 1, &Loss );
            assert( cublasErr == CUBLAS_STATUS_SUCCESS );

            cublas_zcale = -1.0*Loss;
            cublasErr = cublasDaxpy(cublas_handle, Asize, &cublas_zcale, U+Asize*i, 1, U+Asize*(j+1), 1);
            assert( cublasErr == CUBLAS_STATUS_SUCCESS );

        } // end of i

        alpha_tmp = alpha_tmp + Loss;
        Talpha[j] = alpha_tmp;
        // End of Full Reorthogonalization

        cublasErr = cublasDnrm2( cublas_handle, Asize, U+Asize*(j+1), 1, Tbeta+j );
        assert( cublasErr == CUBLAS_STATUS_SUCCESS );

        cublas_scale = 1.0 / Tbeta[j];
        cublasErr = cublasDscal( cublas_handle, Asize, &(cublas_scale), U+Asize*(j+1), 1 );
        assert( cublasErr == CUBLAS_STATUS_SUCCESS );

    } // end of j

    return 0;

}

static __global__ void Lanczos_init_kernel(double *U, const int Asize){

    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if ( idx < Asize ){ 
        U[idx] = 0.0; 
    }

    if ( idx == 0 ){
        U[idx] = 1.0;
    }

}