////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    Lanczos.cuh
/// @brief   The header for Lanczos internal routines.
///
/// @author  William Liao
/// @author  Yuhsiang Mike Tsai

#ifndef SCSC_Lanczos_CUH
#define SCSC_Lanczos_CUH

#include "sgp.hpp"
#include "cublas_v2.h"
#include <string>

int GVqrrq_g(double *v, double *u, double *c, double *s, double shift, int n, 
             double *u_temp, double *uu);

int Lanczos_LockPurge_gpu( double           *Talpha,
                           double           *Tbeta,
                           double           *U,
                           double           *T_d,
                           LSEV_INFO        LSEV_info, 
                           const int        Asize, 
                           cublasHandle_t   cublas_handle);

int Lanczos_decomp_gpu(int         m,
                       int         nnz,
                       double      csrValA,
                       int         csrRowIndA,
                       int         csrColIndA,
                       LSEV_INFO   LSEV_info,
                       double      *U,
                       double      *Talpha,
                       double      *Tbeta,
                       bool        isInit,
                       std::string      solver_settings,
                       cublasHandle_t cublas_handle);
#endif