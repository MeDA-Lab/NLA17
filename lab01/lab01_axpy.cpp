// Copyright [2017] <NLA17>
// Author: Yuhsiang Mike Tsai
#include <mkl.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <iostream>
double diff(int n, double *x, double *y);
int main() {
    // Init
    // n is the length of two vectors.
    // x, y : vectors with length n.
    int n = 1000;
    double alpha = 1.0;
    double *x = nullptr, *y = nullptr;
    double *mkl_ans = nullptr, *cuda_ans = nullptr;
    x = new double[n];
    y = new double[n];
    mkl_ans = new double[n];
    cuda_ans = new double[n];
    unsigned int seed = 2017;
    for (int i = 0; i < n; i++) {
        x[i] = ((double) rand_r(&seed))/RAND_MAX;
        y[i] = ((double) rand_r(&seed))/RAND_MAX;
    }

    // mkl: axpy (y <- a*x + y) by cblas_daxpy
    std::cout << "===== MKL  =====\n";
    for (int i = 0; i < n; i++) {
        mkl_ans[i] = y[i];
    }
    cblas_daxpy(n, x, 1, mkl_ans, 1);

    // cuda: axpy
    // dx, dy: vectors in device (GPU)
    std::cout << "===== CUDA =====\n";
    double *dx = nullptr, *dy = nullptr;

    // Allocate memory in GPU by cudaMalloc
    std::cout << "Allocate device memory\n";
    // todo : allocate memory

    // Transfer data from CPU to GPU by cudaMemcpy
    std::cout << "Transfer data from CPU to GPU\n";
    // todo : transfer data

    // Compute axpy by cublasDaxpy
    std::cout << "Calculate y <- a*x+y\n";
    cublasHandle_t handle;
    // todo : create/destroy handle and use cublasDaxpy.
    // hint : see alpha type carefully

    // Transfer answer from GPU to CPU
    // todo : transfer answer
    // hint : dy -> cuda_ans

    // Compare two answers
    std::cout << "===== DIFF =====\n";
    std::cout << "The diff of two ans: " << diff(n, mkl_ans, cuda_ans) << "\n";
    return 0;
}

double diff(int n, double *x, double *y) {
    // answer = ||(y-x)||/||x||
    double a = 0, b = 0;
    for (int i = 0; i < n; i++) {
        a += (x[i]-y[i])*(x[i]-y[i]);
        b += x[i]*x[i];
    }
    return a/b;
}
