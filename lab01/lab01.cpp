// Copyright [2017] <NLA17>
// Author: Yuhsiang Mike Tsai
#include <mkl.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <iostream>

int main() {
    // Init
    // n is the length of two vectors.
    // x, y : vectors with length n.
    int n = 1000;
    double *x = nullptr, *y = nullptr;
    double mkl_ans = 0, cuda_ans = 0;
    x = new double[n];
    y = new double[n];
    unsigned int seed = 2017;
    for (int i = 0; i < n; i++) {
        x[i] = ((double) rand_r(&seed))/RAND_MAX;
        y[i] = ((double) rand_r(&seed))/RAND_MAX;
    }

    // mkl: dot-product by cblas_ddot
    std::cout << "===== MKL  =====\n";
    mkl_ans = cblas_ddot(n, x, 1, y, 1);
    std::cout << "MKL answer : " << mkl_ans << std::endl;
    // cuda: dot-product
    // dx, dy: vectors in device (GPU)
    std::cout << "===== CUDA =====\n";
    double *dx = nullptr, *dy = nullptr;

    // Allocate memory in GPU by cudaMalloc
    std::cout << "Allocate device memory\n";
    cudaMalloc((void**) &dx, n*sizeof(double));
    cudaMalloc((void**) &dy, n*sizeof(double));

    // Transfer data from CPU to GPU by cudaMemcpy
    std::cout << "Transfer data from CPU to GPU\n";
    cudaMemcpy(dx, x, n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y, n*sizeof(double), cudaMemcpyHostToDevice);

    // Compute dot-product by cublasDdot
    std::cout << "Calculate dot-product\n";
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasDdot(handle, n, dx, 1, dy, 1, &cuda_ans);
    cublasDestroy(handle);
    std::cout << "CUDA answer : " << cuda_ans << std::endl;
    std::cout << "===== DIFF =====\n";
    std::cout << "The diff of two ans: " << cuda_ans-mkl_ans << "\n";
    return 0;
}
