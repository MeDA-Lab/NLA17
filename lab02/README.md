# NLA17 - lab02
This is lab02 of NLA17.

In this lab, you will learn how to use the sparse direct linear solvers in __cuSOLVER__.

## Library
- CUDA

## Compilation and Usage
Please read the [__QuickStart__](QuickStart.md) for more information.

## Exercise
1. Please complete the function `chol_dev()` in the file `chol_cuda.cpp`.
2. Please complete the function `qr_host()` and `qr_dev()` in the file `qr_cuda.cpp`.
3. Please test test the two main programs with  three different direct linear solvers: LU, Cholesky and QR.
4. Please test the graph laplacian program with different graph data.
5. Please test the harmonic parameterization program with different mesh data.
6. Compare the results in direct solvers. EX: solver execution time, accuracy,...