# NLA17 - lab02
This is lab02 of NLA17.

In this lab, you will learn how to use the sparse direct linear solvers in __cuSOLVER__.

## Library
- CUDA

## Compilation and Usage
Please read the [__QuickStart__](QuickStart.md) for more information.

## Exercise
1. Please complete the function `chol_dev()` in the file `chol_cuda.cpp`.

	__Hint__: use [`cusolverSpDcsrlsvchol`](http://docs.nvidia.com/cuda/cusolver/index.html#cusolver-lt-t-gt-csrlsvchol)

2. Please complete the function `qr_host()` and `qr_dev()` in the file `qr_cuda.cpp`.

	__Hint__:
	* For the host function `qr_host()` please refer to the function `chol_host()` in `chol_cuda.cpp` for an example.
	* For the device function `qr_dev()` use [`cusolverSpDcsrlsvqr`](http://docs.nvidia.com/cuda/cusolver/index.html#cusolver-lt-t-gt-csrlsvqr)

	For __1.__ and __2.__, you only need to add codes inside the parts highlighted by
	
	```
	/*======================================================*/
	
	
	
	/*======================================================*/
	```

3. Please test test the two main programs with  three different direct linear solvers: LU, Cholesky and QR.

	__Hint__: 
	* Please refer to the [Setting Parameter](QuickStart.md#setting-parameters) section on how to switch to different solvers.
	* After trying different solvers, you can adjust other parameters listed in [Setting Parameter](QuickStart.md#setting-parameters).

4. Please test the graph laplacian program with different graph data.
5. Please test the harmonic parameterization program with different mesh data.
6. Compare the results in direct solvers. EX: solver execution time, accuracy,...

	__Hint__:
	* The __solver execution time__ and the __absolute residual__ (`||Ax - b||`) will be displayed on your terminal each time you successfully execute the whole program.