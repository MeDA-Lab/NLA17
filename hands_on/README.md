# NLA17 - Hands-on
This is the C++ codes for hands-on of NLA17.


## Requirements
* C++ compiler with C++11 support.
* [CUDA](https://developer.nvidia.com/cuda-zone) 8.0+ (Used for cuSOLVER & cuSPARSE)
* [Intel&reg; Math Kernel Library](https://software.intel.com/en-us/intel-mkl) (Used for BLAS & LAPACK).
* [MAGMA](http://icl.cs.utk.edu/magma/) 2+ (Used for BLAS & LAPACK with GPU support).
* [Doxygen](http://www.stack.nl/~dimitri/doxygen/) (Used for documentation)(optional).
* [OpenMP](http://openmp.org) Library (optional).

## Usage
* You may need to load the required libraries first before building the program. Execute the following commands in a terminal:  
	```
	module load intel-mkl
	module load cuda-dev/8.0
	module load magma-dev/2.2f
	```
## Build the Program

`make` for __Surface Parameterization__ with magma solver and __Spectral Graph Partition__  
`make surfpara.out` for __Surface Parameterization__ without any solvers.  
`make surfpara_magma.out` for __Surface Parameterization__ with magma solver.  
`make sgp.out` for __Spectral Graph Patition__.  

## Surface Parameterizations
Please see [SurfPara.md](SurfPara.md)  

## Spectral Graph Partition
Please see [SGP.md](SGP.md)  