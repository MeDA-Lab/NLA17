# NLA2017fall
Numerical Linear Algebra(NTU, NCKU), Matrix Computation (NTNU)

## Information

### Git
* https://github.com/wlab-pro/scsc17summer/NLA_src

### Author
* Yuhsiang Tsai <<yhmtsai@gmail.com>>
* Mu Yang <<emfomy@gmail.com>>
* Yen-Chen Chen <<yanjen224@gmail.com>>
* Dawei D. Chang <<davidzan830@gmail.com>>
* Ting-Hui Wu <<b99201017@gmail.com>>
* Wei-Chien Liao <<b00201028.ntu@gmail.com>>

## Requirements
* C++ compiler with C++11 support.
* [CUDA](https://developer.nvidia.com/cuda-zone) 8.0+ (Used for cuSOLVER & cuSPARSE)
* [Intel&reg; Math Kernel Library](https://software.intel.com/en-us/intel-mkl) (Used for BLAS & LAPACK).
* [MAGMA](http://icl.cs.utk.edu/magma/) 2+ (Used for BLAS & LAPACK with GPU support).
* [Doxygen](http://www.stack.nl/~dimitri/doxygen/) (Used for documentation)(optional).
* [OpenMP](http://openmp.org) Library (optional).

## Usage
* You may need to load the required libraries first before building the program. Execute the following commands in a terminal:

	`module load cuda-dev/8.0`
	
	`module load intel-mkl`

* To build the program, simply type `make` in terminal.
* For graph laplacian, type the following in terminal:

	`./sgp_main.out [data filename]`

	Example Usage: `./sgp_main.out data/graph/ChicagoRegional`
	
	There are some prepared graph data files in the `data/graph` directory.

* For 3D face animation, the basic usage is

	`./main_3Dface_evp.out [OPTIONS]`

	Type the following in terminal to get more information:
	
	`./main_3Dface_evp.out -h` or `./main_3Dface_evp.out --help`
	
	Example Usage: `./main_3Dface_evp.out -f data/obj/CYHo.obj -t 1`
	
	There are some prepared obj data files in the `data/obj` directory.