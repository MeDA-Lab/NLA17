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

* To build the program, simply type `make` in terminal. After typing `make`, you will see the following

	```
	g++ -c sgp_main_ls.cpp -I include -O3 -m64 -std=c++11 -I/opt/intel/mkl/include
	g++ -c src/core/read_graph.cpp -I include -O3 -m64 -std=c++11 -I/opt/intel/mkl/include -I/usr/local/cuda-8.0/include
	g++ -c src/core/graph_adjacency.cpp -I include -O3 -m64 -std=c++11 -I/opt/intel/mkl/include -I/usr/local/cuda-8.0/include
	g++ -c src/core/graph_laplacian.cpp -I include -O3 -m64 -std=c++11 -I/opt/intel/mkl/include -I/usr/local/cuda-8.0/include
	g++ -c src/cuda/solve_shiftevp_cuda.cpp -I include -O3 -m64 -std=c++11 -I/opt/intel/mkl/include -I/usr/local/cuda-8.0/include
	g++ -c src/core/map_boundary.cpp -I include -O3 -m64 -std=c++11 -I/opt/intel/mkl/include -I/usr/local/cuda-8.0/include
	g++ -c src/core/read_args.cpp -I include -O3 -m64 -std=c++11 -I/opt/intel/mkl/include -I/usr/local/cuda-8.0/include
	g++ -c src/core/read_object.cpp -I include -O3 -m64 -std=c++11 -I/opt/intel/mkl/include -I/usr/local/cuda-8.0/include
	g++ -c src/core/reorder_vertex.cpp -I include -O3 -m64 -std=c++11 -I/opt/intel/mkl/include -I/usr/local/cuda-8.0/include
	g++ -c src/sparse/construct_laplacian_sparse.cpp -I include -O3 -m64 -std=c++11 -I/opt/intel/mkl/include -I/usr/local/cuda-8.0/include
	g++ -c src/sparse/solve_harmonic_sparse.cpp -I include -O3 -m64 -std=c++11 -I/opt/intel/mkl/include -I/usr/local/cuda-8.0/include
	g++ -c src/sparse/verify_boundary_sparse.cpp -I include -O3 -m64 -std=c++11 -I/opt/intel/mkl/include -I/usr/local/cuda-8.0/include
	g++ -c src/core/set_graph_type.cpp -I include -O3 -m64 -std=c++11 -I/opt/intel/mkl/include -I/usr/local/cuda-8.0/include
	g++ -c src/core/residual.cpp -I include -O3 -m64 -std=c++11 -I/opt/intel/mkl/include -I/usr/local/cuda-8.0/include
	g++ -c src/core/generate_RHS.cpp -I include -O3 -m64 -std=c++11 -I/opt/intel/mkl/include -I/usr/local/cuda-8.0/include
	g++ -c src/cuda/solve_ls_cuda.cpp -I include -O3 -m64 -std=c++11 -I/opt/intel/mkl/include -I/usr/local/cuda-8.0/include
	g++ -c src/cuda/lu_cuda.cpp -I include -O3 -m64 -std=c++11 -I/opt/intel/mkl/include -I/usr/local/cuda-8.0/include
	g++ -c src/cuda/qr_cuda.cpp -I include -O3 -m64 -std=c++11 -I/opt/intel/mkl/include -I/usr/local/cuda-8.0/include
	g++ -c src/cuda/chol_cuda.cpp -I include -O3 -m64 -std=c++11 -I/opt/intel/mkl/include -I/usr/local/cuda-8.0/include
	g++ sgp_main_ls.o -o sgp_main_ls.out read_graph.o graph_adjacency.o graph_laplacian.o solve_shiftevp_cuda.o map_boundary.o read_args.o read_object.o reorder_vertex.o construct_laplacian_sparse.o solve_harmonic_sparse.o verify_boundary_sparse.o set_graph_type.o residual.o generate_RHS.o solve_ls_cuda.o lu_cuda.o qr_cuda.o chol_cuda.o -O3 -m64 -std=c++11 -L/opt/intel/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -lcudart -lcublas -lcufft -lcusolver -lcusparse -lgomp -lm -ldl
	g++ -c main_3Dface_ls.cpp -I include -O3 -m64 -std=c++11 -I/opt/intel/mkl/include
	g++ main_3Dface_ls.o -o main_3Dface_ls.out read_graph.o graph_adjacency.o graph_laplacian.o solve_shiftevp_cuda.o map_boundary.o read_args.o read_object.o reorder_vertex.o construct_laplacian_sparse.o solve_harmonic_sparse.o verify_boundary_sparse.o set_graph_type.o residual.o generate_RHS.o solve_ls_cuda.o lu_cuda.o qr_cuda.o chol_cuda.o -O3 -m64 -std=c++11 -L/opt/intel/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -lcudart -lcublas -lcufft -lcusolver -lcusparse -lgomp -lm -ldl 
	```
* For graph laplacian, the usage is:

	`./sgp_main.out [data filename] [type of graph (optional)]`
	
	Currently, 2 types of graph are supported:
	* `0`: __simple graph__
	* `3`: __undirected weighted graph__

	Example Usage: Type the following commands in the terminal
	
	* for __simple graph__, type
	
		`./sgp_main.out data/graph/ChicagoRegional`
		
		or
		
		`./sgp_main.out data/graph/ChicagoRegional 0`
		
	* for __undirected weighted graph__, type
	
		`./sgp_main.out data/graph/moreno_kangaroo_kangaroo`
		
		or
		
		`./sgp_main.out data/graph/moreno_kangaroo_kangaroo 1`
	
	There are some prepared graph data files in the `data/graph` directory.

* For 3D face animation, the basic usage is

	`./main_3Dface_ls.out [OPTIONS]`

	Type the following in terminal to get more information:
	
	`./main_3Dface_ls.out -h` or `./main_3Dface_ls.out --help`
	
	Example Usage: Type the following in the terminal
	
	`./main_3Dface_ls.out -f data/obj/CYHo.obj -t 1`
	
	There are some prepared obj data files in the `data/obj` directory.
	
## Results
For graph laplacian, you will see output like

```
read file......... Done.  
Size of data is 12x2
type of graph: simple graph
Construct adjacency matrix of graph......... Done.  
size of matrix = 8
nnz of A = 24
Construct Laplacian matrix of graph......... Done.  
nnz of L = 32
Solving Linear System......... Done.  
Elapsed time is 0.385103 seconds.
||Ax - b|| =  1.70194e-15
```
For 3D face animation,  you will see output like

```
dos2unix: converting file data/obj/CYHo.obj to Unix format ...
Loads from "data/obj/CYHo.obj" with color.
"data/obj/CYHo.obj" contains 61961 vertices and 123132 faces.

Verifying boundary ..................... Done.  Elapsed time is 0.130582 seconds.
Reordering vertices .................... Done.  Elapsed time is 0.00731587 seconds.
Constructing Laplacian ................. Done.  Elapsed time is 0.074718 seconds.
Mapping Boundary ....................... Done.  Elapsed time is 8.98838e-05 seconds.
Solving Linear System ....................... Done.  Elapsed time is 8.04763 seconds.
n = 61173
nnz = 425981

||Ax - b|| =  1.23811e-12
```

## Setting Parameters
### __Graph Laplacian__

1. Modify `shift_sigma` to set the shift.
2. Modify `flag` to choose solver on GPU or CPU. Possible options are
	* `'H'`: solver on host &nbsp;&nbsp;&nbsp;(CPU)
	* `'D'`: solver on device    (GPU) (default)
3. Modify `solver` to switch between different linear solvers. Possible options are
	* `0`: LU (default)
	* `1`: Cholesky
	* `2`: QR

All parameters mentioned above are in the file `sgp_main_ls.cpp`.

### __3D face animation__

1. Modify `flag` to choose solver on GPU or CPU. Possible options are
	* `'H'`: solver on host &nbsp;&nbsp;&nbsp;(CPU)
	* `'D'`: solver on device    (GPU) (default)
2. Modify `solver` to switch between different linear solvers. Possible options are
	* `0`: LU (default)
	* `1`: Cholesky
	* `2`: QR

All parameters mentioned above are in the file `main_3Dface_ls.cpp`.