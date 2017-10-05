# NLA17 - Hands-on
This is the C codes for hands-on of NLA17.


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
If you want to use `magma_3Dface_evp`, you need to load extra module.  
`module load magma-dev/2.2f`
<!-- 
* To build the program, simply type `make` in terminal. After typing `make`, you will see the following

	```
	g++ -c sgp_main.cpp -I include -O3 -m64 -std=c++11 -I/opt/intel/mkl/include
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
	g++ sgp_main.o -o sgp_main.out read_graph.o graph_adjacency.o graph_laplacian.o solve_shiftevp_cuda.o map_boundary.o read_args.o read_object.o reorder_vertex.o construct_laplacian_sparse.o solve_harmonic_sparse.o verify_boundary_sparse.o set_graph_type.o -O3 -m64 -std=c++11 -L/opt/intel/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -lcudart -lcublas -lcufft -lcusolver -lcusparse -lgomp -lm -ldl
	g++ -c main_3Dface_evp.cpp -I include -O3 -m64 -std=c++11 -I/opt/intel/mkl/include
	g++ main_3Dface_evp.o -o main_3Dface_evp.out read_graph.o graph_adjacency.o graph_laplacian.o solve_shiftevp_cuda.o map_boundary.o read_args.o read_object.o reorder_vertex.o construct_laplacian_sparse.o solve_harmonic_sparse.o verify_boundary_sparse.o set_graph_type.o -O3 -m64 -std=c++11 -L/opt/intel/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -lcudart -lcublas -lcufft -lcusolver -lcusparse -lgomp -lm -ldl
	``` -->
* To build the program, you can select what you need.  
`make sgp_main.out` for bipartition.  
`make main_3Dface_evp.out` for 3Dface animation without any solver.  
`make magma_3Dface_evp.out` for 3Dface animation with magma solver.  

* For graph laplacian, the usage is

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

	`./main_3Dface_evp.out [OPTIONS]`

	Type the following in terminal to get more information:
	
	`./main_3Dface_evp.out -h` or `./main_3Dface_evp.out --help`
	
	Example Usage: Type the following in terminal
	
	`./main_3Dface_evp.out -f data/obj/CYHo.obj -t 1`
	
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
Solving Eigenvalue Problem......... Done.  
Elapsed time is 0.396083 seconds.
The estimated eigenvalue near 0.18 = 0.0000000000000
```

For 3D face animation,  you will see output like

```
dos2unix: converting file data/obj/CYHo.obj to Unix format ...
Loads from "data/obj/CYHo.obj" with color.
"data/obj/CYHo.obj" contains 61961 vertices and 123132 faces.

Verifying boundary ..................... Done.  Elapsed time is 0.131394 seconds.
Reordering vertices .................... Done.  Elapsed time is 0.00719786 seconds.
Constructing Laplacian ................. Done.  Elapsed time is 0.0773089 seconds.
Mapping Boundary ....................... Done.  Elapsed time is 9.60827e-05 seconds.
Solving Eigenvalue Problem ....................... Done.  Elapsed time is 86.1831 seconds.

n = 61173
nnz = 425981
The estimated eigenvalue near 1.1 = 0.0000000000000
```

## Setting Parameters
### __Graph Laplacian__

1. Modify `mu0` to change the initial guess of eigenvalue.
2. Modify `flag` to choose solver on GPU or CPU. Possible options are
	* `'H'`: solver on host &nbsp;&nbsp;&nbsp;(CPU)
	* `'D'`: solver on device    (GPU) (default)
3. Modify `shift_sigma` to set the shift.

All parameters mentioned above are in the file `sgp_main.cpp`.

### __3D face animation__

1. Modify `mu0` to change the initial guess of eigenvalue.
2. Modify `flag` to choose solver on GPU or CPU. Possible options are
	* `'H'`: solver on host &nbsp;&nbsp;&nbsp;(CPU)
	* `'D'`: solver on device    (GPU)
	* Note it is assigned by command-line.

All parameters mentioned above are in the file `main_3Dface_evp.cpp`.