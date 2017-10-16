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

	`./sgp_main.out [OPTIONS]`
	
	The following are current possible options:
	
	```
	-h,       --help           Display this information
  -f<file>, --file <file>    The graph data file
  -t<num>,  --type <num>     0: simple graph(default if the graph data has 2 columns),
                             1: directed (multi) graph (not supported yet)
                             2: directed weighted graph (not supported yet)
                             3: undirected weighted graph (default if the graph data has 3 columns)
  -p<file>, --para <file>    The parameter setting file
  -e<num>,  --evp <num>      0: None(default), 1: Host, 2: Device
  -l<num>,  --ls <num>       0: None, 1: Direct Host, 2: Direct Device(default), 3: Iterative
  ```
	
	Currently, 2 types of graph are supported:
	* `0`: __simple graph__
	* `3`: __undirected weighted graph__

	Example Usage: Type the following commands in the terminal
	
	* For __simple graph__ , type
		1. Solve eigenvalue problem on device
	
			`./sgp_main.out -f data/graph/ChicagoRegional -e 2`
			
			or
			
			`./sgp_main.out data/graph/ChicagoRegional -t 0 -e 2`
			
		2. Solve linear system with direct device solver

			`./sgp_main.out -f data/graph/ChicagoRegional`
			
		3. Solve linear system with iterative solver
		
			`./sgp_main.out -f data/graph/ChicagoRegional -l 3`
		
	* For __undirected weighted graph__, type
	
		1. Solve eigenvalue problem on device
	
			`./sgp_main.out -f data/graph/moreno_kangaroo_kangaroo -e 2`
			
			or
			
			`./sgp_main.out data/graph/moreno_kangaroo_kangaroo -t 3 -e 2`
			
		2. Solve linear system with direct device solver

			`./sgp_main.out -f data/graph/moreno_kangaroo_kangaroo`
			
		3. Solve linear system with iterative solver
		
			`./sgp_main.out -f data/graph/moreno_kangaroo_kangaroo -l 3`

	
	There are some prepared graph data files in the `data/graph` directory. 

	(__UPDATE__)
	
	You can use the downloaded graph data file as input without modifying the format of the file.

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
Read the graph data from file............... Done.  
Size of data is 198050x2
type of graph: simple graph
Construct adjacency matrix of graph......... Done.  
size of matrix = 18771
nnz of A = 396100
Setting Laplacian and solver parameters..... Done.  
Construct Laplacian matrix of graph......... Done.  
nnz of L = 414871
Solver: CUDA Cholesky Device
Solving Linear System....................... Done.  Elapsed time is 5.01492 seconds.
||Ax - b|| =  4.80763e-12
```
or

```
Read the graph data from file............... Done.  
Size of data is 1298x2
type of graph: simple graph
Construct adjacency matrix of graph......... Done.  
size of matrix = 1467
nnz of A = 2596
Setting Laplacian and solver parameters..... Done.  
Construct Laplacian matrix of graph......... Done.  
nnz of L = 4063
Solving Eigenvalue Problem.................. Done.  Elapsed time is 0.396985 seconds.
The estimated eigenvalue near 0.6 = 0.5833032584504
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

Modify the file `solverpara.txt` to set user-defined parameters. Current available parameters and values are listed in the following form:

| Parameter | Value |
| --------- | ----- |
| `sigma`   | The shift sigma $A+\sigma I$. Can be any real number. Needs to be nonzero for solving linear system.|
| `mu0` | Initial guess of eigenvalue. Can be any real number. No effect when `-e` is set to `0`. |
| `eigtol` | Tolerance for eigensolver.  Can be any real number. No effect when `-e` is set to `0`. |
| `eigmaxite` | Maximum number of iterations for the eigensolver. Positive integer. No effect when `-e` is set to `0`. |
| `solver` | Name of the linear solver. Needs to match the `-l` option. If the solver name is an iterative solver, `-l` option must set to `3`. No effect when `-l` is set to `0`. |
| `tol` | Tolerance for cuda direct linear solver. Only take effects when `-l` is set to `1` or `2`. When the value is set to `default`, then `tol = 1e-12` |
| `atol` | Absolute residual for iterative linear solver. Only take effects when `-l` is set to `3` |
| `rtol` | Relative residual for iterative linear solver. Only take effects when `-l` is set to `3` |
| `maxiter` | Set an upper limit for the iteration count. Positive integer. Only take effects when `-l` is set to `3` |
| `precond` | Possibility to choose a preconditioner. Only take effects when `-l` is set to `3`|
| `restart` | For GMRES: possibility to choose the restart.|                                       
|           | For IDR: Number of distinct subspaces (1,2,4,8). 
|           | Only take effects when `-l` is set to `3` and `solver` is set to `GMRES` or `IDR` |

### __3D face animation__

1. Modify `mu0` to change the initial guess of eigenvalue.
2. Modify `flag` to choose solver on GPU or CPU. Possible options are
	* `'H'`: solver on host &nbsp;&nbsp;&nbsp;(CPU)
	* `'D'`: solver on device    (GPU)
	* Note it is assigned by command-line.

All parameters mentioned above are in the file `main_3Dface_evp.cpp`.