## Usage
You may need to load the required libraries first before building the program. Execute the following commands in a terminal:  
	```
	module load intel-mkl  
	module load cuda-dev/8.0  
	module load magma-dev/2.2f  
	```
## Build Spectral Graph Patition

* `make sgp.out` for __Spectral Graph Patition__  
## Command 

`./sgp.out [OPTIONS]`  

The following are current possible options:  

```
Options:
  -h,       --help           Display this information
  -f<file>, --file <file>    The graph data file
  -e<num>,  --evp <num>      0: None (default), 1: Host, 2: Device
  -l<num>,  --ls <num>       0: None, 1: Direct Host, 2: Direct Device(default), 3: Iterative
  -m<mu0>,  --mu0 <mu0>      The initial mu0 (default: 1.5)
  -s"solver_settings", --magmasolver "solver_settings" default: "--solver CG"
  --shift_sigma <value>,     The value of A+sigma*I
  --eigtol <value>,          The tolerance of eigsolver
  --lstol <value>,           The tolerance of direct lssover (not magma_solver)
  --eigmaxiter <iter>,       The maximum iteration of eigsolver
  --lssolver <num>,          0: LU (default), 1: Cholesky, 2: QR 
``` 
## Solver Settings
The magma solver settings is in [SolverSettings.md](SolverSettings.md)

## Example

1. `./surfpara_magma.out -f data/obj/CYHo.obj -o CYHo_surf.obj`  
    Compute Surface Parameterization of CYHo.obj and output is CYHo_surf.obj  
2. `./surfpara_magma.out -f data/obj/CYHo.obj -s "--solver BICGSTAB"`  
    Compute Surface Parameterization of CYHo.obj with BICGSTAB solver  
3. `./surfpara_magma.out -f data/obj/CYHo.obj -s "--solver PBICGSTAB --precond ILU"`  
    Compute with PBICGSTAB solver with ILU preconditioner.
4. `./surfpara_magma.out -f data/obj/CYHo.obj -e 1 -m 3`  
    Compute the inverse power method with mu0 = 3 on HOST.  


* For graph laplacian, the usage is

	`./sgp_main.out [OPTIONS]`
	
	The following are current possible options:
	
	```
	-h,       --help          Display this information
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
