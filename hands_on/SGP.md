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