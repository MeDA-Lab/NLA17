## Data
You can download __undirected__ network type with __unweighted__, __positive weighted__,  __rating__ edge type from [KONECT](http://konect.uni-koblenz.de).  
Note. You do not fix anything of the graph now.  
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
  -m<mu0>,  --mu0 <mu0>      The initial mu0 (default: 0.6)
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

1. `./sgp.out -f data/graph/ChicagoRegional -e 2`  
    Solve eigenvalue problem on device
2. `./sgp.out -f data/graph/ChicagoRegional`  
	Solve linear system with direct device solver  
3. `./sgp.out -f data/graph/moreno_kangaroo_kangaroo -l 3`
	Solve linear system with iterative magma solver
4. `./sgp.out -f data/graph/moreno_kangaroo_kangaroo -l 3 -s "--solver BICGSTAB"`  
    Solve linear system with iterative magma solver with BICGSTAB solver  
5. `./sgp.out -f data/graph/moreno_kangaroo_kangaroo -l 3 -s "--solver PBICGSTAB --precond ILU"`  
    Compute with PBICGSTAB solver with ILU preconditioner.  