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
  -f<file>, --file <file>    The graph data file (defalut: input.obj)
  -o<file>, --output <file>  The Output file (default: output.obj)
  -t<num>,  --target <num>   0: LOBPCG (solve some smallest eigenvectors) (default)
                             1: SIPM - Shift Inverse Power Method
                             2: LS   - Linear System (A+sigmaI)
  -s"solver_settings",    --magmasolver "solver_settings"
                        default settings: "--solver CG" for Iterative Linear System
                                          "--solver LOBPCG --ev 4 --precond ILU" for LOBPCG
  --tol <num>           Tolerance of Direct Eigensolver or Linear System Solver
  --sigma <value>       SIPM: as mu0, LS: as shift element (default: 0)
  --eig_maxiter <value> The maximum iteration of eigensolver (default: 1000)
  --sipm_option <num>   0: Host(default) 1: Device
  --ls_option <num>     Iterative - 0: MAGMA Iterative solver(default)
                        Direct    - 1: HOST_QR   2:HOST_CHOL   3: HOST_LU
                                    4: DEVICE_QR 5:DEVICE_CHOL
``` 
## Solver Settings
The magma solver settings is in [SolverSettings.md](SolverSettings.md)

## Example

1. `./sgp.out -f data/graph/butterfly.txt`  
    Solve SGP of Butterfly.txt
2. `./sgp.out -f data/graph/out.moreno_zebra_zebra -s "--solver LOBPCG --ev 5"`  
	Solve SGP of zebra
3. `./sgp.out -t2 -f data/graph/moreno_kangaroo_kangaroo --sigma 1e-5`  
	Solve linear system A=(L+sigma*I) with iterative magma solver 
4. `./sgp.out -t2 -f data/graph/moreno_kangaroo_kangaroo --sigma 1e-5 -s "--solver BICGSTAB"`  
    Solve linear system A=(L+sigma*I) with iterative magma solver with BICGSTAB solver  
5. `./sgp.out -f data/graph/moreno_kangaroo_kangaroo -l 3 -s "--solver PBICGSTAB --precond ILU"`  
    Compute with PBICGSTAB solver with ILU preconditioner.  
6.  `./sgp.out -t1 -f data/graph/moreno_kangaroo_kangaroo --sigma 0.6`  
    Compute Shift Invese Power Method with sigma(mu0) 0.6