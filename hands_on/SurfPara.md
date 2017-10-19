## Usage
You may need to load the required libraries first before building the program. Execute the following commands in a terminal:  
	```
	module load intel-mkl  
	module load cuda-dev/8.0  
	module load magma-dev/2.2f  
	```
## Build Surface Parameterization

* `make surfpara.out` for __Surface Parameterization__ without any solvers.  
* `make surfpara_magma.out` for __Surface Parameterization__ with magma solver.  

## Command 

`./surfpara_magma.out [OPTIONS]` or `./surfpara.out [OPTIONS]`  

The following are current possible options:  

```
Options:
  -h,       --help           Display this information
  -f<file>, --file <file>    The graph file (default: input.obj)
  -t<num>,  --type <num>     0: KIRCHHOFF(default), 1: COTANGENT
  -o<file>, --output <file>  The output file (default: output.obj)
  -e<num>,  --evp <num>      0: None(default), 1: Host, 2: Device
  -m<mu0>,  --mu0 <mu0>      The initial mu0 (default: 1.5)
  -s"solver_settings", --magmasolver "solver_settings" default: "--solver CG"
  --eigtol <value>,          The tolerance of eigsolver (default: 1e-12)
  --eigmaxiter <iter>,       The maximum iteration of eigsolver (default: 1000)
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