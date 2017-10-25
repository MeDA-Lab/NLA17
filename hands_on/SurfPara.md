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
  -f<file>, --file <file>    The Object file (default: input.obj)
  -o<file>, --output <file>  The Output file (default: output.obj)
  -t<num>,  --target <num>   0: LS   - Linear System (Lii Ui = Lib Ub) (default)
                             1: SIPM - Shift Inverse Power Method
  -s"solver_settings",      --magmasolver "solver_settings"
                        default settings: "--solver CG" for Iterative Linear System
  --method <num>        Laplacian matrix, 0: KIRCHHOFF (default) 1: COTANGENT
  --tol <num>           Tolerance of Direct Eigensolver or Linear System Solver
  --sigma <value>       SIPM: as mu0 (default 1.5)
  --eig_maxiter <value> The maximum iteration of eigensolver (default: 1000)
  --sipm_option <num>   0: Host(default) 1: Device
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
4. `./surfpara_magma.out -t1 -f data/obj/CYHo_simplified.obj --sipm_option 1 --sigma 3`  
    Compute the shift inverse power method with sigma(mu0) = 3 on Device (100 secs).  