 --solver      Possibility to choose a solver:
               CG, PCG, BICGSTAB, PBICGSTAB, GMRES, PGMRES, LOBPCG, JACOBI,
               BAITER, IDR, PIDR, CGS, PCGS, TFQMR, PTFQMR, QMR, PQMR, BICG,
               PBICG, BOMBARDMENT, ITERREF.
 --restart     For GMRES: possibility to choose the restart.
               For IDR: Number of distinct subspaces (1,2,4,8).
 --atol x      Set an absolute residual stopping criterion.
 --verbose x   Possibility to print intermediate residuals every x iteration.
 --maxiter x   Set an upper limit for the iteration count.
 --rtol x      Set a relative residual stopping criterion.
 --precond x   Possibility to choose a preconditioner:
               CG, BICGSTAB, GMRES, LOBPCG, JACOBI,
               BAITER, IDR, CGS, TFQMR, QMR, BICG
               BOMBARDMENT, ITERREF, ILU, PARILU, PARILUT, NONE.
                   --patol atol  Absolute residual stopping criterion for preconditioner.
                   --prtol rtol  Relative residual stopping criterion for preconditioner.
                   --piters k    Iteration count for iterative preconditioner.
                   --plevels k   Number of ILU levels.
                   --triolver k  Solver for triangular ILU factors: e.g. CUSOLVE, JACOBI, ISAI.
                   --ppattern k  Pattern used for ISAI preconditioner.
                   --psweeps x   Number of iterative ParILU sweeps.
 --trisolver   Possibility to choose a triangular solver for ILU preconditioning: 
               e.g. CUSOLVE, ISPTRSV, JACOBI, VBJACOBI, ISAI.
 --ppattern k  Possibility to choose a pattern for the trisolver: ISAI(k) or Block Jacobi.
 --piters k    Number of preconditioner relaxation steps, e.g. for ISAI or (Block) Jacobi trisolver.
 --patol x     Set an absolute residual stopping criterion for the preconditioner.
                      Corresponds to the relative fill-in in PARILUT.
 --prtol x     Set a relative residual stopping criterion for the preconditioner.
                      Corresponds to the replacement ratio in PARILUT.