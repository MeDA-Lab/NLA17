////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    sgp_main.cpp
/// @brief   The main function.
///
/// @author  William Liao
///

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <string>
#include <timer.hpp>
#include "sgp.hpp"

using namespace std;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  Main function for spectral graph partitioning.
///
int main( int argc, char** argv ){
    const char *input = NULL;
    const char *parafile = NULL;
    const char *solver = "CHOL"; // type of linear solver
    LSOLVER solflag = LSOLVER::CHOL;
    Method method;
    EVP evp = EVP::NONE;
    LS  ls  = LS::DEVICE;
    // Flags to check certain conditions
	int tflag, pflag, err_test;

	// Read arguments
    readArgs(argc, argv, input, parafile, method, evp, ls, tflag, pflag);
    assert( (evp != EVP::NONE) || (ls != LS::NONE) );

    // Read file
    int E_size_r, E_size_c, *E;
    cout << "Read the graph data from file..............." << flush;
    err_test = readGraph(input, &E, &E_size_r, &E_size_c);
    assert( err_test == 0 ); cout << " Done.  " << endl;
    cout << "Size of data is " << E_size_r << "x" << E_size_c << endl;

    // Set graph type
    char flag1;
    if ( tflag == 0 )
    {
        setgraphtype(method, E_size_c);
    }

    if ( method == Method::SIMPLE )
    {
        flag1 = 'S';
        cout << "type of graph: simple graph" << endl;
    }else if( method == Method::DIRECTED ){
        flag1 = 'D';
        cout << "type of graph: directed (multi) graph" << endl;
    }else if ( method == Method::WEIGHTED ){
        flag1 = 'W';
        cout << "type of graph: directed weighted graph" << endl;
    }else if ( method == Method::UW )
    {
        flag1 = 'U';
        cout << "type of graph: undirected weighted graph" << endl;
    }

    // Construct adjacency matrix of graph
    int nnz, *cooRowIndA, *cooColIndA, n;
    double *cooValA;
    cout << "Construct adjacency matrix of graph........." << flush;
    err_test = GraphAdjacency(E, E_size_r, &nnz, &cooRowIndA, &cooColIndA, &cooValA, &n, flag1);
    assert( err_test == 0 ); cout << " Done.  " << endl;
    cout << "size of matrix = " << n << endl;
    cout << "nnz of A = " << nnz << endl;

    // Setting parameters for Laplacian and solvers
    double shift_sigma = 1e-5; // Modify shift_sigma to set the
                               // shift
    // Parameters for eigensolver
    double mu0, eigtol;
    int eigmaxite;
    // Parameters for direct linear solver
    double tol;
    // Parameters for iterative linear solver
    string atol, rtol, maxiter, precond, restart;
    cout << "Setting Laplacian and solver parameters....." << flush;
    if ( evp != EVP::NONE && ls != LS::NONE )
    {
        mu0 = 0.6; // Modify mu0 to change the initial
                   // guess of eigenvalue
        if ( pflag == 1 )
        {
            readPara(parafile, shift_sigma, mu0, eigtol, eigmaxite, solflag, solver, tol, atol, rtol, maxiter, precond, restart);
        }
    }else if ( evp != EVP::NONE && ls == LS::NONE )
    {
        mu0 = 0.6; // Modify mu0 to change the initial
                   // guess of eigenvalue
        if ( pflag == 1 )
        {
            readPara(parafile, shift_sigma, mu0, eigtol, eigmaxite);
        }
    }else if ( evp == EVP::NONE && ls != LS::NONE )
    {
        if ( pflag == 1 )
        {
            readPara(parafile, shift_sigma, solflag, solver, tol, atol, rtol, maxiter, precond, restart);
        }
    }
    cout << " Done.  " << endl;

    // Construct Laplacian
    int *csrRowIndA, *csrColIndA;
    double  *csrValA;
    cout << "Construct Laplacian matrix of graph........." << flush;
    GraphLaplacian(&nnz, cooRowIndA, cooColIndA, cooValA, n, &csrRowIndA, &csrColIndA, &csrValA, shift_sigma);
    cout << " Done.  " << endl;
    cout << "nnz of L = " << nnz << endl;

    // Shift to zero-based indexing
    int tmp;
    for (int i = 0; i < nnz; i++)
    {
    	tmp = csrColIndA[i]-1;
        csrColIndA[i] = tmp;
    }
    for (int i = 0; i < n+1; i++)
    {
    	tmp = csrRowIndA[i]-1;
        csrRowIndA[i] = tmp;
    }

    if ( ls != LS::NONE )
    {
        // Generate RHS
        double *b;
        b = new double[n];
        genRHS(b, n, nnz, csrValA, csrRowIndA, csrColIndA);

        // Solve LS
        double *x, timer;
        x = new double[n];

        cudasolverinfo(flag, solver);
        cout << "Solving Linear System......................." << flush;

        switch (ls){
            case LS::HOST:
                solver = static_cast<int>(solflag);
                if ( pflag == 0 )
                {
                    tic(&timer);
                    solvelsHost(n, nnz, csrValA, csrRowIndA, csrColIndA, b, x, solver);
                    cout << " Done.  ";
                    toc(&timer);
                }else if ( pflag == 1 )
                {
                    tic(&timer);
                    solvelsHostCust(n, nnz, csrValA, csrRowIndA, csrColIndA, b, x, solver, tol);
                    cout << " Done.  ";
                    toc(&timer);
                }
                // Compute redsidual
                double res;
                res = residual(n, nnz, csrValA, csrRowIndA, csrColIndA, b, x);
                cout << "||Ax - b|| =  "  << res << endl;
                break;
            case LS::DEVICE:
                solver = static_cast<int>(solflag);
                if ( pflag == 0 )
                {
                    tic(&timer);
                    solvels(n, nnz, csrValA, csrRowIndA, csrColIndA, b, x, solver); cout << " Done.  ";
                    toc(&timer);
                }else if ( pflag == 1 )
                {
                    tic(&timer);
                    solvelsCust(n, nnz, csrValA, csrRowIndA, csrColIndA, b, x, solver, tol);
                    cout << " Done.  ";
                    toc(&timer);
                }
                // Compute redsidual
                double res;
                res = residual(n, nnz, csrValA, csrRowIndA, csrColIndA, b, x);
                cout << "||Ax - b|| =  "  << res << endl;
                break;
            case LS::ITERATIVE:
                if ( pflag == 0 )
                {
                    solveGraph(n, nnz, csrValA, csrRowIndA, csrColIndA, b, x);
                }else if ( pflag == 1 )
                {
                    solveGraphCust(n, nnz, csrValA, csrRowIndA, csrColIndA, b, x, solver, tol, atol, rtol, maxiter, precond, restart);
                }
                break;
        }
    }

    return 0;
}