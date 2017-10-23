////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    sgp.cpp
/// @brief   The main function.
///
/// @author  William Liao
/// @author  Yuhsiang Mike Tsai

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
    args setting;
    setting.eigmaxiter = 1000;
    setting.eigtol = 1e-12;
    setting.evp = EVP::NONE;
    setting.ls = LS::DEVICE;
    setting.file = NULL;
    setting.mu0 = 0.6;
    setting.shift_sigma = 1e-5;
    setting.solver_settings = "--solver CG";
    setting.tol = 1e-12;
    setting.lsover = LSOLVER::LU;
    // Flags to check certain conditions
	// Read arguments
    readArgs(argc, argv, &setting);
    assert( (setting.evp != EVP::NONE) || (setting.ls != LS::NONE) );

    // Read file
    int E_size_r, E_size_c, *E;
    double *W;
    Network network_type = Network::UNDEFINED;
    Edge edge_type = Edge::UNDEFINED;
    cout << "Read the graph data from file..............." << flush;
    readGraph(setting.file, &E_size_r, &E_size_c, &E, &W, &network_type, &edge_type);
    cout << " Done.  " << endl;
    cout << "Size of data is " << E_size_r << "x" << E_size_c << endl;
    printKonectHeader(network_type, edge_type);

    // Construct adjacency matrix of graph
    int nnz, *cooRowIndA, *cooColIndA, n;
    double *cooValA;
    cout << "Construct adjacency matrix of graph........." << flush;
    GraphAdjacency(E_size_r, E, W, &n, &nnz,
        &cooValA, &cooRowIndA, &cooColIndA);
    cout << " Done.  " << endl;
    cout << "size of matrix = " << n << endl;
    cout << "nnz of A = " << nnz << endl;


    // Construct Laplacian
    int *csrRowIndA, *csrColIndA;
    double  *csrValA;
    cout << "Construct Laplacian matrix of graph........." << flush;
    GraphLaplacian(&nnz, cooRowIndA, cooColIndA, cooValA, n, &csrRowIndA, &csrColIndA, &csrValA, setting.shift_sigma);
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
    int ev_num = 5;
    double *eigval = new double[ev_num], *eigvec = new double[ev_num*n];
    solveSMEVP(ev_num, n, nnz, csrValA, csrRowIndA, csrColIndA, eigval, eigvec);
    if ( setting.ls != LS::NONE )
    {
        // Generate RHS
        double *b;
        b = new double[n];
        genRHS(b, n, nnz, csrValA, csrRowIndA, csrColIndA);

        // Solve LS
        double *x, timer;
        double res;
        int solverid;
        x = new double[n];

        if ( setting.ls != LS::ITERATIVE )
        {
            solverid = static_cast<int>(setting.lsover);
            cudasolverinfo(static_cast<int>(setting.ls), solverid);
        }
        cout << "Solving Linear System......................." << flush;

        switch (setting.ls ){
            case LS::HOST:
                tic(&timer);
                solvelsHostCust(n, nnz, csrValA, csrRowIndA, csrColIndA, b, x, solverid, setting.tol);
                cout << " Done.  ";
                toc(&timer);
                // Compute redsidual
                res = residual(n, nnz, csrValA, csrRowIndA, csrColIndA, b, x);
                cout << "||Ax - b|| =  "  << res << endl;
                break;
            case LS::DEVICE:
                tic(&timer);
                solvelsCust(n, nnz, csrValA, csrRowIndA, csrColIndA, b, x, solverid, setting.tol);
                cout << " Done.  ";
                toc(&timer);
                
                // Compute redsidual
                res = residual(n, nnz, csrValA, csrRowIndA, csrColIndA, b, x);
                cout << "||Ax - b|| =  "  << res << endl;
                break;
            case LS::ITERATIVE:
                solveGraph(setting.solver_settings, n, nnz, csrValA, csrRowIndA, csrColIndA, b, x);
                break;
        }
    }

    if ( setting.evp != EVP::NONE )
    {
        // Solve EVP
        double mu;
        double *x, timer;
        x = new double[n];

        cout << "Solving Eigenvalue Problem.................." << flush;

        switch (setting.evp){
            case EVP::HOST:
                tic(&timer);
                solveShiftEVPHost(n, nnz, csrValA, csrRowIndA, csrColIndA, setting.mu0, setting.eigmaxiter, setting.eigtol, &mu, x);
                cout << " Done.  ";
                toc(&timer);
                break;
            case EVP::DEVICE:
                tic(&timer);
                solveShiftEVP(n, nnz, csrValA, csrRowIndA, csrColIndA, setting.mu0, setting.eigmaxiter, setting.eigtol, &mu, x);
                cout << " Done.  ";
                toc(&timer);
                break;
        }

        cout << "The estimated eigenvalue near "  << setting.mu0 << " = ";
        cout << fixed << setprecision(13) << mu << endl;
    }

    return 0;
}