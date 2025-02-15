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
#include <timer.hpp>
#include "sgp.hpp"

using namespace std;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  Main function for spectral graph partitioning.
///
int main( int argc, char** argv ){
	int err_test;

	// need at least 2 argument!
    assert( argc >= 2 );

    // check 2nd argument
    assert( argv[1] != NULL );

    // read file
    int E_size_r, E_size_c, *E;
    cout << "Read the graph data from file..............." << flush;
    err_test = readGraph(argv[1], &E, &E_size_r, &E_size_c);
    assert( err_test == 0 ); cout << " Done.  " << endl;
    cout << "Size of data is " << E_size_r << "x" << E_size_c << endl;

    // set graph type
    int type;
    char flag1;
    if ( argc == 2 )
    {
        type = setgraphtype(E_size_c);
    }else if( argc == 3 ){
        type = setgraphtype(argv[2], E_size_c);
    }

    if ( type == 0 )
    {
        flag1 = 'S';
        cout << "type of graph: simple graph" << endl;
    }else if( type == 1 ){
        flag1 = 'D';
        cout << "type of graph: directed (multi) graph" << endl;
    }else if ( type == 2 ){
        flag1 = 'W';
        cout << "type of graph: directed weighted graph" << endl;
    }else if ( type == 3 )
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

    // Construct Laplacian
    int *csrRowIndA, *csrColIndA;
    double  *csrValA;
    double shift_sigma = 1e-5; // Modify shift_sigma to set the
                               // shift
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

    // Generate RHS
    double *b;
    b = new double[n];
    genRHS(b, n, nnz, csrValA, csrRowIndA, csrColIndA);

    // Solve LS
    double *x, timer;
    x = new double[n];
    char flag = 'H';      // Modify flag to choose solver on GPU
                          // or CPU. Possible options are
                          // 'H': solver on host    (CPU)
                          // 'D': solver on device  (GPU)
    
    int solver = 0;       // Modify solver to switch between
                          // different linear solvers. Possible
                          // options are
                          // 0: LU
                          // 1: Cholesky
                          // 2: QR

    cout << "Solving Linear System......................." << flush;

    switch (flag){
    	case 'H':
    		tic(&timer);
    		solvelsHost(n, nnz, csrValA, csrRowIndA, csrColIndA, b, x, solver);
            cout << " Done.  ";
    		toc(&timer);
    		break;
    	case 'D':
    		tic(&timer);
    		solvels(n, nnz, csrValA, csrRowIndA, csrColIndA, b, x, solver); cout << " Done.  ";
    		toc(&timer);
    		break;
    }

    // Compute redsidual
    double res;
    res = residual(n, nnz, csrValA, csrRowIndA, csrColIndA, b, x);

    cout << "||Ax - b|| =  "  << res << endl;

    return 0;
}