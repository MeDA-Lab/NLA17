////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    cuda_solverinfo.cpp
/// @brief   The main function.
///
/// @author  William Liao
///

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <string>

using namespace std;

int cudasolverinfo(int flag, int solver){
	if ( flag == 1 )
	{
		if ( solver == 0 )
		{
			cout << "Solver: CUDA LU Host" << endl;
		}else if ( solver == 1 )
		{
			cout << "Solver: CUDA Cholesky Host" << endl;
		}else if ( solver == 2 )
		{
			cout << "Solver: CUDA QR Host" << endl;
		}
	}else if ( flag == 2 )
	{
		if ( solver == 0 )
		{
			cout << "Solver: CUDA LU Device" << endl;
		}else if ( solver == 1 )
		{
			cout << "Solver: CUDA Cholesky Device" << endl;
		}else if ( solver == 2 )
		{
			cout << "Solver: CUDA QR Device" << endl;
		}
	}
}