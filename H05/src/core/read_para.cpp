////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    read_para.cpp
/// @brief   Read parameter settings from file
///
/// @author  William Liao
///

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "sgp.hpp"

using namespace std;

void readParaDEVP(const char *parafile,
	double &shift_sigma,
	double &mu0,
	double &eigtol,
	int &eigmaxite,
	LSOLVER &solflag,
	double &tol){
	fstream pfile;
	string str, str1, str2, str3;
	int count = 0, n = 0;

	pfile.open(parafile, ios::in);
    assert( pfile );

    while( !pfile.eof() && pfile.peek()!=EOF ){
    	pfile >> str1;
    	if ( str1 == "#" )
    	{
    		pfile.ignore(4096, '\n');
    	}else if ( str1 == "%" )
    	{
    		pfile >> str2;
    		if ( str2 == "EVP" )
    		{
    			pfile.ignore(4096, '\n');
    			pfile >> str3;
    			if ( str3 == "mu0" )
    			{
    				pfile >> mu0;
    				pfile.ignore(4096, '\n');
    			}else if ( str3 == "eigtol" )
    			{
    				pfile >> eigtol;
    				pfile.ignore(4096, '\n');
    			}else if ( str3 == "eigmaxite")
    			{
    				pfile >> eigmaxite;
    				pfile.ignore(4096, '\n');
    			}else{
    				cout << "Unknown parameter!" << endl;
    				abort();
    			}
    		}else if ( str2 == "LS" )
    		{
    			pfile.ignore(4096, '\n');
    			pfile >> str3;
    			if ( str3 == "solver" )
    			{
    				pfile >> str;
    				if ( str == "LU" )
    				{
    					solflag = LSOLVER::LU;
    				}else if ( str == "CHOL" )
    				{
    					solflag = LSOLVER::CHOL;
    				}else if ( str == "QR" )
    				{
    					solflag = LSOLVER::QR;
    				}else{
    					cout << "Invalid option!" << endl;
    					abort();
    				}
    				pfile.ignore(4096, '\n');
    			}else{
    				cout << "Unknown parameter!" << endl;
    				abort();
    			}
    		}else if ( str2 == "DLS" )
    		{
    			pfile.ignore(4096, '\n');
    			pfile >> str3;
    			if ( str3 == "tol" )
    			{
    				pfile >> str;
    				if ( str == "default" )
    				{
    					tol = 1e-12;
    				}else{
    					tol = stod(str);
    				}
    			}else{
    				cout << "Unknown parameter!" << endl;
    				abort();
    			}
    		}else if ( str2 == "ILS" )
    		{
    			pfile >> count;
    			pfile.ignore(4096, '\n');
    			for (n = 0; n < count; n++)
    			{
    				pfile.ignore(4096, '\n');
    			}
    		}
    	}else{
    		if ( str1 == "mu0" )
			{
				pfile >> mu0;
			}else if ( str1 == "eigtol" )
			{
				pfile >> eigtol;
			}else if ( str1 == "eigmaxite")
			{
				pfile >> eigmaxite;
			}else if ( str1 == "sigma")
			{
				pfile >> shift_sigma;
			} else{
				cout << "Unknown parameter!" << endl;
				abort();
			}
    	}
    }
    pfile.close();
}

void readParaIEVP(const char *parafile,
	double &shift_sigma,
	double &mu0,
	double &eigtol,
	int &eigmaxite,
	std::string &solver,
	std::string &atol,
	std::string &rtol,
	std::string &maxiter,
	std::string &precond,
	std::string &restart){
	fstream pfile;
	string str, str1, str2, str3;
	int count = 0, n = 0;

	pfile.open(parafile, ios::in);
    assert( pfile );

    while( !pfile.eof() && pfile.peek()!=EOF ){
    	pfile >> str1;
    	if ( str1 == "#" )
    	{
    		pfile.ignore(4096, '\n');
    	}else if ( str1 == "%" )
    	{
    		pfile >> str2;
    		if ( str2 == "EVP" )
    		{
    			pfile.ignore(4096, '\n');
    			pfile >> str3;
    			if ( str3 == "mu0" )
    			{
    				pfile >> mu0;
    				pfile.ignore(4096, '\n');
    			}else if ( str3 == "eigtol" )
    			{
    				pfile >> eigtol;
    				pfile.ignore(4096, '\n');
    			}else if ( str3 == "eigmaxite")
    			{
    				pfile >> eigmaxite;
    				pfile.ignore(4096, '\n');
    			}else{
    				cout << "Unknown parameter!" << endl;
    				abort();
    			}
    		}else if ( str2 == "LS" )
    		{
    			pfile.ignore(4096, '\n');
    			pfile >> str3;
    			if ( str3 == "solver" )
    			{
    				pfile >> solver;
    				pfile.ignore(4096, '\n');
    			}else{
    				cout << "Unknown parameter!" << endl;
    				abort();
    			}
    		}else if ( str2 == "DLS" )
    		{
    			pfile >> count;
    			pfile.ignore(4096, '\n');
    			for (n = 0; n < count; n++)
    			{
    				pfile.ignore(4096, '\n');
    			}
    		}else if ( str2 == "ILS" )
    		{
    			pfile.ignore(4096, '\n');
    			pfile >> str3;
    			if ( str3 == "atol" )
    			{
    				pfile >> str;
    				if ( str == "default" )
    				{
    					atol = "";
    				}else{
    					atol = str;
    				}
    			}else if ( str3 == "rtol" )
    			{
    				pfile >> str;
    				if ( str == "default" )
    				{
    					rtol = "";
    				}else{
    					rtol = str;
    				}
    			}else if ( str3 == "maxiter" )
    			{
    				pfile >> str;
    				if ( str == "default" )
    				{
    					maxiter = "";
    				}else{
    					maxiter = str;
    				}
    			}else if ( str3 == "precond" )
    			{
    				pfile >> precond;
    			}else if ( str3 == "restart" )
    			{
    				pfile >> str;
    				if ( str == "default" )
    				{
    					restart = "";
    				}else{
    					restart = str;
    				}
    			}else{
    				cout << "Unknown parameter!" << endl;
    				abort();
    			}
    		}
    	}else{
    		if ( str1 == "mu0" )
			{
				pfile >> mu0;
			}else if ( str1 == "eigtol" )
			{
				pfile >> eigtol;
			}else if ( str1 == "eigmaxite")
			{
				pfile >> eigmaxite;
			}else if ( str1 == "sigma")
			{
				pfile >> shift_sigma;
			}else if ( str3 == "atol" )
			{
				pfile >> str;
				if ( str == "default" )
				{
					atol = "";
				}else{
					atol = str;
				}
			}else if ( str3 == "rtol" )
			{
				pfile >> str;
				if ( str == "default" )
				{
					rtol = "";
				}else{
					rtol = str;
				}
			}else if ( str3 == "maxiter" )
			{
				pfile >> str;
				if ( str == "default" )
				{
					maxiter = "";
				}else{
					maxiter = str;
				}
			}else if ( str3 == "precond" )
			{
				pfile >> precond;
			}else if ( str3 == "restart" )
			{
				pfile >> str;
				if ( str == "default" )
				{
					restart = "";
				}else{
					restart = str;
				}
			} else{
				cout << "Unknown parameter!" << endl;
				abort();
			}
    	}
    }
    pfile.close();
}

void readParaEVP(const char *parafile,
	double &shift_sigma,
	double &mu0,
	double &eigtol,
	int &eigmaxite){
	fstream pfile;
	string str, str1, str2, str3;
	int count = 0, n = 0;

	pfile.open(parafile, ios::in);
    assert( pfile );

    while( !pfile.eof() && pfile.peek()!=EOF ){
    	pfile >> str1;
    	if ( str1 == "#" )
    	{
    		pfile.ignore(4096, '\n');
    	}else if ( str1 == "%" )
    	{
    		pfile >> str2;
    		if ( str2 == "EVP" )
    		{
    			pfile.ignore(4096, '\n');
    			pfile >> str3;
    			if ( str3 == "mu0" )
    			{
    				pfile >> mu0;
    				pfile.ignore(4096, '\n');
    			}else if ( str3 == "eigtol" )
    			{
    				pfile >> eigtol;
    				pfile.ignore(4096, '\n');
    			}else if ( str3 == "eigmaxite")
    			{
    				pfile >> eigmaxite;
    				pfile.ignore(4096, '\n');
    			}else{
    				cout << "Unknown parameter!" << endl;
    				abort();
    			}
    		}else if ( str2 == "LS" )
    		{
    			pfile >> count;
    			pfile.ignore(4096, '\n');
    			for (n = 0; n < count; n++)
    			{
    				pfile.ignore(4096, '\n');
    			}
    		}else if ( str2 == "DLS" )
    		{
    			pfile >> count;
    			pfile.ignore(4096, '\n');
    			for (n = 0; n < count; n++)
    			{
    				pfile.ignore(4096, '\n');
    			}
    		}else if ( str2 == "ILS" )
    		{
    			pfile >> count;
    			pfile.ignore(4096, '\n');
    			for (n = 0; n < count; n++)
    			{
    				pfile.ignore(4096, '\n');
    			}
    		}
    	}else{
    		if ( str1 == "mu0" )
			{
				pfile >> mu0;
			}else if ( str1 == "eigtol" )
			{
				pfile >> eigtol;
			}else if ( str1 == "eigmaxite")
			{
				pfile >> eigmaxite;
			}else if ( str1 == "sigma")
			{
				pfile >> shift_sigma;
			} else{
				cout << "Unknown parameter!" << endl;
				abort();
			}
    	}
    }
    pfile.close();
}

void readParaDLS(const char *parafile,
	double &shift_sigma,
	LSOLVER &solflag,
	double &tol){
	fstream pfile;
	string str, str1, str2, str3;
	int count = 0, n = 0;

	pfile.open(parafile, ios::in);
    assert( pfile );

    while( !pfile.eof() && pfile.peek()!=EOF ){
    	pfile >> str1;
    	if ( str1 == "#" )
    	{
    		pfile.ignore(4096, '\n');
    	}else if ( str1 == "%" )
    	{
    		pfile >> str2;
    		if ( str2 == "EVP" )
    		{
    			pfile >> count;
    			pfile.ignore(4096, '\n');
    			for (n = 0; n < count; n++)
    			{
    				pfile.ignore(4096, '\n');
    			}
    		}else if ( str2 == "LS" )
    		{
    			pfile.ignore(4096, '\n');
    			pfile >> str3;
    			if ( str3 == "solver" )
    			{
    				pfile >> str;
    				if ( str == "LU" )
    				{
    					solflag = LSOLVER::LU;
    				}else if ( str == "CHOL" )
    				{
    					solflag = LSOLVER::CHOL;
    				}else if ( str == "QR" )
    				{
    					solflag = LSOLVER::QR;
    				}else{
    					cout << "Invalid option!" << endl;
    					abort();
    				}
    				pfile.ignore(4096, '\n');
    			}else{
    				cout << "Unknown parameter!" << endl;
    				abort();
    			}
    		}else if ( str2 == "DLS" )
    		{
    			pfile.ignore(4096, '\n');
    			pfile >> str3;
    			if ( str3 == "tol" )
    			{
    				pfile >> str;
    				if ( str == "default" )
    				{
    					tol = 1e-12;
    				}else{
    					tol = stod(str);
    				}
    			}else{
    				cout << "Unknown parameter!" << endl;
    				abort();
    			}
    		}else if ( str2 == "ILS" )
    		{
    			pfile >> count;
    			pfile.ignore(4096, '\n');
    			for (n = 0; n < count; n++)
    			{
    				pfile.ignore(4096, '\n');
    			}
    		}
    	}else{
    		if ( str1 == "sigma")
			{
				pfile >> shift_sigma;
			} else{
				cout << "Unknown parameter!" << endl;
				abort();
			}
    	}
    }
    pfile.close();
}

void readParaILS(const char *parafile,
	double &shift_sigma,
	std::string &solver,
	std::string &atol,
	std::string &rtol,
	std::string &maxiter,
	std::string &precond,
	std::string &restart){
	fstream pfile;
	string str, str1, str2, str3;
	int count = 0, n = 0;

	pfile.open(parafile, ios::in);
    assert( pfile );

    while( !pfile.eof() && pfile.peek()!=EOF ){
    	pfile >> str1;
    	if ( str1 == "#" )
    	{
    		pfile.ignore(4096, '\n');
    	}else if ( str1 == "%" )
    	{
    		pfile >> str2;
    		if ( str2 == "EVP" )
    		{
    			pfile >> count;
    			pfile.ignore(4096, '\n');
    			for (n = 0; n < count; n++)
    			{
    				pfile.ignore(4096, '\n');
    			}
    		}else if ( str2 == "LS" )
    		{
    			pfile.ignore(4096, '\n');
    			pfile >> str3;
    			if ( str3 == "solver" )
    			{
    				pfile >> solver;
    				cout << "solver in para: " << solver << endl;
    				pfile.ignore(4096, '\n');
    			}else{
    				cout << "Unknown parameter!" << endl;
    				abort();
    			}
    		}else if ( str2 == "DLS" )
    		{
    			pfile >> count;
    			pfile.ignore(4096, '\n');
    			for (n = 0; n < count; n++)
    			{
    				pfile.ignore(4096, '\n');
    			}
    		}else if ( str2 == "ILS" )
    		{
    			pfile.ignore(4096, '\n');
    			pfile >> str3;
    			if ( str3 == "atol" )
    			{
    				pfile >> str;
    				if ( str == "default" )
    				{
    					atol = "";
    				}else{
    					atol = str;
    				}
    			}else if ( str3 == "rtol" )
    			{
    				pfile >> str;
    				if ( str == "default" )
    				{
    					rtol = "";
    				}else{
    					rtol = str;
    				}
    			}else if ( str3 == "maxiter" )
    			{
    				pfile >> str;
    				if ( str == "default" )
    				{
    					maxiter = "";
    				}else{
    					maxiter = str;
    				}
    			}else if ( str3 == "precond" )
    			{
    				pfile >> precond;
    			}else if ( str3 == "restart" )
    			{
    				pfile >> str;
    				if ( str == "default" )
    				{
    					restart = "";
    				}else{
    					restart = str;
    				}
    			}else{
    				cout << "Unknown parameter!" << endl;
    				abort();
    			}
    		}
    	}else{
    		if ( str1 == "sigma")
			{
				pfile >> shift_sigma;
			}else if ( str3 == "atol" )
			{
				pfile >> str;
				if ( str == "default" )
				{
					atol = "";
				}else{
					atol = str;
				}
			}else if ( str3 == "rtol" )
			{
				pfile >> str;
				if ( str == "default" )
				{
					rtol = "";
				}else{
					rtol = str;
				}
			}else if ( str3 == "maxiter" )
			{
				pfile >> str;
				if ( str == "default" )
				{
					maxiter = "";
				}else{
					maxiter = str;
				}
			}else if ( str3 == "precond" )
			{
				pfile >> precond;
			}else if ( str3 == "restart" )
			{
				pfile >> str;
				if ( str == "default" )
				{
					restart = "";
				}else{
					restart = str;
				}
			} else{
				cout << "Unknown parameter!" << endl;
				abort();
			}
    	}
    }
    pfile.close();
}