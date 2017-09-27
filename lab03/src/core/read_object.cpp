////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    read_object.cpp
/// @brief   The implementation of object reading.
///
/// @author  Mu Yang  <<emfomy@gmail.com>>
/// @author  Yuhisang Mike Tsai
///

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <harmonic.hpp>
using namespace std;

void readObject(
    const char *input,
    int *ptr_nv,
    int *ptr_nf,
    double **ptr_V,
    double **ptr_C,
    int **ptr_F
) {

  int &nv = *ptr_nv;
  int &nf = *ptr_nf;
  bool mode = 0; // 0: No color; 1: With color

  // CR to LF
  {
    stringstream buffer;
    buffer << "dos2unix " << input;
    int info = system(buffer.str().c_str());
    if ( info ) {
      cerr << "Unable to convert file \"" << input << "\"!" << endl;
    }
  }

  // Open file
  ifstream fin(input);
  if ( fin.fail() ) {
    cerr << "Unable to open file \"" << input << "\"!" << endl;
    abort();
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Determine vertex mode
  {
    // Skip until first vertex
    while ( fin.peek() != 'v' ) {
      fin.ignore(4096, '\n');
    }
    fin.get();

    // Read first vertex
    string str;
    getline(fin, str);
    istringstream sin(str);
    double v;
    int count = 0;
    while (sin >> v) {
      ++count;
    }

    // Determine mode
    if ( count == 3 ) {
      mode = 0;
      cout << "Loads from \"" << input << "\" without color." << endl;
    } else if ( count == 6 ) {
      mode = 1;
      cout << "Loads from \"" << input << "\" with color." << endl;
    } else {
      cerr << "Unable to load vertex: the number of values must be 3 or 6!" << endl;
      abort();
    }
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Count vertices and faces
  fin.clear();
  fin.seekg(0, ios::beg);
  nv = 0; nf = 0;
  while ( !fin.eof() ) {
    char c = fin.peek();
    if ( c == 'v' ) { ++nv; }
    else if ( c == 'f' ) { ++nf; }
    fin.ignore(4096, '\n');
  }
  cout << "\"" << input << "\" contains " << nv << " vertices and " << nf << " faces." << endl;

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Read vertex and faces

  // Return to top of file
  fin.clear();
  fin.seekg(0, ios::beg);

  *ptr_V = new double[3*nv];
  *ptr_C = new double[3*nv];
  *ptr_F = new int[3*nf];

  double *Vx = *ptr_V;
  double *Vy = *ptr_V+nv;
  double *Vz = *ptr_V+2*nv;

  double *Cx = *ptr_C;
  double *Cy = *ptr_C+nv;
  double *Cz = *ptr_C+2*nv;

  int *F1 = *ptr_F;
  int *F2 = *ptr_F+nf;
  int *F3 = *ptr_F+2*nf;

  while ( !fin.eof() ) {
    char c = fin.peek();

    // Read vertex
    if ( c == 'v' ) {
      fin.get();
      if ( mode == 0 ) {
        fin >> *Vx++ >> *Vy++ >> *Vz++;
      } else {
        fin >> *Vx++ >> *Vy++ >> *Vz++ >> *Cx++ >> *Cy++ >> *Cz++;
      }
    }

    // Read face
    if ( c == 'f' ) {
      fin.get();
      fin >> *F1++ >> *F2++ >> *F3++;
    }

    fin.ignore(4096, '\n');
  }

  if ( mode == 0 ) {
    for ( int i = 0; i < 3*nv; ++i ) {
      (*ptr_C)[i] = -1.0;
    }
  }
}
