////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    sgp_read_args.cpp
/// @brief   The implementation of arguments reader.
///
/// @author  Mu Yang <<emfomy@gmail.com>>
/// @author  Yuhsiang Tsai <<yhmtsai@gmail.com>>
/// @author  William Liao  <<b00201028.ntu@gmail.com>>
///

#include <iostream>
#include <iomanip>
#include <sgp.hpp>
#include <getopt.h>
#include <string>
using namespace std;

const char* const short_opt = "hf:o:t:s:";

const struct option long_opt[] = {
  {"help",   0, NULL, 'h'},
  {"file",   1, NULL, 'f'},
  {"output", 1, NULL, 'o'},
  {"target", 1, NULL, 't'},
  {"magmasolver", 1, NULL, 's'},
  {"tol", 1, NULL, 1002},
  {"sigma", 1, NULL, 1003},
  {"eig_maxiter", 1, NULL, 1004},
  {"sipm_option", 1, NULL, 1005},
  {"ls_option", 1, NULL, 1006},
  {"res", 1, NULL, 1007},
  {NULL,     0, NULL, 0}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  Display the usage.
///
/// @param  bin  the name of binary file.
///
void dispUsage( const char *bin ) {
  cout << "Usage: " << bin << " [OPTIONS]" << endl;
  cout << "Options:" << endl;
  cout << "  -h,       --help           Display this information" << endl;
  cout << "  -f<file>, --file <file>    The graph data file (defalut: input.obj)" << endl;
  cout << "  -o<file>, --output <file>  The Output file (default: output.obj)" << endl;
  cout << "  -t<num>,  --target <num>   0: LOBPCG (solve some smallest eigenvectors) (default) \n"
       << "                             1: SIPM - Shift Inverse Power Method\n"
       << "                             2: LS   - Linear System (A+sigmaI)\n";
  cout << "  -s\"solver_settings\",       --magmasolver \"solver_settings\"\n"
       << "                             default settings: \"--solver CG\" for Iterative Linear System\n"
       << "                                               \"--solver LOBPCG --ev 4 --precond ILU\" for LOBPCG\n";
  cout << "  --tol <num>                Tolerance of Direct Eigensolver or Linear System Solver\n";
  cout << "  --sigma <value>            SIPM: as mu0, LS: as shift element (default: 0)\n";
  cout << "  --eig_maxiter <value>      The maximum iteration of eigensolver (default: 1000)\n";
  cout << "  --sipm_option <num>        0: Host(default) 1: Device\n";
  cout << "  --ls_option <num>          Iterative - 0: MAGMA Iterative solver(default)\n"
       << "                             Direct    - 1: HOST_QR   2:HOST_CHOL   3: HOST_LU\n"
       << "                                         4: DEVICE_QR 5:DEVICE_CHOL\n";
  cout << "  --res <filename>           Write the residual vector to the file named <filename>.\n";
  cout << "                             Must be used with the verbose option value > 0 in --magmasolver \n";
}

void readArgs(int argc, char** argv, args *setting) {
  int c = 0;
  bool isSolverSet = false;
  while ( (c = getopt_long(argc, argv, short_opt, long_opt, NULL)) != -1 ) {
    switch (c) {
      case 'h': {
        dispUsage(argv[0]);
        exit(0);
      }
      case 'f': {
        setting->file = optarg;
        break;
      }
      case 'o': {
        setting->output = optarg;
        break;
      }
      case 't': {
        setting->target = static_cast<Target>(atoi(optarg));
        assert(setting->target >= Target::LOBPCG
            && setting->target < Target::COUNT);
      }
      case 's': {
        setting->solver_settings = optarg;
        isSolverSet = true;
        break;
      }
      case 1002: {
        setting->tol = stod(optarg, nullptr);
        break;
      }
      case 1003: {
        setting->sigma = stod(optarg, nullptr);
        break;
      }
      case 1004: {
        setting->eig_maxiter = stoi(optarg, nullptr);
        break;
      }
      case 1005: {
        setting->sipm = static_cast<SIPM>(atoi(optarg));
        assert(setting->sipm >= SIPM::HOST && setting->sipm < SIPM::COUNT);
        break;
      }
      case 1006: {
        setting->ls = static_cast<LS>(stoi(optarg, nullptr));
        assert(setting->ls >= LS::MAGMA && setting->ls < LS::COUNT);
        break;
      }
      case 1007: {
        setting->res_flag = 1;
        setting->res_filename = optarg;
        cout << "residual will be written to " << setting->res_filename << endl;
        break;
      }
      case ':': {
        cout << "Option -" << c << " requires an argument.\n";
        abort();
      }

      case '?': {
        cout << "Unknown option -" << c << endl;
        abort();
      }
    }
  }
  if (isSolverSet == false) {
    if (setting->target == Target::LOBPCG) {
      setting->solver_settings = "--solver LOBPCG --ev 4 --precond ILU";
    } else {
      setting->solver_settings = "--solver CG";
    }
  }
  if (setting->solver_settings.find("LOBPCG") != string::npos
    && setting->target == Target::LS) {
    cerr << "Do not use LOBPCG in linear system\n";
    cerr << "example: --solver LOBPCG ...\n";
    exit(1);
  }
  if (setting->solver_settings.find("LOBPCG") == string::npos
    && setting->target == Target::LOBPCG) {
    cerr << "only use LOBPCG in LOBPCG\n";
    exit(1);
  }
  
}
