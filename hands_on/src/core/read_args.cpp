////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    read_args.cpp
/// @brief   The implementation of arguments reader.
///
/// @author  Mu Yang <<emfomy@gmail.com>>
/// @author  Yuhsiang Tsai <<yhmtsai@gmail.com>>
///

#include <iostream>
#include <string>
#include <harmonic.hpp>
#include <getopt.h>

using namespace std;

const char* const short_opt = "hf:o:t:s:";

const struct option long_opt[] = {
  {"help",   0, NULL, 'h'},
  {"file",   1, NULL, 'f'},
  {"output", 1, NULL, 'o'},
  {"target", 1, NULL, 't'},
  {"magmasolver", 1, NULL, 's'},
  {"method",   1, NULL, 1001},
  {"tol", 1, NULL, 1002},
  {"sigma", 1, NULL, 1003},
  {"eig_maxiter", 1, NULL, 1004},
  {"sipm_option", 1, NULL, 1005},
  {"res", 1, NULL, 1006},
  {"nwant", 1, NULL, 1007},
  {"nstep", 1, NULL, 1008},
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
  cout << "  -f<file>, --file <file>    The Object file (default: input.obj)" << endl;
  cout << "  -o<file>, --output <file>  The Output file (default: output.obj)" << endl;
  cout << "  -t<num>,  --target <num>   0: LS   - Linear System (Lii Ui = Lib Ub) (default) \n"
       << "                             1: SIPM - Shift Inverse Power Method\n"
       << "                             2: LANCZOS   - Lanczos Iteration\n";
  cout << "  -s\"solver_settings\",       --magmasolver \"solver_settings\"\n"
       << "                             default settings: \"--solver CG\" for Iterative Linear System\n";
  cout << "  --method <num>             Laplacian matrix, 0: KIRCHHOFF (default) 1: COTANGENT\n";
  cout << "  --tol <num>                Tolerance of Direct Eigensolver or Linear System Solver\n";
  cout << "  --sigma <value>            SIPM: as mu0 (default 1.5)\n";
  cout << "  --eig_maxiter <value>      The maximum iteration of eigensolver (default: 1000)\n";
  cout << "  --sipm_option <num>        0: Host(default) 1: Device\n";
  cout << "  --res <filename>           Write the residual vector to the file named <filename>.\n";
  cout << "                             Must be used with the verbose option value > 0 in --magmasolver \n";
  cout << "  --nwant <num>              Number of eigenvalues to solve in Lanczos.\n";
  cout << "  --nstep <num>              Number of steps in Lanczos.\n";
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
        assert(setting->target >= Target::LS
            && setting->target < Target::COUNT);
      }
      case 's': {
        setting->solver_settings = optarg;
        isSolverSet = true;
        break;
      }
      case 1001: {
        setting->method = static_cast<Method>(atoi(optarg));
        assert(setting->method >= Method::KIRCHHOFF
          && setting->method < Method::COUNT);
      }
      case 1002: {
        setting->tol = stod(optarg, nullptr);
        (setting->LSEV_info).tol = stod(optarg, nullptr);
        break;
      }
      case 1003: {
        setting->sigma = stod(optarg, nullptr);
        break;
      }
      case 1004: {
        setting->eig_maxiter = stoi(optarg, nullptr);
        (setting->LSEV_info).maxit = stoi(optarg, nullptr);
        break;
      }
      case 1005: {
        setting->sipm = static_cast<SIPM>(atoi(optarg));
        assert(setting->sipm >= SIPM::HOST && setting->sipm < SIPM::COUNT);
        break;
      }
      case 1006: {
        setting->res_flag = 1;
        setting->res_filename = optarg;
        cout << "residual will be written to " << setting->res_filename << endl;
        break;
      }
      case 1007: {
        (setting->LSEV_info).Nwant = stoi(optarg, nullptr);
        break;
      }
      case 1008: {
        (setting->LSEV_info).Nstep = stoi(optarg, nullptr);
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
  if (isSolverSet == false && setting->target == Target::LS) {
      setting->solver_settings = "--solver CG";
  }
  if (setting->solver_settings.find("LOBPCG") != string::npos
    && setting->target == Target::LS) {
    cerr << "Do not use LOBPCG in linear system\n";
    exit(1);
  }

}