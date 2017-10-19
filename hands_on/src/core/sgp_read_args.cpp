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

const char* const short_opt = "hf:e:l:s:m:";

const struct option long_opt[] = {
  {"help",   0, NULL, 'h'},
  {"file",   1, NULL, 'f'},
  {"evp",    1, NULL, 'e'},
  {"ls",     1, NULL, 'l'},
  {"magmasolver", 1, NULL, 's'},
  {"shift_sigma", 1, NULL, 1001},
  {"mu0", 1, NULL, 'm'},
  {"eigtol", 1, NULL, 1002},
  {"lstol", 1, NULL, 1003},
  {"eigmaxiter", 1, NULL, 1004},
  {"lssolver", 1, NULL, 1005},
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
  cout << "  -f<file>, --file <file>    The graph data file" << endl;
  cout << "  -e<num>,  --evp <num>      0: None (default), 1: Host, 2: Device" << endl;
  cout << "  -l<num>,  --ls <num>       0: None, 1: Direct Host, 2: Direct Device(default), 3: Iterative" << endl;
  cout << "  -m<mu0>,  --mu0 <mu0>      The initial mu0 (default: 0.6)" << endl;
  cout << "  -s\"solver_settings\", --magmasolver \"solver_settings\" default: \"--solver CG\"" << endl;
  cout << "  --shift_sigma <value>,     The value of A+sigma*I (default: 1e-5)" << endl;
  cout << "  --eigtol <value>,          The tolerance of eigsolver (default: 1e-12)" << endl;
  cout << "  --lstol <value>,           The tolerance of direct cuda lssover (default: 1e-12)" << endl;
  cout << "  --eigmaxiter <iter>,       The maximum iteration of eigsolver (default: 1000)" << endl;
  cout << "  --lssolver <num>,          0: LU (default), 1: Cholesky, 2: QR " << endl;
}

void readArgs(int argc, char** argv, args *setting) {
  int c = 0;
  int fflag = 0;
  while ( (c = getopt_long(argc, argv, short_opt, long_opt, NULL)) != -1 ) {
    switch ( c ) {
      case 'h': {
        dispUsage(argv[0]);
        exit(0);
      }

      case 'f': {
        setting->file = optarg;
        fflag = 1;
        break;
      }
      case 'e': {
        setting->evp = static_cast<EVP>(atoi(optarg));
        assert(setting->evp >= EVP::NONE && setting->evp < EVP::COUNT );
        break;
      }

      case 'l': {
        setting->ls = static_cast<LS>(atoi(optarg));
        assert(setting->ls >= LS::NONE && setting->ls < LS::COUNT);
        break;
      }
      case 's': {
        setting->solver_settings = optarg;
        break;
      }
      case 'm': {
        setting->mu0 = stod(optarg, nullptr);
        break;
      }
      case 1001: {
        setting->shift_sigma = stod(optarg, nullptr);
        break;
      }
      case 1002: {
        setting->eigtol = stod(optarg, nullptr);
        break;
      }
      case 1003: {
        setting->tol = stod(optarg, nullptr);
        break;
      }
      case 1004: {
        setting->eigmaxiter = stoi(optarg, nullptr);
        break;
      }
      case 1005: {
        setting->lsover = static_cast<LSOLVER>(stoi(optarg, nullptr));
        assert(setting->lsover >= LSOLVER::LU && setting->lsover < LSOLVER::COUNT);
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
  if ( fflag == 0 )
  {
    cout << "Error!!! This program requires input file." << endl;
    abort();
  }
}