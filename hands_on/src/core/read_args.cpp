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

const char* const short_opt = "hf:t:o:e:m:s:";

const struct option long_opt[] = {
  {"help",   0, NULL, 'h'},
  {"file",   1, NULL, 'f'},
  {"type",   1, NULL, 't'},
  {"output", 1, NULL, 'o'},
  {"evp",    1, NULL, 'e'},
  {"mu0",    1, NULL, 'm'},
  {"magmasolver", 1, NULL, 's'},
  {"shift_sigma", 1, NULL, 1001},
  {"eigtol", 1, NULL, 1002},
  {"eigmaxiter", 1, NULL, 1004},
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
  cout << "  -f<file>, --file <file>    The graph file (default: input.obj)" << endl;
  cout << "  -t<num>,  --type <num>     0: KIRCHHOFF(default), 1: COTANGENT" << endl;
  cout << "  -o<file>, --output <file>  The output file (default: output.obj)" << endl;
  cout << "  -e<num>,  --evp <num>      0: None(default), 1: Host, 2: Device" << endl;
  cout << "  -m<mu0>,  --mu0 <mu0>      The initial mu0 (default: 1.5)" << endl;
  cout << "  -s\"solver_settings\", --magmasolver \"solver_settings\" default: \"--solver CG\"" << endl;
  cout << "  --eigtol <value>,          The tolerance of eigsolver (default: 1e-12)" << endl;
  cout << "  --eigmaxiter <iter>,       The maximum iteration of eigsolver (default: 1000)" << endl;
}

void readArgs( int argc, char** argv, args *setting) {
  int c = 0;
  while ( (c = getopt_long(argc, argv, short_opt, long_opt, NULL)) != -1 ) {
    switch (c) {
      case 'h': {
        dispUsage(argv[0]);
        exit(0);
      }

      case 'f': {
        setting->input = optarg;
        break;
      }

      case 't': {
        setting->method = static_cast<Method>(atoi(optarg));
        assert(setting->method >= Method::KIRCHHOFF && setting->method < Method::COUNT );
        break;
      }

      case 'o': {
        setting->output = optarg;
        break;
      }
      
      case 'e': {
        setting->evp = static_cast<EVP>(atoi(optarg));
        assert(setting->evp >= EVP::NONE && setting->evp < EVP::COUNT );
        break;
      }
      case 'm': {
        setting->mu0 = stod(optarg, nullptr);
        break;
      }
      case 's': {
        setting->solver_settings = optarg;
        break;
      }
      case 1002: {
        setting->eigtol = stod(optarg, nullptr);
        break;
      }
      case 1004: {
        setting->eigmaxiter = stoi(optarg, nullptr);
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
}
