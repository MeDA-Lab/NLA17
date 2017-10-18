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

const char* const short_opt = "hf:t:e:l:p:s:";

const struct option long_opt[] = {
  {"help",   0, NULL, 'h'},
  {"file",   1, NULL, 'f'},
  {"type",   1, NULL, 't'},
  {"evp",    1, NULL, 'e'},
  {"ls",     1, NULL, 'l'},
  {"para",   1, NULL, 'p'},
  {"magmasolver", 1, NULL, 's'},
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
  cout << "  -t<num>,  --type <num>     0: simple graph(default if the graph data has 2 columns)," << endl;
  cout << setw(74) << "1: directed (multi) graph (not supported yet)" << endl;
  cout << setw(75) << "2: directed weighted graph (not supported yet)" << endl;
  cout << setw(99) << "3: undirected weighted graph (default if the graph data has 3 columns)" << endl;
  cout << "  -p<file>, --para <file>    The parameter setting file" << endl;
  //cout << "  -o<file>, --output <file>  The output file" << endl;
  cout << "  -e<num>,  --evp <num>      0: None(default), 1: Host, 2: Device" << endl;
  cout << "  -l<num>,  --ls <num>       0: None, 1: Direct Host, 2: Direct Device(default), 3: Iterative" << endl;
}

void readArgs(int argc, char** argv, args *setting) {
  char c = 0;
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