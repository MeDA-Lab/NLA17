////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    sgp_read_args.cpp
/// @brief   The implementation of arguments reader.
///
/// @author  Mu Yang <<emfomy@gmail.com>>
/// @author  Yuhsiang Tsai <<yhmtsai@gmail.com>>
/// @author  William Liao  <<b00201028.ntu@gmail.com>>
///

#include <iostream>
#include <sgp.hpp>
#include <getopt.h>

using namespace std;

const char* const short_opt = "hf:t:e:l:p:";

const struct option long_opt[] = {
  {"help",   0, NULL, 'h'},
  {"file",   1, NULL, 'f'},
  {"type",   1, NULL, 't'},
  //{"output", 1, NULL, 'o'},
  {"evp",    1, NULL, 'e'},
  {"ls",     1, NULL, 'l'},
  {"para",   1, NULL, 'p'},
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
  cout << setw(29) << "1: directed (multi) graph (not supported yet)" << endl;
  cout << setw(29) << "2: directed weighted graph (not supported yet)" << endl;
  cout << setw(29) << "3: undirected weighted graph (default if the graph data has 3 columns)" << endl;
  cout << "  -p<file>, --para <file>  The parameter setting file" << endl;
  //cout << "  -o<file>, --output <file>  The output file" << endl;
  cout << "  -e<num>,  --evp <num>      0: None(default), 1: Host, 2: Device" << endl;
  cout << "  -l<num>,  --ls <num>       0: None, 1: Direct Host, 2: Direct Device(default), 3: Iterative" << endl;
}

void readArgs( int argc, char** argv, const char *&input, const char *&para, Method &method, EVP &evp, LS &ls, int &tflag,
  int &pflag) {
  int fflag = 0;
  tflag = 0, pflag = 0;
  char c = 0;
  while ( (c = getopt_long(argc, argv, short_opt, long_opt, NULL)) != -1 ) {
    switch ( c ) {
      case 'h': {
        dispUsage(argv[0]);
        exit(0);
      }

      case 'f': {
        input = optarg;
        fflag = 1;
        break;
      }

      case 't': {
        method = static_cast<Method>(atoi(optarg));
        assert(method >= Method::SIMPLE && method < Method::COUNT );
        tflag = 1;
        break;
      }

      case 'p': {
        para = optarg;
        pflag = 1;
        break;
      }
      
      case 'e': {
        evp = static_cast<EVP>(atoi(optarg));
        assert(evp >= EVP::NONE && evp < EVP::COUNT );
        break;
      }

      case 'l': {
        ls = static_cast<LS>(atoi(optarg));
        assert(evp >= LS::NONE && evp < LS::COUNT );
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