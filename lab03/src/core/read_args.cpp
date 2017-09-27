////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    read_args.cpp
/// @brief   The implementation of arguments reader.
///
/// @author  Mu Yang <<emfomy@gmail.com>>
/// @author  Yuhsiang Tsai <<yhmtsai@gmail.com>>
///

#include <iostream>
#include <harmonic.hpp>
#include <getopt.h>

using namespace std;

const char* const short_opt = "hf:t:o:";

const struct option long_opt[] = {
  {"help",   0, NULL, 'h'},
  {"file",   1, NULL, 'f'},
  {"type",   1, NULL, 't'},
  {"output", 1, NULL, 'o'},
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
  cout << "  -f<file>, --file <file>    The graph file" << endl;
  cout << "  -t<num>,  --type <num>     0: KIRCHHOFF(default), 1: COTANGENT" << endl;
  cout << "  -o<file>, --output <file>  The output file" << endl;
}

void readArgs( int argc, char** argv, const char *&input, const char *&output, Method &method ) {
  char c = 0;
  while ( (c = getopt_long(argc, argv, short_opt, long_opt, NULL)) != -1 ) {
    switch ( c ) {
      case 'h': {
        dispUsage(argv[0]);
        exit(0);
      }

      case 'f': {
        input = optarg;
        break;
      }

      case 't': {
        method = static_cast<Method>(atoi(optarg));
        assert(method >= Method::KIRCHHOFF && method < Method::COUNT );
        break;
      }

      case 'o': {
        output = optarg;
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
