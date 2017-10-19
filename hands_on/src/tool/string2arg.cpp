#include <string>
#include <iostream>
#include "tool.hpp"
using namespace std;
void string2arg(string str, int *argc, char ***argv) {
    size_t found = -1;
    (*argc) = 0;
    do {
      found = str.find(" ", found+1);
      (*argc)++;
    } while (found != string::npos);
    *argv = new char*[*argc];
    size_t s_start = -1, s_end;
    string temp;
    for (int i = 0; i < *argc; i++) {
      s_end = str.find(" ", s_start+1);
      temp = str.substr(s_start+1, s_end-s_start-1);
      (*argv)[i] = new char[temp.length()+1];
      snprintf((*argv)[i], temp.length()+1, "%s", temp.c_str());
      s_start = s_end;
    }
}
