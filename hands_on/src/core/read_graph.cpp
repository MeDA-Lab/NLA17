////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    read_graph.cpp
/// @brief   Read graph from file
///
/// @author  William Liao
/// @author  Yuhsiang Mike Tsai

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "sgp.hpp"
using namespace std;
void readKonectHeader(string str, Network *network, Edge *edge_type) {
    cout << "The network format is ";
    if (str.find("bip") != string::npos) {
        *network = Network::BIPARTITE;
        cout << "bipartite.\n";
    } else if (str.find("asym") != string::npos) {
        *network = Network::DIRECTED;
        cout << "directed\n";
    } else if (str.find("sym") != string::npos) {
        *network = Network::UNDIRECTED;
        cout << "undirected\n";
    } else {
        *network = Network::UNDEFINED;
        cout << "undefined\n";
    }

    cout << "The edge weight type is ";
    if (str.find("unweighted") != string::npos) {
        *edge_type = Edge::UNWEIGHTED;
        cout << "unweighted edge type\n";
    } else if (str.find("positive") != string::npos) {
        *edge_type = Edge::MULTIPLE;
        cout << "multiple edges type\n";
    } else if (str.find("posweighted") != string::npos) {
        *edge_type = Edge::POSITIVE;
        cout << "positive weighted edge type\n";
    } else if (str.find("multisigned") != string::npos) {
        *edge_type = Edge::MULT_SIGNED;
        cout << "multiple signed weighted edge type\n";
    } else if (str.find("signed") != string::npos) {
        *edge_type = Edge::SIGNED;
        cout << "signed weighted edge type\n";
    } else if (str.find("multiweighted") != string::npos) {
        *edge_type = Edge::MULT_RATING;
        cout << "multiple weighted edge type\n";
    } else if (str.find("weighted") != string::npos) {
        *edge_type = Edge::RATING;
        cout << "rating weighted edge type\n";
    } else if (str.find("dynamic") != string::npos) {
        *edge_type = Edge::DYNAMIC;
        cout << "dynamic edge type\n";
    } else {
        *edge_type = Edge::UNDEFINED;
        cout << "undefined edge type\n";
    }
}
void readGraph(const char *input, int *E_size_r, int *E_size_c, int **E,
    double **W, Network *network_type, Edge *edge_type) {
    std::fstream pfile;
    int count = 0, n = 0;
    int tmp;

    pfile.open(input, std::ios::in);
    assert(pfile);

    std::string str;
    int linecount = 0, i;
    // The first line is header line.
    while (std::getline(pfile, str)) {
        if ( str[0] == '%' ) {
            if (linecount == 0) {
                readKonectHeader(str, network_type, edge_type);
            }
            linecount++;
        }
    }
    assert(*network_type == Network::UNDIRECTED ||
        *network_type == Network::UNDEFINED);
    assert(*edge_type == Edge::UNWEIGHTED || *edge_type == Edge::POSITIVE
        || *edge_type == Edge::RATING || *edge_type == Edge::UNDEFINED);
    pfile.clear();
    pfile.seekg(0, std::ios::beg);
    for (i = 0; i < linecount; i++) {
        pfile.ignore(4096, '\n');
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Count size
    // col
    std::getline(pfile, str);
    std::istringstream sin(str);
    double k;
    while (sin >> k) {
      ++count;
    }
    *E_size_c = count;
    assert(count >= 2);
    assert(!(*edge_type == Edge::UNWEIGHTED && count != 2));
    assert(!(*edge_type == Edge::POSITIVE && count != 3));
    assert(!(*edge_type == Edge::RATING && count != 3));

    // row
    pfile.clear();
    pfile.seekg(0, std::ios::beg);
    for (i = 0; i < linecount; i++) {
        pfile.ignore(4096, '\n');
    }
    count = 0;
    while (!pfile.eof() && pfile.peek() != EOF) {
        count++;
        pfile.ignore(4096, '\n');
    }
    *E_size_r = count;
    // Return to top of file
    pfile.clear();
    pfile.seekg(0, std::ios::beg);
    for (i = 0; i < linecount; i++) {
        pfile.ignore(4096, '\n');
    }
    *E = new int[2*count];
    *W = new double[count];
    string *others = new string[count];
    int nth = 0;
    while (!pfile.eof() && pfile.peek() != EOF) {
        pfile >> (*E)[nth] >> (*E)[nth+count];
        if (*E_size_c > 2) {
            pfile >> (*W)[nth];
        } else {
            (*W)[nth] = 1;
        }
        getline(pfile, others[nth]);
        // cout << (*E)[nth] << " " << (*E)[nth+count] << " " << (*W)[nth] << " " << others[nth] << endl;
        nth++;
    }
    for (int i = 0; i < count*2; i++) {
        (*E)[i]--;
    }

    pfile.close();

    return;
}
