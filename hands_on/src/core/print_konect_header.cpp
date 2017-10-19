#include <iostream>
#include <string>
#include "sgp.hpp"
using namespace std;
void printKonectHeader(Network network, Edge edge_type) {
    cout << "    The network format is ";
    if (network == Network::BIPARTITE) {
        cout << "bipartite.\n";
    } else if (network == Network::DIRECTED) {
        cout << "directed\n";
    } else if (network == Network::UNDIRECTED) {
        cout << "undirected\n";
    } else {
        cout << "undefined (It may be error)\n";
    }

    cout << "    The edge weight type is ";
    if (edge_type == Edge::UNWEIGHTED) {
        cout << "unweighted edge type\n";
    } else if (edge_type == Edge::MULTIPLE) {
        cout << "multiple edges type\n";
    } else if (edge_type == Edge::POSITIVE) {
        cout << "positive weighted edge type\n";
    } else if (edge_type == Edge::MULT_SIGNED) {
        cout << "multiple signed weighted edge type\n";
    } else if (edge_type == Edge::SIGNED) {
        cout << "signed weighted edge type\n";
    } else if (edge_type == Edge::MULT_RATING) {
        cout << "multiple weighted edge type\n";
    } else if (edge_type == Edge::RATING) {
        cout << "rating weighted edge type\n";
    } else if (edge_type == Edge::DYNAMIC) {
        cout << "dynamic edge type\n";
    } else {
        cout << "undefined edge type (It may be error)\n";
    }
}