////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    set_graph_type.cpp
/// @brief   Set graph type
///
/// @author  William Liao
///

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <iostream>
#include <string>

int setgraphtype(int E_size_c){
	int type;
	if (E_size_c == 3)
	{
		type = 3;
	}else{
		type = 0;
	}

	return type;
}

int setgraphtype(char *input, int E_size_c){
	int type, tmp;
	tmp = atoi(input);
	//std::cout << "tmp = " << tmp << std::endl;
	assert( tmp < 3 && tmp >=0 );
	if (tmp == 2)
	{
		assert( E_size_c == 3 );
	}
	type = tmp;

	return type;
}