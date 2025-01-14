#pragma once
#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
#include <queue>
#include "comm/comm.h"

class CPUGraph {

public:
	std::string input_dir;
	uint vertex_count;
	uint edge_count;
	uint max_degree;
	index_t* beg_pos;
	vertex_t* src_list;
	vertex_t* adj_list;
	bool has_edge_list;
	vertex_t* edge_list;

public:
	CPUGraph() {};
	CPUGraph(std::string filename);
	~CPUGraph();
};
