#pragma once
#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>

#include "comm/comm.h"
#include "comm/cuda_comm.h"
#include "graph/cpu_graph.h"

class GPUGraph {
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
    GPUGraph(){};
    GPUGraph(CPUGraph& h_graph);
    void init(CPUGraph& h_graph);
    ~GPUGraph();
};
