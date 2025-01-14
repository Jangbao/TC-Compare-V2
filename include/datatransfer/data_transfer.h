#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "graph/cpu_graph.h"
#include "graph/gpu_graph.h"

class DataTransfer {
   public:
    std::string input_file;
    CPUGraph* h_graph;
    GPUGraph d_graph;

   public:
    static bool degree_less(vertex& a, vertex& b);

    static bool degree_greather(vertex& a, vertex& b);

    static bool edge_less(edge& a, edge& b);

    static bool edge_greather(edge& a, edge& b);

    static void cpu_graph_comparasion(CPUGraph& graph_1, CPUGraph& graph_2) {};

    static void gpu_graph_comparasion(GPUGraph& gpu_graph_1, GPUGraph& gpu_graph_2) {};

    DataTransfer(std::string file, CPUGraph* graph);

    DataTransfer();

    void init(std::string file, CPUGraph* graph);

    bool check_init();

    ~DataTransfer() {};

    virtual void transfer() {};

    int compute_max_degree();
};