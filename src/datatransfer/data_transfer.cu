#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "comm/constant_comm.h"
#include "datatransfer/data_transfer.h"
#include "graph/cpu_graph.h"
#include "graph/cuda_graph_comm.h"
#include "spdlog/spdlog.h"

bool DataTransfer::degree_less(vertex &a, vertex &b) { return a.degree < b.degree; }

bool DataTransfer::degree_greather(vertex &a, vertex &b) { return a.degree > b.degree; }

bool DataTransfer::edge_less(edge &a, edge &b) { return a.u < b.u || (a.u == b.u && a.v < b.v); }

bool DataTransfer::edge_greather(edge &a, edge &b) { return a.u > b.u || (a.u == b.u && a.v > b.v); }

DataTransfer::DataTransfer(std::string file, CPUGraph *graph) {
    init(file, graph);
}

DataTransfer::DataTransfer() {}

void DataTransfer::init(std::string file, CPUGraph *graph) {
    input_file = file;
    h_graph = graph;
    d_graph.init(*graph);
}

bool DataTransfer::check_init() {
    if (input_file.empty()) {
        spdlog::error("Input file is empty, please init the data transfer first.");
        return false;
    }
    return true;
}

int DataTransfer::compute_max_degree() {
    index_t *offset_arr = (index_t *)malloc(sizeof(index_t) * (h_graph->vertex_count + 1));
    HRR(cudaMemcpy(offset_arr, d_graph.beg_pos, sizeof(index_t) * (h_graph->vertex_count + 1), cudaMemcpyDeviceToHost));
    uint vertex_count = h_graph->vertex_count;

    int max_degre = 0;
    for (uint i = 1; i <= vertex_count; i++) {
        if (offset_arr[i] - offset_arr[i - 1] > max_degre) {
            max_degre = offset_arr[i] - offset_arr[i - 1];
        }
    }
    free(offset_arr);
    return max_degre;
}
