#include "graph/cpu_graph.h"

#include <omp.h>

#include <fstream>

#include "comm/constant_comm.h"
#include "spdlog/spdlog.h"

using namespace std;

CPUGraph::CPUGraph(string jsonfile) {
    string s_begin = jsonfile + "/begin.bin";
    string s_src = jsonfile + "/source.bin";
    string s_adj = jsonfile + "/adjacent.bin";

    char* begin_file = const_cast<char*>(s_begin.c_str());
    char* src_file = const_cast<char*>(s_src.c_str());
    char* adj_file = const_cast<char*>(s_adj.c_str());

    input_dir = jsonfile;
    vertex_count = fsize(begin_file) / sizeof(index_t) - 1;
    edge_count = fsize(adj_file) / sizeof(vertex_t);
    has_edge_list = false;

    spdlog::info("Load graph from disk to host, input dir is: {}", input_dir);
    spdlog::info("Graph vertices: {0}, edges: {1}, avg degrees: {2}", vertex_count, edge_count, (float)edge_count * 2 / vertex_count);

    FILE* pFile = fopen(src_file, "rb");
    src_list = (vertex_t*)malloc(fsize(src_file));
    size_t size = fread(src_list, sizeof(vertex_t), edge_count, pFile);
    fclose(pFile);

    FILE* pFile1 = fopen(adj_file, "rb");
    adj_list = (vertex_t*)malloc(fsize(adj_file));
    size = fread(adj_list, sizeof(vertex_t), edge_count, pFile1);
    fclose(pFile1);

    FILE* pFile3 = fopen(begin_file, "rb");
    beg_pos = (index_t*)malloc(fsize(begin_file));
    size = fread(beg_pos, sizeof(index_t), vertex_count + 1, pFile3);
    fclose(pFile3);

    for (int i = 0; i < constant_comm::kPrintArrNum; i++) {
        spdlog::trace("Graph edge[{0}]: ({1}, {2})", i, src_list[i], adj_list[i]);
    }

    spdlog::trace("Graph edge[...]: (..., ...)");

    for (int i = constant_comm::kPrintArrNum; i > 0; i--) {
        spdlog::trace("Graph edge[{0}]: ({1}, {2})", edge_count - i, src_list[edge_count - i], adj_list[edge_count - i]);
    }

    spdlog::info("Load graph from disk to host finished");
}

CPUGraph::~CPUGraph() {
    free(beg_pos);
    free(src_list);
    free(adj_list);
}
