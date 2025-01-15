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
    if (vertex_count == -1 || edge_count == -1) {
        spdlog::error("offset file or edge list file is empty");
        return;
    }

    spdlog::info("Load graph from disk to host, input dir is: {}", input_dir);
    spdlog::info("Graph vertices: {0}, edges: {1}, avg degrees: {2}", vertex_count, edge_count, (float)edge_count * 2 / vertex_count);

    FILE* pFile = fopen(src_file, "rb");
    if (!pFile) {
        spdlog::error("Unable to open file: {}", src_file);
        return;
    }
    src_list = (vertex_t*)malloc(fsize(src_file));
    if (fread(src_list, sizeof(vertex_t), edge_count, pFile) != edge_count) {
        spdlog::error("Error reading file: {}", src_file);
        fclose(pFile);
        return;
    }
    fclose(pFile);

    FILE* pFile1 = fopen(adj_file, "rb");
    if (!pFile1) {
        spdlog::error("Unable to open file: {}", adj_file);
        return;
    }
    adj_list = (vertex_t*)malloc(fsize(adj_file));
    if (fread(adj_list, sizeof(vertex_t), edge_count, pFile1) != edge_count) {
        spdlog::error("Error reading file: {}", adj_file);
        fclose(pFile1);
        return;
    }
    fclose(pFile1);

    FILE* pFile3 = fopen(begin_file, "rb");
    if (!pFile3) {
        spdlog::error("Unable to open file: {}", begin_file);
        return;
    }
    beg_pos = (index_t*)malloc(fsize(begin_file));
    if (fread(beg_pos, sizeof(index_t), vertex_count + 1, pFile3) != vertex_count + 1) {
        spdlog::error("Error reading file: {}", begin_file);
        fclose(pFile3);
        return;
    }
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
