#include "comm/constant_comm.h"
#include "graph/gpu_graph.h"
#include "spdlog/spdlog.h"

GPUGraph::GPUGraph(CPUGraph &graph) { init(graph); }

void GPUGraph::init(CPUGraph &h_graph) {
    input_dir = h_graph.input_dir;
    vertex_count = h_graph.vertex_count;
    edge_count = h_graph.edge_count;
    max_degree = h_graph.max_degree;
    has_edge_list = h_graph.has_edge_list;

    size_t free_byte, total_byte;
    HRR(cudaMemGetInfo(&free_byte, &total_byte));
    spdlog::debug("GPUGraph before transfer, used memory {:.2f} GB", float(total_byte - free_byte) / MEMORY_G);

    size_t beg_sz = sizeof(index_t) * (vertex_count + 1 + 10);
    size_t adj_sz = sizeof(vertex_t) * (edge_count + 10);

    spdlog::info("Load graph from host to device, input dir is: {}", h_graph.input_dir);
    spdlog::info("Graph vertices: {0}, edges: {1}, avg degrees: {2}", vertex_count, edge_count, (float)edge_count * 2 / vertex_count);

    /* Alloc GPU space */
    HRR(cudaMalloc((void **)&src_list, adj_sz));
    HRR(cudaMalloc((void **)&adj_list, adj_sz));
    HRR(cudaMalloc((void **)&beg_pos, beg_sz));

    spdlog::debug("Load graph from host to device, alloc GPU space finished");

    /* copy it to GPU */
    HRR(cudaMemcpy(src_list, h_graph.src_list, sizeof(vertex_t) * (edge_count), cudaMemcpyHostToDevice));
    HRR(cudaMemcpy(adj_list, h_graph.adj_list, sizeof(vertex_t) * (edge_count), cudaMemcpyHostToDevice));
    HRR(cudaMemcpy(beg_pos, h_graph.beg_pos, sizeof(index_t) * (vertex_count + 1), cudaMemcpyHostToDevice));

    if (has_edge_list) {
        size_t edge_sz = adj_sz * 2;
        HRR(cudaMalloc((void **)&edge_list, edge_sz));
        HRR(cudaMemcpy(edge_list, h_graph.edge_list, edge_sz, cudaMemcpyHostToDevice));
    }

    spdlog::debug("Load graph from host to device, copy finished");
    spdlog::info("Load graph from host to device finished");

    HRR(cudaMemGetInfo(&free_byte, &total_byte));
    spdlog::debug("GPUGraph after transfer, used memory {:.2f} GB", float(total_byte - free_byte) / MEMORY_G);
}

GPUGraph::~GPUGraph() {
    HRR(cudaFree(beg_pos));
    HRR(cudaFree(src_list));
    HRR(cudaFree(adj_list));

    if (has_edge_list) {
        HRR(cudaFree(edge_list));
    }
}