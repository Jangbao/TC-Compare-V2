#pragma once
#include <cuda_profiler_api.h>

#include "comm/comm.h"
#include "graph/gpu_graph.h"

namespace tc {
namespace approach {
namespace Fox {

__global__ void getEdgeWorkLoad(uint edge_count, uint16_t *d_edgeWorkLoad, vertex_t *d_src_list, vertex_t *d_adj_list, uint *c_adjLen);

uint binarySearchValue(uint16_t *array, int value, uint arrayLength, int direction);

__global__ void binSearchKernel(index_t *c_offset, vertex_t *d_src_reorder, vertex_t *d_adj_reorder, vertex_t *c_adj_list, uint *c_adjLen,
                                uint edge_count, uint c_edge_start_pos, uint c_edge_end_pos, int c_threadsPerEdge, long *c_sums);

void gpu_run(INIReader &config, GPUGraph &gpu_graph, std::string key_space = "Fox");

void start_up(INIReader &config, GPUGraph &gpu_graph, int argc, char **argv);

}  // namespace Fox
}  // namespace approach
}  // namespace tc