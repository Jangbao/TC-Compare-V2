#pragma once
#include <cuda_profiler_api.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include "comm/comm.h"
#include "graph/gpu_graph.h"

#define TriCore_BLOCKSIZE 64

namespace tc {
namespace approach {
namespace TriCore {

__global__ void warp_binary_kernel(uint edge_count, vertex_t *src_list, vertex_t *adj_list, index_t *beg_pos, unsigned long long *results);

void gpu_run(INIReader &config, GPUGraph &gpu_graph, std::string key_space = "TriCore");

void start_up(INIReader &config, GPUGraph &gpu_graph, int argc, char **argv);

}  // namespace TriCore
}  // namespace approach
}  // namespace tc
