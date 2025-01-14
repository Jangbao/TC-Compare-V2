#pragma once
#include <cuda_profiler_api.h>

#include "comm/comm.h"
#include "graph/gpu_graph.h"

#define GroupTC_BLOCK_BUCKETNUM 256
#define GroupTC_SUBWARP_SIZE 32
#define GroupTC_WARP_STEP (GroupTC_BLOCK_BUCKETNUM / GroupTC_SUBWARP_SIZE)

namespace tc {
namespace approach {
namespace GroupTC {

__device__ int bin_search(vertex_t* arr, int len, int val);

__device__ int bin_search_less_branch(vertex_t* arr, int len, int val);

__device__ int bin_search_with_offset_and_less_branch(vertex_t* arr, int len, int val, int& offset);

__global__ void grouptc(vertex_t* src_list, vertex_t* adj_list, index_t* beg_pos, uint edge_count, uint vertex_count, unsigned long long* result_count);

void gpu_run(INIReader& config, GPUGraph& gpu_graph, std::string key_space = "GroupTC");

void start_up(INIReader& config, GPUGraph& gpu_graph, int argc, char** argv);

}  // namespace GroupTC
}  // namespace approach
}  // namespace tc
