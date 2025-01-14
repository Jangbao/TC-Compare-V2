#pragma once

#include "comm/comm.h"
#include "graph/gpu_graph.h"

namespace tc {
namespace approach {
namespace H_INDEX {

__device__ int linear_search(int neighbor, int *partition, int *bin_count, int bin, int BIN_OFFSET, int BIN_START, int BUCKETS);

__device__ int max_count(int *bin_count, int start, int end, int len);

__global__ void warp_hash_count(vertex_t *adj_list, index_t *beg_pos, vertex_t *edge_list, uint edge_count, uint vertex_count, uint edge_list_count,
                                int *partition, unsigned long long *GLOBAL_COUNT, long long E_START, long long E_END, int device, int BUCKETS,
                                int G_BUCKET_SIZE, int T_Group);

__global__ void CTA_hash_count(vertex_t *adj_list, index_t *beg_pos, vertex_t *edge_list, uint edge_count, uint vertex_count, uint edge_list_count,
                               int *partition, unsigned long long *GLOBAL_COUNT, int E_START, int E_END, int device, int BUCKETS, int BUCKET_SIZE,
                               int T_Group);

void gpu_run(INIReader &config, GPUGraph &gpu_graph, std::string key_space = "H-INDEX");

void start_up(INIReader &config, GPUGraph &gpu_graph, int argc, char **argv);

}  // namespace H_INDEX
}  // namespace approach
}  // namespace tc