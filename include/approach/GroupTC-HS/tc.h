#pragma once

#include "comm/comm.h"
#include "graph/gpu_graph.h"

#define GroupTC_HS_COMBINE 1

#define GroupTC_HS_BLOCK_SIZE 1024
#define GroupTC_HS_GROUP_SIZE 1024
#define GroupTC_HS_Cuckoo_WARP_SIZE 64

#define GroupTC_HS_shared_BLOCK_BUCKET_SIZE 6
#define GroupTC_HS_shared_GROUP_BUCKET_SIZE 4

#define GroupTC_HS_shared_Cuckoo_BUCKET_SIZE 4
#define GroupTC_HS_shared_Basic_BUCKET_SIZE (GroupTC_HS_shared_BLOCK_BUCKET_SIZE - GroupTC_HS_shared_Cuckoo_BUCKET_SIZE)

#define GroupTC_HS_USE_CTA 100
#define GroupTC_HS_USE_WARP 1

#define GroupTC_HS_block_bucketnum 1024
#define GroupTC_HS_group_bucketnum 1024
#define GroupTC_HS_BLOCK_MODULO 1023
#define GroupTC_HS_GROUP_MODULO 1023

#define GroupTC_HS_EDGE_CHUNK 512
#define GroupTC_HS_shared_CHUNK_CACHE_SIZE 640
#define GroupTC_HS_BLOCK_BUCKET_SIZE 25
#define GroupTC_HS_GROUP_BUCKET_SIZE 25

#define GroupTC_HS_H0 1
#define GroupTC_HS_H1 31
#define GroupTC_HS_H2 37
#define GroupTC_HS_H3 43
#define GroupTC_HS_H4 53
#define GroupTC_HS_H5 61
#define GroupTC_HS_H6 83
#define GroupTC_HS_H7 97
// #define GroupTC_HS_H0 0.6180339887
// #define GroupTC_HS_H1 0.123456789
#define GroupTC_HS_Max_Conflict 3
#define GroupTC_HS_BUCKET_NUM (GroupTC_HS_shared_Cuckoo_BUCKET_SIZE * GroupTC_HS_BLOCK_SIZE - 1)
// #define GroupTC_HS_BUCKET_NUM 4093

namespace tc {
namespace approach {
namespace GroupTC_HS {

__device__ int linear_search_block(int neighbor, int *partition, int len, int bin, int BIN_START);

__device__ int linear_search_group(int neighbor, int *partition, int len, int bin, int BIN_START);

int my_binary_search(int len, int val, index_t *beg);

__device__ unsigned fmix32(unsigned int h);


template <const int GroupTC_HS_Group_SUBWARP_SIZE, const int GroupTC_HS_Group_WARP_STEP, const int CHUNK_SIZE>
__global__ void grouptc_hs(vertex_t *src_list, vertex_t *adj_list, index_t *beg_pos, uint edge_count, uint vertex_count, int *partition,
                               unsigned long long *GLOBAL_COUNT, int T_Group, int *G_INDEX, int warpfirstvertex, int warpfirstedge,
                               int nocomputefirstvertex, int nocomputefirstedge);

void gpu_run(INIReader &config, GPUGraph &gpu_graph, std::string key_space = "GroupTC-HS");

void start_up(INIReader &config, GPUGraph &gpu_graph, int argc = 0, char **argv = nullptr);

}  // namespace GroupTC_HS
}  // namespace approach
}  // namespace tc
