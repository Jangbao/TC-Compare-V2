#include <cuda_profiler_api.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include <string>

#include "approach/GroupTC/tc.h"
#include "comm/comm.h"
#include "comm/constant_comm.h"
#include "comm/cuda_comm.h"
#include "spdlog/spdlog.h"

__device__ int tc::approach::GroupTC::bin_search(vertex_t* arr, int len, int val) {
    uint32_t Y;
    int32_t bot = 0;
    int32_t top = len - 1;
    int32_t r;
    while (top >= bot) {
        r = (top + bot) / 2;
        Y = arr[r];

        if (val == Y) {
            return 1;
        }

        if (val < Y) {
            top = r - 1;
        } else {
            bot = r + 1;
        }
    }
    return 0;
}

__device__ int tc::approach::GroupTC::bin_search_less_branch(vertex_t* arr, int len, int val) {
    int ret = 0;
    int halfsize;
    int candidate;
    int temp = len;
    while (temp > 1) {
        halfsize = temp / 2;
        candidate = arr[ret + halfsize];
        ret += (candidate < val) ? halfsize : 0;
        temp -= halfsize;
    }
    ret += (arr[ret] < val);
    return ret < len && arr[ret] == val;
}

__device__ int tc::approach::GroupTC::bin_search_with_offset_and_less_branch(vertex_t* arr, int len, int val, int& offset) {
    int ret = 0;
    int halfsize;
    int candidate;
    int temp = len;
    while (temp > 1) {
        halfsize = temp / 2;
        candidate = arr[ret + halfsize];
        ret += (candidate < val) ? halfsize : 0;
        temp -= halfsize;
    }
    ret += (arr[ret] < val);
    offset = ret;
    return ret < len && arr[ret] == val;
}

__global__ void tc::approach::GroupTC::grouptc(vertex_t* src_list, vertex_t* adj_list, index_t* beg_pos, uint edge_count, uint vertex_count,
                                               unsigned long long* GLOBAL_COUNT) {
    // 共享内存中的 hashTable
    __shared__ int sh_tb_start[GroupTC_BLOCK_BUCKETNUM];
    __shared__ int sh_tb_len[GroupTC_BLOCK_BUCKETNUM];
    __shared__ int sh_ele_start[GroupTC_BLOCK_BUCKETNUM];
    __shared__ int sh_ele_len[GroupTC_BLOCK_BUCKETNUM];

    unsigned long long P_counter = 0;

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    for (int i = bid * GroupTC_BLOCK_BUCKETNUM; i < edge_count; i += gridDim.x * GroupTC_BLOCK_BUCKETNUM) {
        if (i + tid < edge_count) {
            vertex_t src = src_list[i + tid];
            vertex_t dst = adj_list[i + tid];
            int temp;

            int tb_start, tb_len, ele_start, ele_len;
            tb_start = i + tid + 1;
            // tb_start = beg_pos[src];
            tb_len = beg_pos[src + 1] - tb_start;
            ele_start = beg_pos[dst];
            ele_len = beg_pos[dst + 1] - ele_start;

            if (tb_len * 2 < ele_len) {
                temp = tb_start;
                tb_start = ele_start;
                ele_start = temp;

                temp = tb_len;
                tb_len = ele_len;
                ele_len = temp;
            }

            sh_tb_start[tid] = tb_start;
            sh_tb_len[tid] = tb_len;
            sh_ele_start[tid] = ele_start;
            sh_ele_len[tid] = ele_len;
        }

        __syncthreads();

        int now = tid / GroupTC_SUBWARP_SIZE;
        int end = min(edge_count - i, GroupTC_BLOCK_BUCKETNUM);
        int workid = tid % GroupTC_SUBWARP_SIZE;
        int offset = 0;
        int last_now = -1;

        // 获取二跳邻居节点
        int neighbor_degree = sh_ele_len[now];
        while (now < end) {
            // 如果当前一阶邻居节点已被处理完，找下一个一阶邻居节点去处理
            while (now < end && workid >= neighbor_degree) {
                now += GroupTC_WARP_STEP;
                if (now < end) {
                    workid -= neighbor_degree;
                    neighbor_degree = sh_ele_len[now];
                }
            }

            if (now < end) {
                offset = last_now == now ? offset : 0;
                P_counter += tc::approach::GroupTC::bin_search_with_offset_and_less_branch(
                    adj_list + (sh_tb_start[now] + offset), sh_tb_len[now] - offset, adj_list[sh_ele_start[now] + workid], offset);
                last_now = now;
            }
            workid += GroupTC_SUBWARP_SIZE;
        }
        __syncthreads();
    }

    GLOBAL_COUNT[bid * GroupTC_BLOCK_BUCKETNUM + tid] = P_counter;
}

void tc::approach::GroupTC::gpu_run(INIReader& config, GPUGraph& gpu_graph, std::string key_space) {
    std::string file = gpu_graph.input_dir;
    int iteration_count = config.GetInteger(key_space, "iteration_count", 1);
    spdlog::info("Run algorithm {}", key_space);
    spdlog::info("Dataset {}", file);
    spdlog::info("Number of nodes: {0}, number of edges: {1}", gpu_graph.vertex_count, gpu_graph.edge_count);
    int device = config.GetInteger(key_space, "device", 1);
    HRR(cudaSetDevice(device));

    vertex_t* d_src = gpu_graph.src_list;
    vertex_t* d_adj = gpu_graph.adj_list;
    index_t* d_beg_pos = gpu_graph.beg_pos;
    vertex_t vertex_count = gpu_graph.vertex_count;
    vertex_t edge_count = gpu_graph.edge_count;
    int grid_size = NumberOfMPs() * 8;

    double t_start, total_kernel_use = 0;
    uint64_t count;

    unsigned long long* d_results;
    HRR(cudaMalloc(&d_results, grid_size * GroupTC_BLOCK_BUCKETNUM * sizeof(unsigned long long)));

    for (int i = 0; i < iteration_count; i++) {
        HRR(cudaMemset(d_results, 0, grid_size * GroupTC_BLOCK_BUCKETNUM * sizeof(unsigned long long)));

        t_start = wtime();

        tc::approach::GroupTC::grouptc<<<grid_size, GroupTC_BLOCK_BUCKETNUM>>>(d_src, d_adj, d_beg_pos, edge_count, vertex_count, d_results);
        HRR(cudaDeviceSynchronize());

        thrust::device_ptr<unsigned long long> ptr(d_results);
        count = thrust::reduce(ptr, ptr + (grid_size * GroupTC_BLOCK_BUCKETNUM));

        double ee = wtime();
        total_kernel_use += ee - t_start;
        if (i == 0) {
            spdlog::info("Iter 0, kernel use {:.6f} s", total_kernel_use);
            if (ee - t_start > 0.1 && iteration_count != 1) {
                iteration_count = 10;
            }
        }
    }

    // algorithm, dataset, iteration_count, avg compute time/s,
    auto logger = spdlog::get("GroupTC_file_logger");
    if (logger) {
        logger->info("{0}\t{1}\t{2}\t{3}\t{4:.6f}", "GroupTC", gpu_graph.input_dir, count, iteration_count, total_kernel_use / iteration_count);
    } else {
        spdlog::warn("Logger 'GroupTC_file_logger' is not initialized.");
    }

    spdlog::info("Iter {0}, avg kernel use {1:.6f} s", iteration_count, total_kernel_use / iteration_count);
    spdlog::info("Triangle count {:d}", count);

    HRR(cudaFree(d_results));
}

void tc::approach::GroupTC::start_up(INIReader& config, GPUGraph& gpu_graph, int argc, char** argv) {
    bool run = config.GetBoolean("comm", "GroupTC", false);
    if (run) {
        size_t free_byte, total_byte, available_byte;
        HRR(cudaMemGetInfo(&free_byte, &total_byte));
        available_byte = total_byte - free_byte;
        spdlog::debug("GroupTC before compute, used memory {:.2f} GB", float(total_byte - free_byte) / MEMORY_G);

        tc::approach::GroupTC::gpu_run(config, gpu_graph);

        HRR(cudaMemGetInfo(&free_byte, &total_byte));
        spdlog::debug("GroupTC after compute, used memory {:.2f} GB", float(total_byte - free_byte) / MEMORY_G);
        if (available_byte != total_byte - free_byte) {
            spdlog::warn("There is GPU memory that is not freed after GroupTC runs.");
        }
    }
}
