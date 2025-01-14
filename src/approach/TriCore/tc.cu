#include <cuda_profiler_api.h>

#include "approach/TriCore/tc.h"
#include "comm/comm.h"
#include "comm/constant_comm.h"
#include "comm/cuda_comm.h"
#include "graph/cuda_graph_comm.h"
#include "spdlog/spdlog.h"

__global__ void tc::approach::TriCore::warp_binary_kernel(uint edge_count, vertex_t* src_list, vertex_t* adj_list, index_t* beg_pos,
                                                          unsigned long long* results) {
    // phase 1, partition
    uint64_t count = 0;
    __shared__ vertex_t local[TriCore_BLOCKSIZE];

    vertex_t i = threadIdx.x % 32;
    vertex_t p = threadIdx.x / 32;
    for (vertex_t tid = (threadIdx.x + blockIdx.x * blockDim.x) / 32; tid < edge_count; tid += blockDim.x * gridDim.x / 32) {
        vertex_t node_m = src_list[tid];
        vertex_t node_n = adj_list[tid];

        vertex_t degree_m = beg_pos[node_m + 1] - beg_pos[node_m];
        vertex_t degree_n = beg_pos[node_n + 1] - beg_pos[node_n];
        vertex_t* a = adj_list + beg_pos[node_m];
        vertex_t* b = adj_list + beg_pos[node_n];
        if (degree_m < degree_n) {
            vertex_t temp = degree_m;
            degree_m = degree_n;
            degree_n = temp;
            vertex_t* aa = a;
            a = b;
            b = aa;
        }

        // initial cache
        local[p * 32 + i] = a[i * degree_m / 32];
        __syncthreads();

        // search
        vertex_t j = i;
        while (j < degree_n) {
            vertex_t X = b[j];
            vertex_t Y;
            // phase 1: cache
            int32_t bot = 0;
            int32_t top = 32;
            int32_t r;
            while (top > bot + 1) {
                r = (top + bot) / 2;
                Y = local[p * 32 + r];
                if (X == Y) {
                    count++;
                    bot = top + 32;
                }
                if (X < Y) {
                    top = r;
                }
                if (X > Y) {
                    bot = r;
                }
            }
            // phase 2
            bot = bot * degree_m / 32;
            top = top * degree_m / 32 - 1;
            while (top >= bot) {
                r = (top + bot) / 2;
                Y = a[r];
                if (X == Y) {
                    count++;
                }
                if (X <= Y) {
                    top = r - 1;
                }
                if (X >= Y) {
                    bot = r + 1;
                }
            }
            j += 32;
        }
        __syncthreads();
    }
    results[blockDim.x * blockIdx.x + threadIdx.x] = count;
}

void tc::approach::TriCore::gpu_run(INIReader& config, GPUGraph& gpu_graph, std::string key_space) {
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
    vertex_t edge_count = gpu_graph.edge_count;

    int grid_size = 1048576;
    int block_size = TriCore_BLOCKSIZE;
    double t_start, total_kernel_use = 0;
    uint64_t count;

    unsigned long long* d_results;
    HRR(cudaMalloc(&d_results, grid_size * block_size * sizeof(unsigned long long)));

    for (int i = 0; i < iteration_count; i++) {
        HRR(cudaMemset(d_results, 0, grid_size * block_size * sizeof(unsigned long long)));

        cuda_graph_comm::check_array("d_src", d_src, edge_count, edge_count - 10, edge_count);
        cuda_graph_comm::check_array("d_adj", d_adj, edge_count, edge_count - 10, edge_count);
        cuda_graph_comm::check_array("d_beg_pos", d_beg_pos, 10000, 0, 10);
        t_start = wtime();

        tc::approach::TriCore::warp_binary_kernel<<<grid_size, block_size>>>(edge_count, d_src, d_adj, d_beg_pos, d_results);

        HRR(cudaDeviceSynchronize());
        thrust::device_ptr<unsigned long long> ptr(d_results);
        count = thrust::reduce(ptr, ptr + (grid_size * block_size));

        double ee = wtime();
        total_kernel_use += ee - t_start;
        if (i == 0) {
            spdlog::info("Iter 0, kernel use {:.6f} s", total_kernel_use);
            if (ee - t_start > 0.1 && iteration_count != 1) {
                iteration_count = 10;
            }
        }
    }

    // algorithm, dataset, triangle_count, iteration_count, avg kernel time/s
    spdlog::get("TriCore_file_logger")
        ->info("{0}\t{1}\t{2}\t{3}\t{4:.6f}", "TriCore", gpu_graph.input_dir, count, iteration_count, total_kernel_use / iteration_count);

    spdlog::info("Iter {0}, avg kernel use {1:.6f} s", iteration_count, total_kernel_use / iteration_count);
    spdlog::info("Triangle count {:d}", count);

    HRR(cudaFree(d_results));
}

void tc::approach::TriCore::start_up(INIReader& config, GPUGraph& gpu_graph, int argc, char** argv) {
    bool run = config.GetBoolean("comm", "TriCore", false);
    if (run) {
        size_t free_byte, total_byte, available_byte;
        HRR(cudaMemGetInfo(&free_byte, &total_byte));
        available_byte = total_byte - free_byte;
        spdlog::debug("TriCore before compute, used memory {:.2f} GB", float(total_byte - free_byte) / MEMORY_G);

        tc::approach::TriCore::gpu_run(config, gpu_graph);

        HRR(cudaMemGetInfo(&free_byte, &total_byte));
        spdlog::debug("TriCore after compute, used memory {:.2f} GB", float(total_byte - free_byte) / MEMORY_G);
        if (available_byte != total_byte - free_byte) {
            spdlog::warn("There is GPU memory that is not freed after TriCore runs.");
        }
    }
}
