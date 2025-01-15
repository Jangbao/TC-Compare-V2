#include <cuda_profiler_api.h>

#include "approach/Polak/tc.h"
#include "comm/comm.h"
#include "comm/constant_comm.h"
#include "comm/cuda_comm.h"
#include "graph/cuda_graph_comm.h"
#include "spdlog/spdlog.h"

__global__ void tc::approach::Polak::calculate_triangles(uint edge_count, vertex_t* src_list, vertex_t* adj_list, index_t* beg_pos,
                                                         unsigned long long* results, int deviceCount, int deviceIdx) {
    int from = gridDim.x * blockDim.x * deviceIdx + blockDim.x * blockIdx.x + threadIdx.x;
    int step = deviceCount * gridDim.x * blockDim.x;
    unsigned long long count = 0;

    for (int i = from; i < edge_count; i += step) {
        int u = src_list[i], v = adj_list[i];
        int u_it = beg_pos[u], u_end = beg_pos[u + 1];
        int v_it = beg_pos[v], v_end = beg_pos[v + 1];

        int a = adj_list[u_it], b = adj_list[v_it];
        while (u_it < u_end && v_it < v_end) {
            int d = a - b;
            if (d <= 0) a = adj_list[++u_it];
            if (d >= 0) b = adj_list[++v_it];
            if (d == 0) {
                ++count;
            }
        }
    }

    results[blockDim.x * blockIdx.x + threadIdx.x] = count;
}

void tc::approach::Polak::gpu_run(INIReader& config, GPUGraph& gpu_graph, std::string key_space) {
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

    int grid_size = 640;
    int block_size = 64;
    double t_start, total_kernel_use = 0;
    uint64_t count;

    unsigned long long* d_results;
    HRR(cudaMalloc(&d_results, grid_size * block_size * sizeof(unsigned long long)));

    for (int i = 0; i < iteration_count; i++) {
        HRR(cudaMemset(d_results, 0, grid_size * block_size * sizeof(unsigned long long)));

        t_start = wtime();
        tc::approach::Polak::calculate_triangles<<<grid_size, block_size>>>(edge_count, d_src, d_adj, d_beg_pos, d_results);
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
    auto logger = spdlog::get("Polak_file_logger");
    if (logger) {
        logger->info("{0}\t{1}\t{2}\t{3}\t{4:.6f}", "Polak", gpu_graph.input_dir, count, iteration_count, total_kernel_use / iteration_count);
    } else {
        spdlog::warn("Logger 'Polak_file_logger' is not initialized.");
    }

    spdlog::info("Iter {0}, avg kernel use {1:.6f} s", iteration_count, total_kernel_use / iteration_count);
    spdlog::info("Triangle count {:d}", count);

    HRR(cudaFree(d_results));
}

void tc::approach::Polak::start_up(INIReader& config, GPUGraph& gpu_graph, int argc, char** argv) {
    bool run = config.GetBoolean("comm", "Polak", false);
    if (run) {
        size_t free_byte, total_byte, available_byte;
        HRR(cudaMemGetInfo(&free_byte, &total_byte));
        available_byte = total_byte - free_byte;
        spdlog::debug("Polak before compute, used memory {:.2f} GB", float(total_byte - free_byte) / MEMORY_G);

        tc::approach::Polak::gpu_run(config, gpu_graph);

        HRR(cudaMemGetInfo(&free_byte, &total_byte));
        spdlog::debug("Polak after compute, used memory {:.2f} GB", float(total_byte - free_byte) / MEMORY_G);
        if (available_byte != total_byte - free_byte) {
            spdlog::warn("There is GPU memory that is not freed after Polak runs.");
        }
    }
}
