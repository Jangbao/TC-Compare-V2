#include "approach/Green/main.cuh"
#include "graph/cuda_graph_comm.h"
#include "spdlog/spdlog.h"

void tc::approach::Green::gpu_run(INIReader& config, GPUGraph& gpu_graph, std::string key_space) {
    std::string file = gpu_graph.input_dir;
    int iteration_count = config.GetInteger(key_space, "iteration_count", 1);
    spdlog::info("Run algorithm {}", key_space);
    spdlog::info("Dataset {}", file);
    spdlog::info("Number of nodes: {0}, number of edges: {1}", gpu_graph.vertex_count, gpu_graph.edge_count);
    int device = config.GetInteger(key_space, "device", 1);
    HRR(cudaSetDevice(device));

    uint thread_count = config.GetUnsigned(key_space, "thread_count", 512);
    int64_t threadsPerIntsctn = config.GetInteger64(key_space, "threads_per_intsctn", 32);
    int64_t intsctnPerBlock = thread_count / threadsPerIntsctn;
    int64_t threadShift = std::log2(threadsPerIntsctn);

    int64_t vertexCount = (int64_t)gpu_graph.vertex_count;
    int64_t edgeCount = (int64_t)gpu_graph.edge_count;
    int64_t* offsetVector = (int64_t*)gpu_graph.beg_pos;
    int64_t* indexVector;

    int block_size = 1024;
    int edge_grid_size = (edgeCount - 1) / block_size + 1;
    HRR(cudaMalloc((void**)&indexVector, sizeof(int64_t) * edgeCount));
    cuda_graph_comm::copy_32_to_64<<<edge_grid_size, block_size>>>(edgeCount, (int32_t*)gpu_graph.adj_list, indexVector);
    HRR(cudaDeviceSynchronize());

    thrust::device_vector<int64_t> dTriangleOutputVector(vertexCount, 0);
    int64_t* const dTriangle = thrust::raw_pointer_cast(dTriangleOutputVector.data());

    uint blocks = 1000000;
    if (edgeCount / 10 < blocks) {
        blocks = edgeCount / 10;
    }

    uint64_t triangleCount;
    double total_kernel_use = 0;
    double startKernel, ee;

    for (int i = 0; i < iteration_count; i++) {
        startKernel = wtime();
        kernelCall(blocks, thread_count, vertexCount, offsetVector, indexVector, dTriangle, threadsPerIntsctn, intsctnPerBlock, threadShift);
        cudaDeviceSynchronize();
        triangleCount = thrust::reduce(dTriangleOutputVector.begin(), dTriangleOutputVector.end());
        ee = wtime();
        total_kernel_use += ee - startKernel;
        if (i == 0) {
            spdlog::info("Iter 0, kernel use {:.6f} s", total_kernel_use);
            if (ee - startKernel > 0.1 && iteration_count != 1) {
                iteration_count = 10;
            }
        }
    }

    // algorithm, dataset, triangle_count, iteration_count, avg kernel time/s
    auto logger = spdlog::get("Green_file_logger");
    if (logger) {
        logger->info("{0}\t{1}\t{2}\t{3}\t{4:.6f}", "Green", gpu_graph.input_dir, triangleCount, iteration_count, total_kernel_use / iteration_count);
    } else {
        spdlog::warn("Logger 'Green_file_logger' is not initialized.");
    }

    spdlog::info("Iter {0}, avg kernel use {1:.6f} s", iteration_count, total_kernel_use / iteration_count);
    spdlog::info("Triangle count {:d}", triangleCount);

    HRR(cudaFree(indexVector));
}

void tc::approach::Green::start_up(INIReader& config, GPUGraph& gpu_graph, int argc, char** argv) {
    bool run = config.GetBoolean("comm", "Green", false);
    if (run) {
        size_t free_byte, total_byte, available_byte;
        HRR(cudaMemGetInfo(&free_byte, &total_byte));
        available_byte = total_byte - free_byte;
        spdlog::debug("Green before compute, used memory {:.2f} GB", float(total_byte - free_byte) / MEMORY_G);

        tc::approach::Green::gpu_run(config, gpu_graph);

        HRR(cudaMemGetInfo(&free_byte, &total_byte));
        spdlog::debug("Green after compute, used memory {:.2f} GB", float(total_byte - free_byte) / MEMORY_G);
        if (available_byte != total_byte - free_byte) {
            spdlog::warn("There is GPU memory that is not freed after Green runs.");
        }
    }
}