#include <cuda_profiler_api.h>
#include <omp.h>

#include "approach/Fox/tc.h"
#include "comm/comm.h"
#include "comm/config_comm.h"
#include "comm/constant_comm.h"
#include "comm/cuda_comm.h"
#include "graph/cuda_graph_comm.h"
#include "spdlog/spdlog.h"

__global__ void tc::approach::Fox::getEdgeWorkLoad(uint edge_count, uint16_t *d_edgeWorkLoad, vertex_t *d_src_list, vertex_t *d_adj_list,
                                                   uint *c_adjLen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= edge_count) {
        return;
    }

    int srcLen = c_adjLen[d_src_list[idx]];
    int dstLen = c_adjLen[d_adj_list[idx]];
    int large = (srcLen > dstLen) ? srcLen : dstLen;
    uint16_t bWork = (srcLen + dstLen - large) * log2((double)large + 2);
    d_edgeWorkLoad[idx] = bWork;
    return;
}

uint tc::approach::Fox::binarySearchValue(uint16_t *array, int value, uint arrayLength, int direction) {
    uint s = 0, e = arrayLength - 1;
    uint rightPos;
    bool find = false;
    uint mid = (s + e) / 2;
    while (s <= e) {
        if (array[mid] == value) {
            rightPos = mid;
            find = true;
            break;
        } else if (array[mid] < value) {
            s = mid + 1;
        } else {
            e = mid - 1;
        }
        mid = (s + e) / 2;
    }
    if (!find) {
        return s;
    }
    long int tmpValue = rightPos + direction;
    while (tmpValue >= 0 && tmpValue < arrayLength && (int)array[tmpValue] == value) {
        rightPos += direction;
        tmpValue = rightPos + direction;
    }
    return rightPos;
}

__global__ void tc::approach::Fox::binSearchKernel(index_t *c_offset, vertex_t *d_src_reorder, vertex_t *d_adj_reorder, vertex_t *c_adj_list,
                                                   uint *c_adjLen, uint edge_count, uint c_edge_start_pos, uint c_edge_end_pos, int c_threadsPerEdge,
                                                   long *c_sums) {
    long idx = blockDim.x * blockIdx.x + threadIdx.x;
    long sum = 0;
    __shared__ long sh_sum[32];
    int edgeID = idx / c_threadsPerEdge + c_edge_start_pos;
    int inEdgeID = idx % c_threadsPerEdge;

    while (edgeID < c_edge_end_pos) {
        int src = d_src_reorder[edgeID];
        int dst = d_adj_reorder[edgeID];
        if (c_adjLen[src] < c_adjLen[dst]) {
            int tmp = src;
            src = dst;
            dst = tmp;
        }
        int srcAdjListLen = c_adjLen[src];
        int dstAdjListLen = c_adjLen[dst];
        vertex_t *srcAdjList = c_adj_list + c_offset[src];
        vertex_t *dstAdjList = c_adj_list + c_offset[dst];

        for (int i = 0; i < dstAdjListLen; i += c_threadsPerEdge) {
            int dstListIdx = i + inEdgeID;
            if (dstListIdx >= dstAdjListLen) {
                continue;
            }

            int targetValue = dstAdjList[dstListIdx];

            int s = 0, e = srcAdjListLen - 1;
            int mid = (s + e) / 2;
            while (s <= e) {
                if (srcAdjList[mid] == targetValue) {
                    sum++;
                    break;
                } else if (srcAdjList[mid] < targetValue) {
                    s = mid + 1;
                } else {
                    e = mid - 1;
                }
                mid = (s + e) / 2;
            }
        }
        idx += blockDim.x * gridDim.x;
        edgeID = idx / c_threadsPerEdge + c_edge_start_pos;
        inEdgeID = idx % c_threadsPerEdge;
    }
    int tIdx = threadIdx.x;

    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    if (tIdx % 32 == 0) sh_sum[tIdx / 32] = sum;
    __syncthreads();
    if (tIdx == 0) {
        sum = 0;
        for (int i = 0; i < 32; i++) {
            sum += sh_sum[i];
        }
        c_sums[blockIdx.x] = sum;
    }
    return;
}

void tc::approach::Fox::gpu_run(INIReader &config, GPUGraph &gpu_graph, std::string key_space) {
    std::string file = gpu_graph.input_dir;
    int iteration_count = config.GetInteger(key_space, "iteration_count", 1);
    spdlog::info("Run algorithm {}", key_space);
    spdlog::info("Dataset {}", file);
    spdlog::info("Number of nodes: {0}, number of edges: {1}", gpu_graph.vertex_count, gpu_graph.edge_count);
    int device = config.GetInteger(key_space, "device", 1);
    HRR(cudaSetDevice(device));

    long int triangleCount = 0;
    uint *d_adjLength;
    index_t *d_edgeOffset = gpu_graph.beg_pos;
    vertex_t *d_src_list = gpu_graph.src_list;
    vertex_t *d_adj_list = gpu_graph.adj_list;
    uint nodeNum = gpu_graph.vertex_count;
    uint edgeNum = gpu_graph.edge_count;

    HRR(cudaMalloc(&d_adjLength, sizeof(uint) * (nodeNum + 1)));

    int block_size = 1024;
    int vertex_grid_size = (nodeNum - 1) / block_size + 1;
    int edge_grid_size = (edgeNum - 1) / block_size + 1;
    cuda_graph_comm::cal_out_degree_by_offset<<<vertex_grid_size, block_size>>>(edgeNum, nodeNum, d_adjLength, d_edgeOffset);
    HRR(cudaDeviceSynchronize());

    cuda_graph_comm::set_value_by_index(d_edgeOffset, nodeNum + 1, (index_t)edgeNum + 1);
    cuda_graph_comm::set_value_by_index(d_adjLength, nodeNum, (uint)1024);
    cuda_graph_comm::set_value_by_index(d_adj_list, edgeNum, nodeNum);

    uint16_t *edgeWorkLoad;
    vertex_t *d_src_reorder;
    vertex_t *d_adj_reorder;

    size_t edge_workload_size = (size_t)sizeof(uint16_t) * edgeNum;
    size_t edge_src_size = (size_t)sizeof(vertex_t) * edgeNum;

    HRR(cudaMalloc(&d_src_reorder, edge_src_size));
    HRR(cudaMalloc(&d_adj_reorder, edge_src_size));

    double t_start = wtime();

    int iterations = config_comm::cPreprocessingIterations;
    for (int i = 0; i < iterations; i++) {
        uint16_t *d_edgeWorkLoad;
        HRR(cudaMalloc(&d_edgeWorkLoad, edge_workload_size));
        tc::approach::Fox::getEdgeWorkLoad<<<edge_grid_size, block_size>>>(edgeNum, d_edgeWorkLoad, d_src_list, d_adj_list, d_adjLength);
        HRR(cudaDeviceSynchronize());

        size_t free_byte, total_byte;
        HRR(cudaMemGetInfo(&free_byte, &total_byte));
        spdlog::debug("Fox after get edge workload, used memory {:.2f} GB", float(total_byte - free_byte) / MEMORY_G);

        HRR(cudaMemGetInfo(&free_byte, &total_byte));
        spdlog::debug("Fox after get reorder arr, used memory {:.2f} GB", float(total_byte - free_byte) / MEMORY_G);

        HRR(cudaMemcpy(d_src_reorder, d_src_list, edge_src_size, cudaMemcpyDeviceToDevice));
        HRR(cudaMemcpy(d_adj_reorder, d_adj_list, edge_src_size, cudaMemcpyDeviceToDevice));

        vertex_t *h_src_list;
        vertex_t *h_adj_list;

        // The GPU space required for sorting is insufficient, so src_list and adj_list need to be transferred to CPU space first.
        if (edgeNum > constant_comm::kFoxMaxEdgeCount) {
            spdlog::info("Fox's sorting requires more GPU space, so src_list and adj_list are transferred to CPU space.");

            h_src_list = (vertex_t *)malloc(edge_src_size);
            h_adj_list = (vertex_t *)malloc(edge_src_size);
            HRR(cudaMemcpy(h_src_list, d_src_list, edge_src_size, cudaMemcpyDeviceToHost));
            HRR(cudaMemcpy(h_adj_list, d_adj_list, edge_src_size, cudaMemcpyDeviceToHost));
            HRR(cudaFree(d_src_list));
            HRR(cudaFree(d_adj_list));
        }

        uint16_t *d_edgeWorkLoad_copy;
        HRR(cudaMalloc(&d_edgeWorkLoad_copy, edge_workload_size));
        HRR(cudaMemcpy(d_edgeWorkLoad_copy, d_edgeWorkLoad, edge_workload_size, cudaMemcpyDeviceToDevice));

        spdlog::debug("Fox after get edge workload copy, used memory {:.2f} GB", float(total_byte - free_byte) / MEMORY_G);

        thrust::sort_by_key((thrust::device_ptr<uint16_t>)d_edgeWorkLoad_copy, (thrust::device_ptr<uint16_t>)(d_edgeWorkLoad_copy + edgeNum),
                            (thrust::device_ptr<vertex_t>)d_src_reorder);
        HRR(cudaFree(d_edgeWorkLoad_copy));

        thrust::sort_by_key((thrust::device_ptr<uint16_t>)d_edgeWorkLoad, (thrust::device_ptr<uint16_t>)(d_edgeWorkLoad + edgeNum),
                            (thrust::device_ptr<vertex_t>)d_adj_reorder);

        spdlog::debug("Fox sort successed ...");

        // After sorting, src_list and adj_list are transferred back to GPU space.
        if (edgeNum > constant_comm::kFoxMaxEdgeCount) {
            spdlog::info("Fox's sorting is completed, src_list and adj_list are transferred back to the GPU space.");

            HRR(cudaMalloc(&d_src_list, edge_src_size));
            HRR(cudaMalloc(&d_adj_list, edge_src_size));
            HRR(cudaMemcpy(d_src_list, h_src_list, edge_src_size, cudaMemcpyHostToDevice));
            HRR(cudaMemcpy(d_adj_list, h_adj_list, edge_src_size, cudaMemcpyHostToDevice));
            gpu_graph.src_list = d_src_list;
            gpu_graph.adj_list = d_adj_list;
            free(h_src_list);
            free(h_adj_list);
        }

        edgeWorkLoad = (uint16_t *)malloc(edge_workload_size);
        HRR(cudaMemcpy(edgeWorkLoad, d_edgeWorkLoad, edge_workload_size, cudaMemcpyDeviceToHost));
        HRR(cudaFree(d_edgeWorkLoad));
    }

    double t_end = wtime();

    // algorithm, dataset, iterations, avg compute time/s,
    auto preprocessing_logger = spdlog::get("Fox_preprocessing_file_logger");
    if (preprocessing_logger) {
        preprocessing_logger->info("{0}\t{1}\t{2}\t{3:.6f}", "Fox", gpu_graph.input_dir, iterations, (t_end - t_start) / iterations);
    } else {
        spdlog::warn("Logger 'Fox_preprocessing_file_logger' is not initialized.");
    }

    double total_kernel_use = 0;
    double startKernel, ee;
    for (int iter = 0; iter < iteration_count; iter++) {
        triangleCount = 0;
        int binMaxWork = edgeWorkLoad[edgeNum - 1];
        int workPerThreadB = 8;
        int maxThreadsPerEdge = 32;
        double maxPowerOf8 = log(binMaxWork) / log(workPerThreadB);
        int curBinB = maxThreadsPerEdge;
        uint *binStartPos = new uint[curBinB + 2];
        binStartPos[1] = 0;
        for (int i = 2; i <= curBinB; i *= 2) {
            double index = (double)(maxPowerOf8 - 1) * (i - 1) / (double)curBinB + 1;
            int partitionPoint = powl(workPerThreadB, index);
            binStartPos[i] = tc::approach::Fox::binarySearchValue(edgeWorkLoad, partitionPoint, edgeNum, -1);
            binStartPos[i / 2 + 1] = binStartPos[i];
        }

        binStartPos[curBinB + 1] = edgeNum;

        for (int i = 1; i <= curBinB; i *= 2) {
            // for each bin
            if (binStartPos[i + 1] - binStartPos[i] == 0) {
                continue;
            }

            uint c_edge_start_pos = binStartPos[i];
            uint c_edge_end_pos = binStartPos[i + 1];

            uint curGridSize = 20000;
            int maxBlockNumTC = 20000;
            int maxBlockSizeTC = 1024;
            int curThreadsPerEdge = i;
            long *d_sum;
            HRR(cudaMalloc(&d_sum, sizeof(long) * maxBlockNumTC));
            HRR(cudaMemset(d_sum, 0, sizeof(long) * maxBlockNumTC));
            startKernel = wtime();
            tc::approach::Fox::binSearchKernel<<<curGridSize, maxBlockSizeTC>>>(d_edgeOffset, d_src_reorder, d_adj_reorder, d_adj_list, d_adjLength,
                                                                                edgeNum, c_edge_start_pos, c_edge_end_pos, curThreadsPerEdge, d_sum);
            HRR(cudaDeviceSynchronize());
            triangleCount += thrust::reduce((thrust::device_ptr<long>)d_sum, (thrust::device_ptr<long>)(d_sum + curGridSize));
            ee = wtime();
            total_kernel_use += ee - startKernel;
            HRR(cudaFree(d_sum));
        }

        if (iter == 0) {
            spdlog::info("Iter 0, kernel use {:.6f} s", total_kernel_use);
            if (total_kernel_use > 0.1 && iteration_count != 1) {
                iteration_count = 10;
            }
        }
    }

    // algorithm, dataset, triangle_count, iteration_count, avg kernel time/s
    auto logger = spdlog::get("Fox_file_logger");
    if (logger) {
        logger->info("{0}\t{1}\t{2}\t{3}\t{4:.6f}", "Fox", gpu_graph.input_dir, triangleCount, iteration_count, total_kernel_use / iteration_count);
    } else {
        spdlog::warn("Logger 'Fox_file_logger' is not initialized.");
    }

    spdlog::info("Iter {0}, avg kernel use {1:.6f} s", iteration_count, total_kernel_use / iteration_count);
    spdlog::info("Triangle count {:d}", triangleCount);

    free(edgeWorkLoad);
    HRR(cudaFree(d_adjLength));
    HRR(cudaFree(d_src_reorder));
    HRR(cudaFree(d_adj_reorder));
}

void tc::approach::Fox::start_up(INIReader &config, GPUGraph &gpu_graph, int argc, char **argv) {
    bool run = config.GetBoolean("comm", "Fox", false);
    if (run) {
        size_t free_byte, total_byte, available_byte;
        HRR(cudaMemGetInfo(&free_byte, &total_byte));
        available_byte = total_byte - free_byte;
        spdlog::debug("Fox before compute, used memory {:.2f} GB", float(total_byte - free_byte) / MEMORY_G);

        tc::approach::Fox::gpu_run(config, gpu_graph);

        HRR(cudaMemGetInfo(&free_byte, &total_byte));
        spdlog::debug("Fox after compute, used memory {:.2f} GB", float(total_byte - free_byte) / MEMORY_G);
        if (available_byte != total_byte - free_byte) {
            spdlog::warn("There is GPU memory that is not freed after Fox runs.");
        }
    }
}
