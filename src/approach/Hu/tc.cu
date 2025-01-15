#include <cuda_profiler_api.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include <string>

#include "approach/Hu/tc.h"
#include "comm/comm.h"
#include "comm/config_comm.h"
#include "comm/constant_comm.h"
#include "comm/cuda_comm.h"
#include "graph/cuda_graph_comm.h"
#include "spdlog/spdlog.h"

__global__ void tc::approach::Hu::getNodesWorkLoad(int startPos, int threadNum, long int *d_nodeWorkLoad, index_t *c_offset, vertex_t *c_row,
                                                   uint *c_adjLen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= threadNum) return;
    int src = startPos + idx;
    vertex_t *srcList = c_row + c_offset[src];
    uint srcListLen = c_adjLen[src];
    long int totalLength = 0;
    for (int i = 0; i < srcListLen; i++) {
        totalLength += c_adjLen[srcList[i]];
    }
    d_nodeWorkLoad[idx] = totalLength * (unsigned)log2((double)srcListLen);
    return;
}

uint tc::approach::Hu::binarySearchValue(long int *array, long int value, uint arrayLength, int direction) {
    long int s = 0, e = arrayLength - 1;
    long int rightPos;
    bool find = false;
    long int mid = (s + e) / 2;
    while (s <= e) {
        if (array[mid] == value) {
            rightPos = mid;
            find = true;
            break;
        } else if (array[mid] < value) {
            s = mid + 1;
        } else {
            if (e == 0) break;
            e = mid - 1;
        }
        mid = (s + e) / 2;
    }
    if (!find) {
        return s;
    }
    long int tmpValue = rightPos + direction;
    while (tmpValue >= 0 && tmpValue < arrayLength && array[tmpValue] == value) {
        rightPos += direction;
        tmpValue = rightPos + direction;
    }
    return rightPos;
}

__global__ void tc::approach::Hu::triangleCountKernel(index_t *c_offset, vertex_t *c_row, uint *c_adjLen, int *c_blockStartNodeOffset,
                                                      long int *c_sum) {
    int uid = c_blockStartNodeOffset[blockIdx.x];
    int uidThre = c_blockStartNodeOffset[blockIdx.x + 1];
    if (uid == uidThre) return;
    unsigned vpos = c_offset[uid];
    int vid = c_row[vpos];
    int wpos = threadIdx.x;
    // int *srcList = c_row + c_offset[uid];
    // int *dstList = c_row + c_offset[c_row[vpos]];
    long int sum = 0;
    __shared__ int cachedMaxUid;
    __shared__ bool cacheWorked;
    __shared__ int sharedVid[HU_shareMemorySizeInBlock];
    __shared__ int lastCachedMaxUid;
    while (1) {
        __syncthreads();
        if (threadIdx.x == 0) {
            if (uid == c_blockStartNodeOffset[blockIdx.x]) {
                cachedMaxUid = uid;
            }
            lastCachedMaxUid = cachedMaxUid;
            cachedMaxUid--;
            cacheWorked = true;
            int cachedVid = 0;
            int cachedUid = 0;
            while (1) {
                cachedMaxUid++;
                cachedUid++;
                cachedVid += c_adjLen[cachedMaxUid];
                if (cachedVid >= HU_shareMemorySizeInBlock || cachedMaxUid >= uidThre) {
                    break;
                }
            }
            if (cachedUid == 1) {
                cacheWorked = false;
                cachedMaxUid++;
                cachedMaxUid = (uidThre > cachedMaxUid) ? cachedMaxUid : uidThre;
            } else {
                int len = cachedVid - c_adjLen[cachedMaxUid];
                memcpy(sharedVid, c_row + c_offset[lastCachedMaxUid], sizeof(int) * len);
            }
        }
        __syncthreads();

        while (vpos >= c_offset[uid + 1]) uid++;
        while (1) {
            while (wpos >= c_adjLen[vid]) {
                wpos -= c_adjLen[vid];
                vpos++;
                vid = c_row[vpos];
                while (vpos >= c_offset[uid + 1]) {
                    uid++;
                }
            }
            if (uid >= cachedMaxUid) {
                break;
            }
            vertex_t *dstList = c_row + c_offset[vid];
            int targetValue = dstList[wpos];
            if (!cacheWorked) {
                vertex_t *srcList = c_row + c_offset[uid];
                uint s = 0, e = c_adjLen[uid];
                uint mid = (s + e) >> 1;
                while (s + 1 < e) {
                    if (srcList[mid] <= targetValue) {
                        s = mid;
                    } else {
                        e = mid;
                    }
                    mid = (s + e) >> 1;
                }
                if (srcList[s] == targetValue) sum++;
            } else {
                int adjListOff = c_offset[uid] - c_offset[lastCachedMaxUid];
                int s = 0, e = c_adjLen[uid];
                int mid = (s + e) >> 1;
                while (s + 1 < e) {
                    if (sharedVid[adjListOff + mid] <= targetValue) {
                        s = mid;
                    } else {
                        e = mid;
                    }
                    mid = (s + e) >> 1;
                }
                if (sharedVid[adjListOff + s] == targetValue) sum++;
            }
            wpos += HU_threadsPerBlockInTC;
        }
        if (cachedMaxUid >= uidThre) break;
    }
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadIdx.x % 32 == 0) {
        c_sum[idx >> 5] = sum;
    }
    return;
}

void tc::approach::Hu::gpu_run(INIReader &config, GPUGraph &gpu_graph, std::string key_space) {
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
    vertex_t *d_edgeRow = gpu_graph.adj_list;
    uint nodeNum = gpu_graph.vertex_count;
    uint edgeNum = gpu_graph.edge_count;

    HRR(cudaMalloc(&d_adjLength, sizeof(uint) * (nodeNum + 1)));

    int block_size = 1024;
    int vertex_grid_size = (nodeNum - 1) / block_size + 1;
    cuda_graph_comm::cal_out_degree_by_offset<<<vertex_grid_size, block_size>>>(edgeNum, nodeNum, d_adjLength, d_edgeOffset);
    HRR(cudaDeviceSynchronize());

    cuda_graph_comm::set_value_by_index(d_edgeOffset, nodeNum + 1, (index_t)edgeNum + 1);
    cuda_graph_comm::set_value_by_index(d_adjLength, nodeNum, (uint)1024);
    cuda_graph_comm::set_value_by_index(d_edgeRow, edgeNum, nodeNum);

    cuda_graph_comm::check_array("d_adjLength", d_adjLength, nodeNum + 1, nodeNum - 10, nodeNum + 1);

    long int *nodeWorkLoad;
    int maxThreadsPerBlock = 1024;
    int maxBlocksPerGrid = 64000;
    int maxThreadsPerKernel = maxBlocksPerGrid * maxThreadsPerBlock;
    long int *d_nodeWorkLoad;
    int *d_blockStartNodeOffset;

    int threadsPerKernelInTC = HU_threadsPerBlockInTC * HU_blocksPerKernelInTC;
    int maxWarpPerGrid = threadsPerKernelInTC / 32;
    int *blockStartNodeOffset;
    HRR(cudaMalloc(&d_blockStartNodeOffset, sizeof(int) * (HU_blocksPerKernelInTC + 1)));

    double t_start = wtime();

    int iterations = config_comm::cPreprocessingIterations;
    for (int iter = 0; iter < iterations; iter++) {
        HRR(cudaMalloc(&d_nodeWorkLoad, sizeof(long int) * maxThreadsPerKernel));

        nodeWorkLoad = new long int[nodeNum];

        for (int i = 0; i < (nodeNum + maxThreadsPerKernel - 1) / maxThreadsPerKernel; i++) {
            int curThread = maxThreadsPerKernel;
            int remainedNodes = nodeNum - i * maxThreadsPerKernel;
            curThread = (remainedNodes > curThread) ? curThread : remainedNodes;
            HRR(cudaMemset(d_nodeWorkLoad, 0, sizeof(long) * curThread));
            tc::approach::Hu::getNodesWorkLoad<<<(curThread + maxThreadsPerBlock - 1) / maxThreadsPerBlock, maxThreadsPerBlock>>>(
                i * maxThreadsPerKernel, curThread, d_nodeWorkLoad, d_edgeOffset, d_edgeRow, d_adjLength);

            HRR(cudaMemcpy(nodeWorkLoad + i * maxThreadsPerKernel, d_nodeWorkLoad, sizeof(long int) * curThread, cudaMemcpyDeviceToHost));
        }
        HRR(cudaFree(d_nodeWorkLoad));

        for (int i = 1; i < nodeNum; i++) nodeWorkLoad[i] += nodeWorkLoad[i - 1];

        long int workLoadStep = (nodeWorkLoad[nodeNum - 1] + HU_blocksPerKernelInTC - 1) / HU_blocksPerKernelInTC;

        blockStartNodeOffset = new int[HU_blocksPerKernelInTC + 1];
        blockStartNodeOffset[0] = 0;
#pragma omp parallel for
        for (int i = 1; i < HU_blocksPerKernelInTC; i++) {
            blockStartNodeOffset[i] = (int)tc::approach::Hu::binarySearchValue(nodeWorkLoad, i * workLoadStep + 1, nodeNum, -1);
        }
        blockStartNodeOffset[HU_blocksPerKernelInTC] = nodeNum;

        delete[] nodeWorkLoad;

        HRR(cudaMemcpy((void *)d_blockStartNodeOffset, (void *)blockStartNodeOffset, sizeof(int) * (HU_blocksPerKernelInTC + 1),
                       cudaMemcpyHostToDevice));

        delete[] blockStartNodeOffset;
    }

    double t_end = wtime();

    // algorithm, dataset, iterations, avg compute time/s,
    auto preprocessing_logger = spdlog::get("Hu_preprocessing_file_logger");
    if (preprocessing_logger) {
        preprocessing_logger->info("{0}\t{1}\t{2}\t{3:.6f}", "Hu", gpu_graph.input_dir, iterations, (t_end - t_start) / iterations);
    } else {
        spdlog::warn("Logger 'Hu_preprocessing_file_logger' is not initialized.");
    }

    long int *d_sum;
    HRR(cudaMalloc(&d_sum, sizeof(long int) * maxWarpPerGrid));
    HRR(cudaMemset(d_sum, 0, sizeof(long int) * maxWarpPerGrid));

    double total_kernel_use = 0;
    double startKernel, ee;
    for (int i = 0; i < iteration_count; i++) {
        startKernel = wtime();

        tc::approach::Hu::triangleCountKernel<<<HU_blocksPerKernelInTC, HU_threadsPerBlockInTC>>>(d_edgeOffset, d_edgeRow, d_adjLength,
                                                                                                  d_blockStartNodeOffset, d_sum);
        HRR(cudaDeviceSynchronize());
        triangleCount = thrust::reduce((thrust::device_ptr<long>)d_sum, (thrust::device_ptr<long>)(d_sum + maxWarpPerGrid));

        ee = wtime();
        total_kernel_use += ee - startKernel;
        if (i == 0) {
            spdlog::info("Iter 0, kernel use {:.6f} s", total_kernel_use);
            if (ee - startKernel > 0.1 && iteration_count != 1) {
                iteration_count = 10;
            }
        }
    }

    // algorithm, dataset, iteration_count, avg compute time/s,
    auto logger = spdlog::get("Hu_file_logger");
    if (logger) {
        logger->info("{0}\t{1}\t{2}\t{3}\t{4:.6f}", "Hu", gpu_graph.input_dir, triangleCount, iteration_count, total_kernel_use / iteration_count);
    } else {
        spdlog::warn("Logger 'Hu_file_logger' is not initialized.");
    }

    spdlog::info("Iter {0}, avg kernel use {1:.6f} s", iteration_count, total_kernel_use / iteration_count);
    spdlog::info("Triangle count {:d}", triangleCount);

    HRR(cudaFree(d_sum));
    HRR(cudaFree(d_blockStartNodeOffset));
    HRR(cudaFree(d_adjLength));
}

void tc::approach::Hu::start_up(INIReader &config, GPUGraph &gpu_graph, int argc, char **argv) {
    bool run = config.GetBoolean("comm", "Hu", false);
    if (run) {
        size_t free_byte, total_byte, available_byte;
        HRR(cudaMemGetInfo(&free_byte, &total_byte));
        available_byte = total_byte - free_byte;
        spdlog::debug("Hu before compute, used memory {:.2f} GB", float(total_byte - free_byte) / MEMORY_G);

        tc::approach::Hu::gpu_run(config, gpu_graph);

        HRR(cudaMemGetInfo(&free_byte, &total_byte));
        spdlog::debug("Hu after compute, used memory {:.2f} GB", float(total_byte - free_byte) / MEMORY_G);
        if (available_byte != total_byte - free_byte) {
            spdlog::warn("There is GPU memory that is not freed after Hu runs.");
        }
    }
}
