#include <cuda_profiler_api.h>
#include <omp.h>

#include <cassert>

#include "approach/Bisson/tc.h"
#include "comm/comm.h"
#include "comm/constant_comm.h"
#include "comm/cuda_comm.h"
#include "graph/cuda_graph_comm.h"
#include "spdlog/spdlog.h"

__global__ void tc::approach::Bisson::triangleCountKernel(uint nodeNum, index_t *c_offset, vertex_t *c_row, uint *c_adjLen, long int *c_sum,
                                                          int *c_bitmap) {
    long int sum = 0;
    int curRowNum = blockIdx.x;
    int lane_id = threadIdx.x % 32;
    __shared__ int sh_bitMap[Bisson_shareMemorySizeInBlock];
    uint intSizePerBitmap = (nodeNum + 31) / 32;
    int *myBitmap = c_bitmap + blockIdx.x * intSizePerBitmap;
    while (1) {
        //__syncthreads();
        // int privateRowNum = (curRowNum < nonZeroSize) ? c_nonZeroRow[curRowNum] : totalNodeNum;
        int privateRowNum = curRowNum;
        if (privateRowNum >= nodeNum) {
            break;
        }
        // if (c_offset[privateRowNum+1] == c_offset[privateRowNum])
        //	continue;
        vertex_t *curNodeNbr = c_row + c_offset[privateRowNum];
        uint curNodeNbrLength = c_offset[privateRowNum + 1] - c_offset[privateRowNum];
        if (curNodeNbrLength > 256) {
            // if (1) {
            if (threadIdx.x == 0) {
                memset(myBitmap, 0, sizeof(int) * intSizePerBitmap);
                memset(sh_bitMap, 0, sizeof(int) * Bisson_shareMemorySizeInBlock);
            }
            __threadfence();
            for (int i = (curNodeNbrLength + blockDim.x - 1) / blockDim.x - 1; i >= 0; i--) {
                int curIndex = i * blockDim.x + threadIdx.x;
                int curNbr;
                if (curIndex < curNodeNbrLength) {
                    curNbr = curNodeNbr[curIndex];
                    atomicOr(myBitmap + (curNbr / 32), 1 << (31 - curNbr % 32));
                    atomicOr(sh_bitMap + (curNbr / Bisson_hIndex / 32), 1 << (31 - (curNbr / Bisson_hIndex) % 32));
                }
                __syncthreads();
                if (curIndex < curNodeNbrLength) {
                    vertex_t *twoHoopNbr = c_row + c_offset[curNbr];
                    uint twoHoopNbrLength = c_offset[curNbr + 1] - c_offset[curNbr];
                    for (int j = 0; j < twoHoopNbrLength; j++) {
                        vertex_t curValue = twoHoopNbr[j];
                        if (((sh_bitMap[curValue / Bisson_hIndex / 32] >> (31 - (curValue / Bisson_hIndex) % 32)) & 1) &&
                            ((myBitmap[curValue / 32] >> (31 - curValue % 32)) & 1)) {
                            sum++;
                        }
                    }
                }
            }
        } else {
            if (threadIdx.x == 0) memcpy(sh_bitMap, curNodeNbr, sizeof(int) * curNodeNbrLength);
            __threadfence();
            for (int i = lane_id; i < curNodeNbrLength; i += 32) {
                int curNbr = curNodeNbr[i];
                vertex_t *twoHoopNbr = c_row + c_offset[curNbr];
                int twoHoopNbrLength = c_offset[curNbr + 1] - c_offset[curNbr];
                for (int j = 0; j < twoHoopNbrLength; j++) {
                    int targetValue = twoHoopNbr[j];
                    int s = 0, e = curNodeNbrLength, mid;
                    while (s < e) {
                        mid = (s + e) / 2;
                        if (sh_bitMap[mid] > targetValue)
                            e = mid;
                        else if (sh_bitMap[mid] < targetValue)
                            s = mid + 1;
                        else {
                            sum++;
                            break;
                        }
                    }
                }
            }
        }
        curRowNum += gridDim.x;  // atomicAdd(&nextNode, 1);
                                 //__syncthreads();
                                 // if (privateRowNum != curRowNum)
                                 //	printf("private is %d, curRowNum is %d, block %d, index %d\n",privateRowNum,curRowNum,blockIdx.x,threadIdx.x);
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

void tc::approach::Bisson::gpu_run(INIReader &config, GPUGraph &gpu_graph, std::string key_space) {
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

    HRR(cudaMalloc(&d_adjLength, sizeof(int) * (nodeNum + 1)));

    int block_size = 1024;
    int vertex_grid_size = (nodeNum - 1) / block_size + 1;
    cuda_graph_comm::cal_out_degree_by_offset<<<vertex_grid_size, block_size>>>(edgeNum, nodeNum, d_adjLength, d_edgeOffset);
    HRR(cudaDeviceSynchronize());

    cuda_graph_comm::set_value_by_index(d_edgeOffset, nodeNum + 1, (index_t)edgeNum + 1);
    cuda_graph_comm::set_value_by_index(d_adjLength, nodeNum, (uint)1024);
    cuda_graph_comm::set_value_by_index(d_edgeRow, edgeNum, nodeNum);

    int bitPerInt = sizeof(int) * 8;
    int intSizePerBitmap = (nodeNum + bitPerInt - 1) / bitPerInt;
    int blockSize = 32;
    int blockNum = 30 * 2048 / blockSize;

    if (nodeNum > Bisson_hIndex * Bisson_shareMemorySizeInBlock * 32) {
        spdlog::error("The nodeNum is too large: {}", nodeNum);
        HRR(cudaFree(d_adjLength));
        return;
    }
    if ((long long int)blockNum * intSizePerBitmap * sizeof(int) > (long long int)16 * MEMORY_G) {
        spdlog::error("The bitmap is too large: {} bytes", (long long int)blockNum * intSizePerBitmap * sizeof(int));
        HRR(cudaFree(d_adjLength));
        return;
    }

    int *d_bitmaps;
    HRR(cudaMalloc(&d_bitmaps, sizeof(int) * intSizePerBitmap * blockNum));

    long int *d_sum;
    unsigned maxWarpPerGrid = blockNum * blockSize / 32;
    HRR(cudaMalloc(&d_sum, sizeof(long int) * maxWarpPerGrid));
    HRR(cudaMemset(d_sum, 0, sizeof(long int) * maxWarpPerGrid));

    double total_kernel_use = 0;
    double startKernel, ee;
    for (int i = 0; i < iteration_count; i++) {
        startKernel = wtime();
        tc::approach::Bisson::triangleCountKernel<<<blockNum, blockSize>>>(nodeNum, d_edgeOffset, d_edgeRow, d_adjLength, d_sum, d_bitmaps);
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

    // algorithm, dataset, triangle_count, iteration_count, avg kernel time/s
    auto logger = spdlog::get("Bisson_file_logger");
    if (logger) {
        logger->info("{0}\t{1}\t{2}\t{3}\t{4:.6f}", "Bisson", gpu_graph.input_dir, triangleCount, iteration_count, total_kernel_use / iteration_count);
    } else {
        spdlog::warn("Logger 'Bisson_file_logger' is not initialized.");
    }

    spdlog::info("Iter {0}, avg kernel use {1:.6f} s", iteration_count, total_kernel_use / iteration_count);
    spdlog::info("Triangle count {:d}", triangleCount);

    HRR(cudaFree(d_sum));
    HRR(cudaFree(d_adjLength));
    HRR(cudaFree(d_bitmaps));
}

void tc::approach::Bisson::start_up(INIReader &config, GPUGraph &gpu_graph, int argc, char **argv) {
    bool run = config.GetBoolean("comm", "Bisson", false);
    if (run) {
        size_t free_byte, total_byte, available_byte;
        HRR(cudaMemGetInfo(&free_byte, &total_byte));
        available_byte = total_byte - free_byte;
        spdlog::debug("Bisson before compute, used memory {:.2f} GB", float(total_byte - free_byte) / MEMORY_G);

        tc::approach::Bisson::gpu_run(config, gpu_graph);

        HRR(cudaMemGetInfo(&free_byte, &total_byte));
        spdlog::debug("Bisson after compute, used memory {:.2f} GB", float(total_byte - free_byte) / MEMORY_G);
        if (available_byte != total_byte - free_byte) {
            spdlog::warn("There is GPU memory that is not freed after Bisson runs.");
        }
    }
}
