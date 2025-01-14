#include <cuda_profiler_api.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include <string>

#include "approach/TRUST/tc.h"
#include "comm/comm.h"
#include "comm/constant_comm.h"
#include "comm/cuda_comm.h"
#include "spdlog/spdlog.h"

__device__ int tc::approach::TRUST::linear_search(int neighbor, int* shared_partition, int* partition, int* bin_count, int bin, int BIN_START) {
    for (;;) {
        int i = bin;
        int len = bin_count[i];
        int step = 0;
        int nowlen;
        if (len < TRUST_SHARED_BUCKET_SIZE)
            nowlen = len;
        else
            nowlen = TRUST_SHARED_BUCKET_SIZE;
        while (step < nowlen) {
            if (shared_partition[i] == neighbor) {
                return 1;
            }
            i += TRUST_BLOCK_BUCKETNUM;
            step += 1;
        }

        len -= TRUST_SHARED_BUCKET_SIZE;
        i = bin + BIN_START;
        step = 0;
        while (step < len) {
            if (partition[i] == neighbor) {
                return 1;
            }
            i += TRUST_BLOCK_BUCKETNUM;
            step += 1;
        }
        if (len + TRUST_SHARED_BUCKET_SIZE < 99) break;
        bin++;
    }
    return 0;
}

int tc::approach::TRUST::my_binary_search(int len, int val, index_t* beg) {
    int l = 0, r = len;
    while (l < r - 1) {
        int mid = (l + r) / 2;
        if (beg[mid + 1] - beg[mid] > val)
            l = mid;
        else
            r = mid;
    }
    if (beg[l + 1] - beg[l] <= val) return -1;
    return l;
}

__global__ void tc::approach::TRUST::trust(vertex_t* adj_list, index_t* beg_pos, uint edge_count, uint vertex_count, int* partition,
                                           unsigned long long* GLOBAL_COUNT, int* G_INDEX, int CHUNK_SIZE, int warpfirstvertex) {
    // hashTable bucket 计数器
    __shared__ int bin_count[TRUST_BLOCK_BUCKETNUM];
    // 共享内存中的 hashTable
    __shared__ int shared_partition[TRUST_BLOCK_BUCKETNUM * TRUST_SHARED_BUCKET_SIZE + 1];
    unsigned long long __shared__ G_counter;
    int WARPSIZE = 32;
    if (threadIdx.x == 0) {
        G_counter = 0;
    }

    int BIN_START = blockIdx.x * TRUST_BLOCK_BUCKETNUM * TRUST_BUCKET_SIZE;
    // __syncthreads();
    unsigned long long P_counter = 0;

    // CTA for large degree vertex
    int vertex = blockIdx.x * CHUNK_SIZE;
    int vertex_end = vertex + CHUNK_SIZE;
    __shared__ int ver;
    while (vertex < warpfirstvertex){
    // while (0) {
        // if (degree<=TRUST_USE_CTA) break;
        int start = beg_pos[vertex];
        int end = beg_pos[vertex + 1];
        int now = threadIdx.x + start;
        int MODULO = TRUST_BLOCK_BUCKETNUM - 1;
        // int divide=(vert_count/blockDim.x);
        int BIN_OFFSET = 0;
        // clean bin_count
        // 初始化 hashTable bucket 计数器
        for (int i = threadIdx.x; i < TRUST_BLOCK_BUCKETNUM; i += blockDim.x) bin_count[i] = 0;
        __syncthreads();

        // start_time = clock64();
        // count hash bin
        // 生成 hashTable
        while (now < end) {
            int temp = adj_list[now];
            int bin = temp & MODULO;
            int index;
            index = atomicAdd(&bin_count[bin], 1);
            if (index < TRUST_SHARED_BUCKET_SIZE) {
                shared_partition[index * TRUST_BLOCK_BUCKETNUM + bin] = temp;
            } else if (index < TRUST_BUCKET_SIZE) {
                index = index - TRUST_SHARED_BUCKET_SIZE;
                partition[index * TRUST_BLOCK_BUCKETNUM + bin + BIN_START] = temp;
            }
            now += blockDim.x;
        }
        __syncthreads();

        if (0) {
            now = beg_pos[vertex];
            end = beg_pos[vertex + 1];
            int superwarp_ID = threadIdx.x / 64;
            int superwarp_TID = threadIdx.x % 64;
            int workid = superwarp_TID;
            now = now + superwarp_ID;
            // 获取二跳邻居节点
            int neighbor = adj_list[now];
            int neighbor_start = beg_pos[neighbor];
            int neighbor_degree = beg_pos[neighbor + 1] - neighbor_start;
            while (now < end) {
                // 如果当前一阶邻居节点已被处理完，找下一个一阶邻居节点去处理
                while (now < end && workid >= neighbor_degree) {
                    now += 16;
                    workid -= neighbor_degree;
                    neighbor = adj_list[now];
                    neighbor_start = beg_pos[neighbor];
                    neighbor_degree = beg_pos[neighbor + 1] - neighbor_start;
                }
                if (now < end) {
                    int temp = adj_list[neighbor_start + workid];
                    int bin = temp & MODULO;
                    P_counter += linear_search(temp, shared_partition, partition, bin_count, bin + BIN_OFFSET, BIN_START);
                }
                // __syncthreads();
                workid += 64;
            }
        }

        __syncthreads();
        // if (vertex>1) break;
        vertex++;
        if (vertex == vertex_end) {
            if (threadIdx.x == 0) {
                ver = atomicAdd(&G_INDEX[1], CHUNK_SIZE);
            }
            __syncthreads();
            vertex = ver;
            vertex_end = vertex + CHUNK_SIZE;
        }
        // __syncthreads();
    }

    // warp method
    int WARPID = threadIdx.x / WARPSIZE;
    int WARP_TID = threadIdx.x % WARPSIZE;
    vertex = warpfirstvertex + (WARPID + blockIdx.x * blockDim.x / WARPSIZE) * CHUNK_SIZE;
    vertex_end = vertex + CHUNK_SIZE;
    // while (vertex < vertex_count) {
    while (0) {
        int degree = beg_pos[vertex + 1] - beg_pos[vertex];
        if (degree < TRUST_USE_WARP) break;
        int start = beg_pos[vertex];
        int end = beg_pos[vertex + 1];
        int now = WARP_TID + start;
        int MODULO = TRUST_WARP_BUCKETNUM - 1;
        int BIN_OFFSET = WARPID * TRUST_WARP_BUCKETNUM;
        // clean bin_count

        for (int i = BIN_OFFSET + WARP_TID; i < BIN_OFFSET + TRUST_WARP_BUCKETNUM; i += WARPSIZE) bin_count[i] = 0;
        // bin_count[threadIdx.x]=0;
        //__syncwarp();

        // count hash bin
        while (now < end) {
            int temp = adj_list[now];
            int bin = temp & MODULO;
            bin += BIN_OFFSET;
            int index;
            index = atomicAdd(&bin_count[bin], 1);
            if (index < TRUST_SHARED_BUCKET_SIZE) {
                shared_partition[index * TRUST_BLOCK_BUCKETNUM + bin] = temp;
            } else if (index < TRUST_BUCKET_SIZE) {
                index = index - TRUST_SHARED_BUCKET_SIZE;
                partition[index * TRUST_BLOCK_BUCKETNUM + bin + BIN_START] = temp;
            }
            now += WARPSIZE;
        }
        //__syncwarp();

        if (1) {
            now = beg_pos[vertex];
            end = beg_pos[vertex + 1];

            int workid = WARP_TID;
            while (now < end) {
                int neighbor = adj_list[now];
                int neighbor_start = beg_pos[neighbor];
                int neighbor_degree = beg_pos[neighbor + 1] - neighbor_start;

                while (now < end && workid >= neighbor_degree) {
                    now++;
                    workid -= neighbor_degree;
                    neighbor = adj_list[now];
                    neighbor_start = beg_pos[neighbor];
                    neighbor_degree = beg_pos[neighbor + 1] - neighbor_start;
                }
                if (now < end) {
                    int temp = adj_list[neighbor_start + workid];
                    int bin = temp & MODULO;
                    P_counter += linear_search(temp, shared_partition, partition, bin_count, bin + BIN_OFFSET, BIN_START);
                }
                //__syncwarp();
                now = __shfl_sync(0xffffffff, now, 31);
                workid = __shfl_sync(0xffffffff, workid, 31);
                workid += WARP_TID + 1;

                // workid+=WARPSIZE;
            }
        }

        //__syncwarp();
        vertex++;
        if (vertex == vertex_end) {
            if (WARP_TID == 0) {
                vertex = atomicAdd(&G_INDEX[2], CHUNK_SIZE);
            }
            //__syncwarp();
            vertex = __shfl_sync(0xffffffff, vertex, 0);
            vertex_end = vertex + CHUNK_SIZE;
        }
    }

    atomicAdd(&G_counter, P_counter);

    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(&GLOBAL_COUNT[0], G_counter);
    }
}

void tc::approach::TRUST::gpu_run(INIReader& config, GPUGraph& gpu_graph, std::string key_space) {
    std::string file = gpu_graph.input_dir;
    int iteration_count = config.GetInteger(key_space, "iteration_count", 1);
    spdlog::info("Run algorithm {}", key_space);
    spdlog::info("Dataset {}", file);
    spdlog::info("Number of nodes: {0}, number of edges: {1}", gpu_graph.vertex_count, gpu_graph.edge_count);
    int device = config.GetInteger(key_space, "device", 1);
    HRR(cudaSetDevice(device));

    int grid_size = 1024;
    int block_size = 1024;
    int chunk_size = 1;

    uint vertex_count = gpu_graph.vertex_count;
    uint edge_count = gpu_graph.edge_count;

    index_t* h_beg_pos = (index_t*)malloc(sizeof(index_t) * (vertex_count + 1));
    HRR(cudaMemcpy(h_beg_pos, gpu_graph.beg_pos, sizeof(index_t) * (vertex_count + 1), cudaMemcpyDeviceToHost));

    int warpfirstvertex = my_binary_search(vertex_count, TRUST_USE_CTA, h_beg_pos) + 1;

    int* BIN_MEM;
    unsigned long long* GLOBAL_COUNT;
    int* G_INDEX;

    index_t* d_beg_pos = gpu_graph.beg_pos;
    vertex_t* d_adj_list = gpu_graph.adj_list;

    unsigned long long* counter = (unsigned long long*)malloc(sizeof(unsigned long long) * 10);

    HRR(cudaMalloc((void**)&BIN_MEM, sizeof(int) * grid_size * TRUST_BLOCK_BUCKETNUM * TRUST_BUCKET_SIZE));
    HRR(cudaMalloc((void**)&GLOBAL_COUNT, sizeof(unsigned long long) * 10));
    HRR(cudaMalloc((void**)&G_INDEX, sizeof(int) * 3));

    int T_Group = 32;
    int nowindex[3];
    nowindex[0] = chunk_size * grid_size * block_size / T_Group;
    nowindex[1] = chunk_size * grid_size;
    nowindex[2] = warpfirstvertex + chunk_size * (grid_size * block_size / T_Group);

    HRR(cudaMemcpy(G_INDEX, &nowindex, sizeof(int) * 3, cudaMemcpyHostToDevice));

    double total_kernel_use = 0;
    double startKernel, ee = 0;
    for (int i = 0; i < iteration_count; i++) {
        HRR(cudaMemcpy(G_INDEX, &nowindex, sizeof(int) * 3, cudaMemcpyHostToDevice));
        startKernel = wtime();
        cudaMemset(GLOBAL_COUNT, 0, sizeof(unsigned long long) * 10);
        tc::approach::TRUST::trust<<<grid_size, block_size>>>(d_adj_list, d_beg_pos, edge_count, vertex_count, BIN_MEM, GLOBAL_COUNT, G_INDEX,
                                                              chunk_size, warpfirstvertex);
        HRR(cudaDeviceSynchronize());

        ee = wtime();
        total_kernel_use += ee - startKernel;
        if (i == 0) {
            spdlog::info("Iter 0, kernel use {:.6f} s", total_kernel_use);
            if (ee - startKernel > 0.1 && iteration_count != 1) {
                iteration_count = 10;
            }
        }
    }

    HRR(cudaMemcpy(counter, GLOBAL_COUNT, sizeof(unsigned long long) * 10, cudaMemcpyDeviceToHost));

    // algorithm, dataset, triangle_count, iteration_count, avg kernel time/s
    spdlog::get("TRUST_file_logger")
        ->info("{0}\t{1}\t{2}\t{3}\t{4:.6f}", "TRUST", gpu_graph.input_dir, counter[0], iteration_count, total_kernel_use / iteration_count);

    spdlog::info("Iter {0}, avg kernel use {1:.6f} s", iteration_count, total_kernel_use / iteration_count);
    spdlog::info("Triangle count {:d}", counter[0]);

    free(counter);
    free(h_beg_pos);
    HRR(cudaFree(BIN_MEM));
    HRR(cudaFree(GLOBAL_COUNT));
    HRR(cudaFree(G_INDEX));
}

void tc::approach::TRUST::start_up(INIReader& config, GPUGraph& gpu_graph, int argc, char** argv) {
    bool run = config.GetBoolean("comm", "TRUST", false);
    if (run) {
        size_t free_byte, total_byte, available_byte;
        HRR(cudaMemGetInfo(&free_byte, &total_byte));
        available_byte = total_byte - free_byte;
        spdlog::debug("TRUST before compute, used memory {:.2f} GB", float(total_byte - free_byte) / MEMORY_G);

        tc::approach::TRUST::gpu_run(config, gpu_graph);

        HRR(cudaMemGetInfo(&free_byte, &total_byte));
        spdlog::debug("TRUST after compute, used memory {:.2f} GB", float(total_byte - free_byte) / MEMORY_G);
        if (available_byte != total_byte - free_byte) {
            spdlog::warn("There is GPU memory that is not freed after TRUST runs.");
        }
    }
}
