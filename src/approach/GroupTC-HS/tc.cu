#include <cuda_profiler_api.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include <string>

#include "approach/GroupTC-HS/tc.h"
#include "comm/comm.h"
#include "comm/constant_comm.h"
#include "comm/cuda_comm.h"
#include "spdlog/spdlog.h"

typedef struct longint2 {
    long long int x, y;  // 两个 long 类型的成员
} longint2;

__device__ int tc::approach::GroupTC_HS::linear_search_block(int neighbor, int *partition, int len, int bin, int BIN_START) {
    for (;;) {
        int i = bin + BIN_START;
        int step = 0;
        while (step < len) {
            if (partition[i] == neighbor) {
                return 1;
            }
            i += GroupTC_HS_block_bucketnum;
            step += 1;
        }
        if (len + GroupTC_HS_shared_BLOCK_BUCKET_SIZE < 99) break;
        bin++;
    }
    return 0;
}

__device__ int tc::approach::GroupTC_HS::linear_search_group(int neighbor, int *partition, int len, int bin, int BIN_START) {
    len -= GroupTC_HS_shared_GROUP_BUCKET_SIZE;
    int i = bin + BIN_START;
    int step = 0;
    while (step < len) {
        if (partition[i] == neighbor) {
            return 1;
        }
        i += GroupTC_HS_group_bucketnum;
        step += 1;
    }

    return 0;
}

int tc::approach::GroupTC_HS::my_binary_search(int len, int val, index_t *beg) {
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

__device__ unsigned int tc::approach::GroupTC_HS::fmix32(unsigned int h) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

template <const int GroupTC_HS_Group_SUBWARP_SIZE, const int GroupTC_HS_Group_WARP_STEP, const int CHUNK_SIZE>
__global__ void tc::approach::GroupTC_HS::grouptc_hs(vertex_t *src_list, vertex_t *adj_list, index_t *beg_pos, uint edge_count, uint vertex_count,
                                                     int *partition, unsigned long long *GLOBAL_COUNT, int T_Group, int *G_INDEX, int warpfirstvertex,
                                                     int warpfirstedge, int nocomputefirstvertex, int nocomputefirstedge) {
    // hashTable bucket 计数器
    __shared__ int bin_count[GroupTC_HS_block_bucketnum];

    // 共享内存中的 hashTable
    __shared__ int shared_partition[GroupTC_HS_block_bucketnum * GroupTC_HS_shared_BLOCK_BUCKET_SIZE];
    // unsigned long long __shared__ G_counter;

    // if (threadIdx.x == 0) {
    //     G_counter = 0;
    // }
    int __shared__ vertex;

    if (threadIdx.x == 0) {
        vertex = blockIdx.x;
    }
    __syncthreads();

    int BIN_START = blockIdx.x * GroupTC_HS_block_bucketnum * GroupTC_HS_BLOCK_BUCKET_SIZE;
    unsigned long long P_counter = 0;

    // CTA for large degree vertex
    // int vertex = blockIdx.x * CHUNK_SIZE;
    // int vertex_end = vertex + CHUNK_SIZE;

    while (vertex < warpfirstvertex) {
        // for (int vertex = blockIdx.x; vertex < warpfirstvertex; vertex+=gridDim.x) {
        int group_start = beg_pos[vertex];
        int end = beg_pos[vertex + 1];
        int now = threadIdx.x + group_start;

        // clear shared_partition
        bin_count[threadIdx.x] = 0;
        shared_partition[0 * GroupTC_HS_block_bucketnum + threadIdx.x] = -1;
        shared_partition[1 * GroupTC_HS_block_bucketnum + threadIdx.x] = -1;
        shared_partition[2 * GroupTC_HS_block_bucketnum + threadIdx.x] = -1;
        shared_partition[3 * GroupTC_HS_block_bucketnum + threadIdx.x] = -1;
        // shared_partition[4 * GroupTC_HS_block_bucketnum + threadIdx.x] = -1;
        // shared_partition[5 * GroupTC_HS_block_bucketnum + threadIdx.x] = -1;
        __syncthreads();

        // count hash bin
        // 生成 hashTable
        while (now < end) {
            int temp = adj_list[now];
            int table_id = 0;
            int iter = 0;
            while (temp != -1 && iter++ < GroupTC_HS_Max_Conflict) {
                unsigned int bin = (temp * (table_id ? GroupTC_HS_H1 : GroupTC_HS_H0)) & GroupTC_HS_BUCKET_NUM;
                temp = atomicExch(shared_partition + bin, temp);
                table_id = 1 - table_id;
            }

            if (temp != -1) {
                int bin = temp & GroupTC_HS_BLOCK_MODULO;
                int index;
                index = atomicAdd(&bin_count[bin], 1);
                if (index < GroupTC_HS_shared_Basic_BUCKET_SIZE) {
                    shared_partition[(index + GroupTC_HS_shared_Cuckoo_BUCKET_SIZE) * GroupTC_HS_block_bucketnum + bin] = temp;
                } else if (index < GroupTC_HS_BLOCK_BUCKET_SIZE) {
                    partition[(index - GroupTC_HS_shared_Basic_BUCKET_SIZE) * GroupTC_HS_block_bucketnum + bin + BIN_START] = temp;
                }
            }

            now += blockDim.x;
        }
        __syncthreads();

        if (1) {
            // list intersection
            now = beg_pos[vertex];
            end = beg_pos[vertex + 1];
            int superwarp_ID = threadIdx.x / GroupTC_HS_Cuckoo_WARP_SIZE;
            int superwarp_TID = threadIdx.x % GroupTC_HS_Cuckoo_WARP_SIZE;
            int workid = superwarp_TID;
            now = now + superwarp_ID;
            // 获取二跳邻居节点
            int neighbor = adj_list[now];
            int neighbor_start = beg_pos[neighbor];
            int neighbor_degree = beg_pos[neighbor + 1] - neighbor_start;
            while (now < end) {
                // 如果当前一阶邻居节点已被处理完，找下一个一阶邻居节点去处理
                while (now < end && workid >= neighbor_degree) {
                    now += GroupTC_HS_BLOCK_SIZE / GroupTC_HS_Cuckoo_WARP_SIZE;
                    workid -= neighbor_degree;
                    neighbor = adj_list[now];
                    neighbor_start = beg_pos[neighbor];
                    neighbor_degree = beg_pos[neighbor + 1] - neighbor_start;
                }
                if (now < end) {
                    int temp_adj = adj_list[neighbor_start + workid];

                    const int v1 = shared_partition[(temp_adj * GroupTC_HS_H0) & GroupTC_HS_BUCKET_NUM];
                    const int v2 = shared_partition[(temp_adj * GroupTC_HS_H1) & GroupTC_HS_BUCKET_NUM];

                    P_counter += (v1 == temp_adj) | (v2 == temp_adj);

                    if (!(v1 == temp_adj || v2 == temp_adj) && !(v1 == -1 && v2 == -1)) {
                        int bin = temp_adj & GroupTC_HS_BLOCK_MODULO;
                        int len = bin_count[bin];

                        P_counter += len > 0 && shared_partition[bin + GroupTC_HS_block_bucketnum * 4] == temp_adj;
                        P_counter += len > 1 && shared_partition[bin + GroupTC_HS_block_bucketnum * 5] == temp_adj;

                        if (len > 2) {
                            P_counter += tc::approach::GroupTC_HS::linear_search_block(temp_adj, partition, len - GroupTC_HS_shared_Basic_BUCKET_SIZE,
                                                                                       bin, BIN_START);
                        }
                    }
                }
                workid += GroupTC_HS_Cuckoo_WARP_SIZE;
            }
        }

        __syncthreads();
        if (threadIdx.x == 0) {
            vertex = atomicAdd(&G_INDEX[1], CHUNK_SIZE);
        }
        __syncthreads();
        // // if (vertex>1) break;
        // vertex++;
        // if (vertex == vertex_end) {
        //     if (threadIdx.x == 0) {
        //         ver = atomicAdd(&G_INDEX[1], CHUNK_SIZE);
        //     }
        //     __syncthreads();
        //     vertex = ver;
        //     vertex_end = vertex + CHUNK_SIZE;
        // }
    }

    // EDGE CHUNK for small degree vertex
    __shared__ int group_start;
    __shared__ int group_size;

    int *shared_src = shared_partition + GroupTC_HS_group_bucketnum * GroupTC_HS_shared_GROUP_BUCKET_SIZE;
    int *shared_adj_start = shared_src + GroupTC_HS_shared_CHUNK_CACHE_SIZE;
    int *shared_adj_degree = shared_adj_start + GroupTC_HS_shared_CHUNK_CACHE_SIZE;

    if (1) {
        for (int group_offset = warpfirstedge + blockIdx.x * GroupTC_HS_EDGE_CHUNK; group_offset < nocomputefirstedge;
             group_offset += gridDim.x * GroupTC_HS_EDGE_CHUNK) {
            // compute group start and end
            if (threadIdx.x == 0) {
                int src = src_list[group_offset];
                int src_start = beg_pos[src];
                int src_end = beg_pos[src + 1];
                group_start = ((src_start == group_offset) ? src_start : src_end);

                src = src_list[min(group_offset + GroupTC_HS_EDGE_CHUNK, nocomputefirstedge) - 1];
                group_size = min(beg_pos[src + 1], (index_t)nocomputefirstedge) - group_start;
            }

            // cache start
            for (int i = threadIdx.x; i < GroupTC_HS_group_bucketnum; i += blockDim.x) bin_count[i] = 0;

            __syncthreads();

            for (int i = threadIdx.x; i < group_size; i += GroupTC_HS_BLOCK_SIZE) {
                int temp_src = src_list[i + group_start];
                int temp_adj = adj_list[i + group_start];

                longint2 *point_int2 = reinterpret_cast<longint2 *>(beg_pos + temp_adj);
                longint2 pos2 = *point_int2;
                shared_src[i] = temp_src;
                shared_adj_start[i] = pos2.x;
                shared_adj_degree[i] = pos2.y - pos2.x;

                // if (shared_adj_start[i] != beg_pos[temp_adj]) {
                //     printf("shared_adj_start[%d] = %d, beg_pos[%d] = %d\n", i, shared_adj_start[i], temp_adj, beg_pos[temp_adj]);
                // }
                // if(shared_adj_degree[i] != beg_pos[temp_adj + 1] - shared_adj_start[i]) {
                //     printf("shared_adj_degree[%d] = %d, beg_pos[%d + 1] - shared_adj_start[%d] = %d\n", i, shared_adj_degree[i], temp_adj,
                //     temp_adj, beg_pos[temp_adj + 1] - shared_adj_start[i]);
                // }

                // shared_src[i] = temp_src;
                // shared_adj_start[i] = beg_pos[temp_adj];
                // shared_adj_degree[i] = beg_pos[temp_adj + 1] - shared_adj_start[i];

                int bin = (temp_src + temp_adj) & GroupTC_HS_GROUP_MODULO;
                int index = atomicAdd(&bin_count[bin], 1);

                if (index < GroupTC_HS_shared_GROUP_BUCKET_SIZE) {
                    shared_partition[index * GroupTC_HS_group_bucketnum + bin] = temp_adj;
                } else if (index < GroupTC_HS_GROUP_BUCKET_SIZE) {
                    index = index - GroupTC_HS_shared_GROUP_BUCKET_SIZE;
                    partition[index * GroupTC_HS_group_bucketnum + bin + BIN_START] = temp_adj;
                }
            }
            __syncthreads();

            if (1) {
                // compute 2 hop neighbors
                int now = threadIdx.x / GroupTC_HS_Group_SUBWARP_SIZE;
                int workid = threadIdx.x % GroupTC_HS_Group_SUBWARP_SIZE;

                while (now < group_size) {
                    int neighbor_degree = shared_adj_degree[now];
                    while (now < group_size && workid >= neighbor_degree) {
                        now += GroupTC_HS_Group_WARP_STEP;
                        workid -= neighbor_degree;
                        neighbor_degree = shared_adj_degree[now];
                    }

                    if (now < group_size) {
                        int temp_src = shared_src[now];
                        int temp_adj = adj_list[shared_adj_start[now] + workid];
                        int bin = (temp_src + temp_adj) & GroupTC_HS_GROUP_MODULO;
                        int len = bin_count[bin];

                        P_counter += len > 0 && shared_partition[bin + GroupTC_HS_group_bucketnum * 0] == temp_adj;
                        P_counter += len > 1 && shared_partition[bin + GroupTC_HS_group_bucketnum * 1] == temp_adj;
                        P_counter += len > 2 && shared_partition[bin + GroupTC_HS_group_bucketnum * 2] == temp_adj;
                        P_counter += len > 3 && shared_partition[bin + GroupTC_HS_group_bucketnum * 3] == temp_adj;

                        if (len > GroupTC_HS_shared_GROUP_BUCKET_SIZE) {
                            P_counter += tc::approach::GroupTC_HS::linear_search_group(temp_adj, partition, len, bin, BIN_START);
                        }
                    }
                    workid += GroupTC_HS_Group_SUBWARP_SIZE;
                }
            }

            __syncthreads();
        }
    }
    // atomicAdd(&G_counter, P_counter);

    // __syncthreads();
    // if (threadIdx.x == 0) {
    //     atomicAdd(&GLOBAL_COUNT[0], G_counter);
    // }
    GLOBAL_COUNT[blockIdx.x * blockDim.x + threadIdx.x] = P_counter;
}

void tc::approach::GroupTC_HS::gpu_run(INIReader &config, GPUGraph &gpu_graph, std::string key_space) {
    std::string file = gpu_graph.input_dir;
    int iteration_count = config.GetInteger(key_space, "iteration_count", 1);
    spdlog::info("Run algorithm {}", key_space);
    spdlog::info("Dataset {}", file);
    spdlog::info("Number of nodes: {0}, number of edges: {1}", gpu_graph.vertex_count, gpu_graph.edge_count);
    int device = config.GetInteger(key_space, "device", 1);
    HRR(cudaSetDevice(device));

    int grid_size = 2048;
    int block_size = 1024;
    int chunk_size = 1;

    uint vertex_count = gpu_graph.vertex_count;
    uint edge_count = gpu_graph.edge_count;
    index_t *d_beg_pos = gpu_graph.beg_pos;
    vertex_t *d_src_list = gpu_graph.src_list;
    vertex_t *d_adj_list = gpu_graph.adj_list;

    index_t *h_beg_pos = (index_t *)malloc(sizeof(index_t) * (vertex_count + 1));
    HRR(cudaMemcpy(h_beg_pos, gpu_graph.beg_pos, sizeof(index_t) * (vertex_count + 1), cudaMemcpyDeviceToHost));

    // int *h_adj_list = (int *)malloc(sizeof(vertex_t) * (edge_count));
    // HRR(cudaMemcpy(h_adj_list, gpu_graph.adj_list, sizeof(vertex_t) * (edge_count), cudaMemcpyDeviceToHost));

    // for (int i = h_beg_pos[154]; i < h_beg_pos[154 + 1]; i++) {
    //     printf("%d ", h_adj_list[i]);
    // }
    // printf("\n");

    int warpfirstvertex = my_binary_search(vertex_count, GroupTC_HS_USE_CTA, h_beg_pos) + 1;
    int warpfirstedge = h_beg_pos[warpfirstvertex];
    int nocomputefirstvertex = my_binary_search(vertex_count, GroupTC_HS_USE_WARP, h_beg_pos) + 1;
    int nocomputefirstedge = h_beg_pos[nocomputefirstvertex];

    int T_Group = 32;
    int nowindex[3];
    nowindex[0] = chunk_size * grid_size * block_size / T_Group;
    nowindex[1] = chunk_size * grid_size;
    nowindex[2] = warpfirstvertex + chunk_size * (grid_size * block_size / T_Group);

    int *BIN_MEM;
    int *G_INDEX;
    unsigned long long *GLOBAL_COUNT;

    HRR(cudaMalloc((void **)&BIN_MEM, sizeof(int) * grid_size * GroupTC_HS_block_bucketnum * GroupTC_HS_BLOCK_BUCKET_SIZE));
    HRR(cudaMalloc((void **)&G_INDEX, sizeof(int) * 3));
    // HRR(cudaMalloc((void **)&GLOBAL_COUNT, sizeof(unsigned long long) * 10));

    HRR(cudaMemcpy(G_INDEX, &nowindex, sizeof(int) * 3, cudaMemcpyHostToDevice));

    // unsigned long long *counter = (unsigned long long *)malloc(sizeof(unsigned long long) * 10);

    double total_kernel_use = 0;
    double startKernel, ee = 0;
    int block_kernel_grid_size = min(max(warpfirstvertex, 1), grid_size);
    int group_kernel_grid_size = min((nocomputefirstedge - warpfirstedge) / (GroupTC_HS_EDGE_CHUNK * 10), grid_size);
    int kernel_grid_size = max(max(block_kernel_grid_size, group_kernel_grid_size), 320);

    uint64_t count;
    HRR(cudaMalloc((void **)&GLOBAL_COUNT, sizeof(unsigned long long) * kernel_grid_size * GroupTC_HS_BLOCK_SIZE));
    spdlog::info("kernel_grid_size {:d}", kernel_grid_size);

    for (int i = 0; i < iteration_count; i++) {
        HRR(cudaMemcpy(G_INDEX, &nowindex, sizeof(int) * 3, cudaMemcpyHostToDevice));
        HRR(cudaMemset(GLOBAL_COUNT, 0, sizeof(unsigned long long) * kernel_grid_size * GroupTC_HS_BLOCK_SIZE));

        startKernel = wtime();

        tc::approach::GroupTC_HS::grouptc_hs<64, GroupTC_HS_BLOCK_SIZE / 64, 1>
            <<<kernel_grid_size, GroupTC_HS_BLOCK_SIZE>>>(d_src_list, d_adj_list, d_beg_pos, edge_count, vertex_count, BIN_MEM, GLOBAL_COUNT, T_Group,
                                                          G_INDEX, warpfirstvertex, warpfirstedge, nocomputefirstvertex, nocomputefirstedge);

        HRR(cudaDeviceSynchronize());
        thrust::device_ptr<unsigned long long> ptr(GLOBAL_COUNT);
        count = thrust::reduce(ptr, ptr + (kernel_grid_size * GroupTC_HS_BLOCK_SIZE));
        ee = wtime();

        total_kernel_use += ee - startKernel;
        if (i == 0) {
            spdlog::info("Iter 0, kernel use {:.6f} s", total_kernel_use);
            if (ee - startKernel > 0.1 && iteration_count != 1) {
                iteration_count = 10;
            }
        }
    }

    // HRR(cudaMemcpy(counter, GLOBAL_COUNT, sizeof(unsigned long long) * 10, cudaMemcpyDeviceToHost));

    // algorithm, dataset, iteration_count, avg compute time/s,
    auto logger = spdlog::get("GroupTC-HS_file_logger");
    if (logger) {
        logger->info("{0}\t{1}\t{2}\t{3}\t{4:.6f}", "GroupTC-HS", gpu_graph.input_dir, count, iteration_count, total_kernel_use / iteration_count);
    } else {
        spdlog::warn("Logger 'GroupTC-HS_file_logger' is not initialized.");
    }

    // spdlog::get("GroupTC-HS_file_logger")
    // ->info("{0}\t{1}\t{2}\t{3}\t{4:.6f}", "GroupTC-HS", gpu_graph.input_dir, counter[0], iteration_count, total_kernel_use /
    // iteration_count);

    spdlog::info("Iter {0}, avg kernel use {1:.6f} s", iteration_count, total_kernel_use / iteration_count);
    spdlog::info("Triangle count {:d}", count);

    // free(counter);
    free(h_beg_pos);
    HRR(cudaFree(BIN_MEM));
    HRR(cudaFree(GLOBAL_COUNT));
    HRR(cudaFree(G_INDEX));
}

void tc::approach::GroupTC_HS::start_up(INIReader &config, GPUGraph &gpu_graph, int argc, char **argv) {
    bool run = config.GetBoolean("comm", "GroupTC-HS", false);
    if (run) {
        size_t free_byte, total_byte, available_byte;
        HRR(cudaMemGetInfo(&free_byte, &total_byte));
        available_byte = total_byte - free_byte;
        spdlog::debug("GroupTC_HS before compute, used memory {:.2f} GB", float(total_byte - free_byte) / MEMORY_G);

        tc::approach::GroupTC_HS::gpu_run(config, gpu_graph);

        HRR(cudaMemGetInfo(&free_byte, &total_byte));
        spdlog::debug("GroupTC_HS after compute, used memory {:.2f} GB", float(total_byte - free_byte) / MEMORY_G);
        if (available_byte != total_byte - free_byte) {
            spdlog::warn("There is GPU memory that is not freed after GroupTC_HS runs.");
        }
    }
}
