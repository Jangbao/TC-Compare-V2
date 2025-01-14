#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <iterator>
#include <queue>
#include <set>

#include "approach/H-INDEX/tc.h"
#include "comm/comm.h"
#include "comm/constant_comm.h"
#include "comm/cuda_comm.h"
#include "graph/cuda_graph_comm.h"
#include "spdlog/spdlog.h"

__device__ int tc::approach::H_INDEX::linear_search(int neighbor, int *partition1, int *bin_count, int bin, int BIN_OFFSET, int BIN_START,
                                                    int BUCKETS) {
    int len = bin_count[bin + BIN_OFFSET];
    // printf("\nPartStart: %d\n",BIN_START);
    int i = bin + BIN_START;
    int step = 0;
    while (step < len) {
        int test = partition1[i];
        // printf("Neighbor: %d, Test: %d\n",neighbor,test);
        if (test == neighbor) {
            return 1;
        } else {
            i += BUCKETS;
        }
        step += 1;
    }
    return 0;
}

__device__ int tc::approach::H_INDEX::max_count(int *bin_count, int start, int end, int len) {
    int max_count = bin_count[start];
    int min_count = bin_count[start];
    int zero_count = 0;
    for (int i = start; i < end; i++) {
        if (bin_count[i] > max_count) {
            max_count = bin_count[i];
        }
        if (bin_count[i] < min_count) {
            min_count = bin_count[i];
        }
        if (bin_count[i] == 0) {
            zero_count += 1;
        }
    }
    // printf("%d,%d,%d\n",zero_count,max_count,len);
    return max_count;
}

__global__ void tc::approach::H_INDEX::warp_hash_count(vertex_t *adj_list, index_t *beg_pos, vertex_t *edge_list, uint edge_count, uint vertex_count,
                                                       uint edge_list_count, int *partition, unsigned long long *GLOBAL_COUNT, long long E_START,
                                                       long long E_END, int device, int BUCKETS, int G_BUCKET_SIZE, int T_Group) {
    // Uncomment the lines below and change partition to Gpartition for using shared version
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int WARPSIZE = T_Group;
    int __shared__ bin_count[256 * 4];
    // int __shared__ partition[160*4];
    int PER_BLOCK_WARP = blockDim.x / WARPSIZE;
    int G_WARPID = tid / WARPSIZE;
    int WARPID = threadIdx.x / WARPSIZE;
    int __shared__ G_counter;
    G_counter = 0;
    int P_counter = 0;
    int BINsize = BUCKETS * G_BUCKET_SIZE;
    // int BINsize = BUCKETS*5;
    int BIN_START = G_WARPID * BINsize;
    // int BIN_START = WARPID*BINsize;
    long long i = G_WARPID * 2;
    long long RANGE = E_END - E_START;
    int BIN_OFFSET = WARPID * BUCKETS;
    // for(int i=0;i<edge_list_count; i+=2)
    // TODO: Static assignment to dynamic assignment of edges
    //  unsigned long long TT=0,HT=0,IT=0;
    //  unsigned long long __shared__ G_TT,G_HT,G_IT;
    //  G_TT=0,G_HT=0,G_IT=0;
    while (i < (RANGE)) {
        // if(threadIdx.x%32==0){printf("Warp:%d, G_WArp: %d,i: %d \n",WARPID,G_WARPID,i);}
        // if (device==1){printf("Device: %d, i: %d\n",device,i);}
        /* TODO: Divide edge list to multiple blocks*/
        // unsigned long long start_time=clock64();
        int destination = edge_list[i];
        int source = edge_list[i + 1];
        int N1_start = beg_pos[destination];
        int N1_end = beg_pos[destination + 1];
        int L1 = N1_end - N1_start;
        int N2_start = beg_pos[source];
        int N2_end = beg_pos[source + 1];
        int L2 = N2_end - N2_start;

        // if ((L1==0))
        // {
        // 	//printf("continue %d\n",i);
        // 	continue;
        // }
        // // N2 is for hashing and N1 is lookup
        if (L1 > L2) {
            int temp = N1_start;
            N1_start = N2_start;
            N2_start = temp;
            temp = N1_end;
            N1_end = N2_end;
            N2_end = temp;
            temp = L2;
            L2 = L1;
            L1 = temp;
        }

        // unsigned long long hash_start=clock64();
        int id = threadIdx.x % WARPSIZE + BIN_OFFSET;
        int end = BIN_OFFSET + BUCKETS;
        // if(threadIdx.x%32==0){printf("End: %d\n",end);}
        //  We can remove this line

        __syncwarp();
        while (id < (end)) {
            bin_count[id] = 0;
            // printf("BIN: %d\n",id);
            id += WARPSIZE;
        }
        int start = threadIdx.x % WARPSIZE + N2_start;
        // BIN_OFFSET is for count of number of element of each bin for all 4 warps

        __syncwarp();
        // Hash one list
        while (start < N2_end) {
            int temp = adj_list[start];
            int bin = temp % BUCKETS;
            int index = atomicAdd(&bin_count[bin + BIN_OFFSET], 1);
            partition[index * BUCKETS + bin + BIN_START] = temp;
            //{printf("thread: %d,warp:%d, write: %d bin %d, index %d  at: %d\n",threadIdx.x,WARPID,temp,bin,index,(index*WARPSIZE+bin+BIN_START));}
            start += WARPSIZE;
        }
        __syncwarp();
        // unsigned long long hash_time=clock64()-hash_start;
        // int max_len_collision= max_count(bin_count,BIN_OFFSET,BIN_OFFSET+BUCKETS,L2);

        // unsigned long long intersection_start=clock64();
        start = threadIdx.x % WARPSIZE + N1_start;
        int count;
        // if(threadIdx.x==32){printf("start: %d, BIN_OFFSET: %d\n",start,BIN_OFFSET);}
        // P_counter=0;
        while (start < N1_end) {
            count = 0;
            int neighbor = adj_list[start];
            int bin = neighbor % BUCKETS;
            count = tc::approach::H_INDEX::linear_search(neighbor, partition, bin_count, bin, BIN_OFFSET, BIN_START, BUCKETS);
            P_counter += count;
            start += WARPSIZE;
            // printf("Tid: %d, Search:%d\n",threadIdx.x,neighbor);
        }
        // atomicAdd(&GLOBAL_COUNT[0],P_counter);

        __syncwarp();
        // unsigned long long intersection_time=clock64()-intersection_start;
        // if(threadIdx.x%32==0){printf("I: %d, Start:%d, End:%d, Count:%d\n",i,vertex,vertex1,G_counter);}
        i += gridDim.x * PER_BLOCK_WARP * 2;
        // unsigned long long total_time=clock64()-start_time;
        // if(threadIdx.x%32==0){
        // 	// printf("%d %d %d\n",total_time, hash_time, intersection_time);
        // 	TT+=total_time;
        // 	HT+=hash_time;
        // 	IT+=intersection_time;
        // }
    }
    atomicAdd(&G_counter, P_counter);
    // atomicAdd(&G_HT,HT);
    // atomicAdd(&G_TT,TT);
    // atomicAdd(&G_IT,IT);
    __syncthreads();
    if (threadIdx.x == 0) {
        // printf("%d\n",G_TT);
        atomicAdd(&GLOBAL_COUNT[0], G_counter);
        // atomicAdd(&GLOBAL_COUNT[1],G_TT);
        // atomicAdd(&GLOBAL_COUNT[2],G_HT);
        // atomicAdd(&GLOBAL_COUNT[3],G_IT);
    }

    // if(threadIdx.x==0){printf("Device: %d, Count:%d\n",device,GLOBAL_COUNT[0]);}
}

__global__ void tc::approach::H_INDEX::CTA_hash_count(vertex_t *adj_list, index_t *beg_pos, vertex_t *edge_list, uint edge_count, uint vertex_count,
                                                      uint edge_list_count, int *partition, unsigned long long *GLOBAL_COUNT, int E_START, int E_END,
                                                      int device, int BUCKETS, int BUCKET_SIZE, int T_Group) {
    int __shared__ bin_count[512];
    int G_WARPID = blockIdx.x;
    int __shared__ G_counter;
    G_counter = 0;
    int P_counter = 0;
    int BINsize = BUCKETS * BUCKET_SIZE;
    int i = G_WARPID * 2;
    int RANGE = E_END - E_START;
    int BIN_START = G_WARPID * BINsize;
    // for(int i=0;i<edge_list_count; i+=2)
    // TODO: Static assignment to dynamic assignment of edges

    while (i < (RANGE)) {
        /* TODO: Divide edge list to multiple blocks*/
        int destination = edge_list[i];
        int source = edge_list[i + 1];
        int N1_start = beg_pos[destination];
        int N1_end = beg_pos[destination + 1];
        int L1 = N1_end - N1_start;
        int N2_start = beg_pos[source];
        int N2_end = beg_pos[source + 1];
        int L2 = N2_end - N2_start;

        // N2 is for hashing and N1 is lookup
        if (L1 > L2) {
            int temp = N1_start;
            N1_start = N2_start;
            N2_start = temp;
            temp = N1_end;
            N1_end = N2_end;
            N2_end = temp;
            temp = L2;
            L2 = L1;
            L1 = temp;
        }

        int id = threadIdx.x;
        int end = BUCKETS;

        while (id < (end)) {
            bin_count[id] = 0;
            id += blockDim.x;
        }
        __syncthreads();
        int start = threadIdx.x + N2_start;

        // Hash one list
        while (start < N2_end) {
            int temp = adj_list[start];
            int bin = temp % BUCKETS;
            int index = atomicAdd(&bin_count[bin], 1);
            partition[index * BUCKETS + bin + BIN_START] = temp;
            //{printf("thread: %d,warp:%d, write: %d bin %d, index %d  at: %d\n",threadIdx.x,WARPID,temp,bin,index,(index*WARPSIZE+bin+BIN_START));}
            start += blockDim.x;
        }
        __syncthreads();
        start = threadIdx.x + N1_start;
        int count;
        // if(threadIdx.x==32){printf("start: %d, BIN_OFFSET: %d\n",start,BIN_OFFSET);}
        // P_counter=0;
        while (start < N1_end) {
            count = 0;
            int neighbor = adj_list[start];
            int bin = neighbor % BUCKETS;
            count = tc::approach::H_INDEX::linear_search(neighbor, partition, bin_count, bin, 0, BIN_START, BUCKETS);
            P_counter += count;
            start += blockDim.x;
            // printf("Tid: %d, Search:%d\n",threadIdx.x,neighbor);
        }
        // atomicAdd(&GLOBAL_COUNT[0], P_counter);

        // if(threadIdx.x%32==0){printf("I: %d, Start:%d, End:%d, Count:%d\n",i,vertex,vertex1,G_counter);}
        i += gridDim.x * 2;
    }
    atomicAdd(&G_counter, P_counter);
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(&GLOBAL_COUNT[0], G_counter);
        // atomicAdd(&GLOBAL_COUNT[0], max_len_collision);
    }

    // if(threadIdx.x==0){printf("Device: %d, Count:%d\n",device,GLOBAL_COUNT[0]);}
}

void tc::approach::H_INDEX::gpu_run(INIReader &config, GPUGraph &gpu_graph, std::string key_space) {
    std::string file = gpu_graph.input_dir;
    int grid_size = config.GetInteger(key_space, "grid_size", 1024);
    int block_size = config.GetInteger(key_space, "block_size", 1024);
    int BUCKETS = config.GetInteger(key_space, "buckets", 32);
    int select_thread_group = config.GetInteger(key_space, "select_thread_group", 0);
    int select_partition = config.GetInteger(key_space, "select_partition", 0);
    int iteration_count = config.GetInteger(key_space, "iteration_count", 1);
    spdlog::info("Run algorithm {}", key_space);
    spdlog::info("Dataset {}", file);
    spdlog::info("Number of nodes: {0}, number of edges: {1}", gpu_graph.vertex_count, gpu_graph.edge_count);
    int device = config.GetInteger(key_space, "device", 1);
    HRR(cudaSetDevice(device));

    int T_Group = 32;
    int BUCKET_SIZE = 1000;
    int PER_BLOCK_WARP = block_size / T_Group;
    int total = grid_size * PER_BLOCK_WARP * BUCKETS * BUCKET_SIZE;
    unsigned long long *counter = (unsigned long long *)malloc(sizeof(unsigned long long) * 10);

    uint vertex_count = gpu_graph.vertex_count;
    uint edge_count = gpu_graph.edge_count;
    index_t edge_list_count = edge_count * 2;

    int *BIN_MEM;
    unsigned long long *GLOBAL_COUNT;
    index_t *d_beg_pos = gpu_graph.beg_pos;
    vertex_t *d_adj_list = gpu_graph.adj_list;
    vertex_t *d_edge_list;

    int zip_block_size = 1024;
    int zip_edge_grid_size = (edge_count - 1) / zip_block_size + 1;

    HRR(cudaMalloc((void **)&GLOBAL_COUNT, sizeof(unsigned long long) * 10));
    HRR(cudaMalloc((void **)&BIN_MEM, sizeof(int) * total));
    HRR(cudaMalloc((void **)&d_edge_list, (size_t)sizeof(vertex_t) * edge_list_count));

    cuda_graph_comm::zip_edge<<<zip_edge_grid_size, zip_block_size>>>(edge_count, vertex_count, d_edge_list, gpu_graph.src_list, d_adj_list);
    HRR(cudaDeviceSynchronize());

    double total_kernel_use = 0;
    double startKernel, ee;
    for (int i = 0; i < iteration_count; i++) {
        cudaMemset(GLOBAL_COUNT, 0, sizeof(unsigned long long) * 10);
        startKernel = wtime();
        if (select_thread_group == 1) {
            tc::approach::H_INDEX::CTA_hash_count<<<grid_size, block_size>>>(d_adj_list, d_beg_pos, d_edge_list, edge_count, vertex_count,
                                                                             edge_list_count, BIN_MEM, GLOBAL_COUNT, 0, edge_count * 2, 0, BUCKETS,
                                                                             BUCKET_SIZE, T_Group);
            HRR(cudaDeviceSynchronize());
        } else {
            tc::approach::H_INDEX::warp_hash_count<<<grid_size, block_size>>>(d_adj_list, d_beg_pos, d_edge_list, edge_count, vertex_count,
                                                                              edge_list_count, BIN_MEM, GLOBAL_COUNT, 0, edge_count * 2, 0, BUCKETS,
                                                                              BUCKET_SIZE, T_Group);
            HRR(cudaDeviceSynchronize());
        }
        HRR(cudaMemcpy(counter, GLOBAL_COUNT, sizeof(unsigned long long) * 10, cudaMemcpyDeviceToHost));
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
    spdlog::get("H-INDEX_file_logger")
        ->info("{0}\t{1}\t{2}\t{3}\t{4:.6f}", "H-INDEX", gpu_graph.input_dir, counter[0], iteration_count, total_kernel_use / iteration_count);

    spdlog::info("Iter {0}, avg kernel use {1:.6f} s", iteration_count, total_kernel_use / iteration_count);
    spdlog::info("Triangle count {:d}", counter[0]);

    HRR(cudaFree(BIN_MEM));
    HRR(cudaFree(GLOBAL_COUNT));
    HRR(cudaFree(d_edge_list));
}

void tc::approach::H_INDEX::start_up(INIReader &config, GPUGraph &gpu_graph, int argc, char **argv) {
    bool run = config.GetBoolean("comm", "H-INDEX", false);
    if (run) {
        size_t free_byte, total_byte, available_byte;
        HRR(cudaMemGetInfo(&free_byte, &total_byte));
        available_byte = total_byte - free_byte;
        spdlog::debug("H_INDEX before compute, used memory {:.2f} GB", float(total_byte - free_byte) / MEMORY_G);

        tc::approach::H_INDEX::gpu_run(config, gpu_graph);

        HRR(cudaMemGetInfo(&free_byte, &total_byte));
        spdlog::debug("H_INDEX after compute, used memory {:.2f} GB", float(total_byte - free_byte) / MEMORY_G);
        if (available_byte != total_byte - free_byte) {
            spdlog::warn("There is GPU memory that is not freed after H_INDEX runs.");
        }
    }
}
