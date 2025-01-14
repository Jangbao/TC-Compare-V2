#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "comm/config_comm.h"
#include "comm/constant_comm.h"
#include "datatransfer/csr2dcsr_data_transfer.h"
#include "graph/cuda_graph_comm.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/spdlog.h"

void Csr2DcsrDataTransfer::transfer() {

    if (!check_init()) {
        return;
    }

    uint* d_degree_arr;
    vertex_t* d_edge_arr;

    uint vertex_count = d_graph.vertex_count;
    uint edge_count = d_graph.edge_count;
    vertex_t* d_src_arr = d_graph.src_list;
    vertex_t* d_adj_arr = d_graph.adj_list;
    index_t* d_offset_arr = d_graph.beg_pos;

    size_t size_degree_arr = sizeof(uint) * vertex_count;
    size_t size_src_arr = sizeof(vertex_t) * (edge_count + 10);
    size_t size_edge_arr = size_src_arr * 2;

    HRR(cudaMalloc((void**)&d_degree_arr, size_degree_arr));
    HRR(cudaMalloc((void**)&d_edge_arr, size_edge_arr));

    HRR(cudaMemset(d_degree_arr, 0, size_degree_arr));

    int block_size = 1024;
    int grid_size = (edge_count - 1) / block_size + 1;

    int max_degree = compute_max_degree();

    spdlog::info("CSR2DCSR graph transfer start, graph max degree is {}", max_degree);

    double t_start = wtime();

    int iterations = config_comm::cPreprocessingIterations;
    for (int k = 0; k < iterations; k++) {
        cuda_graph_comm::cal_degree_and_zip_edge<<<grid_size, block_size>>>(edge_count, vertex_count, d_degree_arr, d_edge_arr, d_src_arr, d_adj_arr);
        // HRR(cudaDeviceSynchronize());

        cuda_graph_comm::redirect_edge<<<grid_size, block_size>>>(edge_count, vertex_count, d_degree_arr, d_edge_arr);
        // HRR(cudaDeviceSynchronize());

        vertex_t* h_src_arr;
        vertex_t* h_adj_arr;

        // The GPU space required for sorting is insufficient, so src_list and adj_list need to be transferred to CPU space first.
        if (edge_count > constant_comm::kDataTransferMaxEdgeCount) {
            spdlog::info("Csr2DcsrDataTransfer's sorting requires more GPU space, so src_list and adj_list are transferred to CPU space.");
            h_src_arr = (vertex_t*)malloc(size_src_arr);
            h_adj_arr = (vertex_t*)malloc(size_src_arr);
            HRR(cudaMemcpy(h_src_arr, d_src_arr, size_src_arr, cudaMemcpyDeviceToHost));
            HRR(cudaMemcpy(h_adj_arr, d_adj_arr, size_src_arr, cudaMemcpyDeviceToHost));
            HRR(cudaFree(d_src_arr));
            HRR(cudaFree(d_adj_arr));
        }

        thrust::device_ptr<uint64_t> sort_ptr((uint64_t*)d_edge_arr);
        thrust::sort(sort_ptr, sort_ptr + edge_count);

        // After sorting, src_list and adj_list are transferred back to GPU space.
        if (edge_count > constant_comm::kDataTransferMaxEdgeCount) {
            spdlog::info("Csr2DcsrDataTransfer's sorting is completed, src_list and adj_list are transferred back to the GPU space.");
            HRR(cudaMalloc(&d_src_arr, size_src_arr));
            HRR(cudaMalloc(&d_adj_arr, size_src_arr));
            HRR(cudaMemcpy(d_src_arr, h_src_arr, size_src_arr, cudaMemcpyHostToDevice));
            HRR(cudaMemcpy(d_adj_arr, h_adj_arr, size_src_arr, cudaMemcpyHostToDevice));
            d_graph.src_list = d_src_arr;
            d_graph.adj_list = d_adj_arr;
            free(h_src_arr);
            free(h_adj_arr);
        }

        cuda_graph_comm::unzip_edge<<<grid_size, block_size>>>(edge_count, vertex_count, d_edge_arr, d_src_arr, d_adj_arr);
        // HRR(cudaDeviceSynchronize());

        cuda_graph_comm::recal_offset<<<grid_size, block_size>>>(edge_count, vertex_count, d_src_arr, d_offset_arr);
        // HRR(cudaDeviceSynchronize());
    }
    HRR(cudaDeviceSynchronize());
    double t_end = wtime();

    // algorithm, dataset, iterations, avg compute time/s,
    spdlog::get("csr2dcsr_file_logger")->info("{0}\t{1}\t{2}\t{3:.6f}", "csr2dcsr", d_graph.input_dir, iterations, (t_end - t_start) / iterations);

    spdlog::info("Iterate {0} times, avg time consumption {1:.6f} s", iterations, (t_end - t_start) / iterations);

    cuda_graph_comm::check_array("d_src_arr", d_graph.src_list, edge_count, 0, 10);
    cuda_graph_comm::check_array("d_adj_arr", d_adj_arr, edge_count, 0, 10);
    cuda_graph_comm::check_array("d_offset_arr", d_offset_arr, vertex_count + 1, 0, 10);

    max_degree = compute_max_degree();
    spdlog::info("CSR2DCSR graph transfer finished, graph max degree is {}", max_degree);

    HRR(cudaFree(d_degree_arr));
    HRR(cudaFree(d_edge_arr));
}
