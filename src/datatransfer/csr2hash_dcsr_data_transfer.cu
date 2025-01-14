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
#include "datatransfer/csr2hash_dcsr_data_transfer.h"
#include "graph/cuda_graph_comm.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/spdlog.h"

void Csr2HashDcsrDataTransfer::transfer() {

    if (!check_init()) {
        return;
    }

    uint *d_degree_arr;
    vertex_t *d_id_arr;
    vertex_t *d_id_map_arr;

    uint vertex_count = d_graph.vertex_count;
    uint edge_count = d_graph.edge_count;
    vertex_t *d_src_arr = d_graph.src_list;
    vertex_t *d_adj_arr = d_graph.adj_list;
    index_t *d_offset_arr = d_graph.beg_pos;

    size_t size_degree_arr = sizeof(uint) * vertex_count;

    HRR(cudaMalloc((void **)&d_degree_arr, size_degree_arr));
    HRR(cudaMalloc((void **)&d_id_arr, size_degree_arr));
    HRR(cudaMalloc((void **)&d_id_map_arr, size_degree_arr));

    HRR(cudaMemset(d_degree_arr, 0, size_degree_arr));

    int block_size = 1024;
    // int vertex_grid_size = (vertex_count - 1) / block_size + 1;
    int edge_grid_size = (edge_count - 1) / block_size + 1;

    int max_degree = compute_max_degree();

    spdlog::info("CSR2TrustDCSR graph transfer start, graph max degree is {}", max_degree);


    double t_start = wtime();

    int iterations = config_comm::cPreprocessingIterations;
    for (int k = 0; k < iterations; k++) {
        cuda_graph_comm::cal_degree<<<edge_grid_size, block_size>>>(edge_count, vertex_count, d_degree_arr, d_src_arr, d_adj_arr);
        // HRR(cudaDeviceSynchronize());

        cuda_graph_comm::redirect_edge<<<edge_grid_size, block_size>>>(edge_count, vertex_count, d_degree_arr, d_src_arr, d_adj_arr);
        // HRR(cudaDeviceSynchronize());

        // HRR(cudaMemset(d_degree_arr, 0, size_degree_arr));
        // cuda_graph_comm::cal_out_degree_by_src<<<edge_grid_size, block_size>>>(edge_count, vertex_count, d_degree_arr, d_src_arr);
        // HRR(cudaDeviceSynchronize());

        // cuda_graph_comm::record_id_and_part_graph_by_degree<<<vertex_grid_size, block_size>>>(edge_count, vertex_count, d_id_arr, d_degree_arr);
        // HRR(cudaDeviceSynchronize());

        // thrust::device_ptr<vertex_t> d_id_ptr((vertex_t *)d_id_arr);
        // thrust::sort_by_key(d_degree_arr, d_degree_arr + vertex_count, d_id_ptr);

        // cuda_graph_comm::map_id<<<vertex_grid_size, block_size>>>(edge_count, vertex_count, d_id_arr, d_id_map_arr);
        // HRR(cudaDeviceSynchronize());

        // cuda_graph_comm::reassign_id<<<edge_grid_size, block_size>>>(edge_count, vertex_count, d_id_map_arr, d_src_arr, d_adj_arr);
        // HRR(cudaDeviceSynchronize());

        thrust::device_ptr<vertex_t> d_src_ptr((vertex_t *)d_src_arr);
        thrust::device_ptr<vertex_t> d_dst_ptr((vertex_t *)d_adj_arr);
        thrust::sort_by_key(d_src_ptr, d_src_ptr + edge_count, d_adj_arr);

        cuda_graph_comm::recal_offset<<<edge_grid_size, block_size>>>(edge_count, vertex_count, d_src_arr, d_offset_arr);
        // HRR(cudaDeviceSynchronize());
    }
    HRR(cudaDeviceSynchronize());

    double t_end = wtime();

    // algorithm, dataset, iterations, avg compute time/s,
    spdlog::get("csr2hash_dcsr_file_logger")
        ->info("{0}\t{1}\t{2}\t{3:.6f}", "csr2hash_dcsr", d_graph.input_dir, iterations, (t_end - t_start) / iterations);

    spdlog::info("Iterate {0} times, avg time consumption {1:.6f} s", iterations, (t_end - t_start) / iterations);

    cuda_graph_comm::check_array("d_src_arr", d_src_arr, edge_count, 0, 10);
    cuda_graph_comm::check_array("d_adj_arr", d_adj_arr, edge_count, 0, 10);
    cuda_graph_comm::check_array("d_offset_arr", d_offset_arr, vertex_count + 1, 0, 10);

    max_degree = compute_max_degree();
    spdlog::info("CSR2HashDCSR graph transfer finished, graph max degree is {}", max_degree);

    HRR(cudaFree(d_degree_arr));
    HRR(cudaFree(d_id_arr));
    HRR(cudaFree(d_id_map_arr));
}
