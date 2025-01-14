#pragma once
#include <cuda_runtime_api.h>

#include "comm/comm.h"
#include "comm/constant_comm.h"
#include "comm/cuda_comm.h"
#include "spdlog/spdlog.h"

namespace cuda_graph_comm {

__global__ void copy_32_to_64(uint arr_len, int32_t* arr_32, int64_t* arr_64);

__global__ void zip_edge(uint edge_count, uint vertex_count, vertex_t* d_edge_arr, vertex_t* d_src_arr, vertex_t* d_adj_arr);

__global__ void cal_degree_and_zip_edge(uint edge_count, uint vertex_count, uint* d_degree_arr, vertex_t* d_edge_arr, vertex_t* d_src_arr,
                                        vertex_t* d_adj_arr);

__global__ void redirect_edge(uint edge_count, uint vertex_count, uint* d_degree_arr, vertex_t* d_edge_arr);

__global__ void redirect_edge(uint edge_count, uint vertex_count, uint* d_degree_arr, vertex_t* d_src_arr, vertex_t* d_adj_arr);

__global__ void unzip_edge(uint edge_count, uint vertex_count, vertex_t* d_edge_arr, vertex_t* d_src_arr, vertex_t* d_adj_arr);

__global__ void recal_offset(uint edge_count, uint vertex_count, vertex_t* d_src_arr, index_t* d_offset_arr);

__global__ void record_id(uint edge_count, uint vertex_count, vertex_t* d_id_arr);

__global__ void map_id(uint edge_count, uint vertex_count, vertex_t* d_id_arr, vertex_t* d_id_map_arr);

__global__ void redirect_edge_and_reassign_id(uint edge_count, uint vertex_count, vertex_t* d_id_map_arr, vertex_t* d_edge_arr);

__global__ void cal_degree(uint edge_count, uint vertex_count, uint* d_degree_arr, vertex_t* d_src_arr, vertex_t* d_adj_arr);

__global__ void cal_out_degree_by_src(uint edge_count, uint vertex_count, uint* d_degree_arr, vertex_t* d_src_arr);

__global__ void cal_out_degree_by_offset(uint edge_count, uint vertex_count, uint* d_degree_arr, index_t* d_offset_arr);

__global__ void record_id_and_part_graph_by_degree(uint edge_count, uint vertex_count, vertex_t* d_id_arr, uint* d_degree_arr);

__global__ void reassign_id(uint edge_count, uint vertex_count, vertex_t* d_id_map_arr, vertex_t* d_src_arr, vertex_t* d_adj_arr);

__global__ void check_order(vertex_t* arr, uint len);

__global__ void warm_up(int* data, size_t n);

template <typename T>
void check_array(std::string arr_name, T* d_arr, uint len, uint start, uint end);

template <typename T>
void set_value_by_index(T* d_arr, uint index, T value);

template <typename T>
void sort_big_arr(T* arr, uint len);

}  // namespace cuda_graph_comm

template <typename T>
void cuda_graph_comm::check_array(std::string arr_name, T* d_arr, uint len, uint start, uint end) {
    if (spdlog::get_level() <= spdlog::level::trace) {
        T* h_arr = (T*)malloc(sizeof(T) * len);
        HRR(cudaMemcpy(h_arr, d_arr, sizeof(T) * len, cudaMemcpyDeviceToHost));

        for (vertex_t i = start; i < end; i++) {
            spdlog::trace("{0}[{1}]: {2}", arr_name, i, h_arr[i]);
        }

        free(h_arr);
    }
}

template <typename T>
void cuda_graph_comm::set_value_by_index(T* d_arr, uint index, T value) {
    HRR(cudaMemcpy(d_arr + index, &value, sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void cuda_graph_comm::sort_big_arr(T* arr, uint len) {
    thrust::device_ptr<T> sort_ptr((T*)arr);

    uint part_size = constant_comm::kThrustSortThresholdSize / sizeof(T);
    uint part_num = (len - 1) / part_size + 1;

    // 1. part sort
    uint part_arr_start, part_arr_end;
    for (uint i = 0; i < part_num; i++) {
        part_arr_start = i * part_size;
        part_arr_end = min(len, (i + 1) * part_size);

        thrust::sort(sort_ptr + part_arr_start, sort_ptr + part_arr_end);
    }

    // 2. merge
    T* tmp_arr;
    HRR(cudaMalloc(&tmp_arr, sizeof(T) * len));
    uint part_arr1_start, part_arr1_end;
    uint part_arr2_start, part_arr2_end;

    // The initial input and ouput are inverse, waited flip
    T* input_arr = tmp_arr;
    T* output_arr = arr;
    T* tmp_ptr;

    while (part_num > 1) {
        // flip input and output
        tmp_ptr = input_arr;
        input_arr = output_arr;
        output_arr = tmp_ptr;

        for (uint i = 0; i < part_num; i += 2) {
            // merge part i, part i + 1
            part_arr1_start = i * part_size;
            part_arr1_end = (i + 1) * part_size;
            part_arr2_start = (i + 1) * part_size;
            part_arr2_end = min(len, (i + 2) * part_size);
            thrust::merge((thrust::device_ptr<T>)input_arr + part_arr1_start, (thrust::device_ptr<T>)input_arr + part_arr1_end,
                          (thrust::device_ptr<T>)input_arr + part_arr2_start, (thrust::device_ptr<T>)input_arr + part_arr2_end,
                          (thrust::device_ptr<T>)output_arr + i * part_size);

            cuda_graph_comm::check_array("arr", output_arr, len, part_size - 10, part_size + 10);
        }

        part_size *= 2;
        part_num = part_num / 2 + part_num % 2;
    }

    if (output_arr == tmp_arr) {
        HRR(cudaMemcpy(arr, tmp_arr, sizeof(T) * len, cudaMemcpyDeviceToDevice));
    }

    HRR(cudaFree(tmp_arr));
}
