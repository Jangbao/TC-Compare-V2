#include "graph/cuda_graph_comm.h"

__global__ void cuda_graph_comm::copy_32_to_64(uint arr_len, int32_t* arr_32, int64_t* arr_64) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= arr_len) {
        return;
    }
    arr_64[i] = arr_32[i];
}

__global__ void cuda_graph_comm::zip_edge(uint edge_count, uint vertex_count, vertex_t* d_edge_arr, vertex_t* d_src_arr, vertex_t* d_adj_arr) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= edge_count) {
        return;
    }
    int src = d_src_arr[i];
    int adj = d_adj_arr[i];
    d_edge_arr[i * 2 + 1] = src;
    d_edge_arr[i * 2] = adj;
}

__global__ void cuda_graph_comm::cal_degree_and_zip_edge(uint edge_count, uint vertex_count, uint* d_degree_arr, vertex_t* d_edge_arr,
                                                         vertex_t* d_src_arr, vertex_t* d_adj_arr) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= edge_count) {
        return;
    }
    int src = d_src_arr[i];
    int adj = d_adj_arr[i];
    d_edge_arr[i * 2 + 1] = src;
    d_edge_arr[i * 2] = adj;

    atomicAdd(d_degree_arr + src, 1);
    atomicAdd(d_degree_arr + adj, 1);
}

__global__ void cuda_graph_comm::redirect_edge(uint edge_count, uint vertex_count, uint* d_degree_arr, vertex_t* d_edge_arr) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= edge_count) {
        return;
    }
    int src = d_edge_arr[i * 2 + 1];
    int adj = d_edge_arr[i * 2];
    // redirect edge
    if (d_degree_arr[src] > d_degree_arr[adj] || (d_degree_arr[src] == d_degree_arr[adj] && src > adj)) {
        d_edge_arr[i * 2 + 1] = adj;
        d_edge_arr[i * 2] = src;
    }
}

__global__ void cuda_graph_comm::redirect_edge(uint edge_count, uint vertex_count, uint* d_degree_arr, vertex_t* d_src_arr, vertex_t* d_adj_arr) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= edge_count) {
        return;
    }
    int src = d_src_arr[i];
    int dst = d_adj_arr[i];
    // redirect edge
    if (d_degree_arr[src] > d_degree_arr[dst] || (d_degree_arr[src] == d_degree_arr[dst] && src > dst)) {
        d_adj_arr[i] = src;
        d_src_arr[i] = dst;
    }
}

__global__ void cuda_graph_comm::unzip_edge(uint edge_count, uint vertex_count, vertex_t* d_edge_arr, vertex_t* d_src_arr, vertex_t* d_adj_arr) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= edge_count) {
        return;
    }
    d_src_arr[i] = d_edge_arr[i * 2 + 1];
    d_adj_arr[i] = d_edge_arr[i * 2];
}

__global__ void cuda_graph_comm::recal_offset(uint edge_count, uint vertex_count, vertex_t* d_src_arr, index_t* d_offset_arr) {
    uint from = blockDim.x * blockIdx.x + threadIdx.x;
    uint step = gridDim.x * blockDim.x;
    for (uint i = from; i <= edge_count; i += step) {
        int64_t prev = i > 0 ? (int64_t)d_src_arr[i - 1] : -1;
        int64_t next = i < edge_count ? (int64_t)d_src_arr[i] : vertex_count;
        // 前一个元素小于后一个元素，才有可能出现 offset 的计算
        for (int64_t j = prev + 1; j <= next; ++j) d_offset_arr[j] = i;
    }
}

__global__ void cuda_graph_comm::record_id(uint edge_count, uint vertex_count, vertex_t* d_id_arr) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= vertex_count) {
        return;
    }
    d_id_arr[i] = i;
}

__global__ void cuda_graph_comm::map_id(uint edge_count, uint vertex_count, vertex_t* d_id_arr, vertex_t* d_id_map_arr) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= vertex_count) {
        return;
    }
    d_id_map_arr[d_id_arr[i]] = i;
}

__global__ void cuda_graph_comm::redirect_edge_and_reassign_id(uint edge_count, uint vertex_count, vertex_t* d_id_map_arr, vertex_t* d_edge_arr) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= edge_count) {
        return;
    }
    int src = d_id_map_arr[d_edge_arr[i * 2 + 1]];
    int adj = d_id_map_arr[d_edge_arr[i * 2]];
    if (src > adj) {
        int temp = src;
        src = adj;
        adj = temp;
    }
    d_edge_arr[i * 2] = adj;
    d_edge_arr[i * 2 + 1] = src;
}

__global__ void cuda_graph_comm::cal_degree(uint edge_count, uint vertex_count, uint* d_degree_arr, vertex_t* d_src_arr, vertex_t* d_adj_arr) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= edge_count) {
        return;
    }
    int src = d_src_arr[i];
    int adj = d_adj_arr[i];

    atomicAdd(d_degree_arr + src, 1);
    atomicAdd(d_degree_arr + adj, 1);
}

__global__ void cuda_graph_comm::cal_out_degree_by_src(uint edge_count, uint vertex_count, uint* d_degree_arr, vertex_t* d_src_arr) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= edge_count) {
        return;
    }
    atomicAdd(d_degree_arr + d_src_arr[i], 1);
}

__global__ void cuda_graph_comm::cal_out_degree_by_offset(uint edge_count, uint vertex_count, uint* d_degree_arr, index_t* d_offset_arr) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= vertex_count) {
        return;
    }

    d_degree_arr[i] = d_offset_arr[i + 1] - d_offset_arr[i];
}

__global__ void cuda_graph_comm::record_id_and_part_graph_by_degree(uint edge_count, uint vertex_count, vertex_t* d_id_arr, uint* d_degree_arr) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= vertex_count) {
        return;
    }
    d_id_arr[i] = i;
    int degree = d_degree_arr[i];
    if (degree < 2) {
        degree = 2;
    } else if (degree <= 100) {
        degree = 1;
    } else {
        degree = 0;
    }
    d_degree_arr[i] = degree;
}

__global__ void cuda_graph_comm::reassign_id(uint edge_count, uint vertex_count, vertex_t* d_id_map_arr, vertex_t* d_src_arr, vertex_t* d_adj_arr) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= edge_count) {
        return;
    }
    d_src_arr[i] = d_id_map_arr[d_src_arr[i]];
    d_adj_arr[i] = d_id_map_arr[d_adj_arr[i]];
}

__global__ void cuda_graph_comm::check_order(vertex_t* arr, uint len) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len - 1) {
        return;
    }
    if (arr[i + 1] < arr[i]) {
        // printf("xxxxxxxxxxxxxxx error order xxxxxxxxxxxxxxx\n");
    }
}

__global__ void cuda_graph_comm::warm_up(int* data, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(size_t i = tid; i < n; i += blockDim.x * gridDim.x) {
        data[i] = i;
    }
}
