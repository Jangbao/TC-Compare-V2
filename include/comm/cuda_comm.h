#pragma once
#include <cuda_runtime_api.h>
#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/merge.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>


static void HandError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("\n%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HRR(err) (HandError(err, __FILE__, __LINE__))

inline int CudaAttr(cudaDeviceAttr attr) {
    int dev, val;
    HRR(cudaGetDevice(&dev));
    HRR(cudaDeviceGetAttribute(&val, attr, dev));
    return val;
}

inline int NumberOfMPs() { return CudaAttr(cudaDevAttrMultiProcessorCount); }

inline int WarpSize() { return CudaAttr(cudaDevAttrWarpSize); }
