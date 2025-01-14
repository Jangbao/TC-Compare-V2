#pragma once
#include <cuda_profiler_api.h>

#include "comm/comm.h"
#include "graph/gpu_graph.h"

#define HU_threadsPerBlockInTC 256
#define HU_blocksPerKernelInTC 30000  // 100 when under test,10000 when run normal size dataset
#define HU_shareMemorySizeInBlock 3000

namespace tc {
namespace approach {
namespace Hu {

__global__ void getNodesWorkLoad(int startPos, int threadNum, long int *d_nodeWorkLoad, index_t *c_offset, vertex_t *c_row, uint *c_adjLen);

uint binarySearchValue(long int *array, long int value, uint arrayLength, int direction);

__global__ void triangleCountKernel(index_t *c_offset, vertex_t *c_row, uint *c_adjLen, int *c_blockStartNodeOffset, long int *c_sum);

void gpu_run(INIReader &config, GPUGraph &gpu_graph, std::string key_space = "Hu");

void start_up(INIReader &config, GPUGraph &gpu_graph, int argc, char **argv);

}  // namespace Hu
}  // namespace approach
}  // namespace tc
