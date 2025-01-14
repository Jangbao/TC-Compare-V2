#pragma once
#include <cuda_profiler_api.h>

#include "comm/comm.h"
#include "graph/gpu_graph.h"

#define Bisson_shareMemorySizeInBlock 200
#define Bisson_hIndex 2048

namespace tc {
namespace approach {
namespace Bisson {

__global__ void triangleCountKernel(uint nodeNum, index_t *c_offset, vertex_t *c_row, uint *c_adjLen, long int *c_sum, int *c_bitmap);

void gpu_run(INIReader &config, GPUGraph &gpu_graph, std::string key_space = "Bisson");

void start_up(INIReader &config, GPUGraph &gpu_graph, int argc, char **argv);

}  // namespace Bisson
}  // namespace approach
}  // namespace tc
