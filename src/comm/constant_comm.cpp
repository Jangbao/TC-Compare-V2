#include "comm/constant_comm.h"

namespace constant_comm {

int const kPrintArrNum = 5;

int const kTransferIterations = 100;

int const kAlgorithmRunIterations = 10;

uint const kThrustSortThresholdSize = 4e9;

// For Tesla V100 32GB
uint const kMaxGraphEdgeCount = 1.85 * 1e9;

uint const kDataTransferMaxEdgeCount = 1.3 * 1e9;

uint const kFoxMaxEdgeCount = 1.3 * 1e9;

}  // namespace constant_comm
