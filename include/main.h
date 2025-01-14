#include <string>

#include "approach/Bisson/tc.h"
#include "approach/Fox/tc.h"
#include "approach/Green/main.cuh"
#include "approach/GroupTC-HS/tc.h"
#include "approach/GroupTC-BS/tc.h"
#include "approach/GroupTC/tc.h"
#include "approach/H-INDEX/tc.h"
#include "approach/Hu/tc.h"
#include "approach/Polak/tc.h"
#include "approach/TRUST/tc.h"
#include "approach/TriCore/tc.h"
#include "comm/config_comm.h"
#include "datatransfer/csr2dcsr_data_transfer.h"
#include "datatransfer/csr2rid_dcsr_data_transfer.h"
#include "datatransfer/csr2trust_dcsr_data_transfer.h"
#include "graph/cuda_graph_comm.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/spdlog.h"
#include "ini/INIReader.h"

std::string get_program_dir();

spdlog::level::level_enum switch_log_level(std::string log_level_str);

void init_file_logger(std::string logger_name, std::string file, spdlog::level::level_enum log_level);

std::vector<std::string> get_datasets(const std::string& str, const std::string& delimiter = ",");

std::vector<std::string> merge_vectors(const std::vector<std::string>& vec1, const std::vector<std::string>& vec2);

void init_loggers(std::string log_file_path,  INIReader& config);

void flush_loggers();

int main(int argc, char** argv);