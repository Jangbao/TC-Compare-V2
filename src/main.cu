#include <unistd.h>

#include <string>

#include "main.h"

int main(int argc, char** argv) {
    std::string program_dir = get_program_dir();
    std::string config_file = program_dir + "/../config/config.ini";
    std::string config_dataset = "";

    // 存储命令行参数
    std::vector<std::string> args(argv, argv + argc);

    // 遍历命令行参数
    for (int i = 1; i < argc; ++i) {  // 从1开始，跳过程序名称
        std::string arg = args[i];
        if (arg.find("-config=") != std::string::npos) {
            config_file = arg.substr(8);
            spdlog::info("Config file is {}", config_file);
        }
        if (arg.find("-dataset=") != std::string::npos) {
            config_dataset = arg.substr(9);
            spdlog::info("Config_dataset file is {}", config_dataset);
        }
    }

    INIReader config(config_file);
    if (config.ParseError() < 0) {
        spdlog::info("Can't load {}", config_file);
        return 1;
    }

    // set comm config
    std::string dataset_file_path = config.Get("comm", "dataset_file_path", "UNKNOWN");
    std::string log_file_path = config.Get("comm", "log_file_path", "UNKNOWN");
    std::string str_datasets_1 = config.Get("comm", "datasets_1", "");
    std::string str_datasets_2 = config.Get("comm", "datasets_2", "");
    std::string log_level_str = config.Get("comm", "log_level", "info");
    spdlog::level::level_enum log_level = switch_log_level(log_level_str);
    spdlog::set_level(log_level);

    config_comm::cPreprocessingIterations = config.GetInteger("comm", "preprocessing_iterations", 1);

    int device = config.GetInteger("comm", "device", 0);
    HRR(cudaSetDevice(device));
    spdlog::info("Use device {}", device);

    // init loggers
    init_loggers(log_file_path, config);

    std::vector<std::string> datasets_1 = get_datasets(str_datasets_1);
    std::vector<std::string> datasets_2 = get_datasets(str_datasets_2);
    std::vector<std::string> datasets = merge_vectors(datasets_1, datasets_2);

    for (auto dataset : datasets) {
        std::string input_file = dataset_file_path + dataset;
        if (config_dataset != "" && dataset != config_dataset) {
            continue;
        }

        // read dataset
        CPUGraph cpu_graph(input_file);
        if (cpu_graph.vertex_count <= 0 || cpu_graph.edge_count <= 0) {
            spdlog::warn("Invalid graph data, process next graph ...");
            continue;
        }

        if (cpu_graph.edge_count > constant_comm::kMaxGraphEdgeCount) {
            spdlog::info("Input graph is too large! Process next graph ...");
            continue;
        }

        // GPU warm up
        {
            int* d_warmup;
            size_t warmup_size = 1024 * 1024 * 1024;  // 1GB
            HRR(cudaMalloc((void**)&d_warmup, warmup_size));
            HRR(cudaMemset(d_warmup, 0, warmup_size));

            dim3 grid(1024), block(1024);
            cuda_graph_comm::warm_up<<<grid, block>>>(d_warmup, warmup_size / sizeof(int));

            HRR(cudaDeviceSynchronize());
            HRR(cudaFree(d_warmup));
        }
        
        // preprocess datasets and run algorithms
        {
            Csr2DcsrDataTransfer cddt(input_file, &cpu_graph);
            cddt.transfer();
            GPUGraph& dcsr = cddt.d_graph;

            tc::approach::Polak::start_up(config, dcsr, argc, argv);
            tc::approach::TriCore::start_up(config, dcsr, argc, argv);
            tc::approach::H_INDEX::start_up(config, dcsr, argc, argv);
            tc::approach::Green::start_up(config, dcsr, argc, argv);
            tc::approach::Hu::start_up(config, dcsr, argc, argv);
            // tc::approach::GroupTC_BS::start_up(config, dcsr, argc, argv);
            tc::approach::Fox::start_up(config, dcsr, argc, argv);
        }

        {
            Csr2RidDcsrDataTransfer crdt(input_file, &cpu_graph);
            crdt.transfer();
            GPUGraph& riddcsr = crdt.d_graph;

            tc::approach::Bisson::start_up(config, riddcsr, argc, argv);
            tc::approach::GroupTC::start_up(config, riddcsr, argc, argv);
            tc::approach::GroupTC_BS::start_up(config, riddcsr, argc, argv);
        }

        {
            Csr2TrustDcsrDataTransfer ctdt(input_file, &cpu_graph);
            ctdt.transfer();
            GPUGraph& trustdcsr = ctdt.d_graph;

            tc::approach::GroupTC_HS::start_up(config, trustdcsr, argc, argv);
            tc::approach::TRUST::start_up(config, trustdcsr, argc, argv);
        }
        flush_loggers();
    }

    spdlog::info("All input graphs have been processed, the program ends.");

    return 0;
}

std::string get_program_dir() {
    char buffer[1024];
    ssize_t len = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (len != -1) {
        buffer[len] = '\0';
        std::string fullPath(buffer);
        std::string::size_type pos = fullPath.find_last_of("/");
        std::string program_directory = fullPath.substr(0, pos);
        return program_directory;
    }
    return "";
}

std::vector<std::string> get_datasets(const std::string& str, const std::string& delimiter) {
    std::vector<std::string> datasets;
    size_t pos = 0;
    std::string dataset;
    std::string str_copy = str;
    while ((pos = str_copy.find(delimiter)) != std::string::npos) {
        dataset = str_copy.substr(0, pos);
        dataset.erase(std::remove(dataset.begin(), dataset.end(), ' '), dataset.end());
        if (!dataset.empty()) {
            datasets.push_back(dataset);
        }
        str_copy.erase(0, pos + delimiter.length());
    }
    str_copy.erase(std::remove(str_copy.begin(), str_copy.end(), ' '), str_copy.end());
    if (!str_copy.empty()) {
        datasets.push_back(str_copy);
    }
    return datasets;
}

std::vector<std::string> merge_vectors(const std::vector<std::string>& vec1, const std::vector<std::string>& vec2) {
    std::vector<std::string> merged_vec;

    merged_vec.insert(merged_vec.end(), vec1.begin(), vec1.end());
    merged_vec.insert(merged_vec.end(), vec2.begin(), vec2.end());

    return merged_vec;
}

void init_loggers(std::string log_file_path, INIReader& config) {
    try {
        // 定义所有logger名称和对应的文件路径
        auto make_logger_pair = [&](std::string name, bool is_preprocessing = false) {
            if (!config.GetBoolean("comm", name, true)) {
                return std::make_pair(std::string(""), std::string(""));
            }
            if (is_preprocessing) {
                return std::make_pair(name + "_preprocessing_file_logger", name + "/preprocessing_time_output.txt");
            } else {
                return std::make_pair(name + "_file_logger", name + "/time_output.txt");
            }
        };

        const std::vector<std::pair<std::string, std::string>> loggers = {
            // CSR转换相关logger
            make_logger_pair("csr2dcsr"), make_logger_pair("csr2rid_dcsr"), make_logger_pair("csr2trust_dcsr"),

            // 预处理相关logger
            make_logger_pair("Fox", true), make_logger_pair("Hu", true),

            // 算法性能相关logger
            make_logger_pair("Bisson"), make_logger_pair("Fox"), make_logger_pair("Green"), make_logger_pair("GroupTC"),
            make_logger_pair("GroupTC-HASH"), make_logger_pair("GroupTC-BS"), make_logger_pair("H-INDEX"), make_logger_pair("Hu"),
            make_logger_pair("Polak"), make_logger_pair("TriCore"), make_logger_pair("TRUST"), make_logger_pair("GroupTC-HASH-V2"),
            make_logger_pair("GroupTC-HS")};

        // 批量初始化所有logger
        for (const auto& logger : loggers) {
            if (logger.first != "") {
                init_file_logger(logger.first, log_file_path + logger.second, spdlog::level::info);
            }
        }

    } catch (const spdlog::spdlog_ex& ex) {
        spdlog::error("Log init failed: {}", ex.what());
    }
}

void init_file_logger(std::string logger_name, std::string file, spdlog::level::level_enum log_level) {
    std::shared_ptr<spdlog::logger> logger = spdlog::basic_logger_mt(logger_name, file);
    logger->set_level(log_level);
}

void flush_loggers() {
    // 获取所有已注册的logger并刷新
    spdlog::apply_all([](std::shared_ptr<spdlog::logger> logger) { logger->flush(); });
}

spdlog::level::level_enum switch_log_level(std::string log_level_str) {
    static const std::unordered_map<std::string, spdlog::level::level_enum> level_map = {
        {"trace", spdlog::level::trace}, {"debug", spdlog::level::debug},       {"info", spdlog::level::info}, {"warn", spdlog::level::warn},
        {"err", spdlog::level::err},     {"critical", spdlog::level::critical}, {"off", spdlog::level::off}};

    auto it = level_map.find(log_level_str);
    return it != level_map.end() ? it->second : spdlog::level::info;
}
