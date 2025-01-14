#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "data_transfer.h"

class Csr2RidDcsrDataTransfer : public DataTransfer {
   public:
    Csr2RidDcsrDataTransfer(std::string file, CPUGraph* graph) : DataTransfer(file, graph){};

    Csr2RidDcsrDataTransfer() {};
    
    void transfer();
};
