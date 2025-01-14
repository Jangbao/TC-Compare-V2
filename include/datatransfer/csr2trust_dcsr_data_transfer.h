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

class Csr2TrustDcsrDataTransfer : public DataTransfer {
   public:
    Csr2TrustDcsrDataTransfer(std::string file, CPUGraph* graph) : DataTransfer(file, graph){};

    Csr2TrustDcsrDataTransfer() {};
    
    void transfer();
};
