#ifndef OPTIONS_HPP
#define OPTIONS_HPP

#include "types.hpp"
#include <string>
#include <iostream>


struct ProgramOptions{

    bool help = false;
    bool loadFullDBToGpu = false;
    bool usePseudoDB = false;
    bool printLengthPartitions = false;
    bool interactive = false;
    bool verbose = false;
    int numTopOutputs = 10;
    int gop = -11;
    int gex = -1;
    int pseudoDBLength = 0;
    int pseudoDBSize = 0;
    BlosumType blosumType = BlosumType::BLOSUM62_20;
    KernelType singlePassType = KernelType::Half2;
    KernelType manyPassType_small = KernelType::Half2;
    KernelType manyPassType_large = KernelType::Float;
    KernelType overflowType = KernelType::Float;

    size_t maxBatchBytes = 128ull * 1024ull * 1024ull;
    size_t maxBatchSequences = 10'000'000;
    size_t maxTempBytes = 4ull * 1024ull * 1024ull * 1024ull;

    size_t maxGpuMem = std::numeric_limits<size_t>::max();

    std::string dbPrefix;
    std::vector<std::string> queryFiles;
};

void printOptions(const ProgramOptions& options);

bool parseArgs(int argc, char** argv, ProgramOptions& options);

void printHelp(int argc, char** argv);

#endif