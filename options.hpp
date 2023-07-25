#ifndef OPTIONS_HPP
#define OPTIONS_HPP

#include "blosumTypes.hpp"
#include <string>
#include <iostream>

enum class KernelType{
    Half2,
    DPXs16,
    DPXs32,
    Float
};

std::string to_string(KernelType type);

bool isValidSinglePassType(KernelType type);
bool isValidMultiPassType_small(KernelType type);
bool isValidMultiPassType_large(KernelType type);
bool isValidOverflowType(KernelType type);

struct CudaSW4Options{

    bool help = false;
    bool loadFullDBToGpu = false;
    bool usePseudoDB = false;
    bool printLengthPartitions = false;
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

    std::string queryFile;
    std::string dbPrefix;
};

void printOptions(const CudaSW4Options& options);

bool parseArgs(int argc, char** argv, CudaSW4Options& options);

void printHelp(int argc, char** argv);

#endif