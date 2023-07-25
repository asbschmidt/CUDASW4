#include "options.hpp"
#include "blosumTypes.hpp"

#include <string>
#include <iostream>

std::string to_string(KernelType type){
    switch(type){
        case KernelType::Half2: return "Half2"; break;
        case KernelType::DPXs16: return "DPXs16"; break;
        case KernelType::DPXs32: return "DPXs32"; break;
        case KernelType::Float: return "Float"; break;
        default: return "Missing name for KernelType";
    }
}

bool isValidSinglePassType(KernelType type){
    return (type == KernelType::Half2 || type == KernelType::DPXs16);
}

bool isValidMultiPassType_small(KernelType type){
    return (type == KernelType::Half2);
}

bool isValidMultiPassType_large(KernelType type){
    return (type == KernelType::Float);
}

bool isValidOverflowType(KernelType type){
    return (type == KernelType::Float);
}

void printOptions(const CudaSW4Options& options){
    std::cout << "Selected options:\n";
    std::cout << "loadFullDBToGpu: " << options.loadFullDBToGpu << "\n";
    std::cout << "numTopOutputs: " << options.numTopOutputs << "\n";
    std::cout << "gop: " << options.gop << "\n";
    std::cout << "gex: " << options.gex << "\n";
    std::cout << "maxBatchBytes: " << options.maxBatchBytes << "\n";
    std::cout << "maxBatchSequences: " << options.maxBatchSequences << "\n";
    std::cout << "maxTempBytes: " << options.maxTempBytes << "\n";
    std::cout << "queryFile: " << options.queryFile << "\n";
    std::cout << "blosum: " << to_string(options.blosumType) << "\n";
    std::cout << "singlePassType: " << to_string(options.singlePassType) << "\n";
    std::cout << "manyPassType_small: " << to_string(options.manyPassType_small) << "\n";
    std::cout << "manyPassType_large: " << to_string(options.manyPassType_large) << "\n";
    std::cout << "overflowType: " << to_string(options.overflowType) << "\n";
    if(options.usePseudoDB){
        std::cout << "Using built-in pseudo db with " << options.pseudoDBSize << " sequences of length " << options.pseudoDBLength << "\n";
    }else{
        std::cout << "Using db file: " << options.dbPrefix << "\n";
    }
    
}

bool parseArgs(int argc, char** argv, CudaSW4Options& options){

    auto parseMemoryString = [](const std::string& string) -> std::size_t{
        if(string.length() > 0){
            std::size_t factor = 1;
            bool foundSuffix = false;
            switch(string.back()){
                case 'K':{
                    factor = std::size_t(1) << 10; 
                    foundSuffix = true;
                }break;
                case 'M':{
                    factor = std::size_t(1) << 20;
                    foundSuffix = true;
                }break;
                case 'G':{
                    factor = std::size_t(1) << 30;
                    foundSuffix = true;
                }break;
            }
            if(foundSuffix){
                const auto numberString = string.substr(0, string.size()-1);
                return factor * std::stoull(numberString);
            }else{
                return std::stoull(string);
            }
        }else{
            return 0;
        }
    };

    auto stringToKernelType = [&](const std::string& string){
        if(string == "Half2") return KernelType::Half2;
        if(string == "DPXs16") return KernelType::DPXs16;
        if(string == "DPXs32") return KernelType::DPXs32;
        if(string == "Float") return KernelType::Float;
        assert(false);
        return KernelType::Half2;
    };

    bool gotQuery = false;
    bool gotDB = false;
    bool gotGex = false;
    bool gotGop = false;

    for(int i = 1; i < argc; i++){
        const std::string arg = argv[i];
        if(arg == "--help"){
            options.help = true;
        }else if(arg == "--uploadFull"){
            options.loadFullDBToGpu = true;
        }else if(arg == "--top"){
            options.numTopOutputs = std::atoi(argv[++i]);
        }else if(arg == "--gop"){
            options.gop = std::atoi(argv[++i]);
            gotGop = true;
        }else if(arg == "--gex"){
            options.gex = std::atoi(argv[++i]);
            gotGex = true;
        }else if(arg == "--maxBatchBytes"){
            options.maxBatchBytes = parseMemoryString(argv[++i]);
        }else if(arg == "--maxBatchSequences"){
            options.maxBatchSequences = std::atoi(argv[++i]);
        }else if(arg == "--maxTempBytes"){
            options.maxTempBytes = parseMemoryString(argv[++i]);
        }else if(arg == "--query"){
            options.queryFile = argv[++i];
            gotQuery = true;
        }else if(arg == "--db"){
            options.dbPrefix = argv[++i];
            gotDB = true;
        }else if(arg == "--mat"){
            const std::string val = argv[++i];
            #ifdef CAN_USE_FULL_BLOSUM
            if(val == "blosum45") options.blosumType = BlosumType::BLOSUM45;
            if(val == "blosum50") options.blosumType = BlosumType::BLOSUM50;
            if(val == "blosum62") options.blosumType = BlosumType::BLOSUM62;
            if(val == "blosum80") options.blosumType = BlosumType::BLOSUM80;
            if(val == "blosum45_20") options.blosumType = BlosumType::BLOSUM45_20;
            if(val == "blosum50_20") options.blosumType = BlosumType::BLOSUM50_20;
            if(val == "blosum62_20") options.blosumType = BlosumType::BLOSUM62_20;
            if(val == "blosum80_20") options.blosumType = BlosumType::BLOSUM80_20;
            #else
            if(val == "blosum45") options.blosumType = BlosumType::BLOSUM45_20;
            if(val == "blosum50") options.blosumType = BlosumType::BLOSUM50_20;
            if(val == "blosum62") options.blosumType = BlosumType::BLOSUM62_20;
            if(val == "blosum80") options.blosumType = BlosumType::BLOSUM80_20;
            if(val == "blosum45_20") options.blosumType = BlosumType::BLOSUM45_20;
            if(val == "blosum50_20") options.blosumType = BlosumType::BLOSUM50_20;
            if(val == "blosum62_20") options.blosumType = BlosumType::BLOSUM62_20;
            if(val == "blosum80_20") options.blosumType = BlosumType::BLOSUM80_20;
            #endif
        }else if(arg == "--singlePassType"){
            options.singlePassType = stringToKernelType(argv[++i]);
        }else if(arg == "--manyPassType_small"){
            options.manyPassType_small = stringToKernelType(argv[++i]);
        }else if(arg == "--manyPassType_large"){
            options.manyPassType_large = stringToKernelType(argv[++i]);
        }else if(arg == "--overflowType"){
            options.overflowType = stringToKernelType(argv[++i]);
        }else if(arg == "--pseudodb"){
            options.usePseudoDB = true;
            options.pseudoDBSize = std::atoi(argv[++i]);
            options.pseudoDBLength = std::atoi(argv[++i]);
            gotDB = true;
        }else{
            std::cout << "Unexpected arg " << arg << "\n";
        }
    }

    //set specific gop gex for blosum if no gop gex was set
    if(options.blosumType == BlosumType::BLOSUM45 || options.blosumType == BlosumType::BLOSUM45_20){
        if(!gotGop) options.gop = -13;
        if(!gotGex) options.gex = -2;
    }
    if(options.blosumType == BlosumType::BLOSUM50 || options.blosumType == BlosumType::BLOSUM50_20){
        if(!gotGop) options.gop = -13;
        if(!gotGex) options.gex = -2;
    }
    if(options.blosumType == BlosumType::BLOSUM62 || options.blosumType == BlosumType::BLOSUM62_20){
        if(!gotGop) options.gop = -11;
        if(!gotGex) options.gex = -1;
    }
    if(options.blosumType == BlosumType::BLOSUM80 || options.blosumType == BlosumType::BLOSUM80_20){
        if(!gotGop) options.gop = -10;
        if(!gotGex) options.gex = -1;
    }

    if(!gotQuery){
        std::cout << "Query is missing\n";
        return false;
    }
    if(!gotDB){
        std::cout << "DB prefix is missing\n";
        return false;
    }

    if(!isValidSinglePassType(options.singlePassType)){
        std::cout << "Invalid singlepass kernel type\n";
        return false;
    }

    if(!isValidMultiPassType_small(options.manyPassType_small)){
        std::cout << "Invalid manyPassType_small kernel type\n";
        return false;
    }

    if(!isValidMultiPassType_large(options.manyPassType_large)){
        std::cout << "Invalid manyPassType_large kernel type\n";
        return false;
    }

    if(!isValidOverflowType(options.overflowType)){
        std::cout << "Invalid overflow kernel type\n";
        return false;
    }


    return true;
}

void printHelp(int /*argc*/, char** argv){
    CudaSW4Options defaultoptions;

    std::cout << "Usage: " << argv[0] << " [options]\n";
    std::cout << "Options: \n";
    std::cout << "      --query queryfile : Mandatory. Fasta or Fastq\n";
    std::cout << "      --db dbPrefix : Mandatory. The DB to query against. The same dbPrefix as used for makedb\n";
    std::cout << "      --top val : Output the val best scores. Default val = " << defaultoptions.numTopOutputs << "\n";
    std::cout << "      --maxTempBytes val : Size of temp storage in GPU memory. Can use suffix K,M,G. Default val = " << defaultoptions.maxTempBytes << "\n";
    std::cout << "      --maxBatchBytes val : Process DB in batches of at most val bytes. Can use suffix K,M,G. Default val = " << defaultoptions.maxBatchBytes << "\n";
    std::cout << "      --maxBatchSequences val : Process DB in batches of at most val sequences. Default val = " << defaultoptions.maxBatchSequences << "\n";
    std::cout << "      --uploadFull : Do not process DB in smaller batches. Copy full DB to GPU before processing queries.\n";
    //std::cout << "      --blosum50 : Use BLOSUM50 substitution matrix.\n";
    //std::cout << "      --blosum62 : Use BLOSUM62 substitution matrix.\n";
    #ifdef CAN_USE_FULL_BLOSUM
    std::cout << "      --mat val: Set substitution matrix. Supported values: blosum45, blosum50, blosum62, blosum80, blosum45_20, blosum50_20, blosum62_20, blosum80_20. "
                        "Default: " << to_string(defaultoptions.blosumType) << "\n";
    #else 
    std::cout << "      --mat val: Set substitution matrix. Supported values: blosum45, blosum50, blosum62, blosum80. "
                        "Default: " << to_string_nodim(defaultoptions.blosumType) << "\n";
    #endif
    std::cout << "      --gop val : Gap open score. Overwrites our blosum-dependent default score.\n";
    std::cout << "      --gex val : Gap extend score. Overwrites our blosum-dependent default score.\n";
    std::cout << "      --pseudodb num length : Use a generated DB which contains `num` equal sequences of length `length`.\n";
    std::cout << "      --singlePassType val, --manyPassType_small val, --manyPassType_large val, --overflowType val : Select kernel types for different length partitions. "
                        "Valid values: Half2, DPXs16, DPXs32, Float.\n";
}