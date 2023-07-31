


#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "hpc_helpers/all_helpers.cuh"

#include "kseqpp/kseqpp.hpp"
#include "sequence_io.h"
#include "options.hpp"
#include "dbdata.hpp"
#include "cudasw4.cuh"





int main(int argc, char* argv[])
{

    if(argc < 3) {
        std::cout << "Usage:\n  " << argv[0] << " <FASTA filename 1> [dbPrefix]\n";
        return 0;
    }

    CudaSW4Options options;
    bool parseSuccess = parseArgs(argc, argv, options);

    if(!parseSuccess || options.help){
        printHelp(argc, argv);
        return 0;
    }

    printOptions(options);

    std::vector<int> deviceIds;
    {
        int num = 0;
        cudaGetDeviceCount(&num); CUERR
        for(int i = 0; i < num; i++){
            std::cout << "Using device " << i << "\n";
            deviceIds.push_back(i);
        }
    }
 
    using KernelTypeConfig = cudasw4::CudaSW4::KernelTypeConfig;
    using MemoryConfig = cudasw4::CudaSW4::MemoryConfig;
    using ScanResult = cudasw4::CudaSW4::ScanResult;

    KernelTypeConfig kernelTypeConfig;
    kernelTypeConfig.singlePassType = options.singlePassType;
    kernelTypeConfig.manyPassType_small = options.manyPassType_small;
    kernelTypeConfig.manyPassType_large = options.manyPassType_large;
    kernelTypeConfig.overflowType = options.overflowType;

    MemoryConfig memoryConfig;
    memoryConfig.maxBatchBytes = options.maxBatchBytes;
    memoryConfig.maxBatchSequences = options.maxBatchSequences;
    memoryConfig.maxTempBytes = options.maxTempBytes;
    memoryConfig.maxGpuMem = options.maxGpuMem;

    cudasw4::CudaSW4 cudaSW4(deviceIds, options.blosumType, kernelTypeConfig, memoryConfig);

    if(!options.usePseudoDB){
        std::cout << "Reading Database: \n";
        helpers::CpuTimer timer_read_db("READ_DB");
        constexpr bool writeAccess = false;
        constexpr bool prefetchSeq = true;
        //DB fullDB_tmp = loadDB(options.dbPrefix, writeAccess, prefetchSeq);
        auto fullDB_tmp = std::make_shared<DB>(loadDB(options.dbPrefix, writeAccess, prefetchSeq));
        timer_read_db.print();

        cudaSW4.setDatabase(fullDB_tmp);
    }else{
        std::cout << "Generating pseudo db\n";
        helpers::CpuTimer timer_read_db("READ_DB");
        //PseudoDB fullDB_tmp = loadPseudoDB(options.pseudoDBSize, options.pseudoDBLength);
        auto fullDB_tmp = std::make_shared<PseudoDB>(loadPseudoDB(options.pseudoDBSize, options.pseudoDBLength));
        timer_read_db.print();
        
        cudaSW4.setDatabase(fullDB_tmp);
    }

    cudaSW4.printDBInfo();
    if(options.printLengthPartitions){
        cudaSW4.printDBLengthPartitions();
    }

    if(options.loadFullDBToGpu){
        cudaSW4.prefetchFullDBToGpus();
    }


    // 0 load all queries into memory, then process.
    // 1 load and process queries one after another
    #if 0
        kseqpp::KseqPP reader(options.queryFile);
        int query_num = 0;

        cudaSW4.totalTimerStart();

        while(reader.next() >= 0){
            const std::string& header = reader.getCurrentHeader();
            const std::string& sequence = reader.getCurrentSequence();

            ScanResult scanResult = cudaSW4.scan(sequence.data(), sequence.size());
            std::cout << "Query " << query_num << ". Scan time: " << scanResult.stats.seconds << " s, " << scanResult.stats.gcups << " GCUPS\n";

            std::cout << "Query " << query_num << ", header" <<  header
            << ", length " << sequence.size()
            << ", num overflows " << scanResult.stats.numOverflows << "\n";

            const int n = scanResult.scores.size();
            for(int i = 0; i < n; i++){
                const size_t referenceId = scanResult.referenceIds[i];
                std::cout << "Result " << i << ".";
                std::cout << " Score: " << scanResult.scores[i] << ".";
                std::cout << " Length: " << cudaSW4.getReferenceLength(referenceId) << ".";
                std::cout << " Header " << cudaSW4.getReferenceHeader(referenceId) << ".";
                std::cout << "\n";
                //std::cout << " Sequence " << cudaSW4.getReferenceSequence(referenceId) << "\n";

                // std::cout << i << "," 
                //     << scanResult.scores[i] << ","
                //     << cudaSW4.getReferenceLength(referenceId) << ","
                //     << cudaSW4.getReferenceHeader(referenceId) << "\n";
            }

            query_num++;
        }

        auto totalBenchmarkStats = cudaSW4.totalTimerStop();
        std::cout << "Total time: " << totalBenchmarkStats.seconds << " s, " << totalBenchmarkStats.gcups << " GCUPS\n";

    #else
    
        std::vector<ScanResult> scanResults(numQueries);

        cudaSW4.totalTimerStart();

        for(int query_num = 0; query_num < numQueries; ++query_num) {
            const size_t offset = batchOfQueries.offsets[query_num];
            const int length = batchOfQueries.lengths[query_num];
            const char* sequence = batchOfQueries.chars.data() + offset;
            ScanResult scanResult = cudaSW4.scan(sequence, length);
            scanResults[query_num] = scanResult;
            std::cout << "Query " << query_num << ". Scan time: " << scanResult.stats.seconds << " s, " << scanResult.stats.gcups << " GCUPS\n";
        }

        auto totalBenchmarkStats = cudaSW4.totalTimerStop();

        std::cout << "Total time: " << totalBenchmarkStats.seconds << " s, " << totalBenchmarkStats.gcups << " GCUPS\n";

        for(int query_num = 0; query_num < numQueries; ++query_num) {
            const ScanResult& scanResult = scanResults[query_num];

            std::cout << "Query " << query_num << ", header" <<  batchOfQueries.headers[query_num] 
                << ", length " << batchOfQueries.lengths[query_num]
                << ", num overflows " << scanResult.stats.numOverflows << "\n";
            const int n = scanResult.scores.size();
            for(int i = 0; i < n; i++){
                const size_t referenceId = scanResult.referenceIds[i];
                std::cout << "Result " << i << ".";
                std::cout << " Score: " << scanResult.scores[i] << ".";
                std::cout << " Length: " << cudaSW4.getReferenceLength(referenceId) << ".";
                std::cout << " Header " << cudaSW4.getReferenceHeader(referenceId) << ".";
                std::cout << "\n";
                //std::cout << " Sequence " << cudaSW4.getReferenceSequence(referenceId) << "\n";

                // std::cout << i << "," 
                //     << scanResult.scores[i] << ","
                //     << cudaSW4.getReferenceLength(referenceId) << ","
                //     << cudaSW4.getReferenceHeader(referenceId) << "\n";
            }
        }
    #endif

}
