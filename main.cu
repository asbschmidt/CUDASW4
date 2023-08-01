


#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "hpc_helpers/all_helpers.cuh"

#include "kseqpp/kseqpp.hpp"
#include "sequence_io.h"
#include "options.hpp"
#include "dbdata.hpp"
#include "cudasw4.cuh"


std::vector<std::string> split(const std::string& str, char c){
	std::vector<std::string> result;

	std::stringstream ss(str);
	std::string s;

	while (std::getline(ss, s, c)) {
		result.emplace_back(s);
	}

	return result;
}

void printScanResult(const cudasw4::ScanResult& scanResult, const cudasw4::CudaSW4& cudaSW4){
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


int main(int argc, char* argv[])
{

    if(argc < 3) {
        std::cout << "Usage:\n  " << argv[0] << " <FASTA filename 1> [dbPrefix]\n";
        return 0;
    }

    ProgramOptions options;
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
            deviceIds.push_back(i);
        }
        if(deviceIds.size() > 0){
            if(options.verbose){
                std::cout << "Will use GPU";
                for(auto x : deviceIds){
                    std::cout << " " << x;
                }
                std::cout << "\n";
            }
        }else{
            throw std::runtime_error("No GPU found");
        }
    }
 
    using KernelTypeConfig = cudasw4::KernelTypeConfig;
    using MemoryConfig = cudasw4::MemoryConfig;
    using ScanResult = cudasw4::ScanResult;

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

    cudasw4::CudaSW4 cudaSW4(
        deviceIds, 
        options.blosumType, 
        kernelTypeConfig, 
        memoryConfig, 
        options.verbose
    );

    if(!options.usePseudoDB){
        if(options.verbose){
            std::cout << "Reading Database: \n";
        }
        helpers::CpuTimer timer_read_db("Read DB");
        constexpr bool writeAccess = false;
        constexpr bool prefetchSeq = true;
        auto fullDB_tmp = std::make_shared<DB>(loadDB(options.dbPrefix, writeAccess, prefetchSeq));
        if(options.verbose){
            timer_read_db.print();
        }

        cudaSW4.setDatabase(fullDB_tmp);
    }else{
        if(options.verbose){
            std::cout << "Generating pseudo db\n";
        }
        helpers::CpuTimer timer_read_db("Generate DB");
        auto fullDB_tmp = std::make_shared<PseudoDB>(loadPseudoDB(options.pseudoDBSize, options.pseudoDBLength));
        if(options.verbose){
            timer_read_db.print();
        }
        
        cudaSW4.setDatabase(fullDB_tmp);
    }

    if(options.verbose){
        cudaSW4.printDBInfo();
        if(options.printLengthPartitions){
            cudaSW4.printDBLengthPartitions();
        }
    }

    if(options.loadFullDBToGpu){
        cudaSW4.prefetchFullDBToGpus();
    }

    if(!options.interactive){

        for(const auto& queryFile : options.queryFiles){
            std::cout << "Processing query file " << queryFile << "\n";
        // 0 load all queries into memory, then process.
        // 1 load and process queries one after another
        #if 0
            kseqpp::KseqPP reader(queryFile);
            int query_num = 0;

            cudaSW4.totalTimerStart();

            while(reader.next() >= 0){
                std::cout << "Processing query " << query_num << " ... ";
                std::cout.flush();
                const std::string& header = reader.getCurrentHeader();
                const std::string& sequence = reader.getCurrentSequence();

                ScanResult scanResult = cudaSW4.scan(sequence.data(), sequence.size());
                if(options.verbose){
                    std::cout << "Done. Scan time: " << scanResult.stats.seconds << " s, " << scanResult.stats.gcups << " GCUPS\n";
                }else{
                    std::cout << "Done.\n";
                }

                std::cout << "Query " << query_num << ", header" <<  header
                << ", length " << sequence.size()
                << ", num overflows " << scanResult.stats.numOverflows << "\n";

                printScanResult(scanResult, cudaSW4);

                query_num++;
            }

            auto totalBenchmarkStats = cudaSW4.totalTimerStop();
            if(options.verbose){
                std::cout << "Total time: " << totalBenchmarkStats.seconds << " s, " << totalBenchmarkStats.gcups << " GCUPS\n";
            }

        #else

            sequence_batch batchOfQueries;
            {
                //batchOfQueries = read_all_sequences_and_headers_from_file(queryFile);
                constexpr int ALIGN = 4;
                kseqpp::KseqPP reader(queryFile);
                batchOfQueries.offsets.push_back(0);
                while(reader.next() >= 0){
                    const std::string& header = reader.getCurrentHeader();
                    const std::string& sequence = reader.getCurrentSequence();
                    //we ignore quality
                    //const std::string& quality = reader.getCurrentQuality();

                    batchOfQueries.chars.insert(batchOfQueries.chars.end(), sequence.begin(), sequence.end());
                    //padding
                    if(batchOfQueries.chars.size() % ALIGN != 0){
                        batchOfQueries.chars.insert(batchOfQueries.chars.end(), ALIGN - batchOfQueries.chars.size() % ALIGN, ' ');
                    }

                    batchOfQueries.offsets.push_back(batchOfQueries.chars.size());
                    batchOfQueries.lengths.push_back(sequence.size());
                    batchOfQueries.headers.push_back(header);
                }
            }

            int numQueries = batchOfQueries.lengths.size();
            const char* maxNumQueriesString = std::getenv("ALIGNER_MAX_NUM_QUERIES");
            if(maxNumQueriesString != nullptr){
                int maxNumQueries = std::atoi(maxNumQueriesString);
                numQueries = std::min(numQueries, maxNumQueries);
            }
        
            std::vector<ScanResult> scanResults(numQueries);

            cudaSW4.totalTimerStart();

            for(int query_num = 0; query_num < numQueries; ++query_num) {
                std::cout << "Processing query " << query_num << " ... ";
                std::cout.flush();
                const size_t offset = batchOfQueries.offsets[query_num];
                const int length = batchOfQueries.lengths[query_num];
                const char* sequence = batchOfQueries.chars.data() + offset;
                ScanResult scanResult = cudaSW4.scan(sequence, length);
                scanResults[query_num] = scanResult;
                if(options.verbose){
                    std::cout << "Done. Scan time: " << scanResult.stats.seconds << " s, " << scanResult.stats.gcups << " GCUPS\n";
                }else{
                    std::cout << "Done.\n";
                }
            }

            auto totalBenchmarkStats = cudaSW4.totalTimerStop();

            if(options.verbose){
                std::cout << "Total time: " << totalBenchmarkStats.seconds << " s, " << totalBenchmarkStats.gcups << " GCUPS\n";
            }
            for(int query_num = 0; query_num < numQueries; ++query_num) {
                const ScanResult& scanResult = scanResults[query_num];

                std::cout << "Query " << query_num << ", header" <<  batchOfQueries.headers[query_num] 
                    << ", length " << batchOfQueries.lengths[query_num]
                    << ", num overflows " << scanResult.stats.numOverflows << "\n";
                printScanResult(scanResult, cudaSW4);
            }
        #endif

        }
    }else{
        std::cout << "Interactive mode ready\n";
        std::cout << "Use 's inputsequence' to query inputsequence against the database\n";
        std::cout << "Use 'f inputfile' to query all sequences in inputfile\n";
        std::cout << "Use 'exit' to terminate\n";
        std::cout << "Waiting for command...\n";

        std::string line;
        while(std::getline(std::cin, line)){
            auto tokens = split(line, ' ');
            if(tokens.size() == 0) continue;

            const auto& command = tokens[0];
            if(command == "exit"){
                break;
            }else if(command == "s"){
                if(tokens.size() > 1){
                    const auto& sequence = tokens[1];
                    std::cout << "Processing query " << 0 << " ... ";
                    std::cout.flush();
                    ScanResult scanResult = cudaSW4.scan(sequence.data(), sequence.size());
                    if(options.verbose){
                        std::cout << "Done. Scan time: " << scanResult.stats.seconds << " s, " << scanResult.stats.gcups << " GCUPS\n";
                    }else{
                        std::cout << "Done.\n";
                    }

                    printScanResult(scanResult, cudaSW4);
                }else{
                    std::cout << "Missing argument for command 's'\n";
                }
            }else if(command == "f"){
                if(tokens.size() > 1){
                    const auto& filename = tokens[1];
                    try{
                        kseqpp::KseqPP reader(filename);
                        int query_num = 0;

                        while(reader.next() >= 0){
                            std::cout << "Processing query " << query_num << " ... ";
                            std::cout.flush();
                            const std::string& header = reader.getCurrentHeader();
                            const std::string& sequence = reader.getCurrentSequence();

                            ScanResult scanResult = cudaSW4.scan(sequence.data(), sequence.size());
                            if(options.verbose){
                                std::cout << "Done. Scan time: " << scanResult.stats.seconds << " s, " << scanResult.stats.gcups << " GCUPS\n";
                            }else{
                                std::cout << "Done.\n";
                            }

                            std::cout << "Query " << query_num << ", header" <<  header
                            << ", length " << sequence.size()
                            << ", num overflows " << scanResult.stats.numOverflows << "\n";

                            printScanResult(scanResult, cudaSW4);

                            query_num++;
                        }
                    }catch(...){
                        std::cout << "Error\n";
                    }
                }else{
                    std::cout << "Missing argument for command 'f' \n";
                }
            }else{
                std::cout << "Unrecognized command: " << command << "\n";
            }

            std::cout << "Waiting for command...\n";
        }

    }

}
