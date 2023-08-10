


#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "hpc_helpers/all_helpers.cuh"
#include "hpc_helpers/peer_access.cuh"

#include "kseqpp/kseqpp.hpp"
#include "sequence_io.h"
#include "options.hpp"
#include "dbdata.hpp"
#include "cudasw4.cuh"
#include "config.hpp"


std::vector<std::string> split(const std::string& str, char c){
	std::vector<std::string> result;

	std::stringstream ss(str);
	std::string s;

	while (std::getline(ss, s, c)) {
		result.emplace_back(s);
	}

	return result;
}

void printScanResultPlain(std::ostream& os, const cudasw4::ScanResult& scanResult, const cudasw4::CudaSW4& cudaSW4){
    const int n = scanResult.scores.size();
    for(int i = 0; i < n; i++){
        const auto referenceId = scanResult.referenceIds[i];
        os << "Result " << i << ".";
        os << " Score: " << scanResult.scores[i] << ".";
        os << " Length: " << cudaSW4.getReferenceLength(referenceId) << ".";
        os << " Header " << cudaSW4.getReferenceHeader(referenceId) << ".";
        os << " referenceId " << referenceId;
        os << "\n";
        //std::cout << " Sequence " << cudaSW4.getReferenceSequence(referenceId) << "\n";

    }
}

void printTSVHeader(std::ostream& os){
    constexpr char sep = '\t';

    os << "Query number" << sep 
        << "Query length" << sep 
        << "Query header" << sep
        << "Result number" << sep
        << "Result score" << sep
        << "Reference length" << sep
        << "Reference header" << sep
        << "Reference ID in DB" << "\n";
}

void printScanResultTSV(
    std::ostream& os, 
    const cudasw4::ScanResult& scanResult, 
    const cudasw4::CudaSW4& cudaSW4, 
    int64_t queryId,
    cudasw4::SequenceLengthT queryLength,
    std::string_view queryHeader
){
    constexpr char sep = '\t';

    const int n = scanResult.scores.size();
    for(int i = 0; i < n; i++){
        const auto referenceId = scanResult.referenceIds[i];
        
        os << queryId << sep 
            << queryLength << sep
            << queryHeader << sep
            << i << sep
            << scanResult.scores[i] << sep
            << cudaSW4.getReferenceLength(referenceId) << sep
            << cudaSW4.getReferenceHeader(referenceId) << sep
            << referenceId << "\n";

        //std::cout << " Sequence " << cudaSW4.getReferenceSequence(referenceId) << "\n";
    }
}

struct BatchOfQueries{
    std::vector<char> chars;               
    std::vector<std::size_t> offsets;  
    std::vector<cudasw4::SequenceLengthT> lengths;  
    std::vector<std::string> headers;  
};



int main(int argc, char* argv[])
{
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

    helpers::PeerAccess peerAccess(deviceIds, false);
 
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

    std::ofstream outputfile(options.outputfile);
    if(!bool(outputfile)){
        throw std::runtime_error("Cannot open file " + options.outputfile);
    }
    if(options.outputMode == ProgramOptions::OutputMode::TSV){
        printTSVHeader(outputfile);
    }

    cudasw4::CudaSW4 cudaSW4(
        deviceIds, 
        options.numTopOutputs,
        options.blosumType, 
        kernelTypeConfig, 
        memoryConfig, 
        options.verbose
    );

    if(!options.usePseudoDB){
        if(options.verbose){
            std::cout << "Reading Database: \n";
        }
        try{
            helpers::CpuTimer timer_read_db("Read DB");
            constexpr bool writeAccess = false;
            const bool prefetchSeq = options.prefetchDBFile;
            auto fullDB_tmp = std::make_shared<cudasw4::DB>(cudasw4::loadDB(options.dbPrefix, writeAccess, prefetchSeq));
            if(options.verbose){
                timer_read_db.print();
            }

            cudaSW4.setDatabase(fullDB_tmp);
        }catch(cudasw4::LoadDBException& ex){
            if(options.verbose){
                std::cout << "Failed to map db files. Using fallback db. Error message: " << ex.what() << "\n";
            }
            helpers::CpuTimer timer_read_db("Read DB");
            auto fullDB_tmp = std::make_shared<cudasw4::DBWithVectors>(cudasw4::loadDBWithVectors(options.dbPrefix));
            if(options.verbose){
                timer_read_db.print();
            }

            cudaSW4.setDatabase(fullDB_tmp);
        }
    }else{
        if(options.verbose){
            std::cout << "Generating pseudo db\n";
        }
        helpers::CpuTimer timer_read_db("Generate DB");
        auto fullDB_tmp = std::make_shared<cudasw4::PseudoDB>(cudasw4::loadPseudoDB(options.pseudoDBSize, options.pseudoDBLength));
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
            int64_t query_num = 0;

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

                //std::cout << "subject 569751\n";
                //std::cout << cudaSW4.getReferenceSequence(569751) << "\n";

                //  std::ofstream err("errorsequence.fasta");

                // for(int i = 28998523; i < 28998523+1; i++){
                //     std::string s = cudaSW4.getReferenceSequence(i);
                //     err << ">" << i << "\n";
                //     err << s << "\n";
                // }
                // err.flush();
                // err.close();

                // std::vector<int> cpuscores = cudaSW4.computeAllScoresCPU(sequence.data(), sequence.size());

                // std::vector<int> indices(cpuscores.size());
                // std::iota(indices.begin(), indices.end(), 0);

                // std::sort(indices.begin(), indices.end(), [&](int l, int r){return scanResult.referenceIds[l] < scanResult.referenceIds[r];});
                // std::ofstream ofs("scorestmp.txt");
                // int lastMismatch = -1;
                // for(size_t i = 0; i < cpuscores.size(); i++){
                //     bool mismatch = cpuscores[i] != scanResult.scores[indices[i]]; 
                //     ofs << cpuscores[i] << " " << scanResult.scores[indices[i]] << " " << mismatch << "\n";
                //     if(mismatch){
                //         lastMismatch = i;
                //     }
                //     // if(cpuscores[i] != scanResult.scores[indices[i]]){
                //     //     std::cout << "i " << i << ", cpu score " << cpuscores[i] << ", gpu score " << scanResult.scores[indices[i]] << "\n";
                //     //     std::cout << "gpu ref id " << scanResult.referenceIds[indices[i]] << "\n";
                //     //     std::exit(0);
                //     // }
                // }
                // std::cout << "ok\n";
                // std::cout << "last mismatch " << lastMismatch << "\n";

                if(options.numTopOutputs > 0){
                    if(options.outputMode == ProgramOptions::OutputMode::Plain){
                        outputfile << "Query " << query_num << ", header" <<  header
                            << ", length " << sequence.size()
                            << ", num overflows " << scanResult.stats.numOverflows << "\n";

                        printScanResultPlain(outputfile, scanResult, cudaSW4);
                    }else{
                        printScanResultTSV(outputfile, scanResult, cudaSW4, query_num, sequence.size(), header);
                    }
                    outputfile.flush();
                }

                query_num++;
            }

            auto totalBenchmarkStats = cudaSW4.totalTimerStop();
            if(options.verbose){
                std::cout << "Total time: " << totalBenchmarkStats.seconds << " s, " << totalBenchmarkStats.gcups << " GCUPS\n";
            }

        #else

            BatchOfQueries batchOfQueries;
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

            int64_t numQueries = batchOfQueries.lengths.size();
            const char* maxNumQueriesString = std::getenv("ALIGNER_MAX_NUM_QUERIES");
            if(maxNumQueriesString != nullptr){
                int64_t maxNumQueries = std::atoi(maxNumQueriesString);
                numQueries = std::min(numQueries, maxNumQueries);
            }
        
            std::vector<ScanResult> scanResults(numQueries);

            cudaSW4.totalTimerStart();

            for(int64_t query_num = 0; query_num < numQueries; ++query_num) {
                std::cout << "Processing query " << query_num << " ... ";
                std::cout.flush();
                const size_t offset = batchOfQueries.offsets[query_num];
                const cudasw4::SequenceLengthT length = batchOfQueries.lengths[query_num];
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
            if(options.numTopOutputs > 0){
                for(int64_t query_num = 0; query_num < numQueries; ++query_num) {
                    const ScanResult& scanResult = scanResults[query_num];

                    if(options.outputMode == ProgramOptions::OutputMode::Plain){
                        outputfile << "Query " << query_num << ", header" <<  batchOfQueries.headers[query_num] 
                            << ", length " << batchOfQueries.lengths[query_num]
                            << ", num overflows " << scanResult.stats.numOverflows << "\n";
                        printScanResultPlain(outputfile, scanResult, cudaSW4);
                    }else{
                        printScanResultTSV(outputfile, scanResult, cudaSW4, query_num, batchOfQueries.lengths[query_num], batchOfQueries.headers[query_num]);
                    }
                }
            }
        #endif

        }
    }else{
        std::cout << "Interactive mode ready\n";
        std::cout << "Use 's inputsequence' to query inputsequence against the database. Press ENTER twice to begin.\n";
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
                    auto& sequence = tokens[1];

                    //read the remaining lines to catch multi-line sequence input (for example copy&paste fasta sequence)
                    while(std::getline(std::cin, line)){
                        if(line.empty()) break;
                        sequence += line;
                    }

                    std::cout << "sequence: " << sequence << "\n";
                    std::cout << "Processing query " << 0 << " ... ";
                    std::cout.flush();
                    ScanResult scanResult = cudaSW4.scan(sequence.data(), sequence.size());
                    if(options.verbose){
                        std::cout << "Done. Scan time: " << scanResult.stats.seconds << " s, " << scanResult.stats.gcups << " GCUPS\n";
                    }else{
                        std::cout << "Done.\n";
                    }

                    if(options.outputMode == ProgramOptions::OutputMode::Plain){
                        printScanResultPlain(outputfile, scanResult, cudaSW4);
                    }else{
                        printScanResultTSV(outputfile, scanResult, cudaSW4, -1, sequence.size(), "-");
                    }
                }else{
                    std::cout << "Missing argument for command 's'\n";
                }
            }else if(command == "f"){
                if(tokens.size() > 1){
                    const auto& filename = tokens[1];
                    try{
                        kseqpp::KseqPP reader(filename);
                        int64_t query_num = 0;

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

                            if(options.outputMode == ProgramOptions::OutputMode::Plain){
                                std::cout << "Query " << query_num << ", header" <<  header
                                << ", length " << sequence.size()
                                << ", num overflows " << scanResult.stats.numOverflows << "\n";

                                printScanResultPlain(outputfile, scanResult, cudaSW4);
                            }else{
                                printScanResultTSV(outputfile, scanResult, cudaSW4, -1, sequence.size(), "-");
                            }

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
