

#include <algorithm>
#include <iterator>
#include <iostream>
#include <chrono>

#include "sequence_io.h"
#include "dbdata.hpp"
#include "kseqpp/kseqpp.hpp"

#define TIMERSTART(label)                                                  \
    std::chrono::time_point<std::chrono::system_clock>                     \
        timerstart##label,                                                 \
        timerstop##label;                                                  \
    timerstart##label = std::chrono::system_clock::now();

#define TIMERSTOP(label)                                                   \
    timerstop##label = std::chrono::system_clock::now();                   \
    std::chrono::duration<double>                                          \
        timerdelta##label = timerstop##label-timerstart##label;            \
    std::cout << "# elapsed time ("<< #label <<"): "                       \
              << timerdelta##label.count()  << "s" << std::endl;



//void Func(const sequence_batch& batch, int batchId)
template<class Func>
void forEachSequenceBatchInFile(const std::string& inputfilename, Func callback){
    constexpr int ALIGN = 4;
    constexpr size_t maxCharactersInBatch = 50'000'000'000ull;

    auto resetBatch = [](auto& batch){
        batch.chars.clear();
        batch.offsets.clear();
        batch.lengths.clear();
        batch.headers.clear();
        batch.qualities.clear();

        batch.offsets.push_back(0);
    };

    sequence_batch batch;
    resetBatch(batch);

    int batchId = 0;

    kseqpp::KseqPP reader(inputfilename);
    while(reader.next() >= 0){
        const std::string& header = reader.getCurrentHeader();
        const std::string& sequence = reader.getCurrentSequence();
        //we ignore quality
        //const std::string& quality = reader.getCurrentQuality();

        batch.chars.insert(batch.chars.end(), sequence.begin(), sequence.end());
        //padding
        if(batch.chars.size() % ALIGN != 0){
            batch.chars.insert(batch.chars.end(), ALIGN - batch.chars.size() % ALIGN, ' ');
        }

        batch.offsets.push_back(batch.chars.size());
        batch.lengths.push_back(sequence.size());
        batch.headers.push_back(header);

        if(batch.chars.size() >= maxCharactersInBatch){
            callback(batch, batchId);
            resetBatch(batch);
            batchId++;
        }
    }

    //if there is any sequence left (even sequence with 0 chars), process remainder
    if(batch.lengths.size() > 0){
        callback(batch, batchId);
            resetBatch(batch);
            batchId++;
    }
}



int main(int argc, char* argv[])
{
    using std::cout;

    if(argc < 2) {
        cout << "Usage:\n  " << argv[0] << " <FASTA/FASTQ filename> [outputprefix]\n";
        return 0;
    }

    const std::string fastafilename = argv[1];
    std::string outputPrefix = "./mydb_";
    if(argc > 2){
        outputPrefix = argv[2];
    }


    /*
        Partition the input file in N chunks.
        Creates the following files:
        - outputPrefix+metadata
        for chunk i
            -outputPrefix+i+metadata
            -outputPrefix+i+chars
            -outputPrefix+i+lengths
            -outputPrefix+i+offsets
            -outputPrefix+i+headers
            -outputPrefix+i+headeroffsets
    */

    int processedNumBatches = 0;

    auto processBatch = [&](const sequence_batch& batch, int batchId){
        std::cout << "Converting sequence batch " << batchId << "\n";
        std::cout << "Number of input sequences:  " << batch.offsets.size() - 1 << '\n';
        std::cout << "Number of input characters: " << batch.chars.size() << '\n';

        const std::string batchOutputPrefix = outputPrefix + std::to_string(batchId);
        TIMERSTART(CONVERT_TO_DB_FORMAT)
        createDBfilesFromSequenceBatch(batchOutputPrefix, batch);
        TIMERSTOP(CONVERT_TO_DB_FORMAT)

        processedNumBatches++;
    };

    forEachSequenceBatchInFile(
        fastafilename,
        processBatch
    );

    DBGlobalInfo info;
    info.numChunks = processedNumBatches;

    writeGlobalDbInfo(outputPrefix, info);

}
