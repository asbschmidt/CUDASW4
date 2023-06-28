#include "dbdata.hpp"

#include <type_traits>
#include <string>
#include <fstream>
#include <algorithm>

struct DBdataIoConfig{
    static const std::string metadatafilename(){ return "metadata"; }
    static const std::string headerfilename(){ return "headers"; }
    static const std::string headeroffsetsfilename(){ return "headeroffsets"; }
    static const std::string sequencesfilename(){ return "chars"; }
    static const std::string sequenceoffsetsfilename(){ return "offsets"; }
    static const std::string sequencelengthsfilename(){ return "lengths"; }

};

void loadDBdata(const std::string& inputPrefix, DBdata& result){
    // std::ifstream in1(inputPrefix + DBdataIoConfig::metadatafilename());
    // if(!in1) throw std::runtime_error("Cannot open file " + inputPrefix + DBdataIoConfig::metadatafilename());
    // in1 >> result.nSequences;

    //only allow read access. do not prefetch headers into ram
    MappedFile::Options headerOptions;
    headerOptions.readaccess = true;
    headerOptions.writeaccess = false;
    headerOptions.prefault = false;

    result.mappedFileHeaders = std::make_unique<MappedFile>(inputPrefix + DBdataIoConfig::headerfilename(), headerOptions);
    result.mappedFileHeaderOffsets = std::make_unique<MappedFile>(inputPrefix + DBdataIoConfig::headeroffsetsfilename(), headerOptions);


    //only allow read access. prefetch file immediately
    MappedFile::Options sequenceOptions;
    sequenceOptions.readaccess = true;
    sequenceOptions.writeaccess = false;
    sequenceOptions.prefault = true;

    result.mappedFileSequences = std::make_unique<MappedFile>(inputPrefix + DBdataIoConfig::sequencesfilename(), sequenceOptions);
    result.mappedFileLengths = std::make_unique<MappedFile>(inputPrefix + DBdataIoConfig::sequencelengthsfilename(), sequenceOptions);
    result.mappedFileOffsets = std::make_unique<MappedFile>(inputPrefix + DBdataIoConfig::sequenceoffsetsfilename(), sequenceOptions);
}

//write vector to file, overwrites existing file
template<class T>
void writeTrivialVectorToFile(const std::vector<T>& vec, const std::string& filename){
    static_assert(std::is_trivially_copyable<T>::value, "writeTrivialVectorToFile: type not trivially copyable");

    std::ofstream out(filename, std::ios::binary);
    if(!out) throw std::runtime_error("Cannot open output file " + filename);
    out.write((const char*)vec.data(), sizeof(T) * vec.size());
}

void createDBfilesFromSequenceBatch(const std::string& outputPrefix, const sequence_batch& batch){
    const size_t numSequences = batch.lengths.size();

    std::ofstream out1(outputPrefix + DBdataIoConfig::metadatafilename());
    if(!out1) throw std::runtime_error("Cannot open output file " + outputPrefix + DBdataIoConfig::metadatafilename());
    //num sequences
    out1 << numSequences << "\n";
    //num bytes of headers
    out1 << batch.headers.size() << "\n";
    //num bytes of sequences
    out1 << batch.chars.size() << "\n";

    std::ofstream headersout(outputPrefix + DBdataIoConfig::headerfilename(), std::ios::binary);
    if(!headersout) throw std::runtime_error("Cannot open output file " + outputPrefix + DBdataIoConfig::headerfilename());

    std::ofstream headersoffsetsout(outputPrefix + DBdataIoConfig::headeroffsetsfilename(), std::ios::binary);
    if(!headersoffsetsout) throw std::runtime_error("Cannot open output file " + outputPrefix + DBdataIoConfig::headeroffsetsfilename());

    size_t currentHeaderOffset = 0;
    headersoffsetsout.write((const char*)&currentHeaderOffset, sizeof(size_t));
    for(size_t i = 0; i < numSequences; i++){
        headersout.write((const char*)batch.headers[i].data(), batch.headers[i].size());
        currentHeaderOffset += batch.headers[i].size();
        headersoffsetsout.write((const char*)&currentHeaderOffset, sizeof(size_t));
    }

    writeTrivialVectorToFile(batch.chars, outputPrefix + DBdataIoConfig::sequencesfilename());
    writeTrivialVectorToFile(batch.offsets, outputPrefix + DBdataIoConfig::sequenceoffsetsfilename());
    writeTrivialVectorToFile(batch.lengths, outputPrefix + DBdataIoConfig::sequencelengthsfilename());
}






std::vector<DBdataView> partitionDBdata_by_numberOfSequences(const DBdata& dbData, size_t maxNumSequencesPerPartition){

    const size_t numSequences = dbData.numSequences();
    std::vector<DBdataView> result;

    for(size_t i = 0; i < numSequences; i += maxNumSequencesPerPartition){
        const size_t numInPartition = std::min(maxNumSequencesPerPartition, numSequences - i);
        result.emplace_back(dbData, i, i + numInPartition);
    }

    return result;
}

//partitions have the smallest number of chars such that is at least numCharsPerPartition. (with the exception of last partition)
std::vector<DBdataView> partitionDBdata_by_numberOfChars(const DBdata& dbData, size_t numCharsPerPartition){

    const size_t numChars = dbData.numChars();

    std::vector<size_t> bucketLimits(1,0);
    size_t currentBegin = 0;
    while(currentBegin < numChars){
        const size_t searchBegin = currentBegin + numCharsPerPartition;
        const auto it = std::upper_bound(dbData.offsets(), dbData.offsets() + dbData.numSequences()+1, searchBegin);
        if(it == dbData.offsets() + dbData.numSequences()+1){
            bucketLimits.push_back(dbData.numSequences());
            currentBegin = numChars;
        }else{
            const size_t dist = std::distance(dbData.offsets(), it);
            bucketLimits.push_back(dist);
            currentBegin = dbData.offsets()[dist];
        }
    }

    const size_t numPartitions = bucketLimits.size()-1;
    std::vector<DBdataView> result;
    for(int p = 0; p < numPartitions; p++){
        result.emplace_back(dbData, bucketLimits[p], bucketLimits[p+1]);
    }

    return result;
}



void assertValidPartitioning(const std::vector<DBdataView>& views, const DBdata& dbData){
    const int numPartitions = views.size();

    std::vector<size_t> partitionOffsets(numPartitions+1, 0);
    for(int p = 1; p <= numPartitions; p++){
        partitionOffsets[p] = partitionOffsets[p-1] + views[p-1].numSequences();
    }


    const size_t totalNumSequencesInViews = partitionOffsets.back();
    // const size_t totalNumSequencesInViews = std::reduce(views.begin(), views.end(),
    //     [](const auto& v){return v.numSequences();}
    // );

    assert(dbData.numSequences() == totalNumSequencesInViews);

    #pragma omp parallel for
    for(int p = 0; p < numPartitions; p++){
        const DBdataView& view = views[p];

        for(size_t i = 0; i < view.numSequences(); i++){
            assert(view.lengths()[i] == dbData.lengths()[partitionOffsets[p] + i]);
            assert(view.offsets()[i] == dbData.offsets()[partitionOffsets[p] + i]);
            assert(view.headerOffsets()[i] == dbData.headerOffsets()[partitionOffsets[p] + i]);

            const char* const viewSeqEnd = view.chars() + view.offsets()[i] + view.lengths()[i];
            const char* const dbSeqEnd =  dbData.chars() + dbData.offsets()[partitionOffsets[p] + i] + dbData.lengths()[i];
            auto mismatchSeq = std::mismatch(
                view.chars() + view.offsets()[i],
                viewSeqEnd,
                dbData.chars() + dbData.offsets()[partitionOffsets[p] + i],
                dbSeqEnd
            );
            assert(mismatchSeq.first == viewSeqEnd || mismatchSeq.second == dbSeqEnd);

            const char* const viewHeaderEnd = view.headers() + view.headerOffsets()[i+1];
            const char* const dbHeaderEnd =  dbData.headers() + dbData.headerOffsets()[partitionOffsets[p] + i+1];
            auto mismatchHeader = std::mismatch(
                view.headers() + view.headerOffsets()[i],
                viewHeaderEnd,
                dbData.headers() + dbData.headerOffsets()[partitionOffsets[p] + i],
                dbHeaderEnd
            );
            assert(mismatchHeader.first == viewHeaderEnd || mismatchHeader.second == dbHeaderEnd);
        }
    }
}