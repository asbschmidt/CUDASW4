#include "dbdata.hpp"
#include "length_partitions.hpp"

#include <type_traits>
#include <string>
#include <fstream>
#include <algorithm>
#include <numeric>

namespace cudasw4{

void loadDBdata(const std::string& inputPrefix, DBdata& result, bool writeAccess, bool prefetchSeq, size_t globalSequenceOffset){


    MappedFile::Options headerOptions;
    headerOptions.readaccess = true;
    headerOptions.writeaccess = writeAccess;
    headerOptions.prefault = false;

    result.mappedFileHeaders = std::make_unique<MappedFile>(inputPrefix + DBdataIoConfig::headerfilename(), headerOptions);
    result.mappedFileHeaderOffsets = std::make_unique<MappedFile>(inputPrefix + DBdataIoConfig::headeroffsetsfilename(), headerOptions);


    MappedFile::Options sequenceOptions;
    sequenceOptions.readaccess = true;
    sequenceOptions.writeaccess = writeAccess;
    sequenceOptions.prefault = prefetchSeq;

    result.mappedFileSequences = std::make_unique<MappedFile>(inputPrefix + DBdataIoConfig::sequencesfilename(), sequenceOptions);
    result.mappedFileLengths = std::make_unique<MappedFile>(inputPrefix + DBdataIoConfig::sequencelengthsfilename(), sequenceOptions);
    result.mappedFileOffsets = std::make_unique<MappedFile>(inputPrefix + DBdataIoConfig::sequenceoffsetsfilename(), sequenceOptions);

    result.globalSequenceOffset = globalSequenceOffset;

    // std::ifstream metadatain(inputPrefix + DBdataIoConfig::metadatafilename(), std::ios::binary);
    // if(!metadatain) throw std::runtime_error("Cannot open file " + inputPrefix + DBdataIoConfig::metadatafilename());

    // int numPartitions = 0;
    // metadatain.read((char*)&numPartitions, sizeof(int));

    // result.metaData.lengthBoundaries.resize(numPartitions);
    // result.metaData.numSequencesPerLengthPartition.resize(numPartitions);
    // metadatain.read((char*)result.metaData.lengthBoundaries.data(), sizeof(int) * numPartitions);
    // metadatain.read((char*)result.metaData.numSequencesPerLengthPartition.data(), sizeof(size_t) * numPartitions);
    //
    // auto expectedBoundaries = getLengthPartitionBoundaries();
    // if(expectedBoundaries.size() != result.metaData.lengthBoundaries.size()){
    //     throw std::runtime_error("Invalid partition info in metadata.");
    // }
    // for(int i = 0; i < numPartitions; i++){
    //     if(expectedBoundaries[i] != result.metaData.lengthBoundaries[i]){
    //         throw std::runtime_error("Invalid partition info in metadata.");
    //     }
    // }


    auto lengthBoundaries = getLengthPartitionBoundaries();
    // std::vector<int> lengthBoundaries;
    // for(int l = 64; l <= 8192; l += 64){
    //     lengthBoundaries.push_back(l);
    // }
    const int numPartitions = lengthBoundaries.size();
    result.metaData.lengthBoundaries.resize(numPartitions);
    result.metaData.numSequencesPerLengthPartition.resize(numPartitions);

    auto partitionBegin = result.lengths();
    for(int i = 0; i < numPartitions; i++){
        //length k is in partition i if boundaries[i-1] < k <= boundaries[i]
        SequenceLengthT searchFor = lengthBoundaries[i];
        if(searchFor < std::numeric_limits<SequenceLengthT>::max()){
            searchFor += 1;
        }
        auto partitionEnd = std::lower_bound(
            partitionBegin, 
            result.lengths() + result.numSequences(), 
            searchFor
        );
        result.metaData.lengthBoundaries[i] = lengthBoundaries[i];
        result.metaData.numSequencesPerLengthPartition[i] = std::distance(partitionBegin, partitionEnd);
        partitionBegin = partitionEnd;
    }
}

//write vector to file, overwrites existing file
template<class T>
void writeTrivialVectorToFile(const std::vector<T>& vec, const std::string& filename){
    static_assert(std::is_trivially_copyable<T>::value, "writeTrivialVectorToFile: type not trivially copyable");

    std::ofstream out(filename, std::ios::binary);
    if(!out) throw std::runtime_error("Cannot open output file " + filename);
    out.write((const char*)vec.data(), sizeof(T) * vec.size());
}







void writeGlobalDbInfo(const std::string& outputPrefix, const DBGlobalInfo& /*info*/){
    //write info data to metadata file
    std::ofstream metadataout(outputPrefix + DBdataIoConfig::metadatafilename(), std::ios::binary);
    if(!metadataout) throw std::runtime_error("Cannot open output file " + outputPrefix + DBdataIoConfig::metadatafilename());

}

void readGlobalDbInfo(const std::string& prefix, DBGlobalInfo& /*info*/){
    //write info data to metadata file
    std::ifstream metadatain(prefix + DBdataIoConfig::metadatafilename(), std::ios::binary);
    if(!metadatain) throw std::runtime_error("Cannot open file " + prefix + DBdataIoConfig::metadatafilename());

}


DB loadDB(const std::string& prefix, bool writeAccess, bool prefetchSeq){

    DB result;
    readGlobalDbInfo(prefix, result.info);

    const std::string chunkPrefix = prefix + std::to_string(0);
    result.data = DBdata(chunkPrefix, writeAccess, prefetchSeq, 0);

    return result;
}


PseudoDB loadPseudoDB(size_t num, size_t length, int randomseed){
    PseudoDB result;
    result.data = PseudoDBdata(num, length, randomseed);

    return result;
}






std::vector<DBdataView> partitionDBdata_by_numberOfSequences(const DBdataView& parent, size_t maxNumSequencesPerPartition){

    const size_t numSequences = parent.numSequences();
    std::vector<DBdataView> result;

    for(size_t i = 0; i < numSequences; i += maxNumSequencesPerPartition){
        const size_t numInPartition = std::min(maxNumSequencesPerPartition, numSequences - i);
        result.emplace_back(parent, i, i + numInPartition);
    }

    return result;
}

//partitions have the smallest number of chars such that is at least numCharsPerPartition. (with the exception of last partition)
std::vector<DBdataView> partitionDBdata_by_numberOfChars(const DBdataView& parent, size_t numCharsPerPartition){

    const size_t numChars = parent.numChars();

    std::vector<size_t> bucketLimits(1,0);
    size_t currentBegin = parent.offsets()[0];
    const size_t end = parent.offsets()[0] + numChars;
    while(currentBegin < end){
        const size_t searchBegin = currentBegin + numCharsPerPartition;
        const auto it = std::upper_bound(parent.offsets(), parent.offsets() + parent.numSequences()+1, searchBegin);
        if(it == parent.offsets() + parent.numSequences()+1){
            bucketLimits.push_back(parent.numSequences());
            currentBegin = end;
        }else{
            const size_t dist = std::distance(parent.offsets(), it);
            bucketLimits.push_back(dist);
            currentBegin = parent.offsets()[dist];
        }
    }

    const size_t numPartitions = bucketLimits.size()-1;
    std::vector<DBdataView> result;
    for(size_t p = 0; p < numPartitions; p++){
        result.emplace_back(parent, bucketLimits[p], bucketLimits[p+1]);
    }

    return result;
}



void assertValidPartitioning(const std::vector<DBdataView>& views, const DBdataView& parent){
    const int numPartitions = views.size();

    std::vector<size_t> partitionOffsets(numPartitions+1, 0);
    for(int p = 1; p <= numPartitions; p++){
        partitionOffsets[p] = partitionOffsets[p-1] + views[p-1].numSequences();
    }


    const size_t totalNumSequencesInViews = partitionOffsets.back();
    // const size_t totalNumSequencesInViews = std::reduce(views.begin(), views.end(),
    //     [](const auto& v){return v.numSequences();}
    // );

    assert(parent.numSequences() == totalNumSequencesInViews);

    #pragma omp parallel for
    for(int p = 0; p < numPartitions; p++){
        const DBdataView& view = views[p];

        for(size_t i = 0; i < view.numSequences(); i++){
            assert(view.lengths()[i] == parent.lengths()[partitionOffsets[p] + i]);
            assert(view.offsets()[i] == parent.offsets()[partitionOffsets[p] + i]);
            assert(view.headerOffsets()[i] == parent.headerOffsets()[partitionOffsets[p] + i]);

            const char* const viewSeqEnd = view.chars() + view.offsets()[i] + view.lengths()[i];
            const char* const dbSeqEnd =  parent.chars() + parent.offsets()[partitionOffsets[p] + i] + parent.lengths()[i];
            auto mismatchSeq = std::mismatch(
                view.chars() + view.offsets()[i],
                viewSeqEnd,
                parent.chars() + parent.offsets()[partitionOffsets[p] + i],
                dbSeqEnd
            );
            assert(mismatchSeq.first == viewSeqEnd || mismatchSeq.second == dbSeqEnd);

            const char* const viewHeaderEnd = view.headers() + view.headerOffsets()[i+1];
            const char* const dbHeaderEnd =  parent.headers() + parent.headerOffsets()[partitionOffsets[p] + i+1];
            auto mismatchHeader = std::mismatch(
                view.headers() + view.headerOffsets()[i],
                viewHeaderEnd,
                parent.headers() + parent.headerOffsets()[partitionOffsets[p] + i],
                dbHeaderEnd
            );
            assert(mismatchHeader.first == viewHeaderEnd || mismatchHeader.second == dbHeaderEnd);
        }
    }
}

} //namespace cudasw4