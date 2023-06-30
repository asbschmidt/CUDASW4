#include "dbdata.hpp"
#include "length_partitions.hpp"

#include <type_traits>
#include <string>
#include <fstream>
#include <algorithm>
#include <numeric>

#include <thrust/iterator/transform_iterator.h>


void loadDBdata(const std::string& inputPrefix, DBdata& result){
    std::ifstream metadatain(inputPrefix + DBdataIoConfig::metadatafilename(), std::ios::binary);
    if(!metadatain) throw std::runtime_error("Cannot open file " + inputPrefix + DBdataIoConfig::metadatafilename());

    int numPartitions = 0;
    metadatain.read((char*)&numPartitions, sizeof(int));

    result.metaData.lengthBoundaries.resize(numPartitions);
    result.metaData.numSequencesPerLengthPartition.resize(numPartitions);
    metadatain.read((char*)result.metaData.lengthBoundaries.data(), sizeof(int) * numPartitions);
    metadatain.read((char*)result.metaData.numSequencesPerLengthPartition.data(), sizeof(size_t) * numPartitions);

    //
    auto expectedBoundaries = getLengthPartitionBoundaries();
    if(expectedBoundaries.size() != result.metaData.lengthBoundaries.size()){
        throw std::runtime_error("Invalid partition info in metadata.");
    }
    for(int i = 0; i < numPartitions; i++){
        if(expectedBoundaries[i] != result.metaData.lengthBoundaries[i]){
            throw std::runtime_error("Invalid partition info in metadata.");
        }
    }


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

    std::vector<size_t> indices(numSequences);
    std::iota(indices.begin(), indices.end(), 0);

    auto compareIndicesByLength = [&](const auto& l, const auto& r){
        return batch.lengths[l] < batch.lengths[r];
    };

    std::sort(indices.begin(), indices.end(), compareIndicesByLength);

    const auto sortedLengthsIterator = thrust::make_transform_iterator(
        indices.begin(),
        [&](auto i){ return batch.lengths[i]; }
    );
    const auto sortedLengths_endIterator = sortedLengthsIterator + numSequences;

    auto lengthBoundaries = getLengthPartitionBoundaries();
    const int numPartitions = lengthBoundaries.size();

    std::vector<size_t> numSequencesPerPartition(numPartitions);

    auto partitionBegin = sortedLengthsIterator;
    for(int i = 0; i < numPartitions; i++){
        //length k is in partition i if boundaries[i-1] < k <= boundaries[i]
        int searchFor = lengthBoundaries[i];
        if(searchFor < std::numeric_limits<int>::max()){
            searchFor += 1;
        }
        auto partitionEnd = std::lower_bound(
            partitionBegin, 
            sortedLengths_endIterator, 
            searchFor
        );
        numSequencesPerPartition[i] = std::distance(partitionBegin, partitionEnd);
        partitionBegin = partitionEnd;
    }
    for(int i = 0; i < numPartitions; i++){
        std::cout << "numInPartition " << i << " (<= " << lengthBoundaries[i] << " ) : " << numSequencesPerPartition[i] << "\n";
    }

    //write partition data to metadata file
    std::ofstream metadataout(outputPrefix + DBdataIoConfig::metadatafilename(), std::ios::binary);
    if(!metadataout) throw std::runtime_error("Cannot open output file " + outputPrefix + DBdataIoConfig::metadatafilename());

    metadataout.write((const char*)&numPartitions, sizeof(int));
    for(int i = 0; i < numPartitions; i++){
        const int limit = lengthBoundaries[i];
        metadataout.write((const char*)&limit, sizeof(int));
    }
    metadataout.write((const char*)numSequencesPerPartition.data(), sizeof(size_t) * numPartitions);


    //write db files with sequences sorted by length

    std::ofstream headersout(outputPrefix + DBdataIoConfig::headerfilename(), std::ios::binary);
    if(!headersout) throw std::runtime_error("Cannot open output file " + outputPrefix + DBdataIoConfig::headerfilename());
    std::ofstream headersoffsetsout(outputPrefix + DBdataIoConfig::headeroffsetsfilename(), std::ios::binary);
    if(!headersoffsetsout) throw std::runtime_error("Cannot open output file " + outputPrefix + DBdataIoConfig::headeroffsetsfilename());
    std::ofstream charsout(outputPrefix + DBdataIoConfig::sequencesfilename(), std::ios::binary);
    if(!charsout) throw std::runtime_error("Cannot open output file " + outputPrefix + DBdataIoConfig::sequencesfilename());
    std::ofstream offsetsout(outputPrefix + DBdataIoConfig::sequenceoffsetsfilename(), std::ios::binary);
    if(!offsetsout) throw std::runtime_error("Cannot open output file " + outputPrefix + DBdataIoConfig::sequenceoffsetsfilename());
    std::ofstream lengthsout(outputPrefix + DBdataIoConfig::sequencelengthsfilename(), std::ios::binary);
    if(!lengthsout) throw std::runtime_error("Cannot open output file " + outputPrefix + DBdataIoConfig::sequencelengthsfilename());

    size_t currentHeaderOffset = 0;
    size_t currentCharOffset = 0;
    headersoffsetsout.write((const char*)&currentHeaderOffset, sizeof(size_t));
    offsetsout.write((const char*)&currentCharOffset, sizeof(size_t));
    for(size_t i = 0; i < numSequences; i++){
        const size_t sortedIndex = indices[i];

        const char* const header = batch.headers[sortedIndex].data();
        const size_t headerlength = batch.headers[sortedIndex].size();

        headersout.write(header, headerlength);
        currentHeaderOffset += headerlength;
        headersoffsetsout.write((const char*)&currentHeaderOffset, sizeof(size_t));

        const size_t numChars = batch.offsets[sortedIndex+1] - batch.offsets[sortedIndex];
        const size_t length = batch.lengths[sortedIndex];
        const char* const sequence = batch.chars.data() + batch.offsets[sortedIndex];


        charsout.write(sequence, numChars);
        lengthsout.write((const char*)&length, sizeof(size_t));
        currentCharOffset += numChars;
        offsetsout.write((const char*)&currentCharOffset, sizeof(size_t));
    }
}




void writeGlobalDbInfo(const std::string& outputPrefix, const DBGlobalInfo& info){
    //write info data to metadata file
    std::ofstream metadataout(outputPrefix + DBdataIoConfig::metadatafilename(), std::ios::binary);
    if(!metadataout) throw std::runtime_error("Cannot open output file " + outputPrefix + DBdataIoConfig::metadatafilename());

    metadataout.write((const char*)&info.numChunks, sizeof(int));
}

void readGlobalDbInfo(const std::string& prefix, DBGlobalInfo& info){
    //write info data to metadata file
    std::ifstream metadatain(prefix + DBdataIoConfig::metadatafilename(), std::ios::binary);
    if(!metadatain) throw std::runtime_error("Cannot open file " + prefix + DBdataIoConfig::metadatafilename());

    metadatain.read((char*)&info.numChunks, sizeof(int));
}


DB loadDB(const std::string& prefix){

    DB result;
    readGlobalDbInfo(prefix, result.info);

    for(int i = 0; i < result.info.numChunks; i++){
        const std::string chunkPrefix = prefix + std::to_string(i);
        result.chunks.emplace_back(chunkPrefix);
    }

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