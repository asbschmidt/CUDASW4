#ifndef DB_DATA_HPP
#define DB_DATA_HPP

#include "mapped_file.hpp"
#include "sequence_io.h"

#include <memory>
#include <fstream>

struct DBdataIoConfig{
    static const std::string metadatafilename(){ return "metadata"; }
    static const std::string headerfilename(){ return "headers"; }
    static const std::string headeroffsetsfilename(){ return "headeroffsets"; }
    static const std::string sequencesfilename(){ return "chars"; }
    static const std::string sequenceoffsetsfilename(){ return "offsets"; }
    static const std::string sequencelengthsfilename(){ return "lengths"; }
};


struct DBGlobalInfo{
    int numChunks;
};

struct DBdata{
    friend void loadDBdata(const std::string& inputPrefix, DBdata& result);

    struct MetaData{
        std::vector<int> lengthBoundaries;
        std::vector<size_t> numSequencesPerLengthPartition;
    };

    DBdata(const std::string& inputPrefix){
        loadDBdata(inputPrefix, *this);
    }

    size_t numSequences() const noexcept{
        return mappedFileLengths->numElements<size_t>();
    }

    size_t numChars() const noexcept{
        return mappedFileSequences->numElements<char>();
    }

    const char* chars() const noexcept{
        return mappedFileSequences->data();
    }

    const size_t* lengths() const noexcept{
        return reinterpret_cast<const size_t*>(mappedFileLengths->data());
    }

    const size_t* offsets() const noexcept{
        return reinterpret_cast<const size_t*>(mappedFileOffsets->data());
    }

    const char* headers() const noexcept{
        return mappedFileHeaders->data();
    }

    const size_t* headerOffsets() const noexcept{
        return reinterpret_cast<const size_t*>(mappedFileHeaderOffsets->data());
    }

    const MetaData& getMetaData() const noexcept{
        return metaData;
    }
    
private:
    DBdata() = default;


    std::unique_ptr<MappedFile> mappedFileSequences;
    std::unique_ptr<MappedFile> mappedFileLengths;
    std::unique_ptr<MappedFile> mappedFileOffsets;
    std::unique_ptr<MappedFile> mappedFileHeaders;
    std::unique_ptr<MappedFile> mappedFileHeaderOffsets;
    MetaData metaData;
};

struct DB{
    DBGlobalInfo info;
    std::vector<DBdata> chunks;
};

void createDBfilesFromSequenceBatch(const std::string& outputPrefix, const sequence_batch& batch);
void writeGlobalDbInfo(const std::string& outputPrefix, const DBGlobalInfo& info);
void readGlobalDbInfo(const std::string& prefix, DBGlobalInfo& info);

DB loadDB(const std::string& prefix);






/*
    A view of a partion of DBdata.

    The i-th sequence data in the partition begins at chars() + offsets()[i].
    It has length lengths[i].
    Its header begins at headers() + headerOffsets()[i]

    Important note!:
    This view currently simply modifies the pointers to the original dbData arrays.
    It does not contain a copy of access offsets that begin with 0, and chars() returns the original dbData chars() ptr.
    This means when copying the view data to the device, the host sequence src pointer 
    must be chars() + offsets()[0], not chars(). !, i.e. cudaMemcpy(d_chars, view.chars() + offsets()[0], sizeof(char) * view.numChars())

    Because offsets are stored unmodified, offsets()[0] must be substracted from d_offsets after copying to obatin the correct offsets into d_chars

    The same applies to header offsets if they were to be used on the gpu


*/
struct DBdataView{
    DBdataView(const DBdata& parent) 
        : firstSequence(0), 
        lastSequence_excl(parent.numSequences()), 
        parentChars(parent.chars()),
        parentLengths(parent.lengths()),
        parentOffsets(parent.offsets()),
        parentHeaders(parent.headers()),
        parentHeaderOffsets(parent.headerOffsets())
    {

    }

    DBdataView(const DBdataView& parent, size_t first_, size_t last_) 
        : firstSequence(first_), 
        lastSequence_excl(last_), 
        parentChars(parent.chars()),
        parentLengths(parent.lengths()),
        parentOffsets(parent.offsets()),
        parentHeaders(parent.headers()),
        parentHeaderOffsets(parent.headerOffsets())
    {

    }

    size_t numSequences() const noexcept{
        return lastSequence_excl - firstSequence;
    }

    size_t numChars() const noexcept{
        return parentOffsets[lastSequence_excl] - parentOffsets[firstSequence];
    }

    const char* chars() const noexcept{
        return parentChars;
    }

    const size_t* lengths() const noexcept{
        return parentLengths + firstSequence;
    }

    const size_t* offsets() const noexcept{
        return parentOffsets + firstSequence;
    }

    const char* headers() const noexcept{
        return parentHeaders;
    }

    const size_t* headerOffsets() const noexcept{
        return parentHeaderOffsets + firstSequence;
    }
    
private:
    size_t firstSequence;
    size_t lastSequence_excl;

    const char* parentChars;
    const size_t* parentLengths;
    const size_t* parentOffsets;
    const char* parentHeaders;
    const size_t* parentHeaderOffsets;
};


void createDBfilesFromSequenceBatch(const std::string& outputPrefix, const sequence_batch& batch);


std::vector<DBdataView> partitionDBdata_by_numberOfSequences(const DBdataView& parent, size_t maxNumSequencesPerPartition);
//partitions have the smallest number of chars that is at least numCharsPerPartition. (with the exception of last partition)
std::vector<DBdataView> partitionDBdata_by_numberOfChars(const DBdataView& parent, size_t numCharsPerPartition);

void assertValidPartitioning(const std::vector<DBdataView>& views, const DBdataView& parent);

#endif