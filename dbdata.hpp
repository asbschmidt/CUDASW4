#ifndef DB_DATA_HPP
#define DB_DATA_HPP

#include "mapped_file.hpp"
#include "sequence_io.h"

#include <memory>
#include <fstream>

struct DBdata{
    friend void loadDBdata(const std::string& inputPrefix, DBdata& result);

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
    
private:
    DBdata() = default;

    std::unique_ptr<MappedFile> mappedFileSequences;
    std::unique_ptr<MappedFile> mappedFileLengths;
    std::unique_ptr<MappedFile> mappedFileOffsets;
    std::unique_ptr<MappedFile> mappedFileHeaders;
    std::unique_ptr<MappedFile> mappedFileHeaderOffsets;
};


void createDBfilesFromSequenceBatch(const std::string& outputPrefix, const sequence_batch& batch);






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
    DBdataView(const DBdata& dbData_, size_t first_, size_t last_) 
        : firstSequence(first_), lastSequence_excl(last_), dbData(&dbData_){

    }

    size_t numSequences() const noexcept{
        return lastSequence_excl - firstSequence;
    }

    size_t numChars() const noexcept{
        return dbData->offsets()[lastSequence_excl] - dbData->offsets()[firstSequence];
    }

    const char* chars() const noexcept{
        return dbData->chars();
    }

    const size_t* lengths() const noexcept{
        return dbData->lengths() + firstSequence;
    }

    const size_t* offsets() const noexcept{
        return dbData->offsets() + firstSequence;
    }

    const char* headers() const noexcept{
        return dbData->headers();
    }

    const size_t* headerOffsets() const noexcept{
        return dbData->headerOffsets() + firstSequence;
    }
    
private:
    size_t firstSequence;
    size_t lastSequence_excl;

    const DBdata* dbData;
};


void createDBfilesFromSequenceBatch(const std::string& outputPrefix, const sequence_batch& batch);


std::vector<DBdataView> partitionDBdata_by_numberOfSequences(const DBdata& dbData, size_t maxNumSequencesPerPartition);
//partitions have the smallest number of chars that is at least numCharsPerPartition. (with the exception of last partition)
std::vector<DBdataView> partitionDBdata_by_numberOfChars(const DBdata& dbData, size_t numCharsPerPartition);

void assertValidPartitioning(const std::vector<DBdataView>& views, const DBdata& dbData);

#endif