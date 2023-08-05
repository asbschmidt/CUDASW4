#ifndef DB_DATA_HPP
#define DB_DATA_HPP

#include "mapped_file.hpp"
#include "sequence_io.h"
#include "length_partitions.hpp"
#include "convert.cuh"

#include "config.hpp"

#include <memory>
#include <fstream>
#include <random>
#include <algorithm>
#include <numeric>
#include <vector>
#include <iostream>

namespace cudasw4{

struct DBdataIoConfig{
    static const std::string metadatafilename(){ return "metadata"; }
    static const std::string headerfilename(){ return "headers"; }
    static const std::string headeroffsetsfilename(){ return "headeroffsets"; }
    static const std::string sequencesfilename(){ return "chars"; }
    static const std::string sequenceoffsetsfilename(){ return "offsets"; }
    static const std::string sequencelengthsfilename(){ return "lengths"; }
};


struct DBGlobalInfo{
    
};


struct DBdataMetaData{
    std::vector<int> lengthBoundaries;
    std::vector<size_t> numSequencesPerLengthPartition;
};

struct DBdata{
    friend void loadDBdata(const std::string& inputPrefix, DBdata& result, bool writeAccess, bool prefetchSeq, size_t globalSequenceOffset);
    friend struct DB;

    DBdata(const std::string& inputPrefix, bool writeAccess, bool prefetchSeq, size_t globalSequenceOffset = 0){
        loadDBdata(inputPrefix, *this, writeAccess, prefetchSeq, globalSequenceOffset);
    }

    DBdata(const DBdata&) = delete;
    DBdata(DBdata&&) = default;
    DBdata& operator=(const DBdata&) = delete;
    DBdata& operator=(DBdata&&) = default;

    size_t getGlobalSequenceOffset() const noexcept{
        return globalSequenceOffset;
    }

    size_t numSequences() const noexcept{
        return mappedFileLengths->numElements<SequenceLengthT>();
    }

    size_t numChars() const noexcept{
        return mappedFileSequences->numElements<char>();
    }

    const char* chars() const noexcept{
        return mappedFileSequences->data();
    }

    const SequenceLengthT* lengths() const noexcept{
        return reinterpret_cast<const SequenceLengthT*>(mappedFileLengths->data());
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

    const DBdataMetaData& getMetaData() const noexcept{
        return metaData;
    }

    char* chars() noexcept{
        return mappedFileSequences->data();
    }

    SequenceLengthT* lengths() noexcept{
        return reinterpret_cast<SequenceLengthT*>(mappedFileLengths->data());
    }

    size_t* offsets() noexcept{
        return reinterpret_cast<size_t*>(mappedFileOffsets->data());
    }

    char* headers() noexcept{
        return mappedFileHeaders->data();
    }

    size_t* headerOffsets() noexcept{
        return reinterpret_cast<size_t*>(mappedFileHeaderOffsets->data());
    }

    
private:
    DBdata() = default;

    size_t globalSequenceOffset;
    std::unique_ptr<MappedFile> mappedFileSequences;
    std::unique_ptr<MappedFile> mappedFileLengths;
    std::unique_ptr<MappedFile> mappedFileOffsets;
    std::unique_ptr<MappedFile> mappedFileHeaders;
    std::unique_ptr<MappedFile> mappedFileHeaderOffsets;
    DBdataMetaData metaData;
};

struct PseudoDBdata{
    friend struct PseudoDB;

    PseudoDBdata(size_t num, SequenceLengthT length, int randomseed = 42)
    : lengthRounded(((length + 3) / 4) * 4),
        charvec(num * lengthRounded),
        lengthvec(num),
        offsetvec(num+1),
        headervec(num), //headers will be only 1 letter
        headeroffsetvec(num+1)
    {
        const char* letters = "ARNDCQEGHILKMFPSTWYV";

        std::mt19937 gen(randomseed);
        std::uniform_int_distribution<> dist(0,19);


        std::string dummyseq(length, ' ');
        for(SequenceLengthT i = 0; i < length; i++){
            dummyseq[i] = letters[dist(gen)];
        }
        //std::cout << "PseudoDBdata: num " << num << ", length " << length << ", sequence " << dummyseq << "\n";

        for(size_t i = 0; i < num; i++){
            offsetvec[i] = i * lengthRounded;
            std::copy(dummyseq.begin(), dummyseq.end(), charvec.begin() + i * lengthRounded);
        }
        offsetvec[num] = num * lengthRounded;

        std::fill(lengthvec.begin(), lengthvec.end(), length);

        std::fill(headervec.begin(), headervec.end(), 'H');
        std::iota(headeroffsetvec.begin(), headeroffsetvec.end(), size_t(0));

        //convert amino acids to integers
        std::transform(charvec.begin(), charvec.end(), charvec.begin(), ConvertAA_20{});
        

        auto boundaries = getLengthPartitionBoundaries();

        metaData.lengthBoundaries.insert(metaData.lengthBoundaries.end(), boundaries.begin(), boundaries.end());
        metaData.numSequencesPerLengthPartition.resize(boundaries.size());

        for(int i = 0; i < int(boundaries.size()); i++){
            SequenceLengthT lower = i == 0 ? 0 : boundaries[i-1];
            SequenceLengthT upper = boundaries[i];

            if(lower < length && length <= upper){
                metaData.numSequencesPerLengthPartition[i] = num;
            }else{
                metaData.numSequencesPerLengthPartition[i] = 0;
            }
        }
    }

    PseudoDBdata(const PseudoDBdata&) = delete;
    PseudoDBdata(PseudoDBdata&&) = default;
    PseudoDBdata& operator=(const PseudoDBdata&) = delete;
    PseudoDBdata& operator=(PseudoDBdata&&) = default;

    size_t getGlobalSequenceOffset() const noexcept{
        return 0;
    }

    size_t numSequences() const noexcept{
        return lengthvec.size();
    }

    size_t numChars() const noexcept{
        return charvec.size();
    }

    const char* chars() const noexcept{
        return charvec.data();
    }

    const SequenceLengthT* lengths() const noexcept{
        return lengthvec.data();
    }

    const size_t* offsets() const noexcept{
        return offsetvec.data();
    }

    const char* headers() const noexcept{
        return headervec.data();
    }

    const size_t* headerOffsets() const noexcept{
        return headeroffsetvec.data();
    }

    const DBdataMetaData& getMetaData() const noexcept{
        return metaData;
    }
    
private:

    PseudoDBdata() = default;

    size_t lengthRounded;
    std::vector<char> charvec;
    std::vector<SequenceLengthT> lengthvec;
    std::vector<size_t> offsetvec;
    std::vector<char> headervec;
    std::vector<size_t> headeroffsetvec;
    DBdataMetaData metaData;
};

struct DB{
    friend DB loadDB(const std::string& prefix, bool writeAccess, bool prefetchSeq);

    
    DB(const DB&) = delete;
    DB(DB&&) = default;
    DB& operator=(const DB&) = delete;
    DB& operator=(DB&&) = default;

    DBGlobalInfo getInfo() const{
        return info;
    }

    const DBdata& getData() const{
        return data;
    }

    DBdata& getModyfiableData(){
        return data;
    }

private:
    DB() = default;

    DBGlobalInfo info;
    DBdata data;
};

struct PseudoDB{
    friend PseudoDB loadPseudoDB(size_t num, size_t length, int randomseed);

    PseudoDB() = default;
    PseudoDB(const PseudoDB&) = delete;
    PseudoDB(PseudoDB&&) = default;
    PseudoDB& operator=(const PseudoDB&) = delete;
    PseudoDB& operator=(PseudoDB&&) = default;

    DBGlobalInfo getInfo() const{
        return info;
    }

    const PseudoDBdata& getData() const{
        return data;
    }

private:
    DBGlobalInfo info;
    PseudoDBdata data;
};

void writeGlobalDbInfo(const std::string& outputPrefix, const DBGlobalInfo& info);
void readGlobalDbInfo(const std::string& prefix, DBGlobalInfo& info);

DB loadDB(const std::string& prefix, bool writeAccess, bool prefetchSeq);
PseudoDB loadPseudoDB(size_t num, size_t length, int randomseed = 42);







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
    DBdataView(): firstSequence(0), 
        lastSequence_excl(0), 
        globalSequenceOffset(0),
        parentChars(nullptr),
        parentLengths(nullptr),
        parentOffsets(nullptr),
        parentHeaders(nullptr),
        parentHeaderOffsets(nullptr)
    {

    }

    DBdataView(const DBdata& parent, size_t globalSequenceOffset_ = 0) 
        : firstSequence(0), 
        lastSequence_excl(parent.numSequences()), 
        globalSequenceOffset(globalSequenceOffset_),
        parentChars(parent.chars()),
        parentLengths(parent.lengths()),
        parentOffsets(parent.offsets()),
        parentHeaders(parent.headers()),
        parentHeaderOffsets(parent.headerOffsets())
    {

    }

    DBdataView(const PseudoDBdata& parent, size_t globalSequenceOffset_ = 0) 
        : firstSequence(0), 
        lastSequence_excl(parent.numSequences()), 
        globalSequenceOffset(globalSequenceOffset_),
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
        globalSequenceOffset(parent.getGlobalSequenceOffset() + firstSequence),
        parentChars(parent.chars()),
        parentLengths(parent.lengths()),
        parentOffsets(parent.offsets()),
        parentHeaders(parent.headers()),
        parentHeaderOffsets(parent.headerOffsets())
    {

    }

    size_t getGlobalSequenceOffset() const noexcept{
        return globalSequenceOffset;
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

    const SequenceLengthT* lengths() const noexcept{
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
    size_t globalSequenceOffset; //index of firstSequence at the top level, i.e. in the full db

    const char* parentChars;
    const SequenceLengthT* parentLengths;
    const size_t* parentOffsets;
    const char* parentHeaders;
    const size_t* parentHeaderOffsets;
};

struct AnyDBWrapper{
    AnyDBWrapper() = default;

    AnyDBWrapper(std::shared_ptr<DB> db){
        setDB(*db);
        dbPtr = db;
    }

    AnyDBWrapper(std::shared_ptr<PseudoDB> db){
        setDB(*db);
        pseudoDBPtr = db;
    }

    DBGlobalInfo getInfo() const{
        return info;
    }

    const DBdataView& getData() const{
        return data;
    }

private:
    template<class DB>
    void setDB(const DB& db){
        info = db.getInfo();
        data = DBdataView(db.getData());
    }
    std::shared_ptr<DB> dbPtr = nullptr;
    std::shared_ptr<PseudoDB> pseudoDBPtr = nullptr;
    
    DBGlobalInfo info;
    DBdataView data;   

};



std::vector<DBdataView> partitionDBdata_by_numberOfSequences(const DBdataView& parent, size_t maxNumSequencesPerPartition);
//partitions have the smallest number of chars that is at least numCharsPerPartition. (with the exception of last partition)
std::vector<DBdataView> partitionDBdata_by_numberOfChars(const DBdataView& parent, size_t numCharsPerPartition);

void assertValidPartitioning(const std::vector<DBdataView>& views, const DBdataView& parent);

} //namespace cudasw4

#endif