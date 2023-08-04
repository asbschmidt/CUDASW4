#ifndef CONFIG_HPP
#define CONFIG_HPP

//data type to enumerate all sequences in the database
using ReferenceIdT = size_t;

//data type for length of of both query sequences and databases sequences
using SequenceLengthT = size_t;


struct MaxSequencesInDB{
    static constexpr size_t value(){
        return std::numeric_limits<ReferenceIdT>::max() - 1;
    }
};

struct MaxSequenceLength{
    static constexpr size_t value(){
        return std::numeric_limits<SequenceLengthT>::max() - 128 - 4;
    }
};






#endif