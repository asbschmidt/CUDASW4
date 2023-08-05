#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <cstdint>
#include <type_traits>

namespace cudasw4{

//MODIFY AT OWN RISK

//data type to enumerate all sequences in the database
using ReferenceIdT = std::int32_t;

//data type for length of of both query sequences and databases sequences
using SequenceLengthT = std::int32_t;

static_assert(std::is_same_v<ReferenceIdT, std::int32_t>, "unexpected reference type");
static_assert(std::is_same_v<SequenceLengthT, std::int32_t>, "unexpected sequence length type");

struct MaxSequencesInDB{
    static constexpr ReferenceIdT value(){
        return std::numeric_limits<ReferenceIdT>::max() - 1;
    }
};

struct MaxSequenceLength{
    static constexpr SequenceLengthT value(){
        return std::numeric_limits<SequenceLengthT>::max() - 128 - 4;
    }
};



} //namespace cudasw4


#endif