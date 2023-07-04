#ifndef PSEUDO_DB_DATA_HPP
#define PSEUDO_DB_DATA_HPP

#include <random>
#include <algorithm>
#include <numeric>
#include <vector>
#include <iostream>

struct PseudoDBdata{


    PseudoDBdata(size_t num, size_t length, int randomseed = 42)
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
        for(size_t i = 0; i < length; i++){
            dummyseq[i] = letters[dist(gen)];
        }
        std::cout << "PseudoDBdata: num " << num << ", length " << length << ", sequence " << dummyseq << "\n";

        for(size_t i = 0; i < num; i++){
            offsetvec[i] = i * lengthRounded;
            std::copy(dummyseq.begin(), dummyseq.end(), charvec.begin() + i * lengthRounded);
        }
        offsetvec[num] = num * lengthRounded;

        std::fill(lengthvec.begin(), lengthvec.end(), length);

        std::fill(headervec.begin(), headervec.end(), 'H');
        std::iota(headeroffsetvec.begin(), headeroffsetvec.end(), size_t(0));
        
        
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

    const size_t* lengths() const noexcept{
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
    
private:
    PseudoDBdata() = default;

    size_t lengthRounded;
    std::vector<char> charvec;
    std::vector<size_t> lengthvec;
    std::vector<size_t> offsetvec;
    std::vector<char> headervec;
    std::vector<size_t> headeroffsetvec;
};

// int main(){
//     PseudoDBdata db(5, 128);

//     for(int i = 0; i < db.numSequences(); i++){
//         auto offset = db.offsets()[i];
//         auto length = db.lengths()[i];
//         auto begin = db.chars() + offset;
//         for(int k = 0; k < length; k++){
//             std::cout << begin[k];
//         }
//         std::cout << "\n";
//     }
// }

#endif