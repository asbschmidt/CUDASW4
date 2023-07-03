

#include <algorithm>
#include <iterator>
#include <iostream>
#include <chrono>

#include "sequence_io.h"
#include "dbdata.hpp"
#include "convert.cuh"
#include "mapped_file.hpp"
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


int main(int argc, char* argv[])
{
    using std::cout;

    if(argc < 3) {
        cout << "Usage:\n  " << argv[0] << " operation dbprefix\n";
        return 0;
    }

    const std::string operation = argv[1];
    const std::string dbprefix = argv[2];

    constexpr bool writeAccess = true;
    constexpr bool prefetchSeq = false;
    DB fullDB = loadDB(dbprefix, writeAccess, prefetchSeq);

    if(operation == "convertcharstonumber1"){
        TIMERSTART(convertcharstonumber1);

        for(auto& chunk : fullDB.chunks){
            std::transform(
                chunk.chars(),
                chunk.chars() + chunk.numChars(),
                chunk.chars(),
                &convert_AA
            );
        }

        TIMERSTOP(convertcharstonumber1);
    }



}
