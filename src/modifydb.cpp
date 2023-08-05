

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
    cudasw4::DB fullDB = cudasw4::loadDB(dbprefix, writeAccess, prefetchSeq);

    if(operation == "convertcharstonumber1"){
        TIMERSTART(convertcharstonumber1);

        std::transform(
            fullDB.getData().chars(),
            fullDB.getData().chars() + fullDB.getData().numChars(),
            fullDB.getModyfiableData().chars(),
            cudasw4::ConvertAA_20{}
        );

        TIMERSTOP(convertcharstonumber1);
    }else if(operation == "lengthsToI32"){
        std::string filebasename = cudasw4::DBdataIoConfig::sequencelengthsfilename();
        std::string outputfilename = dbprefix + "0" + filebasename + "_i32";
        std::ofstream outputfile(outputfilename);
        assert(bool(outputfile));
        size_t numSequences = fullDB.getData().numSequences();
        for(size_t i = 0; i < numSequences; i++){
            size_t length = fullDB.getData().lengths()[i];
            assert(length < size_t(std::numeric_limits<int>::max() - 1));

            int lengthAsInt = length;
            outputfile.write((const char*)&lengthAsInt, sizeof(int));
        }
    }else if(operation == "lengthsToI64"){
        std::string filebasename = cudasw4::DBdataIoConfig::sequencelengthsfilename();
        std::string outputfilename = dbprefix + "0" + filebasename + "_i64";
        std::ofstream outputfile(outputfilename);
        assert(bool(outputfile));
        size_t numSequences = fullDB.getData().numSequences();
        for(size_t i = 0; i < numSequences; i++){
            size_t length = fullDB.getData().lengths()[i];
            outputfile.write((const char*)&length, sizeof(size_t));
        }
    }



}
