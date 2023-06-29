

#include <algorithm>
#include <iterator>
#include <iostream>
#include <chrono>

#include "sequence_io.h"
#include "dbdata.hpp"

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

    constexpr int ALIGN = 4;

    if(argc < 2) {
        cout << "Usage:\n  " << argv[0] << " <FASTA/FASTQ filename> [outputprefix]\n";
        return 0;
    }

    const std::string fastafilename = argv[1];
    std::string outputPrefix = "./mydb_";
    if(argc > 2){
        outputPrefix = argv[2];
    }


    TIMERSTART(READ_FASTA_DB)
    sequence_batch batch = read_all_sequences_and_headers_from_file(argv[1], ALIGN);
    TIMERSTOP(READ_FASTA_DB)

    cout << "-------------------------------------------------\n";
    cout << "Number of input sequences:  " << batch.offsets.size() - 1 << '\n';
    cout << "Number of input characters: " << batch.chars.size() << '\n';

    TIMERSTART(CONVERT_TO_DB_FORMAT)
    createDBfilesFromSequenceBatch(outputPrefix, batch);
    TIMERSTOP(CONVERT_TO_DB_FORMAT)

}
