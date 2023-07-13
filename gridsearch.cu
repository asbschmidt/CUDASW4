#include "hpc_helpers/cuda_raiiwrappers.cuh"
#include "hpc_helpers/all_helpers.cuh"
#include "hpc_helpers/nvtx_markers.cuh"
#include "hpc_helpers/simple_allocation.cuh"

#include "dbdata.hpp"
#include "length_partitions.hpp"
#include "convert.cuh"
#include "kernels.cuh"
#include "manypass_half2_kernel.cuh"
#include "singlepass_half2_kernel.cuh"
#include "manypass_float_kernel.cuh"
//#include "singlepass_dpx_kernel.cuh"

#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scatter.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/count.h>
#include <thrust/equal.h>


#include <random>
#include <iostream>
#include <string>

template<class T>
using MyPinnedBuffer = helpers::SimpleAllocationPinnedHost<T, 0>;
template<class T>
using MyDeviceBuffer = helpers::SimpleAllocationDevice<T, 0>;

const char low = -4;
const char BLOSUM62_1D[21*21] = {
    // A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V
       4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, low,
      -1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, low,
      -2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3, low,
      -2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3, low,
       0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, low,
      -1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2, low,
      -1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2, low,
       0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, low,
      -2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3, low,
      -1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, low,
      -1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, low,
      -1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2, low,
      -1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, low,
      -2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, low,
      -1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, low,
       1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2, low,
       0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, low,
      -3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, low,
      -2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, low,
       0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, low,
       low, low, low, low, low, low, low, low, low, low, low, low, low, low, low, low, low, low, low, low, low
     };
    
    // A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V
    const int BLOSUM62[21][21] = {
    {  4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, low },
    { -1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, low },
    { -2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3, low },
    { -2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3, low },
    {  0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, low },
    { -1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2, low },
    { -1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2, low },
    {  0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, low },
    { -2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3, low },
    { -1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, low },
    { -1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, low },
    { -1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2, low },
    { -1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, low },
    { -2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, low },
    { -1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, low },
    {  1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2, low },
    {  0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, low },
    { -3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, low },
    { -2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, low },
    {  0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, low },
    { low, low, low, low, low, low, low, low, low, low, low, low, low, low, low, low, low, low, low, low, low}
    };
    

int main(){
    const int deviceId = 0;
    cudaStream_t stream = 0;
    const int queryLength = 256;
    const int numSubjects = 64*1024;

    const int timingLoopIters = 5;

    const int gop = -11;
    const int gex = -1;


    cudaSetDevice(deviceId);
    cudaMemcpyToSymbol(cBLOSUM62_dev, &(BLOSUM62_1D[0]), 21*21*sizeof(char));


    const char* letters = "ARNDCQEGHILKMFPSTWYV";

    std::mt19937 gen(424242);
    std::uniform_int_distribution<> dist(0,19);
    std::string querySeq(queryLength, ' ');
    for(size_t i = 0; i < queryLength; i++){
        querySeq[i] = letters[dist(gen)];
    }

    MyDeviceBuffer<char> d_query(queryLength);
    MyDeviceBuffer<size_t> d_offsets(2);
    MyDeviceBuffer<size_t> d_lengths(1);

    std::vector<size_t> offsets(2);
    offsets[0] = 0;
    offsets[1] = queryLength;
    std::vector<size_t> lengths(1);
    lengths[0] = queryLength;


    cudaMemcpyAsync(d_query.data(), querySeq.data(), queryLength, cudaMemcpyHostToDevice, stream); CUERR
    cudaMemcpyAsync(d_offsets.data(), offsets.data(), sizeof(size_t) * (2), cudaMemcpyHostToDevice, stream); CUERR
    cudaMemcpyAsync(d_lengths.data(), lengths.data(), sizeof(size_t) * (1), cudaMemcpyHostToDevice, stream); CUERR
    NW_convert_protein<<<1, 128, 0, stream>>>(d_query.data(), d_offsets.data()); CUERR

    std::vector<char> FillChar(512*16, 20);

    cudaMemcpyToSymbolAsync(constantQuery4, FillChar.data(), 512*16, 0, cudaMemcpyHostToDevice, stream); CUERR
    cudaMemcpyToSymbolAsync(constantQuery4, d_query.data(), queryLength, 0, cudaMemcpyDeviceToDevice, stream); CUERR

    // SINGLE PASS BENCHMARKS

    #if 1
    std::cout << "NW_local_affine_Protein_single_pass_half2\n";

    for(int pseudodbSeqLength : {64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024}){
    //for(int pseudodbSeqLength : {64}){
    //for(int pseudodbSeqLength = 11; pseudodbSeqLength <= 64; pseudodbSeqLength++){
        std::cout << "pseudodbSeqLength: " << pseudodbSeqLength << "\n";
 
        PseudoDB fullDB = loadPseudoDB(numSubjects, pseudodbSeqLength);
        const auto& dbData = fullDB.chunks[0];

        std::vector<MyDeviceBuffer<float>> d_scores_vec(std::max(2, timingLoopIters));
        std::vector<MyDeviceBuffer<size_t>> d_overflow_positions_vec(std::max(2, timingLoopIters));
        std::vector<MyDeviceBuffer<int>> d_overflow_number_vec(std::max(2, timingLoopIters));
        for(int i = 0; i < std::max(2, timingLoopIters); i++){
            d_scores_vec[i].resize(numSubjects);
            d_overflow_positions_vec[i].resize(numSubjects);
            d_overflow_number_vec[i].resize(1);
            cudaMemsetAsync(d_overflow_number_vec[i].data(), 0, sizeof(int), stream);
        }
        MyDeviceBuffer<size_t> d_selectedPositions(numSubjects);
        thrust::sequence(thrust::cuda::par_nosync.on(stream), d_selectedPositions.begin(), d_selectedPositions.end(), size_t(0));

        MyDeviceBuffer<char> d_subjects(dbData.numChars());
        MyDeviceBuffer<size_t> d_subjectOffsets(numSubjects+1);
        MyDeviceBuffer<size_t> d_subjectLengths(numSubjects);

        cudaMemcpyAsync(d_subjects.data(), dbData.chars(), dbData.numChars(), H2D, stream); CUERR;
        cudaMemcpyAsync(d_subjectOffsets.data(), dbData.offsets(), sizeof(size_t) * (numSubjects+1), H2D, stream); CUERR;
        cudaMemcpyAsync(d_subjectLengths.data(), dbData.lengths(), sizeof(size_t) * numSubjects, H2D, stream); CUERR;

        auto checkIfEqualResultsNew = [&](){
            const float overflowscore = 123456;
            auto overflowiter = thrust::make_constant_iterator(overflowscore);
            for(int i = 0; i < 2; i++){
                int numOverflow = 0;
                cudaMemcpyAsync(&numOverflow, d_overflow_number_vec[i].data(), sizeof(int), D2H, stream); CUERR;
                cudaStreamSynchronize(stream); CUERR;
                // if(i == 0){
                //     std::cout << "Num overflows: " << numOverflow << "\n";
                // }
                thrust::scatter(
                    thrust::cuda::par_nosync.on(stream),
                    overflowiter,
                    overflowiter + numOverflow,
                    d_overflow_positions_vec[i].data(),
                    d_scores_vec[i].data()
                );
            }
            for(int i = 1; i < 2; i++){
                bool equal = thrust::equal(
                    thrust::cuda::par_nosync.on(stream),
                    d_scores_vec[i].data(),
                    d_scores_vec[i].data() + numSubjects,
                    d_scores_vec[0].data()
                );
                if(!equal){
                    std::cout << "i = " << i << ", scores not equal\n";
                }else{
                    std::cout << "ok\n";
                }
            }
        };

        const double timingCups = ((double(queryLength) * pseudodbSeqLength * numSubjects)) * timingLoopIters;

        using GCUPSstats = std::tuple<double, int, int, int>;

        std::vector<GCUPSstats> gcupsVec;

        #define runSinglePassHalf2(blocksize, groupsize, numRegs){ \
            assert(blocksize % groupsize == 0); \
            if(pseudodbSeqLength <= groupsize * numRegs){ \
                constexpr int alignmentsPerBlock = (blocksize / groupsize) * 2; \
                helpers::GpuTimer timer1(stream, "Timer_" + std::to_string(blocksize) + "_" + std::to_string(groupsize) + "_" + std::to_string(numRegs)); \
                for(int i = 0; i < timingLoopIters; i++){ \
                    NW_local_affine_Protein_single_pass_half2<groupsize, numRegs><<<SDIV(numSubjects, alignmentsPerBlock), blocksize, 0, stream>>>( \
                        d_subjects.data(),  \
                        d_scores_vec[i].data(),  \
                        d_subjectOffsets.data(),  \
                        d_subjectLengths.data(),  \
                        d_selectedPositions.data(),  \
                        numSubjects,  \
                        d_overflow_positions_vec[i].data(),  \
                        d_overflow_number_vec[i].data(),  \
                        1,  \
                        queryLength,  \
                        gop,  \
                        gex \
                    ); CUERR \
                } \
                double gcups = timingCups / 1000. / 1000. / 1000.; \
                gcups = gcups / (timer1.elapsed() / 1000); \
                gcupsVec.push_back(std::make_tuple(gcups,blocksize,groupsize, numRegs )); \
            } \
        }
        #define runSinglePassHalf2_new(blocksize, groupsize, numRegs){ \
            assert(blocksize % groupsize == 0); \
            if(pseudodbSeqLength <= groupsize * numRegs){ \
                constexpr int alignmentsPerBlock = (blocksize / groupsize) * 2; \
                helpers::GpuTimer timer1(stream, "Timer_" + std::to_string(blocksize) + "_" + std::to_string(groupsize) + "_" + std::to_string(numRegs)); \
                for(int i = 0; i < timingLoopIters; i++){ \
                    NW_local_affine_Protein_single_pass_half2_new<groupsize, numRegs><<<SDIV(numSubjects, alignmentsPerBlock), blocksize, 0, stream>>>( \
                        d_subjects.data(),  \
                        d_scores_vec[i].data(),  \
                        d_subjectOffsets.data(),  \
                        d_subjectLengths.data(),  \
                        d_selectedPositions.data(),  \
                        numSubjects,  \
                        d_overflow_positions_vec[i].data(),  \
                        d_overflow_number_vec[i].data(),  \
                        0,  \
                        queryLength,  \
                        gop,  \
                        gex \
                    ); CUERR \
                } \
                double gcups = timingCups / 1000. / 1000. / 1000.; \
                gcups = gcups / (timer1.elapsed() / 1000); \
                gcupsVec.push_back(std::make_tuple(gcups,blocksize,groupsize, numRegs )); \
            } \
        }

        #define compareSinglePassHalf2New(blocksize, groupsize, numRegs){ \
            assert(blocksize % groupsize == 0); \
            constexpr int alignmentsPerBlock = (blocksize / groupsize) * 2; \
                helpers::GpuTimer timer1(stream, "old " + std::to_string(blocksize) + "_" + std::to_string(groupsize) + "_" + std::to_string(numRegs)); \
                NW_local_affine_Protein_single_pass_half2<groupsize, numRegs><<<SDIV(numSubjects, alignmentsPerBlock), blocksize, 0, stream>>>( \
                    d_subjects.data(),  \
                    d_scores_vec[0].data(),  \
                    d_subjectOffsets.data(),  \
                    d_subjectLengths.data(),  \
                    d_selectedPositions.data(),  \
                    numSubjects,  \
                    d_overflow_positions_vec[0].data(),  \
                    d_overflow_number_vec[0].data(),  \
                    0,  \
                    queryLength,  \
                    gop,  \
                    gex \
                ); CUERR \
                timer1.printGCUPS(((double(queryLength) * pseudodbSeqLength * numSubjects)));\
                helpers::GpuTimer timer2(stream, "new " + std::to_string(blocksize) + "_" + std::to_string(groupsize) + "_" + std::to_string(numRegs)); \
                NW_local_affine_Protein_single_pass_half2_new<groupsize, numRegs><<<SDIV(numSubjects, alignmentsPerBlock), blocksize, 0, stream>>>( \
                    d_subjects.data(),  \
                    d_scores_vec[1].data(),  \
                    d_subjectOffsets.data(),  \
                    d_subjectLengths.data(),  \
                    d_selectedPositions.data(),  \
                    numSubjects,  \
                    d_overflow_positions_vec[1].data(),  \
                    d_overflow_number_vec[1].data(),  \
                    0,  \
                    queryLength,  \
                    gop,  \
                    gex \
                ); CUERR \
                timer2.printGCUPS(((double(queryLength) * pseudodbSeqLength * numSubjects)));\
            checkIfEqualResultsNew(); \
        }

        #define runSinglePassHalf2_numregs(blocksize, numRegs){ \
            runSinglePassHalf2(blocksize, 1, numRegs); \
            runSinglePassHalf2(blocksize, 2, numRegs); \
            runSinglePassHalf2(blocksize, 4, numRegs); \
            runSinglePassHalf2(blocksize, 8, numRegs); \
            runSinglePassHalf2(blocksize, 16, numRegs); \
            runSinglePassHalf2(blocksize, 32, numRegs); \
        }
        #define runSinglePassHalf2_numregs_new(blocksize, numRegs){ \
            runSinglePassHalf2_new(blocksize, 1, numRegs); \
            runSinglePassHalf2_new(blocksize, 2, numRegs); \
            runSinglePassHalf2_new(blocksize, 4, numRegs); \
            runSinglePassHalf2_new(blocksize, 8, numRegs); \
            runSinglePassHalf2_new(blocksize, 16, numRegs); \
            runSinglePassHalf2_new(blocksize, 32, numRegs); \
        }

       // compareSinglePassHalf2New(256, 32, 32);

        runSinglePassHalf2_numregs(256, 32);
        runSinglePassHalf2_numregs(256, 30);
        runSinglePassHalf2_numregs(256, 28);
        runSinglePassHalf2_numregs(256, 26);
        runSinglePassHalf2_numregs(256, 24);
        runSinglePassHalf2_numregs(256, 22);
        runSinglePassHalf2_numregs(256, 20);
        runSinglePassHalf2_numregs(256, 18);
        runSinglePassHalf2_numregs(256, 16);
        runSinglePassHalf2_numregs(256, 14);
        runSinglePassHalf2_numregs(256, 12);
        runSinglePassHalf2_numregs(256, 10);
        runSinglePassHalf2_numregs(256, 8);
        runSinglePassHalf2_numregs(256, 6);
        runSinglePassHalf2_numregs(256, 4);
        runSinglePassHalf2_numregs(256, 2);

        std::sort(gcupsVec.begin(), gcupsVec.end(), [](const auto& l, const auto& r){ return std::get<0>(l) > std::get<0>(r);});

        std::cout << "old\n";
        for(int i = 0; i < std::min(3, int(gcupsVec.size())); i++){
            GCUPSstats data = gcupsVec[i];
            std::cout << std::get<0>(data) << " GCUPS, " << std::get<1>(data) << " " << std::get<2>(data) << " " << std::get<3>(data) << "\n";
        }
        gcupsVec.clear();

        runSinglePassHalf2_numregs_new(256, 32);
        runSinglePassHalf2_numregs_new(256, 30);
        runSinglePassHalf2_numregs_new(256, 28);
        runSinglePassHalf2_numregs_new(256, 26);
        runSinglePassHalf2_numregs_new(256, 24);
        runSinglePassHalf2_numregs_new(256, 22);
        runSinglePassHalf2_numregs_new(256, 20);
        runSinglePassHalf2_numregs_new(256, 18);
        runSinglePassHalf2_numregs_new(256, 16);
        runSinglePassHalf2_numregs_new(256, 14);
        runSinglePassHalf2_numregs_new(256, 12);
        runSinglePassHalf2_numregs_new(256, 10);
        runSinglePassHalf2_numregs_new(256, 8);
        runSinglePassHalf2_numregs_new(256, 6);
        runSinglePassHalf2_numregs_new(256, 4);
        runSinglePassHalf2_numregs_new(256, 2);

        std::sort(gcupsVec.begin(), gcupsVec.end(), [](const auto& l, const auto& r){ return std::get<0>(l) > std::get<0>(r);});

        std::cout << "new\n";
        for(int i = 0; i < std::min(3, int(gcupsVec.size())); i++){
            GCUPSstats data = gcupsVec[i];
            std::cout << std::get<0>(data) << " GCUPS, " << std::get<1>(data) << " " << std::get<2>(data) << " " << std::get<3>(data) << "\n";
        }
        gcupsVec.clear();
    }

    #endif




    // MANY PASS HALF2 BENCHMARKS

    #if 0

    std::cout << "NW_local_affine_Protein_many_pass_half2\n";

    //for(int pseudodbSeqLength : {1500, 2000, 2048, 3333, 4096, 6666, 7000}){
    for(int pseudodbSeqLength : {4096}){
    //for(int pseudodbSeqLength = 1024+256; pseudodbSeqLength <= 8192; pseudodbSeqLength += 256){
        std::cout << "pseudodbSeqLength: " << pseudodbSeqLength << "\n";
 
        PseudoDB fullDB = loadPseudoDB(numSubjects, pseudodbSeqLength);
        const auto& dbData = fullDB.chunks[0];

        std::vector<MyDeviceBuffer<float>> d_scores_vec(std::max(2, timingLoopIters));
        std::vector<MyDeviceBuffer<size_t>> d_overflow_positions_vec(std::max(2, timingLoopIters));
        std::vector<MyDeviceBuffer<int>> d_overflow_number_vec(std::max(2, timingLoopIters));
        for(int i = 0; i < std::max(2, timingLoopIters); i++){
            d_scores_vec[i].resize(numSubjects);
            d_overflow_positions_vec[i].resize(numSubjects);
            d_overflow_number_vec[i].resize(1);
            cudaMemsetAsync(d_overflow_number_vec[i].data(), 0, sizeof(int), stream);
        }
        MyDeviceBuffer<size_t> d_selectedPositions(numSubjects);
        thrust::sequence(thrust::cuda::par_nosync.on(stream), d_selectedPositions.begin(), d_selectedPositions.end(), size_t(0));

        MyDeviceBuffer<char> d_subjects(dbData.numChars());
        MyDeviceBuffer<size_t> d_subjectOffsets(numSubjects+1);
        MyDeviceBuffer<size_t> d_subjectLengths(numSubjects);

        cudaMemcpyAsync(d_subjects.data(), dbData.chars(), dbData.numChars(), H2D, stream); CUERR;
        cudaMemcpyAsync(d_subjectOffsets.data(), dbData.offsets(), sizeof(size_t) * (numSubjects+1), H2D, stream); CUERR;
        cudaMemcpyAsync(d_subjectLengths.data(), dbData.lengths(), sizeof(size_t) * numSubjects, H2D, stream); CUERR;

        MyDeviceBuffer<__half2> d_tempH(size_t(queryLength) * SDIV(numSubjects, 64) * 64);
        MyDeviceBuffer<__half2> d_tempE(size_t(queryLength) * SDIV(numSubjects, 64) * 64);

        const double timingCups = ((double(queryLength) * pseudodbSeqLength * numSubjects)) * timingLoopIters;

        using GCUPSstats = std::tuple<double, int, int, int>;

        std::vector<GCUPSstats> gcupsVec;

        auto checkIfEqualResults = [&](){
            const float overflowscore = 123456;
            auto overflowiter = thrust::make_constant_iterator(overflowscore);
            for(int i = 0; i < timingLoopIters; i++){
                int numOverflow = 0;
                cudaMemcpyAsync(&numOverflow, d_overflow_number_vec[i].data(), sizeof(int), D2H, stream); CUERR;
                cudaStreamSynchronize(stream); CUERR;
                // if(i == 0){
                //     std::cout << "Num overflows: " << numOverflow << "\n";
                // }
                thrust::scatter(
                    thrust::cuda::par_nosync.on(stream),
                    overflowiter,
                    overflowiter + numOverflow,
                    d_overflow_positions_vec[i].data(),
                    d_scores_vec[i].data()
                );
            }
            for(int i = 1; i < timingLoopIters; i++){
                bool equal = thrust::equal(
                    thrust::cuda::par_nosync.on(stream),
                    d_scores_vec[i].data(),
                    d_scores_vec[i].data() + numSubjects,
                    d_scores_vec[0].data()
                );
                if(!equal){
                    std::cout << "i = " << i << ", scores not equal\n";
                }
            }
        };

        auto checkIfEqualResultsNew = [&](){
            const float overflowscore = 123456;
            auto overflowiter = thrust::make_constant_iterator(overflowscore);
            for(int i = 0; i < 2; i++){
                int numOverflow = 0;
                cudaMemcpyAsync(&numOverflow, d_overflow_number_vec[i].data(), sizeof(int), D2H, stream); CUERR;
                cudaStreamSynchronize(stream); CUERR;
                // if(i == 0){
                //     std::cout << "Num overflows: " << numOverflow << "\n";
                // }
                thrust::scatter(
                    thrust::cuda::par_nosync.on(stream),
                    overflowiter,
                    overflowiter + numOverflow,
                    d_overflow_positions_vec[i].data(),
                    d_scores_vec[i].data()
                );
            }
            for(int i = 1; i < 2; i++){
                bool equal = thrust::equal(
                    thrust::cuda::par_nosync.on(stream),
                    d_scores_vec[i].data(),
                    d_scores_vec[i].data() + numSubjects,
                    d_scores_vec[0].data()
                );
                if(!equal){
                    std::cout << "i = " << i << ", scores not equal\n";
                }else{
                    std::cout << "ok\n";
                }
            }
        };

        #define runManyPassHalf2(blocksize, groupsize, numRegs){ \
            assert(blocksize % groupsize == 0); \
            constexpr int alignmentsPerBlock = (blocksize / groupsize) * 2; \
            helpers::GpuTimer timer1(stream, "Timer_" + std::to_string(blocksize) + "_" + std::to_string(groupsize) + "_" + std::to_string(numRegs)); \
            for(int i = 0; i < timingLoopIters; i++){ \
                cudaMemsetAsync(d_tempH.data(), 0, d_tempH.size() * sizeof(__half2), stream); CUERR; \
                cudaMemsetAsync(d_tempE.data(), 0, d_tempE.size() * sizeof(__half2), stream); CUERR; \
                NW_local_affine_Protein_many_pass_half2<groupsize, numRegs><<<SDIV(numSubjects, alignmentsPerBlock), blocksize, 0, stream>>>( \
                    d_subjects.data(),  \
                    d_scores_vec[i].data(),  \
                    d_tempH.data(), \
                    d_tempE.data(), \
                    d_subjectOffsets.data(),  \
                    d_subjectLengths.data(),  \
                    d_selectedPositions.data(),  \
                    numSubjects,  \
                    d_overflow_positions_vec[i].data(),  \
                    d_overflow_number_vec[i].data(),  \
                    0,  \
                    queryLength,  \
                    gop,  \
                    gex \
                ); CUERR \
            } \
            timer1.stop(); \
            double gcups = timingCups / 1000. / 1000. / 1000.; \
            gcups = gcups / (timer1.elapsed() / 1000); \
            gcupsVec.push_back(std::make_tuple(gcups,blocksize,groupsize, numRegs )); \
        }

                //cudaMemsetAsync(d_tempH.data(), 0, d_tempH.size() * sizeof(__half2), stream); CUERR; 
                //cudaMemsetAsync(d_tempE.data(), 0, d_tempE.size() * sizeof(__half2), stream); CUERR; 
        #define runManyPassHalf2_new(blocksize, groupsize, numRegs){ \
            assert(blocksize % groupsize == 0); \
            constexpr int alignmentsPerBlock = (blocksize / groupsize) * 2; \
            helpers::GpuTimer timer1(stream, "Timer_" + std::to_string(blocksize) + "_" + std::to_string(groupsize) + "_" + std::to_string(numRegs)); \
            for(int i = 0; i < timingLoopIters; i++){ \
                    \
                NW_local_affine_Protein_many_pass_half2_new<groupsize, numRegs><<<SDIV(numSubjects, alignmentsPerBlock), blocksize, 0, stream>>>( \
                    d_subjects.data(),  \
                    d_scores_vec[i].data(),  \
                    d_tempH.data(), \
                    d_tempE.data(), \
                    d_subjectOffsets.data(),  \
                    d_subjectLengths.data(),  \
                    d_selectedPositions.data(),  \
                    numSubjects,  \
                    d_overflow_positions_vec[i].data(),  \
                    d_overflow_number_vec[i].data(),  \
                    0,  \
                    queryLength,  \
                    gop,  \
                    gex \
                ); CUERR \
            } \
            timer1.stop(); \
            double gcups = timingCups / 1000. / 1000. / 1000.; \
            gcups = gcups / (timer1.elapsed() / 1000); \
            gcupsVec.push_back(std::make_tuple(gcups,blocksize,groupsize, numRegs )); \
        }

        #define compareManyPassHalf2New(blocksize, groupsize, numRegs){ \
            assert(blocksize % groupsize == 0); \
            constexpr int alignmentsPerBlock = (blocksize / groupsize) * 2; \
                cudaMemsetAsync(d_tempH.data(), 0, d_tempH.size() * sizeof(__half2), stream); CUERR; \
                cudaMemsetAsync(d_tempE.data(), 0, d_tempE.size() * sizeof(__half2), stream); CUERR; \
                helpers::GpuTimer timer1(stream, "old " + std::to_string(blocksize) + "_" + std::to_string(groupsize) + "_" + std::to_string(numRegs)); \
                NW_local_affine_Protein_many_pass_half2<groupsize, numRegs><<<SDIV(numSubjects, alignmentsPerBlock), blocksize, 0, stream>>>( \
                    d_subjects.data(),  \
                    d_scores_vec[0].data(),  \
                    d_tempH.data(), \
                    d_tempE.data(), \
                    d_subjectOffsets.data(),  \
                    d_subjectLengths.data(),  \
                    d_selectedPositions.data(),  \
                    numSubjects,  \
                    d_overflow_positions_vec[0].data(),  \
                    d_overflow_number_vec[0].data(),  \
                    0,  \
                    queryLength,  \
                    gop,  \
                    gex \
                ); CUERR \
                timer1.printGCUPS(((double(queryLength) * pseudodbSeqLength * numSubjects)));\
                cudaMemsetAsync(d_tempH.data(), 0, d_tempH.size() * sizeof(__half2), stream); CUERR; \
                cudaMemsetAsync(d_tempE.data(), 0, d_tempE.size() * sizeof(__half2), stream); CUERR; \
                helpers::GpuTimer timer2(stream, "new " + std::to_string(blocksize) + "_" + std::to_string(groupsize) + "_" + std::to_string(numRegs)); \
                NW_local_affine_Protein_many_pass_half2_new<groupsize, numRegs><<<SDIV(numSubjects, alignmentsPerBlock), blocksize, 0, stream>>>( \
                    d_subjects.data(),  \
                    d_scores_vec[1].data(),  \
                    d_tempH.data(), \
                    d_tempE.data(), \
                    d_subjectOffsets.data(),  \
                    d_subjectLengths.data(),  \
                    d_selectedPositions.data(),  \
                    numSubjects,  \
                    d_overflow_positions_vec[1].data(),  \
                    d_overflow_number_vec[1].data(),  \
                    0,  \
                    queryLength,  \
                    gop,  \
                    gex \
                ); CUERR \
                timer2.printGCUPS(((double(queryLength) * pseudodbSeqLength * numSubjects)));\
            checkIfEqualResultsNew(); \
        }


        #define runManyPassHalf2_numregs(blocksize, numRegs){ \
            runManyPassHalf2(blocksize, 1, numRegs); \
            runManyPassHalf2(blocksize, 2, numRegs); \
            runManyPassHalf2(blocksize, 4, numRegs); \
            runManyPassHalf2(blocksize, 8, numRegs); \
            runManyPassHalf2(blocksize, 16, numRegs); \
            runManyPassHalf2(blocksize, 32, numRegs); \
        }

        //runManyPassHalf2(256, 32, 32);

        //compareManyPassHalf2New(256, 32, 32);

        // std::cout << "start 4\n"; runManyPassHalf2(256, 32, 4);
        // std::cout << "start 6\n"; runManyPassHalf2(256, 32, 6);
        // std::cout << "start 8\n"; runManyPassHalf2(256, 32, 8);
        // std::cout << "start 10\n"; runManyPassHalf2(256, 32, 10);
        // std::cout << "start 12\n"; runManyPassHalf2(256, 32, 12);
        // std::cout << "start 14\n"; runManyPassHalf2(256, 32, 14);
        // std::cout << "start 16\n"; runManyPassHalf2(256, 32, 16);
        // runManyPassHalf2(256, 32, 2);
        // runManyPassHalf2(256, 32, 4);
        // runManyPassHalf2(256, 32, 6);
        // runManyPassHalf2(256, 32, 8);
        // runManyPassHalf2(256, 32, 10);
        // runManyPassHalf2(256, 32, 12);
        runManyPassHalf2(256, 32, 14);
        runManyPassHalf2(256, 32, 16);
        runManyPassHalf2(256, 32, 18);
        // runManyPassHalf2(256, 32, 20);
        // runManyPassHalf2(256, 32, 22);
        // runManyPassHalf2(256, 32, 24);
        // runManyPassHalf2(256, 32, 26);
        // runManyPassHalf2(256, 32, 28);
        // runManyPassHalf2(256, 32, 30);
        // runManyPassHalf2(256, 32, 32);

        std::sort(gcupsVec.begin(), gcupsVec.end(), [](const auto& l, const auto& r){ return std::get<0>(l) > std::get<0>(r);});
        //for(int i = 0; i < std::min(3, int(gcupsVec.size())); i++){
        for(int i = 0; i < int(gcupsVec.size()); i++){
            GCUPSstats data = gcupsVec[i];
            std::cout << std::get<0>(data) << " GCUPS, " << std::get<1>(data) << " " << std::get<2>(data) << " " << std::get<3>(data) << "\n";
        }
        std::cout << "\n";
        gcupsVec.clear();


        // runManyPassHalf2_new(256, 32, 6);
        // runManyPassHalf2_new(256, 32, 8);
        // runManyPassHalf2_new(256, 32, 10);
        // runManyPassHalf2_new(256, 32, 12);
        runManyPassHalf2_new(256, 32, 14);
        runManyPassHalf2_new(256, 32, 16);
        runManyPassHalf2_new(256, 32, 18);
        // runManyPassHalf2_new(256, 32, 20);
        // runManyPassHalf2_new(256, 32, 22);
        // runManyPassHalf2_new(256, 32, 24);
        // runManyPassHalf2_new(256, 32, 26);
        // runManyPassHalf2_new(256, 32, 28);
        // runManyPassHalf2_new(256, 32, 30);
        // runManyPassHalf2_new(256, 32, 32);

        std::sort(gcupsVec.begin(), gcupsVec.end(), [](const auto& l, const auto& r){ return std::get<0>(l) > std::get<0>(r);});
        //for(int i = 0; i < std::min(3, int(gcupsVec.size())); i++){
        for(int i = 0; i < int(gcupsVec.size()); i++){
            GCUPSstats data = gcupsVec[i];
            std::cout << std::get<0>(data) << " GCUPS, " << std::get<1>(data) << " " << std::get<2>(data) << " " << std::get<3>(data) << "\n";
        }
        std::cout << "\n";
        gcupsVec.clear();
    }

    #endif






    // MANY PASS FLOAT BENCHMARKS

    #if 0

    std::cout << "NW_local_affine_read4_float_query_Protein\n";

    //for(int pseudodbSeqLength : {1500, 2000, 2048, 3333, 4096, 6666, 7000}){
    for(int pseudodbSeqLength : {4096}){
    //for(int pseudodbSeqLength = 1024+256; pseudodbSeqLength <= 8192; pseudodbSeqLength += 256){
        std::cout << "pseudodbSeqLength: " << pseudodbSeqLength << "\n";
 
        PseudoDB fullDB = loadPseudoDB(numSubjects, pseudodbSeqLength);
        const auto& dbData = fullDB.chunks[0];

        std::vector<MyDeviceBuffer<float>> d_scores_vec(std::max(2, timingLoopIters));
        for(int i = 0; i < std::max(2, timingLoopIters); i++){
            d_scores_vec[i].resize(numSubjects);
        }
        MyDeviceBuffer<size_t> d_selectedPositions(numSubjects);
        thrust::sequence(thrust::cuda::par_nosync.on(stream), d_selectedPositions.begin(), d_selectedPositions.end(), size_t(0));

        MyDeviceBuffer<char> d_subjects(dbData.numChars());
        MyDeviceBuffer<size_t> d_subjectOffsets(numSubjects+1);
        MyDeviceBuffer<size_t> d_subjectLengths(numSubjects);

        cudaMemcpyAsync(d_subjects.data(), dbData.chars(), dbData.numChars(), H2D, stream); CUERR;
        cudaMemcpyAsync(d_subjectOffsets.data(), dbData.offsets(), sizeof(size_t) * (numSubjects+1), H2D, stream); CUERR;
        cudaMemcpyAsync(d_subjectLengths.data(), dbData.lengths(), sizeof(size_t) * numSubjects, H2D, stream); CUERR;

        MyDeviceBuffer<short2> d_tempH(size_t(queryLength) * SDIV(numSubjects, 64) * 64);
        MyDeviceBuffer<short2> d_tempE(size_t(queryLength) * SDIV(numSubjects, 64) * 64);

        const double timingCups = ((double(queryLength) * pseudodbSeqLength * numSubjects)) * timingLoopIters;

        using GCUPSstats = std::tuple<double, int, int, int>;

        std::vector<GCUPSstats> gcupsVec;

        auto checkIfEqualResults = [&](){
            for(int i = 1; i < timingLoopIters; i++){
                bool equal = thrust::equal(
                    thrust::cuda::par_nosync.on(stream),
                    d_scores_vec[i].data(),
                    d_scores_vec[i].data() + numSubjects,
                    d_scores_vec[0].data()
                );
                if(!equal){
                    std::cout << "i = " << i << ", scores not equal\n";
                }
            }
        };

        auto checkIfEqualResultsNew = [&](){
            for(int i = 1; i < 2; i++){
                float s1 = 0;
                cudaMemcpy(&s1, d_scores_vec[i].data(), sizeof(float), D2H);
                float s0 = 0;
                cudaMemcpy(&s0, d_scores_vec[0].data(), sizeof(float), D2H);
                std::cout << "s0 " << s0 << " " << "s1 " << s1 << "\n";
                bool equal = thrust::equal(
                    thrust::cuda::par_nosync.on(stream),
                    d_scores_vec[i].data(),
                    d_scores_vec[i].data() + numSubjects,
                    d_scores_vec[0].data()
                );
                if(!equal){
                    std::cout << "i = " << i << ", scores not equal\n";
                }else{
                    std::cout << "ok\n";
                }
            }
        };

        #define runManyPassFloat(blocksize, groupsize, numRegs){ \
            assert(blocksize % groupsize == 0); \
            constexpr int alignmentsPerBlock = 1; \
            helpers::GpuTimer timer1(stream, "Timer_" + std::to_string(blocksize) + "_" + std::to_string(groupsize) + "_" + std::to_string(numRegs)); \
            for(int i = 0; i < timingLoopIters; i++){ \
                cudaMemsetAsync(d_tempH.data(), 0, d_tempH.size() * sizeof(short2), stream); CUERR; \
                cudaMemsetAsync(d_tempE.data(), 0, d_tempE.size() * sizeof(short2), stream); CUERR; \
                NW_local_affine_read4_float_query_Protein<groupsize, numRegs><<<SDIV(numSubjects, alignmentsPerBlock), blocksize, 0, stream>>>( \
                    d_subjects.data(),  \
                    d_scores_vec[i].data(),  \
                    d_tempH.data(), \
                    d_tempE.data(), \
                    d_subjectOffsets.data(),  \
                    d_subjectLengths.data(),  \
                    d_selectedPositions.data(),  \
                    queryLength,  \
                    gop,  \
                    gex \
                ); CUERR \
            } \
            timer1.stop(); \
            double gcups = timingCups / 1000. / 1000. / 1000.; \
            gcups = gcups / (timer1.elapsed() / 1000); \
            gcupsVec.push_back(std::make_tuple(gcups,blocksize,groupsize, numRegs )); \
        }


        #define runManyPassFloat_new(blocksize, groupsize, numRegs){ \
            assert(groupsize == 32); \
            assert(blocksize == 32); \
            constexpr int alignmentsPerBlock = 1; \
            helpers::GpuTimer timer1(stream, "Timer_" + std::to_string(blocksize) + "_" + std::to_string(groupsize) + "_" + std::to_string(numRegs)); \
            for(int i = 0; i < timingLoopIters; i++){ \
                cudaMemsetAsync(d_tempH.data(), 0, d_tempH.size() * sizeof(short2), stream); CUERR; \
                cudaMemsetAsync(d_tempE.data(), 0, d_tempE.size() * sizeof(short2), stream); CUERR; \
                NW_local_affine_read4_float_query_Protein_new<numRegs><<<SDIV(numSubjects, alignmentsPerBlock), blocksize, 0, stream>>>( \
                    d_subjects.data(),  \
                    d_scores_vec[i].data(),  \
                    d_tempH.data(), \
                    d_tempE.data(), \
                    d_subjectOffsets.data(),  \
                    d_subjectLengths.data(),  \
                    d_selectedPositions.data(),  \
                    queryLength,  \
                    gop,  \
                    gex \
                ); CUERR \
            } \
            timer1.stop(); \
            double gcups = timingCups / 1000. / 1000. / 1000.; \
            gcups = gcups / (timer1.elapsed() / 1000); \
            gcupsVec.push_back(std::make_tuple(gcups,blocksize,groupsize, numRegs )); \
        }

        #define compareManyPassFloatNew(blocksize, groupsize, numRegs){ \
            assert(groupsize == 32); \
            assert(blocksize == 32); \
            constexpr int alignmentsPerBlock = 1; \
                cudaMemsetAsync(d_tempH.data(), 0, d_tempH.size() * sizeof(short2), stream); CUERR; \
                cudaMemsetAsync(d_tempE.data(), 0, d_tempE.size() * sizeof(short2), stream); CUERR; \
                helpers::GpuTimer timer1(stream, "old " + std::to_string(blocksize) + "_" + std::to_string(groupsize) + "_" + std::to_string(numRegs)); \
                NW_local_affine_read4_float_query_Protein<groupsize, numRegs><<<SDIV(numSubjects, alignmentsPerBlock), blocksize, 0, stream>>>( \
                    d_subjects.data(),  \
                    d_scores_vec[0].data(),  \
                    d_tempH.data(), \
                    d_tempE.data(), \
                    d_subjectOffsets.data(),  \
                    d_subjectLengths.data(),  \
                    d_selectedPositions.data(),  \
                    queryLength,  \
                    gop,  \
                    gex \
                ); CUERR \
                timer1.printGCUPS(((double(queryLength) * pseudodbSeqLength * numSubjects)));\
                cudaMemsetAsync(d_tempH.data(), 0, d_tempH.size() * sizeof(short2), stream); CUERR; \
                cudaMemsetAsync(d_tempE.data(), 0, d_tempE.size() * sizeof(short2), stream); CUERR; \
                helpers::GpuTimer timer2(stream, "new " + std::to_string(blocksize) + "_" + std::to_string(groupsize) + "_" + std::to_string(numRegs)); \
                NW_local_affine_read4_float_query_Protein_new<numRegs><<<SDIV(numSubjects, alignmentsPerBlock), blocksize, 0, stream>>>( \
                    d_subjects.data(),  \
                    d_scores_vec[1].data(),  \
                    d_tempH.data(), \
                    d_tempE.data(), \
                    d_subjectOffsets.data(),  \
                    d_subjectLengths.data(),  \
                    d_selectedPositions.data(),  \
                    queryLength,  \
                    gop,  \
                    gex \
                ); CUERR \
                timer2.printGCUPS(((double(queryLength) * pseudodbSeqLength * numSubjects)));\
            checkIfEqualResultsNew(); \
        }


        compareManyPassFloatNew(32, 32, 12);

        // std::cout << "start 4\n"; runManyPassFloat(32, 32, 4);
        // std::cout << "start 6\n"; runManyPassFloat(32, 32, 6);
        // std::cout << "start 8\n"; runManyPassFloat(32, 32, 8);
        // std::cout << "start 10\n"; runManyPassFloat(32, 32, 10);
        // std::cout << "start 12\n"; runManyPassFloat(32, 32, 12);
        // std::cout << "start 14\n"; runManyPassFloat(32, 32, 14);
        // std::cout << "start 16\n"; runManyPassFloat(32, 32, 16);
        // runManyPassFloat(32, 32, 2);
        // runManyPassFloat(32, 32, 4);
        // runManyPassFloat(32, 32, 6);
        // runManyPassFloat(32, 32, 8);
        // runManyPassFloat(32, 32, 10);
        // runManyPassFloat(32, 32, 12);
        // runManyPassFloat(32, 32, 14);
        // runManyPassFloat(32, 32, 16);
        // runManyPassFloat(32, 32, 18);
        // runManyPassFloat(32, 32, 20);
        // runManyPassFloat(32, 32, 22);
        // runManyPassFloat(32, 32, 24);
        // runManyPassFloat(32, 32, 26);
        // runManyPassFloat(32, 32, 28);
        // runManyPassFloat(32, 32, 30);
        // runManyPassFloat(32, 32, 32);

        std::sort(gcupsVec.begin(), gcupsVec.end(), [](const auto& l, const auto& r){ return std::get<0>(l) > std::get<0>(r);});
        //for(int i = 0; i < std::min(3, int(gcupsVec.size())); i++){
        for(int i = 0; i < int(gcupsVec.size()); i++){
            GCUPSstats data = gcupsVec[i];
            std::cout << std::get<0>(data) << " GCUPS, " << std::get<1>(data) << " " << std::get<2>(data) << " " << std::get<3>(data) << "\n";
        }
        std::cout << "\n";
        gcupsVec.clear();


        // runManyPassFloat_new(32, 32, 6);
        // runManyPassFloat_new(32, 32, 8);
        // runManyPassFloat_new(32, 32, 10);
        // runManyPassFloat_new(32, 32, 12);
        // runManyPassFloat_new(32, 32, 14);
        // runManyPassFloat_new(32, 32, 16);
        // runManyPassFloat_new(32, 32, 18);
        // runManyPassFloat_new(32, 32, 20);
        // runManyPassFloat_new(32, 32, 22);
        // runManyPassFloat_new(32, 32, 24);
        // runManyPassFloat_new(32, 32, 26);
        // runManyPassFloat_new(32, 32, 28);
        // runManyPassFloat_new(32, 32, 30);
        // runManyPassFloat_new(32, 32, 32);

        std::sort(gcupsVec.begin(), gcupsVec.end(), [](const auto& l, const auto& r){ return std::get<0>(l) > std::get<0>(r);});
        //for(int i = 0; i < std::min(3, int(gcupsVec.size())); i++){
        for(int i = 0; i < int(gcupsVec.size()); i++){
            GCUPSstats data = gcupsVec[i];
            std::cout << std::get<0>(data) << " GCUPS, " << std::get<1>(data) << " " << std::get<2>(data) << " " << std::get<3>(data) << "\n";
        }
        std::cout << "\n";
        gcupsVec.clear();
    }

    #endif
}