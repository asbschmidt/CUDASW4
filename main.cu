
#include <iostream>
#include <algorithm>
#include <iterator>
#include <string>
#include <cuda_fp16.h>
#include <future>
#include <cstdlib>
#include <numeric>
#include <memory>

//#include <cuda_fp8.h>
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
#include <thrust/logical.h>

#include "hpc_helpers/cuda_raiiwrappers.cuh"
#include "hpc_helpers/all_helpers.cuh"
#include "hpc_helpers/nvtx_markers.cuh"
#include "hpc_helpers/simple_allocation.cuh"

#include <cuda/annotated_ptr>

#include "sequence_io.h"

#include <omp.h>
#include "dbdata.hpp"
#include "length_partitions.hpp"
#include "util.cuh"
#include "convert.cuh"
#include "kernels.cuh"

#include "manypass_half2_kernel.cuh"
#include "singlepass_half2_kernel.cuh"
#include "manypass_float_kernel.cuh"

template<class T>
using MyPinnedBuffer = helpers::SimpleAllocationPinnedHost<T, 0>;
template<class T>
using MyDeviceBuffer = helpers::SimpleAllocationDevice<T, 0>;





#define TIMERSTART_CUDA(label)                                                  \
    cudaEvent_t start##label, stop##label;                                 \
    float time##label;                                                     \
    cudaEventCreate(&start##label);                                        \
    cudaEventCreate(&stop##label);                                         \
    cudaEventRecord(start##label, 0);

#define TIMERSTOP_CUDA(label)                                                   \
            cudaEventRecord(stop##label, 0);                                   \
            cudaEventSynchronize(stop##label);                                 \
            cudaEventElapsedTime(&time##label, start##label, stop##label);     \
            std::cout << "TIMING: " << time##label << " ms " << (dp_cells)/(time##label*1e6) << " GCUPS (" << #label << ")" << std::endl;


#define TIMERSTART_CUDA_STREAM(label, stream)                                                  \
        cudaEvent_t start##label, stop##label;                                 \
        float time##label;                                                     \
        cudaEventCreate(&start##label);                                        \
        cudaEventCreate(&stop##label);                                         \
        cudaEventRecord(start##label, stream);
    
#define TIMERSTOP_CUDA_STREAM(label, stream)                                                   \
            cudaEventRecord(stop##label, stream);                                   \
            cudaEventSynchronize(stop##label);                                 \
            cudaEventElapsedTime(&time##label, start##label, stop##label);     \
            std::cout << "TIMING: " << time##label << " ms " << (dp_cells)/(time##label*1e6) << " GCUPS (" << #label << ")" << std::endl;

                

using std::cout;
using std::copy;




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


template<size_t size>
struct TopNMaximaArray{
    struct Ref{
        size_t index;
        size_t indexOffset;
        int* d_locks;
        volatile float* d_scores;
        size_t* d_indices;

        __device__
        Ref& operator=(float newscore){            

            const size_t slot = (indexOffset + index) % size;

            // if(index + indexOffset == 51766021){
            //     printf("Ref operator=(%f), index %lu indexOffset %lu, slot %lu, griddimx %d, blockdimx %d, blockIdxx %d, threadidxx %d\n", 
            //         newscore, index, indexOffset, slot, gridDim.x, blockDim.x, blockIdx.x, threadIdx.x);
            // }

            int* const lock = &d_locks[slot];

            while (0 != (atomicCAS(lock, 0, 1))) {}

            const float currentScore = d_scores[slot];
            if(currentScore < newscore){
                d_scores[slot] = newscore;
                d_indices[slot] = indexOffset + index;
            }

            atomicExch(lock, 0);
        }
    };

    TopNMaximaArray(float* d_scores_, size_t* d_indices_, int* d_locks_, size_t offset)
        : indexOffset(offset), d_locks(d_locks_), d_scores(d_scores_), d_indices(d_indices_){}

    template<class Index>
    __device__
    Ref operator[](Index index) const{
        Ref r;
        r.index = index;
        r.indexOffset = indexOffset;
        r.d_locks = d_locks;
        r.d_scores = d_scores;
        r.d_indices = d_indices;
        return r;
    }

    size_t indexOffset = 1000;
    int* d_locks;
    volatile float* d_scores;
    size_t* d_indices;
};

struct CudaSW4Options{
    /*
        if loadFullDBToGpu == false, create a double buffer for maxBatchBytes and maxBatchSequences
        in both gpu mem and pinned host mem to stream the DB to the gpu for each query.
        if loadFullDBToGpu == true, attempt to allocate enough gpu memory to store the full db. if
        this fails, fall back to double buffer mode
    */
    bool help = false;
    bool loadFullDBToGpu = false;
    int numTopOutputs = 10;
    int gop = -11;
    int gex = -1;
    size_t maxBatchBytes = 32ull * 1024ull * 1024ull;
    size_t maxBatchSequences = 10'000'000;
    size_t maxTempBytes = 4ull * 1024ull * 1024ull * 1024ull;

    std::string queryFile;
    std::string dbPrefix;
};


struct GpuWorkingSet{

    static constexpr int maxReduceArraySize = 512 * 1024;
    using MaxReduceArray = TopNMaximaArray<maxReduceArraySize>;

    GpuWorkingSet(
        size_t num_queries,
        size_t bytesForQueries,
        const CudaSW4Options& options,
        const std::vector<DBdataView>& dbPartitions
    ){
        cudaGetDevice(&deviceId);

        size_t numSubjects = 0;
        size_t numSubjectBytes = 0;
        for(const auto& p : dbPartitions){
            numSubjects += p.numSequences();
            numSubjectBytes += p.numChars();
        }

        devChars.resize(bytesForQueries); CUERR
        devOffsets.resize(num_queries+1); CUERR
        devLengths.resize(num_queries); CUERR

        d_maxReduceArrayLocks.resize(maxReduceArraySize);
        d_maxReduceArrayScores.resize(maxReduceArraySize);
        d_maxReduceArrayIndices.resize(maxReduceArraySize);

        cudaMemset(d_maxReduceArrayLocks.data(), 0, sizeof(int) * maxReduceArraySize);

        //devAlignmentScoresFloat.resize(numSubjects);
        Fillchar.resize(16*512);
        cudaMemset(Fillchar.data(), 20, 16*512);

        forkStreamEvent = CudaEvent{cudaEventDisableTiming}; CUERR;
        numWorkStreamsWithoutTemp = 1;
        workstreamIndex = 0;
        workStreamsWithoutTemp.resize(numWorkStreamsWithoutTemp);

        if(!options.loadFullDBToGpu){
            singleBatchDBisOnGpu = false;
            MAX_CHARDATA_BYTES = options.maxBatchBytes;
            MAX_SEQ = options.maxBatchSequences;
            numCopyBuffers = 2;
        }else{
            singleBatchDBisOnGpu = true;
            MAX_CHARDATA_BYTES = numSubjectBytes;
            MAX_SEQ = numSubjects;
            numCopyBuffers = 1;
        }

        h_chardata_vec.resize(numCopyBuffers);
        h_lengthdata_vec.resize(numCopyBuffers);
        h_offsetdata_vec.resize(numCopyBuffers);
        d_chardata_vec.resize(numCopyBuffers);
        d_lengthdata_vec.resize(numCopyBuffers);
        d_offsetdata_vec.resize(numCopyBuffers);
        copyStreams.resize(numCopyBuffers);
        pinnedBufferEvents.resize(numCopyBuffers);
        deviceBufferEvents.resize(numCopyBuffers);

        for(int i = 0; i < numCopyBuffers; i++){
            if(!options.loadFullDBToGpu){
                h_chardata_vec[i].resize(MAX_CHARDATA_BYTES);
                h_lengthdata_vec[i].resize(MAX_SEQ);
                h_offsetdata_vec[i].resize(MAX_SEQ+1);
            }
            d_chardata_vec[i].resize(MAX_CHARDATA_BYTES);
            d_lengthdata_vec[i].resize(MAX_SEQ);
            d_offsetdata_vec[i].resize(MAX_SEQ+1);
            pinnedBufferEvents[i] = CudaEvent{cudaEventDisableTiming}; CUERR;
            deviceBufferEvents[i] = CudaEvent{cudaEventDisableTiming}; CUERR;
        }
        d_selectedPositions.resize(MAX_SEQ);
        thrust::sequence(
            thrust::device,
            d_selectedPositions.begin(),
            d_selectedPositions.end(),
            size_t(0)
        );
        d_new_overflow_number.resize(numCopyBuffers);
        h_new_overflow_number.resize(numCopyBuffers);
        d_new_overflow_positions_vec.resize(numCopyBuffers);
        for(int i = 0; i < numCopyBuffers; i++){
            d_new_overflow_positions_vec[i].resize(MAX_SEQ);
        }

        size_t free, total;
        cudaMemGetInfo(&free, &total); CUERR;
        if(free > options.maxTempBytes){
            numTempBytes = options.maxTempBytes;
        }else{
            numTempBytes = free;
        }
        std::cerr << "numTempBytes: " << numTempBytes << "\n";
        d_tempStorageHE.resize(numTempBytes);
    }

    MaxReduceArray getMaxReduceArray(size_t offset){
        return MaxReduceArray(
            d_maxReduceArrayScores.data(), 
            d_maxReduceArrayIndices.data(), 
            d_maxReduceArrayLocks.data(), 
            offset
        );
    }

    void resetMaxReduceArray(cudaStream_t stream){
        thrust::fill(thrust::cuda::par_nosync.on(stream),
            d_maxReduceArrayScores.data(),
            d_maxReduceArrayScores.data() + maxReduceArraySize,
            -1.f
        );
        cudaMemsetAsync(d_maxReduceArrayIndices.data(), 0, sizeof(int) * maxReduceArraySize, stream);
    }

    bool singleBatchDBisOnGpu = false;
    int deviceId;
    int numCopyBuffers;
    int numWorkStreamsWithoutTemp = 1;
    int workstreamIndex;
    int copyBufferIndex = 0;
    size_t numTempBytes;
    size_t MAX_CHARDATA_BYTES;
    size_t MAX_SEQ;



    MyDeviceBuffer<int> d_maxReduceArrayLocks;
    MyDeviceBuffer<float> d_maxReduceArrayScores;
    MyDeviceBuffer<size_t> d_maxReduceArrayIndices;

    MyDeviceBuffer<char> devChars;
    MyDeviceBuffer<size_t> devOffsets;
    MyDeviceBuffer<size_t> devLengths;
    MyDeviceBuffer<char> d_tempStorageHE;
    //MyDeviceBuffer<float> devAlignmentScoresFloat;
    MyDeviceBuffer<char> Fillchar;
    MyDeviceBuffer<size_t> d_selectedPositions;
    MyDeviceBuffer<int> d_new_overflow_number;
    MyPinnedBuffer<int> h_new_overflow_number;
    CudaStream hostFuncStream;
    CudaStream workStreamForTempUsage;
    CudaEvent forkStreamEvent;

    //bool hasFullDb = false;
    // MyDeviceBuffer<char> d_fulldb_chardata;
    // MyDeviceBuffer<size_t> d_fulldb_lengthdata;
    // MyDeviceBuffer<size_t> d_fulldb_offsetdata;
    
    std::vector<MyPinnedBuffer<char>> h_chardata_vec;
    std::vector<MyPinnedBuffer<size_t>> h_lengthdata_vec;
    std::vector<MyPinnedBuffer<size_t>> h_offsetdata_vec;
    std::vector<MyDeviceBuffer<char>> d_chardata_vec;
    std::vector<MyDeviceBuffer<size_t>> d_lengthdata_vec;
    std::vector<MyDeviceBuffer<size_t>> d_offsetdata_vec;
    std::vector<CudaStream> copyStreams;
    std::vector<CudaEvent> pinnedBufferEvents;
    std::vector<CudaEvent> deviceBufferEvents;
    std::vector<CudaStream> workStreamsWithoutTemp;
    std::vector<MyDeviceBuffer<size_t>> d_new_overflow_positions_vec;

};






struct DoMinus_size_t{
    size_t val;
    __host__ __device__
    DoMinus_size_t(size_t v) : val(v){}

    __host__ __device__
    void operator()(size_t& i){
        i -= val;
    }
};

struct DeviceBatchCopyToPinnedPlan{
    struct CopyRange{
        int lengthPartitionId;
        int currentCopyPartition;
        int currentCopySeqInPartition;
        int numToCopy;
    };
    size_t usedBytes = 0;
    size_t usedSeq = 0;
    std::vector<int> h_partitionIds;
    std::vector<int> h_numPerPartition;
    std::vector<CopyRange> copyRanges;
};

std::ostream& operator<<(std::ostream& os, const DeviceBatchCopyToPinnedPlan& plan){
    os << "usedBytes " << plan.usedBytes << ", usedSeq " << plan.usedSeq << " ";
    for(int i = 0; i < int(plan.h_partitionIds.size()); i++){
        os << "(" << plan.h_partitionIds[i] << "," << plan.h_numPerPartition[i] << ") ";
    }
    
    return os;
}

std::vector<DeviceBatchCopyToPinnedPlan> computeDbCopyPlan(
    const std::vector<DBdataView>& dbPartitions,
    const std::vector<int>& lengthPartitionIds,
    size_t MAX_CHARDATA_BYTES,
    size_t MAX_SEQ
){
    std::vector<DeviceBatchCopyToPinnedPlan> result;

    size_t currentCopyPartition = 0;
    size_t currentCopySeqInPartition = 0;

    size_t processedSequences = 0;
    while(currentCopyPartition < dbPartitions.size()){
        
        size_t usedBytes = 0;
        size_t usedSeq = 0;

        DeviceBatchCopyToPinnedPlan plan;

        while(currentCopyPartition < dbPartitions.size()){
            //figure out how many sequences to copy to pinned
            size_t remainingBytes = MAX_CHARDATA_BYTES - usedBytes;
            
            auto dboffsetsBegin = dbPartitions[currentCopyPartition].offsets() + currentCopySeqInPartition;
            auto dboffsetsEnd = dbPartitions[currentCopyPartition].offsets() + dbPartitions[currentCopyPartition].numSequences() + 1;
            
            auto searchFor = dbPartitions[currentCopyPartition].offsets()[currentCopySeqInPartition] + remainingBytes + 1; // +1 because remainingBytes is inclusive
            auto it = std::lower_bound(
                dboffsetsBegin,
                dboffsetsEnd,
                searchFor
            );

            size_t numToCopyByBytes = 0;
            if(it != dboffsetsBegin){
                numToCopyByBytes = std::distance(dboffsetsBegin, it) - 1;
            }
            if(numToCopyByBytes == 0 && currentCopySeqInPartition == 0){
                std::cout << "Warning. copy buffer size too small. skipped a db portion\n";
                break;
            }
            
            size_t remainingSeq = MAX_SEQ - usedSeq;            
            size_t numToCopyBySeq = std::min(dbPartitions[currentCopyPartition].numSequences() - currentCopySeqInPartition, remainingSeq);
            size_t numToCopy = std::min(numToCopyByBytes,numToCopyBySeq);

            if(numToCopy > 0){
                DeviceBatchCopyToPinnedPlan::CopyRange copyRange;
                copyRange.lengthPartitionId = lengthPartitionIds[currentCopyPartition];
                copyRange.currentCopyPartition = currentCopyPartition;
                copyRange.currentCopySeqInPartition = currentCopySeqInPartition;
                copyRange.numToCopy = numToCopy;
                plan.copyRanges.push_back(copyRange);

                if(usedSeq == 0){
                    plan.h_partitionIds.push_back(lengthPartitionIds[currentCopyPartition]);
                    plan.h_numPerPartition.push_back(numToCopy);
                }else{
                    //if is same length partition as previous copy 
                    if(plan.h_partitionIds.back() == lengthPartitionIds[currentCopyPartition]){
                        plan.h_numPerPartition.back() += numToCopy;
                    }else{
                        //new length partition
                        plan.h_partitionIds.push_back(lengthPartitionIds[currentCopyPartition]);
                        plan.h_numPerPartition.push_back(numToCopy);
                    }
                }
                usedBytes += (dbPartitions[currentCopyPartition].offsets()[currentCopySeqInPartition+numToCopy] 
                    - dbPartitions[currentCopyPartition].offsets()[currentCopySeqInPartition]);
                usedSeq += numToCopy;

                currentCopySeqInPartition += numToCopy;
                if(currentCopySeqInPartition == dbPartitions[currentCopyPartition].numSequences()){
                    currentCopySeqInPartition = 0;
                    currentCopyPartition++;
                }
            }else{
                break;
            }
        }

        // {
        //     std::sort(plan.copyRanges.begin(), plan.copyRanges.end(), [](const auto& l, const auto& r){
        //         return l.lengthPartitionId > r.lengthPartitionId;
        //     });
        //     plan.h_partitionIds.clear();
        //     plan.h_numPerPartition.clear();
        //     for(size_t i = 0; i < plan.copyRanges.size(); i++){
        //         const auto& r = plan.copyRanges[i];
        //         if(i == 0){
        //             plan.h_partitionIds.push_back(r.lengthPartitionId);
        //             plan.h_numPerPartition.push_back(r.numToCopy);
        //         }else{
        //             if(plan.h_partitionIds.back() == r.lengthPartitionId){
        //                 plan.h_numPerPartition.back() += r.numToCopy;
        //             }else{
        //                 //new length partition
        //                 plan.h_partitionIds.push_back(r.lengthPartitionId);
        //                 plan.h_numPerPartition.push_back(r.numToCopy);
        //             }
        //         }
        //     }
        // }

        plan.usedBytes = usedBytes;
        plan.usedSeq = usedSeq;

        
        if(usedSeq == 0 && currentCopyPartition < dbPartitions.size()){
            std::cout << "Warning. copy buffer size too small. skipped a db portion. stop\n";
            break;
        }

        result.push_back(plan);
    }

    return result;
}

#if 0
std::vector<DeviceBatchCopyToPinnedPlan> computeDbCopyPlanMaybeOptimized1(
    const std::vector<DBdataView>& dbPartitions,
    const std::vector<int>& lengthPartitionIds,
    size_t MAX_CHARDATA_BYTES,
    size_t MAX_SEQ
){
    using P = std::pair<DBdataView, int>;
    std::vector<DeviceBatchCopyToPinnedPlan> result;

    const int numPartitions = dbPartitions.size();
    std::vector<P> partitionsWithLengthIds;
    
    for(int i = 0; i < numPartitions; i++){
        partitionsWithLengthIds.emplace_back(dbPartitions[i], lengthPartitionIds[i]);
    }

    std::vector<int> sortedIndices(numPartitions);
    std::iota(sortedIndices.begin(), sortedIndices.end(), 0);
    //sort by length partition id
    std::sort(sortedIndices.begin(), sortedIndices.end(), [&](const auto& l, const auto& r){return partitionsWithLengthIds[l].second < partitionsWithLengthIds[r].second;});

    //sort by length partition id
    //std::sort(partitionsWithLengthIds.begin(), partitionsWithLengthIds.end(), [](const auto& l, const auto& r){return l.second < r.second;});

    constexpr int firstLengthPartitionWithTempStorage = 13;

    auto largeBeginIt = std::stable_partition(
        sortedIndices.begin(), 
        sortedIndices.end(), 
        [&](const auto& l){return  partitionsWithLengthIds[l].second < firstLengthPartitionWithTempStorage;}
    );
    //sort large partitions from largest to smallest
    std::sort(largeBeginIt, sortedIndices.end(), [&](const auto& l, const auto& r){return partitionsWithLengthIds[l].second > partitionsWithLengthIds[r].second;});

    const int numSmallPartitions = std::distance(sortedIndices.begin(), largeBeginIt);
    const int numLargePartitions = std::distance(largeBeginIt, sortedIndices.end());
    const int firstLargePartition = numSmallPartitions;

    // auto largeBeginIt = std::stable_partition(
    //     partitionsWithLengthIds.begin(), 
    //     partitionsWithLengthIds.end(), 
    //     [](const auto& l){return l.second < firstLengthPartitionWithTempStorage;}
    // );
    // //sort large partitions from largest to smallest
    // std::sort(largeBeginIt, partitionsWithLengthIds.end(), [](const auto& l, const auto& r){return l.second > r.second;});

    // const int numSmallPartitions = std::distance(partitionsWithLengthIds.begin(), largeBeginIt);
    // const int numLargePartitions = std::distance(largeBeginIt, partitionsWithLengthIds.end());
    // const int firstLargePartition = numSmallPartitions;

    size_t numSequences = 0;
    for(const auto& p : partitionsWithLengthIds){
        numSequences += p.first.numSequences();
    }

    std::vector<size_t> currentCopySeqInPartition_vec(numPartitions, 0);


    size_t processedSequences = 0;
    while(processedSequences < numSequences){
        size_t oldProcessedSequences = processedSequences;
        size_t usedBytes = 0;
        size_t usedSeq = 0;

        DeviceBatchCopyToPinnedPlan plan;

        auto processPartition = [&](int index, size_t charmemoryLimit) -> size_t{
            const int sortedIndex = sortedIndices[index];
            const auto& dbPartition = partitionsWithLengthIds[sortedIndex].first;
            const int lengthPartitionId = partitionsWithLengthIds[sortedIndex].second;
            auto& currentCopySeqInPartition = currentCopySeqInPartition_vec[sortedIndex];
            size_t remainingBytes = charmemoryLimit - usedBytes;
            
            auto dboffsetsBegin = dbPartition.offsets() + currentCopySeqInPartition;
            auto dboffsetsEnd = dbPartition.offsets() + dbPartition.numSequences() + 1;
            
            auto searchFor = dbPartition.offsets()[currentCopySeqInPartition] + remainingBytes + 1; // +1 because remainingBytes is inclusive
            auto it = std::lower_bound(
                dboffsetsBegin,
                dboffsetsEnd,
                searchFor
            );

            size_t numToCopyByBytes = 0;
            if(it != dboffsetsBegin){
                numToCopyByBytes = std::distance(dboffsetsBegin, it) - 1;
            }
            if(numToCopyByBytes == 0 && currentCopySeqInPartition == 0){
                //std::cout << "Warning. copy buffer size too small. skipped a db portion\n";
                return 0;
            }
            
            size_t remainingSeq = MAX_SEQ - usedSeq;            
            size_t numToCopyBySeq = std::min(dbPartition.numSequences() - currentCopySeqInPartition, remainingSeq);
            size_t numToCopy = std::min(numToCopyByBytes,numToCopyBySeq);

            if(numToCopy > 0){
                DeviceBatchCopyToPinnedPlan::CopyRange copyRange;
                copyRange.currentCopyPartition = sortedIndex;
                copyRange.currentCopySeqInPartition = currentCopySeqInPartition;
                copyRange.numToCopy = numToCopy;
                plan.copyRanges.push_back(copyRange);

                if(usedSeq == 0){
                    plan.h_partitionIds.push_back(lengthPartitionId);
                    plan.h_numPerPartition.push_back(numToCopy);
                }else{
                    //if is same length partition as previous copy 
                    if(plan.h_partitionIds.back() == lengthPartitionId){
                        plan.h_numPerPartition.back() += numToCopy;
                    }else{
                        //new length partition
                        plan.h_partitionIds.push_back(lengthPartitionId);
                        plan.h_numPerPartition.push_back(numToCopy);
                    }
                }
                usedBytes += (dbPartition.offsets()[currentCopySeqInPartition+numToCopy] 
                    - dbPartition.offsets()[currentCopySeqInPartition]);
                usedSeq += numToCopy;

                currentCopySeqInPartition += numToCopy;

                processedSequences += numToCopy;
            }

            return numToCopy;
        };

        //add large partitions, up to half the memory
        for(int i = firstLargePartition; i < numPartitions; i++){
            const int sortedIndex = sortedIndices[i];
            if(currentCopySeqInPartition_vec[sortedIndex] < partitionsWithLengthIds[sortedIndex].first.numSequences()){
                //large partitions are sorted descending, if no sequence could be added here, a shorter sequence might be added from a smaller large partition
               processPartition(i, MAX_CHARDATA_BYTES / 2);
            }
        }
        //fill up with small partitions
        for(int i = 0; i < numSmallPartitions; i++){
            const int sortedIndex = sortedIndices[i];
            if(currentCopySeqInPartition_vec[sortedIndex] < partitionsWithLengthIds[sortedIndex].first.numSequences()){
                size_t numAdded = processPartition(i, MAX_CHARDATA_BYTES);
                if(numAdded == 0){
                    break;
                }
            }
        }

        if(oldProcessedSequences == processedSequences){
            // bool allAreDone = true;
            // for(int i = 0; i < numPartitions; i++){
            //     allAreDone &= (currentCopySeqInPartition_vec[i] == partitionsWithLengthIds[i].first.numSequences());
            // }
            std::cout << "Warning. copy buffer size too small. skipped a db portion. stop\n";
            break;
        }else{
            plan.usedBytes = usedBytes;
            plan.usedSeq = usedSeq;

            // std::cout << "plan " << result.size() << "\n";
            // std::cout << plan << "\n";

            result.push_back(plan);

        }

    }

    return result;
}
#endif


void executeCopyPlanH2DDirect(
    const DeviceBatchCopyToPinnedPlan& plan, 
    GpuWorkingSet& ws, 
    int currentBuffer, 
    const std::vector<DBdataView>& dbPartitions,
    cudaStream_t stream
){
    auto& d_chardata_vec = ws.d_chardata_vec;
    auto& d_lengthdata_vec = ws.d_lengthdata_vec;
    auto& d_offsetdata_vec = ws.d_offsetdata_vec;

    size_t usedBytes = 0;
    size_t usedSeq = 0;
    for(const auto& copyRange : plan.copyRanges){
        const auto& dbPartition = dbPartitions[copyRange.currentCopyPartition];
        const auto& firstSeq = copyRange.currentCopySeqInPartition;
        const auto& numToCopy = copyRange.numToCopy;
        size_t numBytesToCopy = dbPartition.offsets()[firstSeq + numToCopy] - dbPartition.offsets()[firstSeq];

        cudaMemcpyAsync(
            d_chardata_vec[currentBuffer].data() + usedBytes,
            dbPartition.chars() + dbPartition.offsets()[firstSeq],
            numBytesToCopy,
            H2D,
            stream
        ); CUERR;
        cudaMemcpyAsync(
            d_lengthdata_vec[currentBuffer].data() + usedSeq,
            dbPartition.lengths() + firstSeq,
            sizeof(size_t) * numToCopy,
            H2D,
            stream
        ); CUERR;
        cudaMemcpyAsync(
            d_offsetdata_vec[currentBuffer].data() + usedSeq,
            dbPartition.offsets() + firstSeq,
            sizeof(size_t) * (numToCopy+1),
            H2D,
            stream
        ); CUERR;
        thrust::for_each(
            thrust::cuda::par_nosync.on(stream),
            d_offsetdata_vec[currentBuffer].data() + usedSeq,
            d_offsetdata_vec[currentBuffer].data() + usedSeq + (numToCopy+1),
            [
                usedBytes,
                firstOffset = dbPartition.offsets()[firstSeq]
            ] __device__ (size_t& off){
                off = off - firstOffset + usedBytes;
            }
        );

        usedBytes += numBytesToCopy;
        usedSeq += numToCopy;
    }
};

void executePinnedCopyPlanSerial(
    const DeviceBatchCopyToPinnedPlan& plan, 
    GpuWorkingSet& ws, 
    int currentBuffer, 
    const std::vector<DBdataView>& dbPartitions
){
    auto& h_chardata_vec = ws.h_chardata_vec;
    auto& h_lengthdata_vec = ws.h_lengthdata_vec;
    auto& h_offsetdata_vec = ws.h_offsetdata_vec;

    size_t usedBytes = 0;
    size_t usedSeq = 0;
    for(const auto& copyRange : plan.copyRanges){
        const auto& dbPartition = dbPartitions[copyRange.currentCopyPartition];
        const auto& firstSeq = copyRange.currentCopySeqInPartition;
        const auto& numToCopy = copyRange.numToCopy;
        size_t numBytesToCopy = dbPartition.offsets()[firstSeq + numToCopy] - dbPartition.offsets()[firstSeq];

        auto end = std::copy(
            dbPartition.chars() + dbPartition.offsets()[firstSeq],
            dbPartition.chars() + dbPartition.offsets()[firstSeq + numToCopy],
            h_chardata_vec[currentBuffer].data() + usedBytes
        );
        std::copy(
            dbPartition.lengths() + firstSeq,
            dbPartition.lengths() + firstSeq+numToCopy,
            h_lengthdata_vec[currentBuffer].data() + usedSeq
        );
        std::transform(
            dbPartition.offsets() + firstSeq,
            dbPartition.offsets() + firstSeq + (numToCopy+1),
            h_offsetdata_vec[currentBuffer].data() + usedSeq,
            [&](size_t off){
                return off - dbPartition.offsets()[firstSeq] + usedBytes;
            }
        );
        usedBytes += std::distance(h_chardata_vec[currentBuffer].data() + usedBytes, end);
        usedSeq += numToCopy;
    }
};

void executePinnedCopyPlanSerialAndTransferToGpu(
    const DeviceBatchCopyToPinnedPlan& plan, 
    GpuWorkingSet& ws, 
    int currentBuffer, 
    const std::vector<DBdataView>& dbPartitions,
    cudaStream_t H2DcopyStream
){
    //std::cout << "executePinnedCopyPlanSerialAndTransferToGpu(" << currentBuffer << ")\n";
    auto& h_chardata_vec = ws.h_chardata_vec;
    auto& h_lengthdata_vec = ws.h_lengthdata_vec;
    auto& h_offsetdata_vec = ws.h_offsetdata_vec;

    auto& d_chardata_vec = ws.d_chardata_vec;
    auto& d_lengthdata_vec = ws.d_lengthdata_vec;
    auto& d_offsetdata_vec = ws.d_offsetdata_vec;

    size_t usedBytes = 0;
    size_t usedSeq = 0;
    for(const auto& copyRange : plan.copyRanges){
        const auto& dbPartition = dbPartitions[copyRange.currentCopyPartition];
        const auto& firstSeq = copyRange.currentCopySeqInPartition;
        const auto& numToCopy = copyRange.numToCopy;
        size_t numBytesToCopy = dbPartition.offsets()[firstSeq + numToCopy] - dbPartition.offsets()[firstSeq];

        auto end = std::copy(
            dbPartition.chars() + dbPartition.offsets()[firstSeq],
            dbPartition.chars() + dbPartition.offsets()[firstSeq + numToCopy],
            h_chardata_vec[currentBuffer].data() + usedBytes
        );
        // cudaMemcpyAsync(
        //     d_chardata_vec[currentBuffer].data() + usedBytes,
        //     h_chardata_vec[currentBuffer].data() + usedBytes,
        //     numBytesToCopy,
        //     H2D,
        //     H2DcopyStream
        // ); CUERR;

        std::copy(
            dbPartition.lengths() + firstSeq,
            dbPartition.lengths() + firstSeq+numToCopy,
            h_lengthdata_vec[currentBuffer].data() + usedSeq
        );
        // cudaMemcpyAsync(
        //     d_lengthdata_vec[currentBuffer].data() + usedSeq,
        //     h_lengthdata_vec[currentBuffer].data() + usedSeq,
        //     sizeof(size_t) * numToCopy,
        //     H2D,
        //     H2DcopyStream
        // ); CUERR;

        std::transform(
            dbPartition.offsets() + firstSeq,
            dbPartition.offsets() + firstSeq + (numToCopy+1),
            h_offsetdata_vec[currentBuffer].data() + usedSeq,
            [&](size_t off){
                return off - dbPartition.offsets()[firstSeq] + usedBytes;
            }
        );
        // cudaMemcpyAsync(
        //     d_offsetdata_vec[currentBuffer].data() + usedSeq,
        //     h_offsetdata_vec[currentBuffer].data() + usedSeq,
        //     sizeof(size_t) * (numToCopy+1),
        //     H2D,
        //     H2DcopyStream
        // ); CUERR;

        usedBytes += numBytesToCopy;
        usedSeq += numToCopy;
    }

    assert(plan.usedBytes == usedBytes);
    assert(plan.usedSeq == usedSeq);

    //std::cout << plan.usedBytes << " " << plan.usedSeq << "\n";

    cudaMemcpyAsync(
        d_chardata_vec[currentBuffer].data(),
        h_chardata_vec[currentBuffer].data(),
        plan.usedBytes,
        H2D,
        H2DcopyStream
    ); CUERR;
    cudaMemcpyAsync(
        d_lengthdata_vec[currentBuffer].data(),
        h_lengthdata_vec[currentBuffer].data(),
        sizeof(size_t) * plan.usedSeq,
        H2D,
        H2DcopyStream
    ); CUERR;
    cudaMemcpyAsync(
        d_offsetdata_vec[currentBuffer].data(),
        h_offsetdata_vec[currentBuffer].data(),
        sizeof(size_t) * (plan.usedSeq+1),
        H2D,
        H2DcopyStream
    ); CUERR;
};

struct ExecutePinnedCopyCallbackData{
    int currentBuffer;
    const DeviceBatchCopyToPinnedPlan* planPtr; 
    GpuWorkingSet* wsPtr;
    const std::vector<DBdataView>* dbPartitionsPtr;
};

void executePinnedCopyPlanCallback(void* args){
    ExecutePinnedCopyCallbackData* callbackData = (ExecutePinnedCopyCallbackData*)args;
    int currentBuffer = callbackData->currentBuffer;
    const auto& plan = *callbackData->planPtr;
    auto& ws = *callbackData->wsPtr;
    auto& dbPartitions = *callbackData->dbPartitionsPtr;
    

    executePinnedCopyPlanSerial(
        plan, 
        ws, 
        currentBuffer, 
        dbPartitions
    );

    delete callbackData;
}

void executePinnedCopyPlanWithHostCallback(
    const DeviceBatchCopyToPinnedPlan& plan, 
    GpuWorkingSet& ws, 
    int currentBuffer, 
    const std::vector<DBdataView>& dbPartitions, 
    cudaStream_t stream
){
    ExecutePinnedCopyCallbackData* data = new ExecutePinnedCopyCallbackData;
    data->currentBuffer = currentBuffer;
    data->planPtr = &plan;
    data->wsPtr = &ws;
    data->dbPartitionsPtr = &dbPartitions;

    cudaLaunchHostFunc(
        stream,
        executePinnedCopyPlanCallback,
        (void*)data
    ); CUERR;
}






#if 0

void processQueryOnGpu(
    GpuWorkingSet& ws,
    const std::vector<DBdataView>& dbPartitions,
    const std::vector<int>& lengthPartitionIds, // dbPartitions[i] belongs to the length partition lengthPartitionIds[i]
    const std::vector<DeviceBatchCopyToPinnedPlan>& batchPlan,
    const char* d_query,
    const int queryLength,
    bool useExtraThreadForBatchTransfer,
    const CudaSW4Options& options,
    cudaStream_t mainStream
){
    constexpr auto boundaries = getLengthPartitionBoundaries();
    constexpr int numLengthPartitions = boundaries.size();


    cudaSetDevice(ws.deviceId); CUERR;

    const int numCopyBuffers = ws.numCopyBuffers;
    const size_t MAX_CHARDATA_BYTES = ws.MAX_CHARDATA_BYTES;
    const size_t MAX_SEQ = ws.MAX_SEQ;

    cudaMemcpyToSymbolAsync(constantQuery4 ,ws.Fillchar.data(), 512*16, 0, cudaMemcpyDeviceToDevice, mainStream); CUERR
    cudaMemcpyToSymbolAsync(constantQuery4, d_query, queryLength, 0, cudaMemcpyDeviceToDevice, mainStream); CUERR

    //create dependency on mainStream
    cudaEventRecord(ws.forkStreamEvent, mainStream); CUERR;
    cudaStreamWaitEvent(ws.workStreamForTempUsage, ws.forkStreamEvent, 0); CUERR;
    for(auto& stream : ws.workStreamsWithoutTemp){
        cudaStreamWaitEvent(stream, ws.forkStreamEvent, 0); CUERR;
    }
    cudaStreamWaitEvent(ws.hostFuncStream, ws.forkStreamEvent, 0); CUERR;
    // for(int i = 0; i < ws.numCopyBuffers; i++){
    //     cudaStreamWaitEvent(ws.copyStreams[i], ws.forkStreamEvent, 0); CUERR;
    // }
    

    auto& h_chardata_vec = ws.h_chardata_vec;
    auto& h_lengthdata_vec = ws.h_lengthdata_vec;
    auto& h_offsetdata_vec = ws.h_offsetdata_vec;
    auto& d_chardata_vec = ws.d_chardata_vec;
    auto& d_lengthdata_vec = ws.d_lengthdata_vec;
    auto& d_offsetdata_vec = ws.d_offsetdata_vec;
    auto& copyStreams = ws.copyStreams;




    size_t processedSequences = 0;

    for(size_t compute_batch = 0, fetch_batch = 0; compute_batch < batchPlan.size(); compute_batch++){
        for (; fetch_batch < batchPlan.size() && fetch_batch < (compute_batch + numCopyBuffers); ++fetch_batch) {
            const auto& plan = batchPlan[fetch_batch];
            if(!ws.singleBatchDBisOnGpu){
                const int currentBuffer = fetch_batch % numCopyBuffers;
                const cudaStream_t H2DcopyStream = copyStreams[currentBuffer];
                //can only overwrite device buffer if it is no longer in use on workstream
                cudaStreamWaitEvent(H2DcopyStream, ws.deviceBufferEvents[currentBuffer], 0); CUERR;
                cudaStreamWaitEvent(ws.hostFuncStream, ws.pinnedBufferEvents[currentBuffer]); CUERR;
                executePinnedCopyPlanWithHostCallback(plan, ws, currentBuffer, dbPartitions, ws.hostFuncStream);
                cudaEventRecord(ws.forkStreamEvent, ws.hostFuncStream); CUERR;
                cudaStreamWaitEvent(H2DcopyStream, ws.forkStreamEvent, 0);

                cudaMemcpyAsync(
                    d_chardata_vec[currentBuffer].data(),
                    h_chardata_vec[currentBuffer].data(),
                    plan.usedBytes,
                    H2D,
                    H2DcopyStream
                ); CUERR;
                cudaMemcpyAsync(
                    d_lengthdata_vec[currentBuffer].data(),
                    h_lengthdata_vec[currentBuffer].data(),
                    sizeof(size_t) * plan.usedSeq,
                    H2D,
                    H2DcopyStream
                ); CUERR;
                cudaMemcpyAsync(
                    d_offsetdata_vec[currentBuffer].data(),
                    h_offsetdata_vec[currentBuffer].data(),
                    sizeof(size_t) * (plan.usedSeq+1),
                    H2D,
                    H2DcopyStream
                ); CUERR;
                cudaEventRecord(ws.pinnedBufferEvents[currentBuffer], H2DcopyStream); CUERR;                
            }

        }

        const auto& plan = batchPlan[compute_batch];

        const int currentBuffer = compute_batch % numCopyBuffers;
        const int previousBuffer = currentBuffer == 0 ? (numCopyBuffers-1) : (currentBuffer - 1);
        const cudaStream_t H2DcopyStream = copyStreams[currentBuffer];
        if(ws.singleBatchDBisOnGpu){
            assert(batchPlan.size() == 1);
            //can only overwrite device buffer if it is no longer in use on workstream
            //for d_overflow_number
            cudaStreamWaitEvent(H2DcopyStream, ws.deviceBufferEvents[currentBuffer], 0); CUERR;
        }

        const char* const inputChars = d_chardata_vec[currentBuffer].data();
        const size_t* const inputOffsets = d_offsetdata_vec[currentBuffer].data();
        const size_t* const inputLengths = d_lengthdata_vec[currentBuffer].data();
        int* const h_overflow_number = ws.h_new_overflow_number.data() + currentBuffer;
        int* const d_overflow_number = ws.d_new_overflow_number.data() + currentBuffer;
        size_t* const d_overflow_positions = ws.d_new_overflow_positions_vec[currentBuffer].data();


        cudaMemsetAsync(d_overflow_number, 0, sizeof(int), H2DcopyStream);

        //all data is ready for alignments. create dependencies for work streams
        cudaEventRecord(ws.forkStreamEvent, H2DcopyStream); CUERR;
        cudaStreamWaitEvent(ws.workStreamForTempUsage, ws.forkStreamEvent, 0); CUERR;
        for(auto& stream : ws.workStreamsWithoutTemp){
            cudaStreamWaitEvent(stream, ws.forkStreamEvent, 0); CUERR;
        }
        //wait for previous batch to finish
        cudaStreamWaitEvent(ws.workStreamForTempUsage, ws.deviceBufferEvents[previousBuffer], 0); CUERR;
        for(auto& stream : ws.workStreamsWithoutTemp){
            cudaStreamWaitEvent(stream, ws.deviceBufferEvents[previousBuffer], 0); CUERR;
        }






        const float gop = options.gop;
        const float gex = options.gex;

        auto runAlignmentKernels = [&](auto& d_scores, size_t* d_overflow_positions, int* d_overflow_number){
            auto nextWorkStreamNoTemp = [&](){
                ws.workstreamIndex = (ws.workstreamIndex + 1) % ws.numWorkStreamsWithoutTemp;
                return (cudaStream_t)ws.workStreamsWithoutTemp[ws.workstreamIndex];
            };
            std::vector<int> numPerPartitionPrefixSum(plan.h_numPerPartition.size());
            for(int i = 0; i < int(plan.h_numPerPartition.size())-1; i++){
                numPerPartitionPrefixSum[i+1] = numPerPartitionPrefixSum[i] + plan.h_numPerPartition[i];
            }
            //size_t exclPs = 0;
            //for(int lp = 0; lp < int(plan.h_partitionIds.size()); lp++){
            for(int lp = plan.h_partitionIds.size() - 1; lp >= 0; lp--){
                const int partId = plan.h_partitionIds[lp];
                const int numSeq = plan.h_numPerPartition[lp];
                const int start = numPerPartitionPrefixSum[lp];
                //std::cout << "partId " << partId << " numSeq " << numSeq << "\n";
                
                const size_t* const d_selectedPositions = ws.d_selectedPositions.data() + start;


                #if 0
                if (partId == 0){NW_local_affine_Protein_single_pass_half2<8, 8><<<(numSeq+63)/(2*8*4), 32*8, 0, nextWorkStreamNoTemp()>>>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex); CUERR }
                if (partId == 1){NW_local_affine_Protein_single_pass_half2<8, 16><<<(numSeq+63)/(2*8*4), 32*8, 0, nextWorkStreamNoTemp()>>>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex); CUERR }
                if (partId == 2){NW_local_affine_Protein_single_pass_half2<8, 24><<<(numSeq+63)/(2*8*4), 32*8, 0, nextWorkStreamNoTemp()>>>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex); CUERR }
                if (partId == 3){NW_local_affine_Protein_single_pass_half2<8, 32><<<(numSeq+63)/(2*8*4), 32*8, 0, nextWorkStreamNoTemp()>>>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex); CUERR }
                if (partId == 4){NW_local_affine_Protein_single_pass_half2<16, 20><<<(numSeq+31)/(2*8*2), 32*8, 0, nextWorkStreamNoTemp()>>>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex); CUERR }
                if (partId == 5){NW_local_affine_Protein_single_pass_half2<16, 24><<<(numSeq+31)/(2*8*2), 32*8, 0, nextWorkStreamNoTemp()>>>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex); CUERR }
                if (partId == 6){NW_local_affine_Protein_single_pass_half2<16, 28><<<(numSeq+31)/(2*8*2), 32*8, 0, nextWorkStreamNoTemp()>>>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex); CUERR }
                if (partId == 7){NW_local_affine_Protein_single_pass_half2<16, 32><<<(numSeq+31)/(2*8*2), 32*8, 0, nextWorkStreamNoTemp()>>>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex); CUERR }
                if (partId == 8){NW_local_affine_Protein_single_pass_half2<32, 18><<<(numSeq+15)/(2*8), 32*8, 0, nextWorkStreamNoTemp()>>>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex); CUERR }
                if (partId == 9){NW_local_affine_Protein_single_pass_half2<32, 20><<<(numSeq+15)/(2*8), 32*8, 0, nextWorkStreamNoTemp()>>>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex); CUERR }
                if (partId == 10){NW_local_affine_Protein_single_pass_half2<32, 24><<<(numSeq+15)/(2*8), 32*8, 0, nextWorkStreamNoTemp()>>>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex); CUERR }
                if (partId == 11){NW_local_affine_Protein_single_pass_half2<32, 28><<<(numSeq+15)/(2*8), 32*8, 0, nextWorkStreamNoTemp()>>>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex); CUERR }
                if (partId == 12){NW_local_affine_Protein_single_pass_half2<32, 32><<<(numSeq+15)/(2*8), 32*8, 0, nextWorkStreamNoTemp()>>>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex); CUERR }
                #endif

                #if 0
                if (partId == 0){call_NW_local_affine_Protein_single_pass_half2_new<256, 4, 16>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 1){call_NW_local_affine_Protein_single_pass_half2_new<256, 4, 32>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 2){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 12>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 3){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 16>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 4){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 20>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 5){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 24>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 6){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 28>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 7){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 32>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 8){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 18>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 9){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 20>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 10){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 24>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 11){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 24>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 12){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 26>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 13){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 28>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 14){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 30>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 15){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 32>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }

                #endif

                #if 1
                if (partId == 0){call_NW_local_affine_Protein_single_pass_half2_new<256, 8, 8>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 1){call_NW_local_affine_Protein_single_pass_half2_new<256, 8, 12>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 2){call_NW_local_affine_Protein_single_pass_half2_new<256, 8, 16>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 3){call_NW_local_affine_Protein_single_pass_half2_new<256, 8, 20>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 4){call_NW_local_affine_Protein_single_pass_half2_new<256, 8, 24>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 5){call_NW_local_affine_Protein_single_pass_half2_new<256, 8, 28>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 6){call_NW_local_affine_Protein_single_pass_half2_new<256, 8, 32>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 7){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 18>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 8){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 20>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 9){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 22>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 10){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 24>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 11){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 26>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 12){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 28>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 13){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 30>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 14){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 32>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 15){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 18>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 16){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 20>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 17){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 22>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 18){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 24>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 19){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 26>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 20){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 28>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 21){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 30>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 22){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 32>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 23){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 34>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 24){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 36>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                
                #endif

                //if (partId > 15 && partId < 44){
                //if (partId == 16){
                if(partId == numLengthPartitions - 2){
                    constexpr int blocksize = 32 * 8;
                    constexpr int groupsize = 32;
                    constexpr int groupsPerBlock = blocksize / groupsize;
                    constexpr int alignmentsPerGroup = 2;
                    constexpr int alignmentsPerBlock = groupsPerBlock * alignmentsPerGroup;
                    
                    const size_t tempBytesPerBlockPerBuffer = sizeof(__half2) * alignmentsPerBlock * queryLength;

                    const size_t maxNumBlocks = ws.numTempBytes / (tempBytesPerBlockPerBuffer * 2);
                    const size_t maxSubjectsPerIteration = std::min(maxNumBlocks * alignmentsPerBlock, size_t(numSeq));

                    const size_t numBlocksPerIteration = SDIV(maxSubjectsPerIteration, alignmentsPerBlock);
                    const size_t requiredTempBytes = tempBytesPerBlockPerBuffer * 2 * numBlocksPerIteration;
                    // std::cout << "queryLength " << queryLength << " alignmentsPerBlock " << alignmentsPerBlock << "\n";
                    // std::cout << "ws.numTempBytes " << ws.numTempBytes << " maxNumBlocks " << maxNumBlocks << "\n";
                    // std::cout << "maxSubjectsPerIteration " << maxSubjectsPerIteration << " numBlocksPerIteration " << numBlocksPerIteration << "\n";

                    __half2* d_temp = (__half2*)ws.d_tempStorageHE.data();
                    __half2* d_tempHcol2 = d_temp;
                    __half2* d_tempEcol2 = (__half2*)(((char*)d_tempHcol2) + requiredTempBytes / 2);

                    const int numIters =  SDIV(numSeq, maxSubjectsPerIteration);

                    for(int iter = 0; iter < numIters; iter++){
                        const size_t begin = iter * maxSubjectsPerIteration;
                        const size_t end = iter < numIters-1 ? (iter+1) * maxSubjectsPerIteration : numSeq;
                        const size_t num = end - begin;      
                        
                        cudaMemsetAsync(d_temp, 0, requiredTempBytes, ws.workStreamForTempUsage); CUERR;
                        // std::cout << "manypass iter " << iter << " / " << numIters << " gridsize " << SDIV(num, alignmentsPerBlock) << "\n";
                        // std::cout << "begin " << begin << " end " << end << " num " << num << "\n";         
                        // std::cout << "requiredTempBytes " << requiredTempBytes << " numBlocksPerIteration " << numBlocksPerIteration << "\n";       
                        
                        //NW_local_affine_Protein_many_pass_half2<groupsize, 12><<<SDIV(num, alignmentsPerBlock), blocksize, 0, ws.workStreamForTempUsage>>>(
                        NW_local_affine_Protein_many_pass_half2_new<groupsize, 22><<<SDIV(num, alignmentsPerBlock), blocksize, 0, ws.workStreamForTempUsage>>>(
                            inputChars, 
                            d_scores, 
                            d_tempHcol2, 
                            d_tempEcol2, 
                            inputOffsets, 
                            inputLengths, 
                            d_selectedPositions + begin, 
                            num, 
                            d_overflow_positions, 
                            d_overflow_number, 
                            1, 
                            queryLength, 
                            gop, 
                            gex
                        ); CUERR

                        // cudaDeviceSynchronize(); CUERR;
                        // cudaMemcpyAsync(h_overflow_number, d_overflow_number, sizeof(int), D2H, ws.workStreamForTempUsage); CUERR;
                        // cudaDeviceSynchronize(); CUERR;
                        // std::cout << "Overflow number after manypass iter " << iter << " is " << *h_overflow_number << "\n";
                    }
                }

                //if (partId == 44){
                //if (partId == 17){
                if(partId == numLengthPartitions - 1){
                    const size_t tempBytesPerSubjectPerBuffer = sizeof(short2) * SDIV(queryLength,32) * 32;
                    const size_t maxSubjectsPerIteration = std::min(size_t(numSeq), ws.numTempBytes / (tempBytesPerSubjectPerBuffer * 2));

                    short2* d_temp = (short2*)ws.d_tempStorageHE.data();
                    short2* d_tempHcol2 = d_temp;
                    short2* d_tempEcol2 = (short2*)(((char*)d_tempHcol2) + maxSubjectsPerIteration * tempBytesPerSubjectPerBuffer);

                    const int numIters =  SDIV(numSeq, maxSubjectsPerIteration);
                    for(int iter = 0; iter < numIters; iter++){
                        const size_t begin = iter * maxSubjectsPerIteration;
                        const size_t end = iter < numIters-1 ? (iter+1) * maxSubjectsPerIteration : numSeq;
                        const size_t num = end - begin;

                        cudaMemsetAsync(d_temp, 0, tempBytesPerSubjectPerBuffer * 2 * num, ws.workStreamForTempUsage); CUERR;

                        //NW_local_affine_read4_float_query_Protein<32, 32><<<num, 32, 0, ws.workStreamForTempUsage>>>(
                        NW_local_affine_read4_float_query_Protein_new<32><<<num, 32, 0, ws.workStreamForTempUsage>>>(
                            inputChars, 
                            d_scores, 
                            d_tempHcol2, 
                            d_tempEcol2, 
                            inputOffsets, 
                            inputLengths, 
                            d_selectedPositions + begin, 
                            queryLength, 
                            gop, 
                            gex
                        ); CUERR 
                    }
                }
                
                // cudaDeviceSynchronize(); CUERR;
                // cudaMemcpyAsync(h_overflow_number, d_overflow_number, sizeof(int), D2H, ws.workStreamForTempUsage); CUERR;
                // cudaDeviceSynchronize(); CUERR;
                // std::cout << "Overflow number after partId " << partId << " is " << *h_overflow_number << "\n";

                //exclPs += numSeq;
            }
        };

        
        //cudaDeviceSynchronize(); CUERR;
        auto maxReduceArray = ws.getMaxReduceArray(processedSequences);
        runAlignmentKernels(maxReduceArray, d_overflow_positions, d_overflow_number);


        //alignments are done in workstreams. now, join all workstreams into workStreamForTempUsage to process overflow alignments
        for(auto& stream : ws.workStreamsWithoutTemp){
            cudaEventRecord(ws.forkStreamEvent, stream); CUERR;
            cudaStreamWaitEvent(ws.workStreamForTempUsage, ws.forkStreamEvent, 0); CUERR;    
        }

        // cudaMemcpyAsync(h_overflow_number, d_overflow_number, sizeof(int), D2H, ws.workStreamForTempUsage); CUERR;
        // cudaStreamSynchronize(ws.workStreamForTempUsage); CUERR;
        // if(*h_overflow_number > 0){
        //     //std::cout << "numOverflow " << *h_overflow_number << "\n";
        //     const size_t tempBytesPerSubjectPerBuffer = sizeof(short2) * SDIV(queryLength,32) * 32;
        //     const size_t maxSubjectsPerIteration = std::min(size_t(*h_overflow_number), ws.numTempBytes / (tempBytesPerSubjectPerBuffer * 2));

        //     short2* d_temp = (short2*)ws.d_tempStorageHE.data();
        //     short2* d_tempHcol2 = d_temp;
        //     short2* d_tempEcol2 = (short2*)(((char*)d_tempHcol2) + maxSubjectsPerIteration * tempBytesPerSubjectPerBuffer);

        //     const int numIters =  SDIV(*h_overflow_number, maxSubjectsPerIteration);
        //     //std::cout << "numIters " << numIters << "\n";
        //     for(int iter = 0; iter < numIters; iter++){
        //         const size_t begin = iter * maxSubjectsPerIteration;
        //         const size_t end = iter < numIters-1 ? (iter+1) * maxSubjectsPerIteration : *h_overflow_number;
        //         const size_t num = end - begin;

        //         cudaMemsetAsync(d_temp, 0, tempBytesPerSubjectPerBuffer * 2 * num, ws.workStreamForTempUsage); CUERR;

        //         //NW_local_affine_read4_float_query_Protein<32, 32><<<num, 32, 0, ws.workStreamForTempUsage>>>(
        //         NW_local_affine_read4_float_query_Protein_new<12><<<num, 32, 0, ws.workStreamForTempUsage>>>(
        //             inputChars, 
        //             maxReduceArray, 
        //             d_tempHcol2, 
        //             d_tempEcol2, 
        //             inputOffsets, 
        //             inputLengths, 
        //             d_overflow_positions + begin, 
        //             queryLength, 
        //             gop, 
        //             gex
        //         ); CUERR 
        //     }
        // }

        {
            short2* d_temp = (short2*)ws.d_tempStorageHE.data();

            launch_process_overflow_alignments_kernel_NW_local_affine_read4_float_query_Protein<32,12><<<1,1,0, ws.workStreamForTempUsage>>>(
                d_overflow_number,
                d_temp, 
                ws.numTempBytes,
                inputChars, 
                maxReduceArray, 
                inputOffsets, 
                inputLengths, 
                d_overflow_positions, 
                queryLength, gop, gex
            ); CUERR
        }

        //after processing overflow alignments, the batch is done and its data can be resused
        cudaEventRecord(ws.deviceBufferEvents[currentBuffer], ws.workStreamForTempUsage); CUERR;

        //let other workstreams depend on temp usage stream
        for(auto& stream : ws.workStreamsWithoutTemp){
            cudaStreamWaitEvent(ws.workStreamForTempUsage, ws.deviceBufferEvents[currentBuffer], 0); CUERR;    
        }

        processedSequences += plan.usedSeq;
    }

    //create dependency for mainStream
    cudaEventRecord(ws.forkStreamEvent, ws.workStreamForTempUsage); CUERR;
    cudaStreamWaitEvent(mainStream, ws.forkStreamEvent, 0); CUERR;

    for(auto& stream : ws.workStreamsWithoutTemp){
        cudaEventRecord(ws.forkStreamEvent, stream); CUERR;
        cudaStreamWaitEvent(mainStream, ws.forkStreamEvent, 0); CUERR;
    }

    // for(auto& stream : ws.copyStreams){
    //     cudaEventRecord(ws.forkStreamEvent, stream); CUERR;
    //     cudaStreamWaitEvent(mainStream, ws.forkStreamEvent, 0); CUERR;
    // }

    cudaEventRecord(ws.forkStreamEvent, ws.hostFuncStream); CUERR;
    cudaStreamWaitEvent(mainStream, ws.forkStreamEvent, 0); CUERR;

}

#else 

void processQueryOnGpu(
    GpuWorkingSet& ws,
    const std::vector<DBdataView>& dbPartitions,
    const std::vector<int>& lengthPartitionIds, // dbPartitions[i] belongs to the length partition lengthPartitionIds[i]
    const std::vector<DeviceBatchCopyToPinnedPlan>& batchPlan,
    const char* d_query,
    const int queryLength,
    bool useExtraThreadForBatchTransfer,
    const CudaSW4Options& options,
    cudaStream_t mainStream
){
    constexpr auto boundaries = getLengthPartitionBoundaries();
    constexpr int numLengthPartitions = boundaries.size();

    struct CopyParams{
        int numBuffers = 3;
        //chars, lengths, and offsets
        const void* src[3];
        void* dst[3];
        size_t bytes[3];
    };
    auto copyBuffersFunc = [](void* args){
        const CopyParams* params = (const CopyParams*)args;
        for(int i = 0; i < params->numBuffers; i++){
            std::memcpy(params->dst[i], params->src[i], params->bytes[i]);
        }
        delete params;
    };


    cudaSetDevice(ws.deviceId); CUERR;

    const int numCopyBuffers = ws.numCopyBuffers;
    const size_t MAX_CHARDATA_BYTES = ws.MAX_CHARDATA_BYTES;
    const size_t MAX_SEQ = ws.MAX_SEQ;

    cudaMemcpyToSymbolAsync(constantQuery4 ,ws.Fillchar.data(), 512*16, 0, cudaMemcpyDeviceToDevice, mainStream); CUERR
    cudaMemcpyToSymbolAsync(constantQuery4, d_query, queryLength, 0, cudaMemcpyDeviceToDevice, mainStream); CUERR

    //create dependency on mainStream
    cudaEventRecord(ws.forkStreamEvent, mainStream); CUERR;
    cudaStreamWaitEvent(ws.workStreamForTempUsage, ws.forkStreamEvent, 0); CUERR;
    for(auto& stream : ws.workStreamsWithoutTemp){
        cudaStreamWaitEvent(stream, ws.forkStreamEvent, 0); CUERR;
    }
    cudaStreamWaitEvent(ws.hostFuncStream, ws.forkStreamEvent, 0); CUERR;
    // for(int i = 0; i < ws.numCopyBuffers; i++){
    //     cudaStreamWaitEvent(ws.copyStreams[i], ws.forkStreamEvent, 0); CUERR;
    // }
    

    auto& h_chardata_vec = ws.h_chardata_vec;
    auto& h_lengthdata_vec = ws.h_lengthdata_vec;
    auto& h_offsetdata_vec = ws.h_offsetdata_vec;
    auto& d_chardata_vec = ws.d_chardata_vec;
    auto& d_lengthdata_vec = ws.d_lengthdata_vec;
    auto& d_offsetdata_vec = ws.d_offsetdata_vec;
    auto& copyStreams = ws.copyStreams;




    size_t processedSequences = 0;
    for(size_t planIndex = 0; planIndex < batchPlan.size(); planIndex++){
        //std::cout << "planIndex " << planIndex << "\n";
        const auto& plan = batchPlan[planIndex];
        int currentBuffer = 0;
        int previousBuffer = 0;
        cudaStream_t H2DcopyStream = copyStreams[currentBuffer];
        if(!ws.singleBatchDBisOnGpu){
            currentBuffer = ws.copyBufferIndex;
            if(currentBuffer == 0){
                previousBuffer = numCopyBuffers - 1;
            }else{
                previousBuffer = (currentBuffer - 1);
            }
            ws.copyBufferIndex = (ws.copyBufferIndex+1) % numCopyBuffers;
            H2DcopyStream = copyStreams[currentBuffer];

            
            //can only overwrite device buffer if it is no longer in use on workstream
            cudaStreamWaitEvent(H2DcopyStream, ws.deviceBufferEvents[currentBuffer], 0); CUERR;

            if(useExtraThreadForBatchTransfer){
                cudaStreamWaitEvent(ws.hostFuncStream, ws.pinnedBufferEvents[currentBuffer]); CUERR;
                executePinnedCopyPlanWithHostCallback(plan, ws, currentBuffer, dbPartitions, ws.hostFuncStream);
                cudaEventRecord(ws.forkStreamEvent, ws.hostFuncStream); CUERR;
                cudaStreamWaitEvent(H2DcopyStream, ws.forkStreamEvent, 0);

                cudaMemcpyAsync(
                    d_chardata_vec[currentBuffer].data(),
                    h_chardata_vec[currentBuffer].data(),
                    plan.usedBytes,
                    H2D,
                    H2DcopyStream
                ); CUERR;
                cudaMemcpyAsync(
                    d_lengthdata_vec[currentBuffer].data(),
                    h_lengthdata_vec[currentBuffer].data(),
                    sizeof(size_t) * plan.usedSeq,
                    H2D,
                    H2DcopyStream
                ); CUERR;
                cudaMemcpyAsync(
                    d_offsetdata_vec[currentBuffer].data(),
                    h_offsetdata_vec[currentBuffer].data(),
                    sizeof(size_t) * (plan.usedSeq+1),
                    H2D,
                    H2DcopyStream
                ); CUERR;
            }else{
                //synchronize to avoid overwriting pinned buffer of target before it has been fully transferred
                cudaEventSynchronize(ws.pinnedBufferEvents[currentBuffer]); CUERR;
                //executePinnedCopyPlanSerial(plan, ws, currentBuffer, dbPartitions);

                executePinnedCopyPlanSerialAndTransferToGpu(
                    plan, ws, currentBuffer, dbPartitions, H2DcopyStream
                );
            }
            
            // cudaMemcpyAsync(
            //     d_chardata_vec[currentBuffer].data(),
            //     h_chardata_vec[currentBuffer].data(),
            //     plan.usedBytes,
            //     H2D,
            //     H2DcopyStream
            // ); CUERR;
            // cudaMemcpyAsync(
            //     d_lengthdata_vec[currentBuffer].data(),
            //     h_lengthdata_vec[currentBuffer].data(),
            //     sizeof(size_t) * plan.usedSeq,
            //     H2D,
            //     H2DcopyStream
            // ); CUERR;
            // cudaMemcpyAsync(
            //     d_offsetdata_vec[currentBuffer].data(),
            //     h_offsetdata_vec[currentBuffer].data(),
            //     sizeof(size_t) * (plan.usedSeq+1),
            //     H2D,
            //     H2DcopyStream
            // ); CUERR;
            cudaEventRecord(ws.pinnedBufferEvents[currentBuffer], H2DcopyStream); CUERR;
        }else{
            assert(batchPlan.size() == 1);
            //can only overwrite device buffer if it is no longer in use on workstream
            //for d_overflow_number
            cudaStreamWaitEvent(H2DcopyStream, ws.deviceBufferEvents[currentBuffer], 0); CUERR;
        }

        const char* const inputChars = d_chardata_vec[currentBuffer].data();
        const size_t* const inputOffsets = d_offsetdata_vec[currentBuffer].data();
        const size_t* const inputLengths = d_lengthdata_vec[currentBuffer].data();
        int* const d_overflow_number = ws.d_new_overflow_number.data() + currentBuffer;
        size_t* const d_overflow_positions = ws.d_new_overflow_positions_vec[currentBuffer].data();

        // std::cout << (void*)inputChars << " " << (void*)inputOffsets << " " << (void*)inputLengths << "\n";

        // assert(thrust::all_of(
        //     thrust::device,
        //     inputChars,
        //     inputChars + plan.usedBytes,
        //     []__host__ __device__ (char c){
        //         return int(c) <= 20;
        //     }
        // ));


        cudaMemsetAsync(d_overflow_number, 0, sizeof(int), H2DcopyStream);
        // cudaMemsetAsync(d_overflow_positions, 0, sizeof(size_t) * 10000, H2DcopyStream);
        
        // cudaDeviceSynchronize(); CUERR;

        //all data is ready for alignments. create dependencies for work streams
        cudaEventRecord(ws.forkStreamEvent, H2DcopyStream); CUERR;
        cudaStreamWaitEvent(ws.workStreamForTempUsage, ws.forkStreamEvent, 0); CUERR;
        for(auto& stream : ws.workStreamsWithoutTemp){
            cudaStreamWaitEvent(stream, ws.forkStreamEvent, 0); CUERR;
        }
        //wait for previous batch to finish
        cudaStreamWaitEvent(ws.workStreamForTempUsage, ws.deviceBufferEvents[previousBuffer], 0); CUERR;
        for(auto& stream : ws.workStreamsWithoutTemp){
            cudaStreamWaitEvent(stream, ws.deviceBufferEvents[previousBuffer], 0); CUERR;
        }






        const float gop = options.gop;
        const float gex = options.gex;

        auto runAlignmentKernels = [&](auto& d_scores, size_t* d_overflow_positions, int* d_overflow_number){
            auto nextWorkStreamNoTemp = [&](){
                ws.workstreamIndex = (ws.workstreamIndex + 1) % ws.numWorkStreamsWithoutTemp;
                return (cudaStream_t)ws.workStreamsWithoutTemp[ws.workstreamIndex];
            };
            std::vector<int> numPerPartitionPrefixSum(plan.h_numPerPartition.size());
            for(int i = 0; i < int(plan.h_numPerPartition.size())-1; i++){
                numPerPartitionPrefixSum[i+1] = numPerPartitionPrefixSum[i] + plan.h_numPerPartition[i];
            }
            //size_t exclPs = 0;
            //for(int lp = 0; lp < int(plan.h_partitionIds.size()); lp++){
            for(int lp = plan.h_partitionIds.size() - 1; lp >= 0; lp--){
                const int partId = plan.h_partitionIds[lp];
                const int numSeq = plan.h_numPerPartition[lp];
                const int start = numPerPartitionPrefixSum[lp];
                //std::cout << "partId " << partId << " numSeq " << numSeq << "\n";
                
                const size_t* const d_selectedPositions = ws.d_selectedPositions.data() + start;


                #if 0
                if (partId == 0){NW_local_affine_Protein_single_pass_half2<8, 8><<<(numSeq+63)/(2*8*4), 32*8, 0, nextWorkStreamNoTemp()>>>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex); CUERR }
                if (partId == 1){NW_local_affine_Protein_single_pass_half2<8, 16><<<(numSeq+63)/(2*8*4), 32*8, 0, nextWorkStreamNoTemp()>>>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex); CUERR }
                if (partId == 2){NW_local_affine_Protein_single_pass_half2<8, 24><<<(numSeq+63)/(2*8*4), 32*8, 0, nextWorkStreamNoTemp()>>>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex); CUERR }
                if (partId == 3){NW_local_affine_Protein_single_pass_half2<8, 32><<<(numSeq+63)/(2*8*4), 32*8, 0, nextWorkStreamNoTemp()>>>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex); CUERR }
                if (partId == 4){NW_local_affine_Protein_single_pass_half2<16, 20><<<(numSeq+31)/(2*8*2), 32*8, 0, nextWorkStreamNoTemp()>>>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex); CUERR }
                if (partId == 5){NW_local_affine_Protein_single_pass_half2<16, 24><<<(numSeq+31)/(2*8*2), 32*8, 0, nextWorkStreamNoTemp()>>>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex); CUERR }
                if (partId == 6){NW_local_affine_Protein_single_pass_half2<16, 28><<<(numSeq+31)/(2*8*2), 32*8, 0, nextWorkStreamNoTemp()>>>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex); CUERR }
                if (partId == 7){NW_local_affine_Protein_single_pass_half2<16, 32><<<(numSeq+31)/(2*8*2), 32*8, 0, nextWorkStreamNoTemp()>>>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex); CUERR }
                if (partId == 8){NW_local_affine_Protein_single_pass_half2<32, 18><<<(numSeq+15)/(2*8), 32*8, 0, nextWorkStreamNoTemp()>>>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex); CUERR }
                if (partId == 9){NW_local_affine_Protein_single_pass_half2<32, 20><<<(numSeq+15)/(2*8), 32*8, 0, nextWorkStreamNoTemp()>>>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex); CUERR }
                if (partId == 10){NW_local_affine_Protein_single_pass_half2<32, 24><<<(numSeq+15)/(2*8), 32*8, 0, nextWorkStreamNoTemp()>>>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex); CUERR }
                if (partId == 11){NW_local_affine_Protein_single_pass_half2<32, 28><<<(numSeq+15)/(2*8), 32*8, 0, nextWorkStreamNoTemp()>>>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex); CUERR }
                if (partId == 12){NW_local_affine_Protein_single_pass_half2<32, 32><<<(numSeq+15)/(2*8), 32*8, 0, nextWorkStreamNoTemp()>>>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex); CUERR }
                #endif

                #if 0
                if (partId == 0){call_NW_local_affine_Protein_single_pass_half2_new<256, 4, 16>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 1){call_NW_local_affine_Protein_single_pass_half2_new<256, 4, 32>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 2){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 12>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 3){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 16>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 4){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 20>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 5){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 24>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 6){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 28>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 7){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 32>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 8){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 18>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 9){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 20>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 10){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 24>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 11){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 24>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 12){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 26>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 13){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 28>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 14){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 30>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 15){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 32>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }

                #endif

                #if 1
                if (partId == 0){call_NW_local_affine_Protein_single_pass_half2_new<256, 8, 8>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 1){call_NW_local_affine_Protein_single_pass_half2_new<256, 8, 12>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 2){call_NW_local_affine_Protein_single_pass_half2_new<256, 8, 16>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 3){call_NW_local_affine_Protein_single_pass_half2_new<256, 8, 20>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 4){call_NW_local_affine_Protein_single_pass_half2_new<256, 8, 24>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 5){call_NW_local_affine_Protein_single_pass_half2_new<256, 8, 28>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 6){call_NW_local_affine_Protein_single_pass_half2_new<256, 8, 32>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 7){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 18>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 8){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 20>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 9){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 22>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 10){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 24>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 11){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 26>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 12){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 28>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 13){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 30>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 14){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 32>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 15){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 18>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 16){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 20>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 17){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 22>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 18){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 24>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 19){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 26>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 20){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 28>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 21){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 30>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 22){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 32>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 23){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 34>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                if (partId == 24){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 36>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }

                #endif

                //if (partId > 15 && partId < 44){
                //if (partId == 16){
                if(partId == numLengthPartitions - 2){
                    constexpr int blocksize = 32 * 8;
                    constexpr int groupsize = 32;
                    constexpr int groupsPerBlock = blocksize / groupsize;
                    constexpr int alignmentsPerGroup = 2;
                    constexpr int alignmentsPerBlock = groupsPerBlock * alignmentsPerGroup;
                    
                    const size_t tempBytesPerBlockPerBuffer = sizeof(__half2) * alignmentsPerBlock * queryLength;

                    const size_t maxNumBlocks = ws.numTempBytes / (tempBytesPerBlockPerBuffer * 2);
                    const size_t maxSubjectsPerIteration = std::min(maxNumBlocks * alignmentsPerBlock, size_t(numSeq));

                    const size_t numBlocksPerIteration = SDIV(maxSubjectsPerIteration, alignmentsPerBlock);
                    const size_t requiredTempBytes = tempBytesPerBlockPerBuffer * 2 * numBlocksPerIteration;

                    __half2* d_temp = (__half2*)ws.d_tempStorageHE.data();
                    __half2* d_tempHcol2 = d_temp;
                    __half2* d_tempEcol2 = (__half2*)(((char*)d_tempHcol2) + requiredTempBytes / 2);

                    const int numIters =  SDIV(numSeq, maxSubjectsPerIteration);

                    for(int iter = 0; iter < numIters; iter++){
                        const size_t begin = iter * maxSubjectsPerIteration;
                        const size_t end = iter < numIters-1 ? (iter+1) * maxSubjectsPerIteration : numSeq;
                        const size_t num = end - begin;                      

                        cudaMemsetAsync(d_temp, 0, requiredTempBytes, ws.workStreamForTempUsage); CUERR;
                        //std::cout << "iter " << iter << " / " << numIters << " gridsize " << SDIV(num, alignmentsPerBlock) << "\n";

                        //cudaDeviceSynchronize(); CUERR;
                        
                        //NW_local_affine_Protein_many_pass_half2<groupsize, 12><<<SDIV(num, alignmentsPerBlock), blocksize, 0, ws.workStreamForTempUsage>>>(
                        NW_local_affine_Protein_many_pass_half2_new<groupsize, 22><<<SDIV(num, alignmentsPerBlock), blocksize, 0, ws.workStreamForTempUsage>>>(
                            inputChars, 
                            d_scores, 
                            d_tempHcol2, 
                            d_tempEcol2, 
                            inputOffsets, 
                            inputLengths, 
                            d_selectedPositions + begin, 
                            num, 
                            d_overflow_positions, 
                            d_overflow_number, 
                            1, 
                            queryLength, 
                            gop, 
                            gex
                        ); CUERR
                    }
                }

                //if (partId == 44){
                //if (partId == 17){
                if(partId == numLengthPartitions - 1){
                    const size_t tempBytesPerSubjectPerBuffer = sizeof(short2) * SDIV(queryLength,32) * 32;
                    const size_t maxSubjectsPerIteration = std::min(size_t(numSeq), ws.numTempBytes / (tempBytesPerSubjectPerBuffer * 2));

                    short2* d_temp = (short2*)ws.d_tempStorageHE.data();
                    short2* d_tempHcol2 = d_temp;
                    short2* d_tempEcol2 = (short2*)(((char*)d_tempHcol2) + maxSubjectsPerIteration * tempBytesPerSubjectPerBuffer);

                    const int numIters =  SDIV(numSeq, maxSubjectsPerIteration);
                    for(int iter = 0; iter < numIters; iter++){
                        const size_t begin = iter * maxSubjectsPerIteration;
                        const size_t end = iter < numIters-1 ? (iter+1) * maxSubjectsPerIteration : numSeq;
                        const size_t num = end - begin;

                        cudaMemsetAsync(d_temp, 0, tempBytesPerSubjectPerBuffer * 2 * num, ws.workStreamForTempUsage); CUERR;

                        //cudaDeviceSynchronize(); CUERR;

                        //NW_local_affine_read4_float_query_Protein<32, 32><<<num, 32, 0, ws.workStreamForTempUsage>>>(
                        NW_local_affine_read4_float_query_Protein_new<32><<<num, 32, 0, ws.workStreamForTempUsage>>>(
                            inputChars, 
                            d_scores, 
                            d_tempHcol2, 
                            d_tempEcol2, 
                            inputOffsets, 
                            inputLengths, 
                            d_selectedPositions + begin, 
                            queryLength, 
                            gop, 
                            gex
                        ); CUERR 
                    }
                }

                // std::cout << "partId = " << partId << ",numSeq " << numSeq << "\n";
                //cudaDeviceSynchronize(); CUERR;

                //exclPs += numSeq;
            }
        };

        
        auto maxReduceArray = ws.getMaxReduceArray(processedSequences);
        runAlignmentKernels(maxReduceArray, d_overflow_positions, d_overflow_number);


        //alignments are done in workstreams. now, join all workstreams into workStreamForTempUsage to process overflow alignments
        for(auto& stream : ws.workStreamsWithoutTemp){
            cudaEventRecord(ws.forkStreamEvent, stream); CUERR;
            cudaStreamWaitEvent(ws.workStreamForTempUsage, ws.forkStreamEvent, 0); CUERR;    
        }
        // int numOverflow = 0;
        // cudaMemcpyAsync(&numOverflow, d_overflow_number, sizeof(int), D2H, ws.workStreamForTempUsage); CUERR;
        // cudaStreamSynchronize(ws.workStreamForTempUsage); CUERR;
        // if(numOverflow > 0){
        //     std::cout << "numOverflow " << numOverflow << "\n";
        //     const size_t tempBytesPerSubjectPerBuffer = sizeof(short2) * SDIV(queryLength,32) * 32;
        //     const size_t maxSubjectsPerIteration = std::min(size_t(numOverflow), ws.numTempBytes / (tempBytesPerSubjectPerBuffer * 2));

        //     short2* d_temp = (short2*)ws.d_tempStorageHE.data();
        //     short2* d_tempHcol2 = d_temp;
        //     short2* d_tempEcol2 = (short2*)(((char*)d_tempHcol2) + maxSubjectsPerIteration * tempBytesPerSubjectPerBuffer);

        //     const int numIters =  SDIV(numOverflow, maxSubjectsPerIteration);
        //     for(int iter = 0; iter < numIters; iter++){
        //         const size_t begin = iter * maxSubjectsPerIteration;
        //         const size_t end = iter < numIters-1 ? (iter+1) * maxSubjectsPerIteration : numOverflow;
        //         const size_t num = end - begin;

        //         cudaMemsetAsync(d_temp, 0, tempBytesPerSubjectPerBuffer * 2 * num, ws.workStreamForTempUsage); CUERR;

        //         //NW_local_affine_read4_float_query_Protein<32, 32><<<num, 32, 0, ws.workStreamForTempUsage>>>(
        //         NW_local_affine_read4_float_query_Protein_new<12><<<num, 32, 0, ws.workStreamForTempUsage>>>(
        //             inputChars, 
        //             maxReduceArray, 
        //             d_tempHcol2, 
        //             d_tempEcol2, 
        //             inputOffsets, 
        //             inputLengths, 
        //             d_overflow_positions + begin, 
        //             queryLength, 
        //             gop, 
        //             gex
        //         ); CUERR 
        //     }
        // }

        {
            //std::cerr << "overflow processing\n";
            short2* d_temp = (short2*)ws.d_tempStorageHE.data();

            //launch_process_overflow_alignments_kernel_NW_local_affine_read4_float_query_Protein<32,32><<<1,1,0, ws.workStreamForTempUsage>>>(
            launch_process_overflow_alignments_kernel_NW_local_affine_read4_float_query_Protein_new<32><<<1,1,0, ws.workStreamForTempUsage>>>(
                d_overflow_number,
                d_temp, 
                ws.numTempBytes,
                inputChars, 
                maxReduceArray, 
                inputOffsets, 
                inputLengths, 
                d_overflow_positions, 
                queryLength, gop, gex
            ); CUERR
            //cudaDeviceSynchronize(); CUERR;
        }

        //after processing overflow alignments, the batch is done and its data can be resused
        cudaEventRecord(ws.deviceBufferEvents[currentBuffer], ws.workStreamForTempUsage); CUERR;

        //let other workstreams depend on temp usage stream
        for(auto& stream : ws.workStreamsWithoutTemp){
            cudaStreamWaitEvent(ws.workStreamForTempUsage, ws.deviceBufferEvents[currentBuffer], 0); CUERR;    
        }

        processedSequences += plan.usedSeq;
    }

    //create dependency for mainStream
    cudaEventRecord(ws.forkStreamEvent, ws.workStreamForTempUsage); CUERR;
    cudaStreamWaitEvent(mainStream, ws.forkStreamEvent, 0); CUERR;

    for(auto& stream : ws.workStreamsWithoutTemp){
        cudaEventRecord(ws.forkStreamEvent, stream); CUERR;
        cudaStreamWaitEvent(mainStream, ws.forkStreamEvent, 0); CUERR;
    }

    // for(auto& stream : ws.copyStreams){
    //     cudaEventRecord(ws.forkStreamEvent, stream); CUERR;
    //     cudaStreamWaitEvent(mainStream, ws.forkStreamEvent, 0); CUERR;
    // }

    cudaEventRecord(ws.forkStreamEvent, ws.hostFuncStream); CUERR;
    cudaStreamWaitEvent(mainStream, ws.forkStreamEvent, 0); CUERR;

}

#endif








void processQueryOnGpus(
    const std::vector<int>& deviceIds,
    const std::vector<cudaStream_t>& gpuStreams,
    const std::vector<std::unique_ptr<GpuWorkingSet>>& workingSets,
    const std::vector<std::vector<DBdataView>>& dbPartitionsPerGpu,
    const std::vector<std::vector<int>>& lengthPartitionIdsPerGpu, // dbPartitions[i] belongs to the length partition lengthPartitionIds[i]
    const std::vector<std::vector<DeviceBatchCopyToPinnedPlan>>& batchPlanPerGpu,
    const std::vector<size_t>& numberOfSequencesPerGpu,
    const std::vector<size_t>& numberOfSequencesPerGpuPrefixSum,
    const char* query, // may be host or device
    const int queryLength,
    bool useExtraThreadForBatchTransfer,
    const CudaSW4Options& options
){
    constexpr auto boundaries = getLengthPartitionBoundaries();
    constexpr int numLengthPartitions = boundaries.size();
    const int numGpus = deviceIds.size();

    size_t totalNumberOfSequencesToProcess = std::reduce(numberOfSequencesPerGpu.begin(), numberOfSequencesPerGpu.end());
    
    size_t totalNumberOfProcessedSequences = 0;
    std::vector<size_t> processedSequencesPerGpu(numGpus, 0);

    std::vector<size_t> processedBatchesPerGpu(numGpus, 0);

    for(int gpu = 0; gpu < numGpus; gpu++){
        cudaSetDevice(deviceIds[gpu]); CUERR;
        auto& ws = *workingSets[gpu];

        cudaMemcpyAsync(ws.devChars.data(), query, queryLength, cudaMemcpyDefault, gpuStreams[gpu]); CUERR
        NW_convert_protein_single<<<SDIV(queryLength, 128), 128, 0, gpuStreams[gpu]>>>(ws.devChars.data(), queryLength); CUERR

        cudaMemcpyToSymbolAsync(constantQuery4 ,ws.Fillchar.data(), 512*16, 0, cudaMemcpyDeviceToDevice, gpuStreams[gpu]); CUERR
        cudaMemcpyToSymbolAsync(constantQuery4, ws.devChars.data(), queryLength, 0, cudaMemcpyDeviceToDevice, gpuStreams[gpu]); CUERR

        ws.resetMaxReduceArray(gpuStreams[gpu]);

        //create dependency on mainStream
        cudaEventRecord(ws.forkStreamEvent, gpuStreams[gpu]); CUERR;
        cudaStreamWaitEvent(ws.workStreamForTempUsage, ws.forkStreamEvent, 0); CUERR;
        for(auto& stream : ws.workStreamsWithoutTemp){
            cudaStreamWaitEvent(stream, ws.forkStreamEvent, 0); CUERR;
        }
        cudaStreamWaitEvent(ws.hostFuncStream, ws.forkStreamEvent, 0); CUERR;
    }

    const float gop = options.gop;
    const float gex = options.gex;

    
    while(totalNumberOfProcessedSequences < totalNumberOfSequencesToProcess){
        //std::cout << "processed batches " << processedBatchesPerGpu[0] << ", processed sequences " << totalNumberOfProcessedSequences << "\n";
        //upload batch
        for(int gpu = 0; gpu < numGpus; gpu++){
            if(processedBatchesPerGpu[gpu] < batchPlanPerGpu[gpu].size()){
                cudaSetDevice(deviceIds[gpu]); CUERR;
                auto& ws = *workingSets[gpu];
                const auto& plan = batchPlanPerGpu[gpu][processedBatchesPerGpu[gpu]];
                int currentBuffer = 0;
                int previousBuffer = 0;
                cudaStream_t H2DcopyStream = ws.copyStreams[currentBuffer];
                if(!ws.singleBatchDBisOnGpu){
                    currentBuffer = ws.copyBufferIndex;
                    if(currentBuffer == 0){
                        previousBuffer = ws.numCopyBuffers - 1;
                    }else{
                        previousBuffer = (currentBuffer - 1);
                    }                    
                    H2DcopyStream = ws.copyStreams[currentBuffer];

                    //std::cout << "copy to buffer " << currentBuffer << "\n";
                    
                    //can only overwrite device buffer if it is no longer in use on workstream
                    cudaStreamWaitEvent(H2DcopyStream, ws.deviceBufferEvents[currentBuffer], 0); CUERR;

                    if(useExtraThreadForBatchTransfer){
                        cudaStreamWaitEvent(ws.hostFuncStream, ws.pinnedBufferEvents[currentBuffer]); CUERR;
                        executePinnedCopyPlanWithHostCallback(plan, ws, currentBuffer, dbPartitionsPerGpu[gpu], ws.hostFuncStream);
                        cudaEventRecord(ws.forkStreamEvent, ws.hostFuncStream); CUERR;
                        cudaStreamWaitEvent(H2DcopyStream, ws.forkStreamEvent, 0);

                        cudaMemcpyAsync(
                            ws.d_chardata_vec[currentBuffer].data(),
                            ws.h_chardata_vec[currentBuffer].data(),
                            plan.usedBytes,
                            H2D,
                            H2DcopyStream
                        ); CUERR;
                        cudaMemcpyAsync(
                            ws.d_lengthdata_vec[currentBuffer].data(),
                            ws.h_lengthdata_vec[currentBuffer].data(),
                            sizeof(size_t) * plan.usedSeq,
                            H2D,
                            H2DcopyStream
                        ); CUERR;
                        cudaMemcpyAsync(
                            ws.d_offsetdata_vec[currentBuffer].data(),
                            ws.h_offsetdata_vec[currentBuffer].data(),
                            sizeof(size_t) * (plan.usedSeq+1),
                            H2D,
                            H2DcopyStream
                        ); CUERR;
                    }else{
                        //synchronize to avoid overwriting pinned buffer of target before it has been fully transferred
                        cudaEventSynchronize(ws.pinnedBufferEvents[currentBuffer]); CUERR;
                        //executePinnedCopyPlanSerial(plan, ws, currentBuffer, dbPartitionsPerGpu[gpu]);

                        executePinnedCopyPlanSerialAndTransferToGpu(
                            plan, ws, currentBuffer, dbPartitionsPerGpu[gpu], H2DcopyStream
                        );
                    }
                    
                    cudaEventRecord(ws.pinnedBufferEvents[currentBuffer], H2DcopyStream); CUERR;
                }else{
                    assert(batchPlanPerGpu[gpu].size() == 1);
                    //can only overwrite device buffer if it is no longer in use on workstream
                    //for d_overflow_number
                    cudaStreamWaitEvent(H2DcopyStream, ws.deviceBufferEvents[currentBuffer], 0); CUERR;
                }
                // const char* const inputChars = d_chardata_vec[currentBuffer].data();
                // const size_t* const inputOffsets = d_offsetdata_vec[currentBuffer].data();
                // const size_t* const inputLengths = d_lengthdata_vec[currentBuffer].data();
                int* const d_overflow_number = ws.d_new_overflow_number.data() + currentBuffer;
                //size_t* const d_overflow_positions = ws.d_new_overflow_positions_vec[currentBuffer].data();


                cudaMemsetAsync(d_overflow_number, 0, sizeof(int), H2DcopyStream);
                // cudaMemsetAsync(d_overflow_positions, 0, sizeof(size_t) * 10000, H2DcopyStream);
                
                //all data is ready for alignments. create dependencies for work streams
                cudaEventRecord(ws.forkStreamEvent, H2DcopyStream); CUERR;
                cudaStreamWaitEvent(ws.workStreamForTempUsage, ws.forkStreamEvent, 0); CUERR;
                for(auto& stream : ws.workStreamsWithoutTemp){
                    cudaStreamWaitEvent(stream, ws.forkStreamEvent, 0); CUERR;
                }
                //wait for previous batch to finish
                cudaStreamWaitEvent(ws.workStreamForTempUsage, ws.deviceBufferEvents[previousBuffer], 0); CUERR;
                for(auto& stream : ws.workStreamsWithoutTemp){
                    cudaStreamWaitEvent(stream, ws.deviceBufferEvents[previousBuffer], 0); CUERR;
                }
            }
        }

        //launch alignment kernels
        for(int gpu = 0; gpu < numGpus; gpu++){
            if(processedBatchesPerGpu[gpu] < batchPlanPerGpu[gpu].size()){
                cudaSetDevice(deviceIds[gpu]); CUERR;
                auto& ws = *workingSets[gpu];
                const auto& plan = batchPlanPerGpu[gpu][processedBatchesPerGpu[gpu]];
                int currentBuffer = 0;
                int previousBuffer = 0;
                cudaStream_t H2DcopyStream = ws.copyStreams[currentBuffer];
                if(!ws.singleBatchDBisOnGpu){
                    currentBuffer = ws.copyBufferIndex;
                    H2DcopyStream = ws.copyStreams[currentBuffer];
                }

                const char* const inputChars = ws.d_chardata_vec[currentBuffer].data();
                const size_t* const inputOffsets = ws.d_offsetdata_vec[currentBuffer].data();
                const size_t* const inputLengths = ws.d_lengthdata_vec[currentBuffer].data();
                int* const d_overflow_number = ws.d_new_overflow_number.data() + currentBuffer;
                size_t* const d_overflow_positions = ws.d_new_overflow_positions_vec[currentBuffer].data();

                auto runAlignmentKernels = [&](auto& d_scores, size_t* d_overflow_positions, int* d_overflow_number){
                    auto nextWorkStreamNoTemp = [&](){
                        ws.workstreamIndex = (ws.workstreamIndex + 1) % ws.numWorkStreamsWithoutTemp;
                        return (cudaStream_t)ws.workStreamsWithoutTemp[ws.workstreamIndex];
                    };
                    std::vector<int> numPerPartitionPrefixSum(plan.h_numPerPartition.size());
                    for(int i = 0; i < int(plan.h_numPerPartition.size())-1; i++){
                        numPerPartitionPrefixSum[i+1] = numPerPartitionPrefixSum[i] + plan.h_numPerPartition[i];
                    }
                    //size_t exclPs = 0;
                    //for(int lp = 0; lp < int(plan.h_partitionIds.size()); lp++){
                    for(int lp = plan.h_partitionIds.size() - 1; lp >= 0; lp--){
                        const int partId = plan.h_partitionIds[lp];
                        const int numSeq = plan.h_numPerPartition[lp];
                        const int start = numPerPartitionPrefixSum[lp];
                        //std::cout << "partId " << partId << " numSeq " << numSeq << "\n";
                        
                        const size_t* const d_selectedPositions = ws.d_selectedPositions.data() + start;


                        #if 0
                        if (partId == 0){NW_local_affine_Protein_single_pass_half2<8, 8><<<(numSeq+63)/(2*8*4), 32*8, 0, nextWorkStreamNoTemp()>>>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex); CUERR }
                        if (partId == 1){NW_local_affine_Protein_single_pass_half2<8, 16><<<(numSeq+63)/(2*8*4), 32*8, 0, nextWorkStreamNoTemp()>>>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex); CUERR }
                        if (partId == 2){NW_local_affine_Protein_single_pass_half2<8, 24><<<(numSeq+63)/(2*8*4), 32*8, 0, nextWorkStreamNoTemp()>>>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex); CUERR }
                        if (partId == 3){NW_local_affine_Protein_single_pass_half2<8, 32><<<(numSeq+63)/(2*8*4), 32*8, 0, nextWorkStreamNoTemp()>>>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex); CUERR }
                        if (partId == 4){NW_local_affine_Protein_single_pass_half2<16, 20><<<(numSeq+31)/(2*8*2), 32*8, 0, nextWorkStreamNoTemp()>>>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex); CUERR }
                        if (partId == 5){NW_local_affine_Protein_single_pass_half2<16, 24><<<(numSeq+31)/(2*8*2), 32*8, 0, nextWorkStreamNoTemp()>>>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex); CUERR }
                        if (partId == 6){NW_local_affine_Protein_single_pass_half2<16, 28><<<(numSeq+31)/(2*8*2), 32*8, 0, nextWorkStreamNoTemp()>>>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex); CUERR }
                        if (partId == 7){NW_local_affine_Protein_single_pass_half2<16, 32><<<(numSeq+31)/(2*8*2), 32*8, 0, nextWorkStreamNoTemp()>>>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex); CUERR }
                        if (partId == 8){NW_local_affine_Protein_single_pass_half2<32, 18><<<(numSeq+15)/(2*8), 32*8, 0, nextWorkStreamNoTemp()>>>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex); CUERR }
                        if (partId == 9){NW_local_affine_Protein_single_pass_half2<32, 20><<<(numSeq+15)/(2*8), 32*8, 0, nextWorkStreamNoTemp()>>>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex); CUERR }
                        if (partId == 10){NW_local_affine_Protein_single_pass_half2<32, 24><<<(numSeq+15)/(2*8), 32*8, 0, nextWorkStreamNoTemp()>>>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex); CUERR }
                        if (partId == 11){NW_local_affine_Protein_single_pass_half2<32, 28><<<(numSeq+15)/(2*8), 32*8, 0, nextWorkStreamNoTemp()>>>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex); CUERR }
                        if (partId == 12){NW_local_affine_Protein_single_pass_half2<32, 32><<<(numSeq+15)/(2*8), 32*8, 0, nextWorkStreamNoTemp()>>>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex); CUERR }
                        #endif

                        #if 0
                        if (partId == 0){call_NW_local_affine_Protein_single_pass_half2_new<256, 4, 16>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        if (partId == 1){call_NW_local_affine_Protein_single_pass_half2_new<256, 4, 32>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        if (partId == 2){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 12>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        if (partId == 3){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 16>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        if (partId == 4){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 20>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        if (partId == 5){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 24>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        if (partId == 6){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 28>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        if (partId == 7){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 32>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        if (partId == 8){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 18>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        if (partId == 9){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 20>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        if (partId == 10){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 24>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        if (partId == 11){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 24>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        if (partId == 12){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 26>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        if (partId == 13){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 28>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        if (partId == 14){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 30>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        if (partId == 15){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 32>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }

                        #endif

                        #if 1
                        if (partId == 0){call_NW_local_affine_Protein_single_pass_half2_new<256, 8, 8>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        if (partId == 1){call_NW_local_affine_Protein_single_pass_half2_new<256, 8, 12>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        if (partId == 2){call_NW_local_affine_Protein_single_pass_half2_new<256, 8, 16>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        if (partId == 3){call_NW_local_affine_Protein_single_pass_half2_new<256, 8, 20>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        if (partId == 4){call_NW_local_affine_Protein_single_pass_half2_new<256, 8, 24>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        if (partId == 5){call_NW_local_affine_Protein_single_pass_half2_new<256, 8, 28>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        if (partId == 6){call_NW_local_affine_Protein_single_pass_half2_new<256, 8, 32>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        if (partId == 7){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 18>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        if (partId == 8){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 20>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        if (partId == 9){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 22>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        if (partId == 10){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 24>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        if (partId == 11){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 26>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        if (partId == 12){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 28>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        if (partId == 13){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 30>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        if (partId == 14){call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 32>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        if (partId == 15){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 18>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        if (partId == 16){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 20>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        if (partId == 17){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 22>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        if (partId == 18){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 24>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        if (partId == 19){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 26>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        if (partId == 20){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 28>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        if (partId == 21){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 30>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        if (partId == 22){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 32>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        if (partId == 23){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 34>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        if (partId == 24){call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 36>(inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }

                        #endif

                        //if (partId > 15 && partId < 44){
                        //if (partId == 16){
                        if(partId == numLengthPartitions - 2){
                            constexpr int blocksize = 32 * 8;
                            constexpr int groupsize = 32;
                            constexpr int groupsPerBlock = blocksize / groupsize;
                            constexpr int alignmentsPerGroup = 2;
                            constexpr int alignmentsPerBlock = groupsPerBlock * alignmentsPerGroup;
                            
                            const size_t tempBytesPerBlockPerBuffer = sizeof(__half2) * alignmentsPerBlock * queryLength;

                            const size_t maxNumBlocks = ws.numTempBytes / (tempBytesPerBlockPerBuffer * 2);
                            const size_t maxSubjectsPerIteration = std::min(maxNumBlocks * alignmentsPerBlock, size_t(numSeq));

                            const size_t numBlocksPerIteration = SDIV(maxSubjectsPerIteration, alignmentsPerBlock);
                            const size_t requiredTempBytes = tempBytesPerBlockPerBuffer * 2 * numBlocksPerIteration;

                            __half2* d_temp = (__half2*)ws.d_tempStorageHE.data();
                            __half2* d_tempHcol2 = d_temp;
                            __half2* d_tempEcol2 = (__half2*)(((char*)d_tempHcol2) + requiredTempBytes / 2);

                            const int numIters =  SDIV(numSeq, maxSubjectsPerIteration);

                            for(int iter = 0; iter < numIters; iter++){
                                const size_t begin = iter * maxSubjectsPerIteration;
                                const size_t end = iter < numIters-1 ? (iter+1) * maxSubjectsPerIteration : numSeq;
                                const size_t num = end - begin;                      

                                cudaMemsetAsync(d_temp, 0, requiredTempBytes, ws.workStreamForTempUsage); CUERR;
                                //std::cout << "iter " << iter << " / " << numIters << " gridsize " << SDIV(num, alignmentsPerBlock) << "\n";

                                //cudaDeviceSynchronize(); CUERR;
                                
                                //NW_local_affine_Protein_many_pass_half2<groupsize, 12><<<SDIV(num, alignmentsPerBlock), blocksize, 0, ws.workStreamForTempUsage>>>(
                                NW_local_affine_Protein_many_pass_half2_new<groupsize, 22><<<SDIV(num, alignmentsPerBlock), blocksize, 0, ws.workStreamForTempUsage>>>(
                                    inputChars, 
                                    d_scores, 
                                    d_tempHcol2, 
                                    d_tempEcol2, 
                                    inputOffsets, 
                                    inputLengths, 
                                    d_selectedPositions + begin, 
                                    num, 
                                    d_overflow_positions, 
                                    d_overflow_number, 
                                    1, 
                                    queryLength, 
                                    gop, 
                                    gex
                                ); CUERR
                            }
                        }

                        //if (partId == 44){
                        //if (partId == 17){
                        if(partId == numLengthPartitions - 1){
                            const size_t tempBytesPerSubjectPerBuffer = sizeof(short2) * SDIV(queryLength,32) * 32;
                            const size_t maxSubjectsPerIteration = std::min(size_t(numSeq), ws.numTempBytes / (tempBytesPerSubjectPerBuffer * 2));

                            short2* d_temp = (short2*)ws.d_tempStorageHE.data();
                            short2* d_tempHcol2 = d_temp;
                            short2* d_tempEcol2 = (short2*)(((char*)d_tempHcol2) + maxSubjectsPerIteration * tempBytesPerSubjectPerBuffer);

                            const int numIters =  SDIV(numSeq, maxSubjectsPerIteration);
                            for(int iter = 0; iter < numIters; iter++){
                                const size_t begin = iter * maxSubjectsPerIteration;
                                const size_t end = iter < numIters-1 ? (iter+1) * maxSubjectsPerIteration : numSeq;
                                const size_t num = end - begin;

                                cudaMemsetAsync(d_temp, 0, tempBytesPerSubjectPerBuffer * 2 * num, ws.workStreamForTempUsage); CUERR;

                                //cudaDeviceSynchronize(); CUERR;

                                NW_local_affine_read4_float_query_Protein<32, 32><<<num, 32, 0, ws.workStreamForTempUsage>>>(
                                //NW_local_affine_read4_float_query_Protein_new<12><<<num, 32, 0, ws.workStreamForTempUsage>>>(
                                    inputChars, 
                                    d_scores, 
                                    d_tempHcol2, 
                                    d_tempEcol2, 
                                    inputOffsets, 
                                    inputLengths, 
                                    d_selectedPositions + begin, 
                                    queryLength, 
                                    gop, 
                                    gex
                                ); CUERR 
                            }
                        }

                        // std::cout << "partId = " << partId << ",numSeq " << numSeq << "\n";
                        //cudaDeviceSynchronize(); CUERR;

                        //exclPs += numSeq;
                    }
                };

                
                auto maxReduceArray = ws.getMaxReduceArray(numberOfSequencesPerGpuPrefixSum[gpu] + processedSequencesPerGpu[gpu]);
                //runAlignmentKernels(ws.devAlignmentScoresFloat.data() + processedSequencesPerGpu[gpu], d_overflow_positions, d_overflow_number);
                runAlignmentKernels(maxReduceArray, d_overflow_positions, d_overflow_number);


                //alignments are done in workstreams. now, join all workstreams into workStreamForTempUsage to process overflow alignments
                for(auto& stream : ws.workStreamsWithoutTemp){
                    cudaEventRecord(ws.forkStreamEvent, stream); CUERR;
                    cudaStreamWaitEvent(ws.workStreamForTempUsage, ws.forkStreamEvent, 0); CUERR;    
                }
            }
        }

        //process overflow alignments and finish processing of batch
        for(int gpu = 0; gpu < numGpus; gpu++){
            if(processedBatchesPerGpu[gpu] < batchPlanPerGpu[gpu].size()){
                cudaSetDevice(deviceIds[gpu]); CUERR;
                auto& ws = *workingSets[gpu];
                const auto& plan = batchPlanPerGpu[gpu][processedBatchesPerGpu[gpu]];
                int currentBuffer = 0;
                cudaStream_t H2DcopyStream = ws.copyStreams[currentBuffer];
                if(!ws.singleBatchDBisOnGpu){
                    currentBuffer = ws.copyBufferIndex;
                    H2DcopyStream = ws.copyStreams[currentBuffer];
                }

                const char* const inputChars = ws.d_chardata_vec[currentBuffer].data();
                const size_t* const inputOffsets = ws.d_offsetdata_vec[currentBuffer].data();
                const size_t* const inputLengths = ws.d_lengthdata_vec[currentBuffer].data();
                int* const d_overflow_number = ws.d_new_overflow_number.data() + currentBuffer;
                size_t* const d_overflow_positions = ws.d_new_overflow_positions_vec[currentBuffer].data();

                auto maxReduceArray = ws.getMaxReduceArray(numberOfSequencesPerGpuPrefixSum[gpu] + processedSequencesPerGpu[gpu]);

                // cudaMemcpyAsync(h_overflow_number, d_overflow_number, sizeof(int), D2H, ws.workStreamForTempUsage); CUERR;
                // cudaStreamSynchronize(ws.workStreamForTempUsage); CUERR;
                // if(*h_overflow_number > 0){
                //     //std::cout << "numOverflow " << *h_overflow_number << "\n";
                //     const size_t tempBytesPerSubjectPerBuffer = sizeof(short2) * SDIV(queryLength,32) * 32;
                //     const size_t maxSubjectsPerIteration = std::min(size_t(*h_overflow_number), ws.numTempBytes / (tempBytesPerSubjectPerBuffer * 2));

                //     short2* d_temp = (short2*)ws.d_tempStorageHE.data();
                //     short2* d_tempHcol2 = d_temp;
                //     short2* d_tempEcol2 = (short2*)(((char*)d_tempHcol2) + maxSubjectsPerIteration * tempBytesPerSubjectPerBuffer);

                //     const int numIters =  SDIV(*h_overflow_number, maxSubjectsPerIteration);
                //     //std::cout << "numIters " << numIters << "\n";
                //     for(int iter = 0; iter < numIters; iter++){
                //         const size_t begin = iter * maxSubjectsPerIteration;
                //         const size_t end = iter < numIters-1 ? (iter+1) * maxSubjectsPerIteration : *h_overflow_number;
                //         const size_t num = end - begin;

                //         cudaMemsetAsync(d_temp, 0, tempBytesPerSubjectPerBuffer * 2 * num, ws.workStreamForTempUsage); CUERR;

                //         //NW_local_affine_read4_float_query_Protein<32, 32><<<num, 32, 0, ws.workStreamForTempUsage>>>(
                //         NW_local_affine_read4_float_query_Protein_new<12><<<num, 32, 0, ws.workStreamForTempUsage>>>(
                //             inputChars, 
                //             //ws.devAlignmentScoresFloat.data() + processedSequences, 
                //             maxReduceArray,
                //             d_tempHcol2, 
                //             d_tempEcol2, 
                //             inputOffsets, 
                //             inputLengths, 
                //             d_overflow_positions + begin, 
                //             queryLength, 
                //             gop, 
                //             gex
                //         ); CUERR 
                //     }
                // }

                {
                    //std::cerr << "overflow processing\n";
                    short2* d_temp = (short2*)ws.d_tempStorageHE.data();

                    launch_process_overflow_alignments_kernel_NW_local_affine_read4_float_query_Protein<32,32><<<1,1,0, ws.workStreamForTempUsage>>>(
                    //launch_process_overflow_alignments_kernel_NW_local_affine_read4_float_query_Protein_new<12><<<1,1,0, ws.workStreamForTempUsage>>>(
                        d_overflow_number,
                        d_temp, 
                        ws.numTempBytes,
                        inputChars, 
                        //ws.devAlignmentScoresFloat.data() + processedSequencesPerGpu[gpu], 
                        maxReduceArray,
                        inputOffsets, 
                        inputLengths, 
                        d_overflow_positions, 
                        queryLength, gop, gex
                    ); CUERR
                }

                //after processing overflow alignments, the batch is done and its data can be resused
                cudaEventRecord(ws.deviceBufferEvents[currentBuffer], ws.workStreamForTempUsage); CUERR;

                //let other workstreams depend on temp usage stream
                for(auto& stream : ws.workStreamsWithoutTemp){
                    cudaStreamWaitEvent(ws.workStreamForTempUsage, ws.deviceBufferEvents[currentBuffer], 0); CUERR;    
                }

                ws.copyBufferIndex = (ws.copyBufferIndex+1) % ws.numCopyBuffers;
                processedBatchesPerGpu[gpu]++;
                processedSequencesPerGpu[gpu] += plan.usedSeq;
                totalNumberOfProcessedSequences += plan.usedSeq;
            }
        }
    }

    for(int gpu = 0; gpu < numGpus; gpu++){
        cudaSetDevice(deviceIds[gpu]); CUERR;
        auto& ws = *workingSets[gpu];
        //create dependency for gpuStreams[gpu]
        cudaEventRecord(ws.forkStreamEvent, ws.workStreamForTempUsage); CUERR;
        cudaStreamWaitEvent(gpuStreams[gpu], ws.forkStreamEvent, 0); CUERR;

        for(auto& stream : ws.workStreamsWithoutTemp){
            cudaEventRecord(ws.forkStreamEvent, stream); CUERR;
            cudaStreamWaitEvent(gpuStreams[gpu], ws.forkStreamEvent, 0); CUERR;
        }

        // for(auto& stream : ws.copyStreams){
        //     cudaEventRecord(ws.forkStreamEvent, stream); CUERR;
        //     cudaStreamWaitEvent(gpuStreams[gpu], ws.forkStreamEvent, 0); CUERR;
        // }

        cudaEventRecord(ws.forkStreamEvent, ws.hostFuncStream); CUERR;
        cudaStreamWaitEvent(gpuStreams[gpu], ws.forkStreamEvent, 0); CUERR;
    }
    

}





bool parseArgs(int argc, char** argv, CudaSW4Options& options){

    auto parseMemoryString = [](const std::string& string) -> std::size_t{
        if(string.length() > 0){
            std::size_t factor = 1;
            bool foundSuffix = false;
            switch(string.back()){
                case 'K':{
                    factor = std::size_t(1) << 10; 
                    foundSuffix = true;
                }break;
                case 'M':{
                    factor = std::size_t(1) << 20;
                    foundSuffix = true;
                }break;
                case 'G':{
                    factor = std::size_t(1) << 30;
                    foundSuffix = true;
                }break;
            }
            if(foundSuffix){
                const auto numberString = string.substr(0, string.size()-1);
                return factor * std::stoull(numberString);
            }else{
                return std::stoull(string);
            }
        }else{
            return 0;
        }
    };

    bool gotQuery = false;
    bool gotDB = false;

    for(int i = 1; i < argc; i++){
        const std::string arg = argv[i];
        if(arg == "--help"){
            options.help = true;
        }else if(arg == "--uploadFull"){
            options.loadFullDBToGpu = true;
        }else if(arg == "--top"){
            options.numTopOutputs = std::atoi(argv[++i]);
        }else if(arg == "--gop"){
            options.gop = std::atoi(argv[++i]);
        }else if(arg == "--gex"){
            options.gex = std::atoi(argv[++i]);
        }else if(arg == "--maxBatchBytes"){
            options.maxBatchBytes = parseMemoryString(argv[++i]);
        }else if(arg == "--maxBatchSequences"){
            options.maxBatchSequences = std::atoi(argv[++i]);
        }else if(arg == "--maxTempBytes"){
            options.maxTempBytes = parseMemoryString(argv[++i]);
        }else if(arg == "--query"){
            options.queryFile = argv[++i];
            gotQuery = true;
        }else if(arg == "--db"){
            options.dbPrefix = argv[++i];
            gotDB = true;
        }else{
            std::cout << "Unexpected arg " << arg << "\n";
        }
    }

    if(!gotQuery){
        std::cout << "Query is missing\n";
        return false;
    }
    if(!gotDB){
        std::cout << "DB prefix is missing\n";
        return false;
    }

    return true;
}

void printHelp(int argc, char** argv){
    CudaSW4Options defaultoptions;

    std::cout << "Usage: " << argv[0] << " [options]\n";
    std::cout << "Options: \n";
    std::cout << "      --query queryfile : Mandatory. Fasta or Fastq\n";
    std::cout << "      --db dbPrefix : Mandatory. The DB to query against. The same dbPrefix as used for makedb\n";
    std::cout << "      --gop val : Gap open score. Default val = " << defaultoptions.gop << "\n";
    std::cout << "      --gex val : Gap extend score. Default val = " << defaultoptions.gex << "\n";
    std::cout << "      --top val : Output the val best scores. Default val = " << defaultoptions.numTopOutputs << "\n";
    std::cout << "      --maxTempBytes val : Size of temp storage in GPU memory. Can use suffix K,M,G. Default val = " << defaultoptions.maxTempBytes << "\n";
    std::cout << "      --maxBatchBytes val : Process DB in batches of at most val bytes. Can use suffix K,M,G. Default val = " << defaultoptions.maxBatchBytes << "\n";
    std::cout << "      --maxBatchSequences val : Process DB in batches of at most val sequences. Default val = " << defaultoptions.maxBatchSequences << "\n";
    std::cout << "      --uploadFull : Do not process DB in smaller batches. Copy full DB to GPU before processing queries.\n";
}


int main(int argc, char* argv[])
{

    if(argc < 3) {
        cout << "Usage:\n  " << argv[0] << " <FASTA filename 1> [dbPrefix]\n";
        return 0;
    }

    CudaSW4Options options;
    bool parseSuccess = parseArgs(argc, argv, options);

    if(!parseSuccess || options.help){
        printHelp(argc, argv);
        return 0;
    }

    std::cout << "Selected options:\n";
    std::cout << "loadFullDBToGpu: " << options.loadFullDBToGpu << "\n";
    std::cout << "numTopOutputs: " << options.numTopOutputs << "\n";
    std::cout << "gop: " << options.gop << "\n";
    std::cout << "gex: " << options.gex << "\n";
    std::cout << "maxBatchBytes: " << options.maxBatchBytes << "\n";
    std::cout << "maxBatchSequences: " << options.maxBatchSequences << "\n";
    std::cout << "maxTempBytes: " << options.maxTempBytes << "\n";
    std::cout << "queryFile: " << options.queryFile << "\n";
    std::cout << "dbPrefix: " << options.dbPrefix << "\n";

    



    std::vector<int> deviceIds;
    {
        int num = 0;
        cudaGetDeviceCount(&num); CUERR
        for(int i = 0; i < num; i++){
            std::cout << "Using device " << i << "\n";
            deviceIds.push_back(i);
        }
    }
    const int numGpus = deviceIds.size();
    assert(numGpus > 0);

    const int masterDeviceId = deviceIds[0];

    for(int i = 0; i < numGpus; i++){
        cudaSetDevice(deviceIds[i]); CUERR
        helpers::init_cuda_context(); CUERR
        cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);CUERR
        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte); CUERR

        cudaMemPool_t mempool;
        cudaDeviceGetDefaultMemPool(&mempool, deviceIds[i]); CUERR
        uint64_t threshold = UINT64_MAX;
        cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);CUERR
    }
    std::vector<CudaStream> gpuStreams(numGpus);
    for(int i = 0; i < numGpus; i++){
        cudaSetDevice(deviceIds[i]); CUERR
        gpuStreams[i] = CudaStream{};
    }

    cudaSetDevice(masterDeviceId); CUERR

    CudaEvent masterevent1{cudaEventDisableTiming};
    CudaStream masterStream1;






	// read all sequences from FASTA or FASTQ file: query file
	sequence_batch batch = read_all_sequences_and_headers_from_file(options.queryFile);
	cout << "Read Protein Query File 1\n";

    // chars   = concatenation of all sequences
    // offsets = starting indices of individual sequences (1st: 0, last: one behind end of 'chars')
    char*   queryChars       = batch.chars.data();
    const size_t* queryOffsets     = batch.offsets.data();
    const size_t* queryLengths      = batch.lengths.data();
    const size_t  totalNumQueryBytes   = batch.chars.size();
    int numQueries = batch.lengths.size();
    const char* maxNumQueriesString = std::getenv("ALIGNER_MAX_NUM_QUERIES");
    if(maxNumQueriesString != nullptr){
        int maxNumQueries = std::atoi(maxNumQueriesString);
        numQueries = std::min(numQueries, maxNumQueries);
    }

    const size_t  offsetBytes = (numQueries+1) * sizeof(size_t);


    cout << "Number of input sequences Query-File:  " << numQueries<< '\n';
    cout << "Number of input characters Query-File: " << totalNumQueryBytes << '\n';


    #if 1
	cout << "Reading Database: \n";
	helpers::CpuTimer timer_read_db("READ_DB");
    constexpr bool writeAccess = false;
    constexpr bool prefetchSeq = true;
    DB fullDB = loadDB(options.dbPrefix, writeAccess, prefetchSeq);
	timer_read_db.print();
    {
    #else
    //for(int pseudodbSeqLength : {64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 768, 896, 1024}){
    for(int pseudodbSeqLength : {251}){
       std::cout << "pseudodbSeqLength: " << pseudodbSeqLength << "\n";

    PseudoDB fullDB = loadPseudoDB(1, pseudodbSeqLength);
    #endif

    


    const int numDBChunks = fullDB.info.numChunks;
    std::cout << "Number of DB chunks: " << numDBChunks << "\n";
    for(int i = 0; i < numDBChunks; i++){
        const auto& chunkData = fullDB.chunks[i];
        const auto& dbMetaData = chunkData.getMetaData();
        std::cout << "DB chunk " << i << ": " << chunkData.numSequences() << " sequences, " << chunkData.numChars() << " characters\n";
        for(int i = 0; i < int(dbMetaData.lengthBoundaries.size()); i++){
            std::cout << "<= " << dbMetaData.lengthBoundaries[i] << ": " << dbMetaData.numSequencesPerLengthPartition[i] << "\n";
        }
    }

    size_t totalNumberOfSequencesInDB = 0;
    size_t maximumNumberOfSequencesInDBChunk = 0;
    for(int i = 0; i < numDBChunks; i++){
        const auto& chunkData = fullDB.chunks[i];
        totalNumberOfSequencesInDB += chunkData.numSequences();
        maximumNumberOfSequencesInDBChunk = std::max(maximumNumberOfSequencesInDBChunk, chunkData.numSequences());
    }


    //determine maximal and minimal read lengths
    int64_t max_length = 0, min_length = std::numeric_limits<int64_t>::max(), avg_length = 0;
    int64_t max_length_2 = 0, min_length_2 = std::numeric_limits<int64_t>::max(), avg_length_2 = 0;
    for (int i=0; i<numQueries; i++) {
        if (queryLengths[i] > max_length) max_length = queryLengths[i];
        if (queryLengths[i] < min_length) min_length = queryLengths[i];
        avg_length += queryLengths[i];
    }

    for(int i = 0; i < numDBChunks; i++){
        const auto& chunkData = fullDB.chunks[i];
        size_t numSeq = chunkData.numSequences();

        for (size_t i=0; i < numSeq; i++) {
            if (chunkData.lengths()[i] > max_length_2) max_length_2 = chunkData.lengths()[i];
            if (chunkData.lengths()[i] < min_length_2) min_length_2 = chunkData.lengths()[i];
            avg_length_2 += chunkData.lengths()[i];
        }
    }




    cout << "Max Length 1: " << max_length << ", Max Length 2: " << max_length_2 <<"\n";
    cout << "Min Length 1: " << min_length << ", Min Length 2: " << min_length_2 <<"\n";
    cout << "Avg Length 1: " << avg_length/numQueries << ", Avg Length 2: " << avg_length_2/totalNumberOfSequencesInDB <<"\n";



    // auto printPartition = [](const auto& view){
    //     std::cout << "Sequences: " << view.numSequences() << "\n";
    //     std::cout << "Chars: " << view.offsets()[0] << " - " << view.offsets()[view.numSequences()] << " (" << (view.offsets()[view.numSequences()] - view.offsets()[0]) << ")"
    //         << " " << view.numChars() << "\n";
    // };
    // auto printPartitions = [printPartition](const auto& dbPartitions){
    //     size_t numPartitions = dbPartitions.size();
    //     for(size_t p = 0; p < numPartitions; p++){
    //         const DBdataView& view = dbPartitions[p];
    
    //         std::cout << "Partition " << p << "\n";
    //         printPartition(view);
    //     }
    // };

   

    const int numLengthPartitions = getLengthPartitionBoundaries().size();

    //partition chars of whole DB amongst the gpus
    std::vector<std::vector<size_t>> numSequencesPerLengthPartitionPrefixSum_perDBchunk(numDBChunks);
    std::vector<std::vector<DBdataView>> dbPartitionsByLengthPartitioning_perDBchunk(numDBChunks);
    std::vector<std::vector<std::vector<DBdataView>>> subPartitionsForGpus_perDBchunk(numDBChunks);
    std::vector<std::vector<std::vector<int>>> lengthPartitionIdsForGpus_perDBchunk(numDBChunks);
    std::vector<std::vector<size_t>> numSequencesPerGpu_perDBchunk(numDBChunks);
    std::vector<std::vector<size_t>> numSequencesPerGpuPrefixSum_perDBchunk(numDBChunks);

    for(int chunkId = 0; chunkId < numDBChunks; chunkId++){
        const auto& dbChunk = fullDB.chunks[chunkId];

        auto& numSequencesPerLengthPartitionPrefixSum = numSequencesPerLengthPartitionPrefixSum_perDBchunk[chunkId];
        auto& dbPartitionsByLengthPartitioning = dbPartitionsByLengthPartitioning_perDBchunk[chunkId];
        auto& subPartitionsForGpus = subPartitionsForGpus_perDBchunk[chunkId];
        auto& lengthPartitionIdsForGpus = lengthPartitionIdsForGpus_perDBchunk[chunkId];
        auto& numSequencesPerGpu = numSequencesPerGpu_perDBchunk[chunkId];
        auto& numSequencesPerGpuPrefixSum = numSequencesPerGpuPrefixSum_perDBchunk[chunkId];

        subPartitionsForGpus.resize(numGpus);
        lengthPartitionIdsForGpus.resize(numGpus);
        numSequencesPerGpu.resize(numGpus, 0);
        numSequencesPerGpuPrefixSum.resize(numGpus, 0);

        numSequencesPerLengthPartitionPrefixSum.resize(numLengthPartitions, 0);
        for(int i = 0; i < numLengthPartitions-1; i++){
            numSequencesPerLengthPartitionPrefixSum[i+1] = numSequencesPerLengthPartitionPrefixSum[i] + dbChunk.getMetaData().numSequencesPerLengthPartition[i];
        }

        for(int i = 0; i < numLengthPartitions; i++){
            size_t begin = numSequencesPerLengthPartitionPrefixSum[i];
            size_t end = begin + dbChunk.getMetaData().numSequencesPerLengthPartition[i];
            dbPartitionsByLengthPartitioning.emplace_back(dbChunk, begin, end);        
        }

        std::vector<std::vector<int>> numSubPartitionsPerLengthPerGpu(numGpus, std::vector<int>(numLengthPartitions, 0));
       
        for(int lengthPartitionId = 0; lengthPartitionId < numLengthPartitions; lengthPartitionId++){
            const auto& lengthPartition = dbPartitionsByLengthPartitioning[lengthPartitionId];
    
            // std::cout << "length partition " << i << "\n";
            // printPartition(lengthPartition);
    
            const auto partitionedByGpu = partitionDBdata_by_numberOfChars(lengthPartition, lengthPartition.numChars() / numGpus);
            // std::cout << "partitionedByGpu \n";
            // printPartitions(partitionedByGpu);
    
            assert(int(partitionedByGpu.size()) <= numGpus);
    
            // for(int gpu = 0; gpu < int(partitionedByGpu.size()); gpu++){
            //     const auto partitionedBySeq = partitionDBdata_by_numberOfSequences(partitionedByGpu[gpu], batch_size);        
            //     subPartitionsForGpus[gpu].insert(subPartitionsForGpus[gpu].end(), partitionedBySeq.begin(), partitionedBySeq.end());    
            //     for(size_t x = 0; x < partitionedBySeq.size(); x++){
            //         lengthPartitionIdsForGpus[gpu].push_back(lengthPartitionId);
            //     }
            // }

            // for(int gpu = 0; gpu < int(partitionedByGpu.size()); gpu++){
            //     const auto partitionedBySeq = partitionDBdata_by_numberOfChars(partitionedByGpu[gpu], 32*1024 * 1024);        
            //     subPartitionsForGpus[gpu].insert(subPartitionsForGpus[gpu].end(), partitionedBySeq.begin(), partitionedBySeq.end());    
            //     for(size_t x = 0; x < partitionedBySeq.size(); x++){
            //         lengthPartitionIdsForGpus[gpu].push_back(lengthPartitionId);
            //     }
            //     numSubPartitionsPerLengthPerGpu[gpu][lengthPartitionId] = partitionedBySeq.size();
            // }

            for(int gpu = 0; gpu < int(partitionedByGpu.size()); gpu++){     
                subPartitionsForGpus[gpu].push_back(partitionedByGpu[gpu]);
                lengthPartitionIdsForGpus[gpu].push_back(lengthPartitionId);
            }
        }

        // for(int gpu = 0; gpu < numGpus; gpu++){
        //     // std::vector<int> indices(lengthPartitionIdsForGpus_perDBchunk[chunkId][gpu].size());
        //     // std::iota(indices.begin(), indices.end(), 0);
        //     // std::sort(indices.begin(), indices.end(), [](const auto& l, const auto& r){
        //     //     return lengthPartitionIdsForGpus[gpu][l] > lengthPartitionIdsForGpus[gpu][r];
        //     // });
        //     std::vector<int> numSubPartitionsPerLengthPrefixSum(numLengthPartitions, 0);
        //     for(int i = 1; i < numLengthPartitions; i++){
        //         numSubPartitionsPerLengthPrefixSum[i] = numSubPartitionsPerLengthPrefixSum[i-1] + numSubPartitionsPerLengthPerGpu[gpu][i-1];
        //     }

        //     std::vector<DBdataView> newSubPartitions;
        //     newSubPartitions.reserve(subPartitionsForGpus[gpu].size());
        //     std::vector<int> newLengthPartitionIds;
        //     newLengthPartitionIds.reserve(subPartitionsForGpus[gpu].size());

        //     int numRemaining = subPartitionsForGpus[gpu].size();
        //     while(numRemaining > 0){
        //         for(int lengthPartitionId = numLengthPartitions-1; lengthPartitionId >= 0; lengthPartitionId--){
        //             if(numSubPartitionsPerLengthPerGpu[gpu][lengthPartitionId] > 0){
        //                 const int k = numSubPartitionsPerLengthPrefixSum[lengthPartitionId] + numSubPartitionsPerLengthPerGpu[gpu][lengthPartitionId] - 1;
        //                 newSubPartitions.push_back(subPartitionsForGpus[gpu][k]);
        //                 newLengthPartitionIds.push_back(lengthPartitionIdsForGpus[gpu][k]);
        //                 numSubPartitionsPerLengthPerGpu[gpu][lengthPartitionId]--;
        //                 numRemaining--;
        //             }
        //         }
        //     }

        //     std::swap(newSubPartitions, subPartitionsForGpus[gpu]);
        //     std::swap(newLengthPartitionIds, lengthPartitionIdsForGpus[gpu]);
        // }

        // for(int gpu = 0; gpu < numGpus; gpu++){
        //     std::rotate(subPartitionsForGpus[gpu].rbegin(), subPartitionsForGpus[gpu].rbegin() + 1, subPartitionsForGpus[gpu].rend());
        //     std::rotate(lengthPartitionIdsForGpus[gpu].rbegin(), lengthPartitionIdsForGpus[gpu].rbegin() + 1, lengthPartitionIdsForGpus[gpu].rend());
        // }
        // for(int gpu = 0; gpu < numGpus; gpu++){
        //     std::reverse(subPartitionsForGpus[gpu].begin(), subPartitionsForGpus[gpu].end());
        //     std::reverse(lengthPartitionIdsForGpus[gpu].begin(),  lengthPartitionIdsForGpus[gpu].end());
        // }

        for(int i = 0; i < numGpus; i++){
            for(const auto& p : subPartitionsForGpus[i]){
                numSequencesPerGpu[i] += p.numSequences();
            }
        }
        for(int i = 0; i < numGpus-1; i++){
            numSequencesPerGpuPrefixSum[i+1] = numSequencesPerGpuPrefixSum[i] + numSequencesPerGpu[i];
        }
    }

    std::vector<size_t> numSequencesPerGpu_total(numGpus, 0);
    std::vector<size_t> numSequencesPerGpuPrefixSum_total(numGpus, 0);

    for(int i = 0; i < numGpus; i++){
        size_t num = 0;
        for(int chunkId = 0; chunkId < numDBChunks; chunkId++){
            num += numSequencesPerGpu_perDBchunk[chunkId][i];
        }
        numSequencesPerGpu_total[i] = num;
        if(i < numGpus - 1){
            numSequencesPerGpuPrefixSum_total[i+1] = numSequencesPerGpuPrefixSum_total[i] + num;
        }
    }






    // std::cout << "Top level partioning:\n";
    // for(size_t i = 0; i < dbPartitionsForGpus.size(); i++){
    //     std::cout << "Partition for gpu " << i << ":\n";
    //     printPartition(dbPartitionsForGpus[i]);
    // }
    // for(size_t i = 0; i < dbPartitionsForGpus.size(); i++){
    //     for(size_t k = 0; k < subPartitionsForGpus[i].size(); k++){
    //         std::cout << "Subpartition " << k << " for gpu " << i << ":\n";
    //         printPartition(subPartitionsForGpus[i][k]);
    //     }
    // }






    


    const uint results_per_query = std::min(size_t(options.numTopOutputs), totalNumberOfSequencesInDB);
    MyPinnedBuffer<float> alignment_scores_float(numQueries *numDBChunks * results_per_query);
    MyPinnedBuffer<size_t> sorted_indices(numQueries *numDBChunks * results_per_query);
    MyPinnedBuffer<int> resultDbChunkIndices(numQueries *numDBChunks * results_per_query);

    //set up gpus

    assert(numDBChunks == 1);


    std::vector<std::unique_ptr<GpuWorkingSet>> workingSets(numGpus);  


    cout << "Allocate Memory: \n";
    //nvtx::push_range("ALLOC_MEM", 0);
	helpers::CpuTimer allocTimer("ALLOC_MEM");

    for(int i = 0; i < numGpus; i++){
        cudaSetDevice(deviceIds[i]);

        // size_t max_batch_char_bytes = 0;
        // size_t max_batch_num_sequences = 0;

        // for(int chunkId = 0; chunkId < numDBChunks; chunkId++){
        //     const size_t max_batch_char_bytes_chunk = std::max_element(subPartitionsForGpus_perDBchunk[chunkId][i].begin(), subPartitionsForGpus_perDBchunk[chunkId][i].end(),
        //         [](const auto& l, const auto& r){ return l.numChars() < r.numChars(); }
        //     )->numChars();

        //     max_batch_char_bytes = std::max(max_batch_char_bytes, max_batch_char_bytes_chunk);

        //     const size_t max_batch_num_sequences_chunk = std::max_element(subPartitionsForGpus_perDBchunk[chunkId][i].begin(), subPartitionsForGpus_perDBchunk[chunkId][i].end(),
        //         [](const auto& l, const auto& r){ return l.numSequences() < r.numSequences(); }
        //     )->numSequences();

        //     max_batch_num_sequences = std::max(max_batch_num_sequences, max_batch_num_sequences_chunk);
        // }

        // const size_t max_batch_offset_bytes = sizeof(size_t) * max_batch_num_sequences;

        // std::cout << "max_batch_char_bytes " << max_batch_char_bytes << "\n";
        // std::cout << "max_batch_num_sequences " << max_batch_num_sequences << "\n";
        // std::cout << "max_batch_offset_bytes " << max_batch_offset_bytes << "\n";

        workingSets[i] = std::make_unique<GpuWorkingSet>(
            numQueries,
            totalNumQueryBytes,
            options,
            subPartitionsForGpus_perDBchunk[0][i]
        );

        //spin up the host callback thread
        auto noop = [](void*){};
        cudaLaunchHostFunc(
            cudaStreamLegacy, 
            noop, 
            nullptr
        ); CUERR
    }    

	allocTimer.print();
    //nvtx::pop_range();


    std::vector<std::vector<std::vector<DeviceBatchCopyToPinnedPlan>>> batchPlans_perChunk(numDBChunks);

    for(int chunkId = 0; chunkId < numDBChunks; chunkId++){
        batchPlans_perChunk[chunkId].resize(numGpus);

        for(int gpu = 0; gpu < numGpus; gpu++){
            const auto& ws = *workingSets[gpu];
            
            batchPlans_perChunk[chunkId][gpu] = computeDbCopyPlan(
                subPartitionsForGpus_perDBchunk[chunkId][gpu],
                lengthPartitionIdsForGpus_perDBchunk[chunkId][gpu],
                ws.MAX_CHARDATA_BYTES,
                ws.MAX_SEQ
            );

            std::cout << "Batch plan chunk " << chunkId << ", gpu " << gpu << ": " << batchPlans_perChunk[chunkId][gpu].size() << " batches\n";
            // for(int i = 0; i < std::min(5, int(batchPlans_perChunk[chunkId][gpu].size())); i++){
            //     std::cout << batchPlans_perChunk[chunkId][gpu][i] << "\n";
            // }
        }
    }

    
    for(int gpu = 0; gpu < numGpus; gpu++){
        cudaSetDevice(deviceIds[gpu]);
        const int chunkId = 0;
        if(true && batchPlans_perChunk[chunkId][gpu].size() == 1){
            auto& ws = *workingSets[gpu];
            const auto& plan = batchPlans_perChunk[chunkId][gpu][0];
            std::cout << "Upload single batch DB to gpu " << gpu << "\n";
            helpers::CpuTimer copyTimer("copy db");
            auto& h_chardata_vec = ws.h_chardata_vec;
            auto& h_lengthdata_vec = ws.h_lengthdata_vec;
            auto& h_offsetdata_vec = ws.h_offsetdata_vec;
            auto& d_chardata_vec = ws.d_chardata_vec;
            auto& d_lengthdata_vec = ws.d_lengthdata_vec;
            auto& d_offsetdata_vec = ws.d_offsetdata_vec;
            auto& copyStreams = ws.copyStreams;
            const int currentBuffer = 0;

            cudaStream_t H2DcopyStream = ws.copyStreams[currentBuffer];

            executeCopyPlanH2DDirect(
                plan,
                ws,
                currentBuffer,
                subPartitionsForGpus_perDBchunk[chunkId][gpu],
                H2DcopyStream
            );
            
            cudaStreamSynchronize(H2DcopyStream); CUERR;
            copyTimer.print();
            //ws.singleBatchDBisOnGpu = true;
        }
    }


    for(int i = 0; i < numGpus; i++){
        cudaSetDevice(deviceIds[i]);

        // const int permutation[21] = {0,20,4,3,6,13,7,8,9,11,10,12,2,14,5,1,15,16,19,17,18};
        // char BLOSUM62_1D_permutation[21*21];
        // perumte_columns_BLOSUM(BLOSUM62_1D,21,permutation,BLOSUM62_1D_permutation);
        // cudaMemcpyToSymbol(cBLOSUM62_dev, &(BLOSUM62_1D_permutation[0]), 21*21*sizeof(char));
        cudaMemcpyToSymbol(cBLOSUM62_dev, &(BLOSUM62_1D[0]), 21*21*sizeof(char));
    }

    cudaSetDevice(masterDeviceId);

    int totalOverFlowNumber = 0;
    MyDeviceBuffer<float> devAllAlignmentScoresFloat(512 * 1024 * numGpus);
    MyDeviceBuffer<size_t> dev_sorted_indices(512 * 1024 * numGpus);


    cudaSetDevice(masterDeviceId);

    std::vector<std::unique_ptr<helpers::GpuTimer>> queryTimers;
    for(int i = 0; i < numQueries; i++){
        queryTimers.emplace_back(std::make_unique<helpers::GpuTimer>(masterStream1, "Query " + std::to_string(i)));
    }

    cout << "Starting FULLSCAN_CUDA: \n";
    helpers::GpuTimer fullscanTimer(masterStream1, "FULLSCAN_CUDA");

    for(int i = 0; i < numGpus; i++){
        cudaSetDevice(deviceIds[i]);
        auto& ws = *workingSets[i];
        cudaMemcpyAsync(ws.devChars.data(), queryChars, totalNumQueryBytes, cudaMemcpyHostToDevice, gpuStreams[i]); CUERR
        cudaMemcpyAsync(ws.devOffsets.data(), queryOffsets, sizeof(size_t) * (numQueries+1), cudaMemcpyHostToDevice, gpuStreams[i]); CUERR
        cudaMemcpyAsync(ws.devLengths.data(), queryLengths, sizeof(size_t) * (numQueries), cudaMemcpyHostToDevice, gpuStreams[i]); CUERR
        NW_convert_protein<<<numQueries, 128, 0, gpuStreams[i]>>>(ws.devChars.data(), ws.devOffsets.data()); CUERR
    }

    std::vector<cudaStream_t> rawGpuStreams;
    for(const auto& s : gpuStreams){
        rawGpuStreams.push_back(s.getStream());
    }
	
	const int FIRST_QUERY_NUM = 0;


	for(int query_num = FIRST_QUERY_NUM; query_num < numQueries; ++query_num) {

        const bool useExtraThreadForBatchTransfer = numGpus > 1;

        cudaSetDevice(masterDeviceId);
        thrust::fill(
            thrust::cuda::par_nosync.on(masterStream1),
            devAllAlignmentScoresFloat.begin(),
            devAllAlignmentScoresFloat.end(),
            0
        );

        std::cout << "Starting NW_local_affine_half2 for Query " << query_num << "\n";

        //nvtx::push_range("QUERY " + std::to_string(query_num), 0);
        queryTimers[query_num]->start();

        for(int chunkId = 0; chunkId < numDBChunks; chunkId++){
            cudaSetDevice(masterDeviceId);           

            cudaEventRecord(masterevent1, masterStream1); CUERR;

            #if 0
                //std::cout << "start old func\n";
                for(int gpu = 0; gpu < numGpus; gpu++){
                    cudaSetDevice(deviceIds[gpu]); CUERR;

                    auto& ws = *workingSets[gpu];
                    assert(ws.deviceId == deviceIds[gpu]);

                    cudaStreamWaitEvent(gpuStreams[gpu], masterevent1, 0); CUERR;

                    processQueryOnGpu(
                        ws,
                        subPartitionsForGpus_perDBchunk[chunkId][gpu],
                        lengthPartitionIdsForGpus_perDBchunk[chunkId][gpu],
                        batchPlans_perChunk[chunkId][gpu],
                        ws.devChars.data() + queryOffsets[query_num],
                        queryLengths[query_num],
                        useExtraThreadForBatchTransfer,
                        options,
                        gpuStreams[gpu]
                    );
                    CUERR;

                    cudaMemcpyAsync(
                        devAllAlignmentScoresFloat.data() + numSequencesPerGpuPrefixSum_perDBchunk[chunkId][gpu],
                        ws.devAlignmentScoresFloat.data(),
                        sizeof(float) * numSequencesPerGpu_perDBchunk[chunkId][gpu],
                        cudaMemcpyDeviceToDevice,
                        gpuStreams[gpu]
                    ); CUERR;

                    cudaEventRecord(ws.forkStreamEvent, gpuStreams[gpu]); CUERR;

                    cudaSetDevice(masterDeviceId);
                    cudaStreamWaitEvent(masterStream1, ws.forkStreamEvent, 0); CUERR;
                }
            #else
            //std::cout << "start new func\n";
            for(int gpu = 0; gpu < numGpus; gpu++){
                cudaSetDevice(deviceIds[gpu]); CUERR;
                cudaStreamWaitEvent(gpuStreams[gpu], masterevent1, 0); CUERR;
            }

            processQueryOnGpus(
                deviceIds,
                rawGpuStreams,
                workingSets,
                subPartitionsForGpus_perDBchunk[chunkId],
                lengthPartitionIdsForGpus_perDBchunk[chunkId], // dbPartitions[i] belongs to the length partition lengthPartitionIds[i]
                batchPlans_perChunk[chunkId],
                numSequencesPerGpu_perDBchunk[chunkId],
                numSequencesPerGpuPrefixSum_perDBchunk[chunkId],
                queryChars + queryOffsets[query_num],
                queryLengths[query_num],
                useExtraThreadForBatchTransfer,
                options
            );

            for(int gpu = 0; gpu < numGpus; gpu++){
                cudaSetDevice(deviceIds[gpu]); CUERR;
                auto& ws = *workingSets[gpu];
                // cudaMemcpyAsync(
                //     devAllAlignmentScoresFloat.data() + numSequencesPerGpuPrefixSum_perDBchunk[chunkId][gpu],
                //     ws.devAlignmentScoresFloat.data(),
                //     sizeof(float) * numSequencesPerGpu_perDBchunk[chunkId][gpu],
                //     cudaMemcpyDeviceToDevice,
                //     gpuStreams[gpu]
                // ); CUERR;

                //we could sort the maxReduceArray here as well and only send the best results_per_query entries

                cudaMemcpyAsync(
                    devAllAlignmentScoresFloat.data() + 512*1024*gpu,
                    ws.d_maxReduceArrayScores.data(),
                    sizeof(float) * 512*1024,
                    cudaMemcpyDeviceToDevice,
                    gpuStreams[gpu]
                ); CUERR;
                cudaMemcpyAsync(
                    dev_sorted_indices.data() + 512*1024*gpu,
                    ws.d_maxReduceArrayIndices.data(),
                    sizeof(size_t) * 512*1024,
                    cudaMemcpyDeviceToDevice,
                    gpuStreams[gpu]
                ); CUERR;

                cudaEventRecord(ws.forkStreamEvent, gpuStreams[gpu]); CUERR;

                cudaSetDevice(masterDeviceId);
                cudaStreamWaitEvent(masterStream1, ws.forkStreamEvent, 0); CUERR;
            }

            #endif

            cudaSetDevice(masterDeviceId);

            // thrust::sequence(
            //     thrust::cuda::par_nosync.on(masterStream1),
            //     dev_sorted_indices.begin(), 
            //     dev_sorted_indices.end(),
            //     0
            // );
            // thrust::sort_by_key(
            //     thrust::cuda::par_nosync(thrust_async_allocator<char>(masterStream1)).on(masterStream1),
            //     // thrust::cuda::par_nosync(thrust_preallocated_single_allocator<char>((void*)workingSets[0]->d_tempStorageHE, 
            //     //     workingSets[0]->numTempBytes)).on(masterStream1),
            //     devAllAlignmentScoresFloat.begin(),
            //     devAllAlignmentScoresFloat.end(),
            //     dev_sorted_indices.begin(),
            //     thrust::greater<float>()
            // );

            thrust::sort_by_key(
                thrust::cuda::par_nosync(thrust_async_allocator<char>(masterStream1)).on(masterStream1),
                // thrust::cuda::par_nosync(thrust_preallocated_single_allocator<char>((void*)workingSets[0]->d_tempStorageHE, 
                //     workingSets[0]->numTempBytes)).on(masterStream1),
                devAllAlignmentScoresFloat.begin(),
                devAllAlignmentScoresFloat.begin() + 512 * 1024 * numGpus,
                dev_sorted_indices.begin(),
                thrust::greater<float>()
            );

            std::fill(
                &resultDbChunkIndices[query_num*numDBChunks*results_per_query + chunkId * results_per_query],
                &resultDbChunkIndices[query_num*numDBChunks*results_per_query + chunkId * results_per_query] + results_per_query,
                chunkId
            );

            cudaMemcpyAsync(
                &(alignment_scores_float[query_num*numDBChunks*results_per_query + chunkId * results_per_query]), 
                devAllAlignmentScoresFloat.data(), 
                results_per_query*sizeof(float), 
                cudaMemcpyDeviceToHost, 
                masterStream1
            );  CUERR
            cudaMemcpyAsync(
                &(sorted_indices[query_num*numDBChunks*results_per_query + chunkId * results_per_query]), 
                dev_sorted_indices.data(), 
                results_per_query*sizeof(size_t), 
                cudaMemcpyDeviceToHost, 
                masterStream1
            );  CUERR

        }

        queryTimers[query_num]->stop();
        queryTimers[query_num]->printGCUPS(avg_length_2 * queryLengths[query_num]);
        //nvtx::pop_range();
    }
    cudaSetDevice(masterDeviceId);
    cudaStreamSynchronize(masterStream1); CUERR

    for(int i = 0; i < numQueries; i++){
        queryTimers[i]->printGCUPS(avg_length_2 * queryLengths[i]);
    }
    fullscanTimer.printGCUPS(avg_length_2 * avg_length);

    CUERR;

    const char* alignerDisableOutputString = std::getenv("ALIGNER_DISABLE_OUTPUT");
    bool outputDisabled = std::atoi(alignerDisableOutputString);

    if(!outputDisabled){

        //sort the chunk results per query to find overall top results
        std::vector<float> final_alignment_scores_float(numQueries * results_per_query);
        std::vector<size_t> final_sorted_indices(numQueries * results_per_query);
        std::vector<int> final_resultDbChunkIndices(numQueries * results_per_query);

        for (int query_num=0; query_num < numQueries; query_num++) {
            float* scores =  &alignment_scores_float[query_num * numDBChunks * results_per_query];
            size_t* indices =  &sorted_indices[query_num * numDBChunks * results_per_query];
            int* chunkIds =  &resultDbChunkIndices[query_num * numDBChunks * results_per_query];

            std::vector<int> permutation(results_per_query * numDBChunks);
            std::iota(permutation.begin(), permutation.end(), 0);
            std::sort(permutation.begin(), permutation.end(),
                [&](const auto& l, const auto& r){
                    return scores[l] > scores[r];
                }
            );

            for(int i = 0; i < results_per_query; i++){
                final_alignment_scores_float[query_num * results_per_query + i] = scores[permutation[i]];
                final_sorted_indices[query_num * results_per_query + i] = indices[permutation[i]];
                final_resultDbChunkIndices[query_num * results_per_query + i] = chunkIds[permutation[i]];
            }        
        }

        auto convert_AA_back = [&](const auto& AA) {
            if (AA == 0) return 'A';
            if (AA == 1) return 'R';
            if (AA == 2) return 'N';
            if (AA == 3) return 'D';
            if (AA == 4) return 'C';
            if (AA == 5) return 'Q';
            if (AA == 6) return 'E';
            if (AA == 7) return 'G';
            if (AA == 8) return 'H';
            if (AA == 9) return 'I';
            if (AA == 10) return 'L';
            if (AA == 11) return 'K';
            if (AA == 12) return 'M';
            if (AA == 13) return 'F';
            if (AA == 14) return 'P';
            if (AA == 15) return 'S';
            if (AA == 16) return 'T';
            if (AA == 17) return 'W';
            if (AA == 18) return 'Y';
            if (AA == 19) return 'V';
            return '-'; //  else
        };


        for (int i=0; i<numQueries; i++) {
            std::cout << totalOverFlowNumber << " total overflow positions \n";
            cout << "Query Length:" << queryLengths[i] << " Header: ";
            cout << batch.headers[i] << '\n';

            for(int j = 0; j < results_per_query; ++j) {
                const int arrayIndex = i*results_per_query+j;
                const size_t sortedIndex = final_sorted_indices[arrayIndex];
                
                const int dbChunkIndex = final_resultDbChunkIndices[arrayIndex];
                
                const auto& chunkData = fullDB.chunks[dbChunkIndex];

                const char* headerBegin = chunkData.headers() + chunkData.headerOffsets()[sortedIndex];
                const char* headerEnd = chunkData.headers() + chunkData.headerOffsets()[sortedIndex+1];
                cout << "Result: "<< j <<", Length: " << chunkData.lengths()[sortedIndex] << " Score: " << final_alignment_scores_float[arrayIndex] << " : ";
                std::copy(headerBegin, headerEnd,std::ostream_iterator<char>{cout});
                //cout << "\n";

                std::cout << " dbChunkIndex " << dbChunkIndex << "\n";
                //std::cout << "sortedIndex " << sortedIndex << "\n";

                // std::copy(chunkData.chars() + chunkData.offsets()[sortedIndex], 
                //     chunkData.chars() + chunkData.offsets()[sortedIndex] + chunkData.lengths()[sortedIndex] ,
                //     std::ostream_iterator<char>{cout});
                // std::transform(chunkData.chars() + chunkData.offsets()[sortedIndex], 
                //     chunkData.chars() + chunkData.offsets()[sortedIndex] + chunkData.lengths()[sortedIndex] ,
                //     std::ostream_iterator<char>{cout},
                //     convert_AA_back);
                // std::cout << "\n";
            }
        }

        CUERR;

    }

    } //pseudodb length loop

}
