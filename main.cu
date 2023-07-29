
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
#include <thrust/binary_search.h>

#include "hpc_helpers/cuda_raiiwrappers.cuh"
#include "hpc_helpers/all_helpers.cuh"
#include "hpc_helpers/nvtx_markers.cuh"
#include "hpc_helpers/simple_allocation.cuh"

#include "sequence_io.h"
#include "kseqpp/kseqpp.hpp"

#include <omp.h>
#include "options.hpp"
#include "dbdata.hpp"
#include "length_partitions.hpp"
#include "util.cuh"
//#include "convert.cuh"
//#include "kernels.cuh"


#include "new_kernels.cuh"
#include "blosum.hpp"

template<class T>
using MyPinnedBuffer = helpers::SimpleAllocationPinnedHost<T, 0>;
template<class T>
using MyDeviceBuffer = helpers::SimpleAllocationDevice<T, 0>;



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

    size_t indexOffset = 0;
    int* d_locks;
    volatile float* d_scores;
    size_t* d_indices;
};





struct HostGpuPartitionOffsets{
    int numGpus;
    int numLengthPartitions;
    std::vector<size_t> partitionSizes;
    std::vector<size_t> horizontalPS;
    std::vector<size_t> verticalPS;
    std::vector<size_t> totalPerLengthPartitionPS;

    HostGpuPartitionOffsets(int numGpus_, int numLengthpartitions_, std::vector<size_t> partitionSizes_)
        : numGpus(numGpus_), 
        numLengthPartitions(numLengthpartitions_), 
        partitionSizes(std::move(partitionSizes_)),
        horizontalPS(numGpus * numLengthPartitions, 0),
        verticalPS(numGpus * numLengthPartitions, 0),
        totalPerLengthPartitionPS(numLengthPartitions, 0)
    {
        assert(partitionSizes.size() == numGpus * numLengthPartitions);

        for(int gpu = 0; gpu < numGpus; gpu++){
            for(int l = 1; l < numLengthPartitions; l++){
                horizontalPS[gpu * numLengthPartitions + l] = horizontalPS[gpu * numLengthPartitions + l-1] + partitionSizes[gpu * numLengthPartitions + l-1];
            }
        }
        for(int l = 0; l < numLengthPartitions; l++){
            for(int gpu = 1; gpu < numGpus; gpu++){
                verticalPS[gpu * numLengthPartitions + l] = verticalPS[(gpu-1) * numLengthPartitions + l] + partitionSizes[(gpu-1) * numLengthPartitions + l];
            }
        }
        for(int l = 1; l < numLengthPartitions; l++){
            totalPerLengthPartitionPS[l] = totalPerLengthPartitionPS[l-1] 
                + (verticalPS[(numGpus - 1) * numLengthPartitions + (l-1)] + partitionSizes[(numGpus-1) * numLengthPartitions + (l-1)]);
        }
    }

    size_t getGlobalIndex(int gpu, size_t localIndex) const {
        const size_t* const myHorizontalPS = horizontalPS.data() + gpu * numLengthPartitions;
        const auto it = std::lower_bound(myHorizontalPS, myHorizontalPS + numLengthPartitions, localIndex+1);
        const int whichPartition = std::distance(myHorizontalPS, it) - 1;
        const size_t occurenceInPartition = localIndex - myHorizontalPS[whichPartition];
        const size_t globalPartitionBegin = totalPerLengthPartitionPS[whichPartition];
        const size_t elementsOfOtherPreviousGpusInPartition = verticalPS[gpu * numLengthPartitions + whichPartition];
        //std::cout << "whichPartition " << whichPartition << ", occurenceInPartition " << occurenceInPartition 
        //    << ", globalPartitionBegin " << globalPartitionBegin << ", elementsOfOtherPreviousGpusInPartition " << elementsOfOtherPreviousGpusInPartition << "\n";
        return globalPartitionBegin + elementsOfOtherPreviousGpusInPartition + occurenceInPartition;
    };

    void print(std::ostream& os){
        os << "numGpus " << numGpus << "\n";
        os << "numLengthPartitions " << numLengthPartitions << "\n";
        os << "partitionSizes\n";
        for(int gpu = 0; gpu < numGpus; gpu++){
            for(int l = 0; l < numLengthPartitions; l++){
                os << partitionSizes[gpu * numLengthPartitions + l] << " ";
            }
            os << "\n";
        }
        os << "\n";
        os << "horizontalPS\n";
        for(int gpu = 0; gpu < numGpus; gpu++){
            for(int l = 0; l < numLengthPartitions; l++){
                os << horizontalPS[gpu * numLengthPartitions + l] << " ";
            }
            os << "\n";
        }
        os << "\n";
        os << "verticalPS\n";
        for(int gpu = 0; gpu < numGpus; gpu++){
            for(int l = 0; l < numLengthPartitions; l++){
                os << verticalPS[gpu * numLengthPartitions + l] << " ";
            }
            os << "\n";
        }
        os << "\n";
        os << "totalPerLengthPartitionPS\n";
        for(int l = 0; l < numLengthPartitions; l++){
            os << totalPerLengthPartitionPS[l] << " ";
        }
        os << "\n";
    }
};

struct DeviceGpuPartitionOffsets{
    int numGpus;
    int numLengthPartitions;
    MyDeviceBuffer<size_t> partitionSizes;
    MyDeviceBuffer<size_t> horizontalPS;
    MyDeviceBuffer<size_t> verticalPS;
    MyDeviceBuffer<size_t> totalPerLengthPartitionPS;

    struct View{
        int numGpus;
        int numLengthPartitions;
        const size_t* partitionSizes;
        const size_t* horizontalPS;
        const size_t* verticalPS;
        const size_t* totalPerLengthPartitionPS;

        __device__
        size_t getGlobalIndex(int gpu, size_t localIndex) const {
            const size_t* const myHorizontalPS = horizontalPS + gpu * numLengthPartitions;
            const auto it = thrust::lower_bound(thrust::seq, myHorizontalPS, myHorizontalPS + numLengthPartitions, localIndex+1);
            const int whichPartition = thrust::distance(myHorizontalPS, it) - 1;
            const size_t occurenceInPartition = localIndex - myHorizontalPS[whichPartition];
            const size_t globalPartitionBegin = totalPerLengthPartitionPS[whichPartition];
            const size_t elementsOfOtherPreviousGpusInPartition = verticalPS[gpu * numLengthPartitions + whichPartition];
            return globalPartitionBegin + elementsOfOtherPreviousGpusInPartition + occurenceInPartition;
        };
    };

    DeviceGpuPartitionOffsets() = default;
    DeviceGpuPartitionOffsets(const HostGpuPartitionOffsets& hostData)
        : numGpus(hostData.numGpus),
        numLengthPartitions(hostData.numLengthPartitions),
        partitionSizes(numGpus * numLengthPartitions),
        horizontalPS(numGpus * numLengthPartitions),
        verticalPS(numGpus * numLengthPartitions),
        totalPerLengthPartitionPS(numLengthPartitions)
    {
        cudaMemcpyAsync(partitionSizes.data(), hostData.partitionSizes.data(), sizeof(size_t) * numGpus * numLengthPartitions, H2D, cudaStreamLegacy); CUERR;
        cudaMemcpyAsync(horizontalPS.data(), hostData.horizontalPS.data(), sizeof(size_t) * numGpus * numLengthPartitions, H2D, cudaStreamLegacy); CUERR;
        cudaMemcpyAsync(verticalPS.data(), hostData.verticalPS.data(), sizeof(size_t) * numGpus * numLengthPartitions, H2D, cudaStreamLegacy); CUERR;
        cudaMemcpyAsync(totalPerLengthPartitionPS.data(), hostData.totalPerLengthPartitionPS.data(), sizeof(size_t) * numLengthPartitions, H2D, cudaStreamLegacy); CUERR;
    }

    View getDeviceView() const{
        View view;
        view.numGpus = numGpus;
        view.numLengthPartitions = numLengthPartitions;
        view.partitionSizes = partitionSizes.data();
        view.horizontalPS = horizontalPS.data();
        view.verticalPS = verticalPS.data();
        view.totalPerLengthPartitionPS = totalPerLengthPartitionPS.data();
        return view;
    }
};


struct GpuWorkingSet{

    static constexpr int maxReduceArraySize = 512 * 1024;
    using MaxReduceArray = TopNMaximaArray<maxReduceArraySize>;

    GpuWorkingSet(
        size_t gpumemlimit,
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

        d_query.resize(1024*1024 + 128); CUERR

        size_t usedGpuMem = 0;
        usedGpuMem += sizeof(int) * maxReduceArraySize; // d_maxReduceArrayLocks
        usedGpuMem += sizeof(float) * maxReduceArraySize; // d_maxReduceArrayScores
        usedGpuMem += sizeof(size_t) * maxReduceArraySize; // d_maxReduceArrayIndices

        d_maxReduceArrayLocks.resize(maxReduceArraySize);
        d_maxReduceArrayScores.resize(maxReduceArraySize);
        d_maxReduceArrayIndices.resize(maxReduceArraySize);

        cudaMemset(d_maxReduceArrayLocks.data(), 0, sizeof(int) * maxReduceArraySize);

        //devAlignmentScoresFloat.resize(numSubjects);
        Fillchar.resize(16*512);
        cudaMemset(Fillchar.data(), 20, 16*512);

        forkStreamEvent = CudaEvent{cudaEventDisableTiming}; CUERR;
        numWorkStreamsWithoutTemp = 10;
        workstreamIndex = 0;
        workStreamsWithoutTemp.resize(numWorkStreamsWithoutTemp);

        size_t memoryRequiredForFullDB = 0;
        memoryRequiredForFullDB += numSubjectBytes; // d_fulldb_chardata
        memoryRequiredForFullDB += sizeof(size_t) * numSubjects; //d_fulldb_lengthdata
        memoryRequiredForFullDB += sizeof(size_t) * (numSubjects+1); //d_fulldb_offsetdata
        memoryRequiredForFullDB += sizeof(size_t) * numSubjects * 2; //d_overflow_positions_vec

        size_t memoryRequiredForBatchedProcessing = 0;
        memoryRequiredForBatchedProcessing += options.maxBatchBytes * 2; // d_chardata_vec
        memoryRequiredForBatchedProcessing += sizeof(size_t) * options.maxBatchSequences * 2; //d_lengthdata_vec
        memoryRequiredForBatchedProcessing += sizeof(size_t) * (options.maxBatchSequences+1) * 2; //d_offsetdata_vec
        memoryRequiredForBatchedProcessing += sizeof(size_t) * options.maxBatchSequences * 2; //d_overflow_positions_vec

        //const bool hasEnoughMemoryForFullDB = options.loadFullDBToGpu;
        const bool hasEnoughMemoryForFullDB = usedGpuMem + memoryRequiredForFullDB + options.maxTempBytes <= gpumemlimit;
        const bool hasEnoughMemoryForBatchedDB = usedGpuMem + memoryRequiredForBatchedProcessing + options.maxTempBytes <= gpumemlimit;

        numCopyBuffers = 2;

        h_chardata_vec.resize(numCopyBuffers);
        h_lengthdata_vec.resize(numCopyBuffers);
        h_offsetdata_vec.resize(numCopyBuffers);
        d_chardata_vec.resize(numCopyBuffers);
        d_lengthdata_vec.resize(numCopyBuffers);
        d_offsetdata_vec.resize(numCopyBuffers);
        copyStreams.resize(numCopyBuffers);
        pinnedBufferEvents.resize(numCopyBuffers);
        deviceBufferEvents.resize(numCopyBuffers);
        d_total_overflow_number.resize(1);
        d_overflow_number.resize(numCopyBuffers);
        h_overflow_number.resize(numCopyBuffers);
        d_overflow_positions_vec.resize(numCopyBuffers);

        auto allocateForFullDBProcessing = [&](){
            d_fulldb_chardata.resize(numSubjectBytes);
            d_fulldb_lengthdata.resize(numSubjects);
            d_fulldb_offsetdata.resize(numSubjects+1);
            
            //d_selectedPositions.resize(numSubjects);
            
            for(int i = 0; i < numCopyBuffers; i++){
                h_chardata_vec[i].resize(options.maxBatchBytes);
                h_lengthdata_vec[i].resize(options.maxBatchSequences);
                h_offsetdata_vec[i].resize(options.maxBatchSequences+1);               
                pinnedBufferEvents[i] = CudaEvent{cudaEventDisableTiming}; CUERR;
                deviceBufferEvents[i] = CudaEvent{cudaEventDisableTiming}; CUERR;
                d_overflow_positions_vec[i].resize(numSubjects);
            }
            canStoreFullDB = true;
            usedGpuMem += memoryRequiredForFullDB;
        };

        auto allocateForBatchedProcessing = [&](){
            //d_selectedPositions.resize(options.maxBatchSequences);

            for(int i = 0; i < numCopyBuffers; i++){
                h_chardata_vec[i].resize(options.maxBatchBytes);
                h_lengthdata_vec[i].resize(options.maxBatchSequences);
                h_offsetdata_vec[i].resize(options.maxBatchSequences+1);
                d_chardata_vec[i].resize(options.maxBatchBytes);
                d_lengthdata_vec[i].resize(options.maxBatchSequences);
                d_offsetdata_vec[i].resize(options.maxBatchSequences+1);
                pinnedBufferEvents[i] = CudaEvent{cudaEventDisableTiming}; CUERR;
                deviceBufferEvents[i] = CudaEvent{cudaEventDisableTiming}; CUERR;
                d_overflow_positions_vec[i].resize(options.maxBatchSequences);
            }
            canStoreFullDB = false;
            usedGpuMem += memoryRequiredForBatchedProcessing;
        };
        

        if(hasEnoughMemoryForFullDB){
            try{
                allocateForFullDBProcessing();
            }catch(...){
                if(hasEnoughMemoryForBatchedDB){
                    allocateForBatchedProcessing();
                }else{
                    throw std::runtime_error("Not enough GPU memory available on device id " + std::to_string(deviceId));
                }   
            }
        }else{
            if(hasEnoughMemoryForBatchedDB){
                allocateForBatchedProcessing();
            }else{
                throw std::runtime_error("Not enough GPU memory available on device id " + std::to_string(deviceId));
            }
        }        
        // thrust::sequence(
        //     thrust::device,
        //     d_selectedPositions.begin(),
        //     d_selectedPositions.end(),
        //     size_t(0)
        // );
        //allocations for db processing did not actually

        assert(usedGpuMem <= gpumemlimit);

        numTempBytes = std::min(options.maxTempBytes, gpumemlimit - usedGpuMem);

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

    void setPartitionOffsets(const HostGpuPartitionOffsets& offsets){
        deviceGpuPartitionOffsets = DeviceGpuPartitionOffsets(offsets);
    }        

    bool singleBatchDBisOnGpu = false;
    int deviceId;
    int numCopyBuffers;
    int numWorkStreamsWithoutTemp = 1;
    int workstreamIndex;
    int copyBufferIndex = 0;
    size_t numTempBytes;

    MyDeviceBuffer<int> d_maxReduceArrayLocks;
    MyDeviceBuffer<float> d_maxReduceArrayScores;
    MyDeviceBuffer<size_t> d_maxReduceArrayIndices;

    MyDeviceBuffer<char> d_query;
    MyDeviceBuffer<char> d_tempStorageHE;
    //MyDeviceBuffer<float> devAlignmentScoresFloat;
    MyDeviceBuffer<char> Fillchar;
    MyDeviceBuffer<size_t> d_selectedPositions;
    MyDeviceBuffer<int> d_total_overflow_number;
    MyDeviceBuffer<int> d_overflow_number;
    MyPinnedBuffer<int> h_overflow_number;
    CudaStream hostFuncStream;
    CudaStream workStreamForTempUsage;
    CudaEvent forkStreamEvent;

    bool canStoreFullDB = false;
    bool fullDBisUploaded = false;
    MyDeviceBuffer<char> d_fulldb_chardata;
    MyDeviceBuffer<size_t> d_fulldb_lengthdata;
    MyDeviceBuffer<size_t> d_fulldb_offsetdata;
    
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
    std::vector<MyDeviceBuffer<size_t>> d_overflow_positions_vec;

    DeviceGpuPartitionOffsets deviceGpuPartitionOffsets;

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

    //size_t processedSequences = 0;
    while(currentCopyPartition < dbPartitions.size()){
        
        size_t usedBytes = 0;
        size_t usedSeq = 0;

        DeviceBatchCopyToPinnedPlan plan;

        while(currentCopyPartition < dbPartitions.size()){
            if(dbPartitions[currentCopyPartition].numSequences() == 0){
                currentCopyPartition++;
                continue;
            }

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

        
        if(usedSeq == 0 && currentCopyPartition < dbPartitions.size() && dbPartitions[currentCopyPartition].numSequences() > 0){
            std::cout << "Warning. copy buffer size too small. skipped a db portion. stop\n";
            break;
        }

        if(plan.usedSeq > 0){
            result.push_back(plan);
        }
    }

    return result;
}




void executeCopyPlanH2DDirect(
    const DeviceBatchCopyToPinnedPlan& plan, 
    char* d_chardata,
    size_t* d_lengthdata,
    size_t* d_offsetdata,
    const std::vector<DBdataView>& dbPartitions,
    cudaStream_t stream
){

    size_t usedBytes = 0;
    size_t usedSeq = 0;
    for(const auto& copyRange : plan.copyRanges){
        const auto& dbPartition = dbPartitions[copyRange.currentCopyPartition];
        const auto& firstSeq = copyRange.currentCopySeqInPartition;
        const auto& numToCopy = copyRange.numToCopy;
        size_t numBytesToCopy = dbPartition.offsets()[firstSeq + numToCopy] - dbPartition.offsets()[firstSeq];

        cudaMemcpyAsync(
            d_chardata + usedBytes,
            dbPartition.chars() + dbPartition.offsets()[firstSeq],
            numBytesToCopy,
            H2D,
            stream
        ); CUERR;
        cudaMemcpyAsync(
            d_lengthdata + usedSeq,
            dbPartition.lengths() + firstSeq,
            sizeof(size_t) * numToCopy,
            H2D,
            stream
        ); CUERR;
        cudaMemcpyAsync(
            d_offsetdata + usedSeq,
            dbPartition.offsets() + firstSeq,
            sizeof(size_t) * (numToCopy+1),
            H2D,
            stream
        ); CUERR;
        thrust::for_each(
            thrust::cuda::par_nosync.on(stream),
            d_offsetdata + usedSeq,
            d_offsetdata + usedSeq + (numToCopy+1),
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
    char* h_chardata,
    size_t* h_lengthdata,
    size_t* h_offsetdata,
    const std::vector<DBdataView>& dbPartitions
){
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
            h_chardata + usedBytes
        );
        std::copy(
            dbPartition.lengths() + firstSeq,
            dbPartition.lengths() + firstSeq+numToCopy,
            h_lengthdata + usedSeq
        );
        std::transform(
            dbPartition.offsets() + firstSeq,
            dbPartition.offsets() + firstSeq + (numToCopy+1),
            h_offsetdata + usedSeq,
            [&](size_t off){
                return off - dbPartition.offsets()[firstSeq] + usedBytes;
            }
        );
        usedBytes += std::distance(h_chardata + usedBytes, end);
        usedSeq += numToCopy;
    }
};

void executePinnedCopyPlanSerialAndTransferToGpu(
    const DeviceBatchCopyToPinnedPlan& plan, 
    char* h_chardata,
    size_t* h_lengthdata,
    size_t* /*h_offsetdata*/,
    char* d_chardata,
    size_t* d_lengthdata,
    size_t* d_offsetdata,
    const std::vector<DBdataView>& dbPartitions,
    cudaStream_t H2DcopyStream
){

    size_t usedBytes = 0;
    for(const auto& copyRange : plan.copyRanges){
        const auto& dbPartition = dbPartitions[copyRange.currentCopyPartition];
        const auto& firstSeq = copyRange.currentCopySeqInPartition;
        const auto& numToCopy = copyRange.numToCopy;
        size_t numBytesToCopy = dbPartition.offsets()[firstSeq + numToCopy] - dbPartition.offsets()[firstSeq];
        constexpr size_t maxTransferBatchSize = 8 * 1024 * 1024;
        for(size_t i = 0; i < numBytesToCopy; i += maxTransferBatchSize){
            const size_t x = std::min(numBytesToCopy - i, maxTransferBatchSize);

            std::copy_n(
                dbPartition.chars() + dbPartition.offsets()[firstSeq] + i,
                x,
                h_chardata + usedBytes + i
            );
            cudaMemcpyAsync(
                d_chardata + usedBytes + i,
                h_chardata + usedBytes + i,
                x,
                H2D,
                H2DcopyStream
            ); CUERR;
        }

        // auto end = std::copy(
        //     dbPartition.chars() + dbPartition.offsets()[firstSeq],
        //     dbPartition.chars() + dbPartition.offsets()[firstSeq + numToCopy],
        //     h_chardata + usedBytes
        // );
        // cudaMemcpyAsync(
        //     d_chardata + usedBytes,
        //     h_chardata + usedBytes,
        //     numBytesToCopy,
        //     H2D,
        //     H2DcopyStream
        // ); CUERR;

        usedBytes += numBytesToCopy;
    }

    size_t usedSeq = 0;
    for(const auto& copyRange : plan.copyRanges){
        const auto& dbPartition = dbPartitions[copyRange.currentCopyPartition];
        const auto& firstSeq = copyRange.currentCopySeqInPartition;
        const auto& numToCopy = copyRange.numToCopy;

        std::copy(
            dbPartition.lengths() + firstSeq,
            dbPartition.lengths() + firstSeq+numToCopy,
            h_lengthdata + usedSeq
        );
        // cudaMemcpyAsync(
        //     d_lengthdata + usedSeq,
        //     h_lengthdata + usedSeq,
        //     sizeof(size_t) * numToCopy,
        //     H2D,
        //     H2DcopyStream
        // ); CUERR;

        usedSeq += numToCopy;
    }
    cudaMemcpyAsync(
        d_lengthdata,
        h_lengthdata,
        sizeof(size_t) * plan.usedSeq,
        H2D,
        H2DcopyStream
    ); CUERR;

    cudaMemsetAsync(d_offsetdata, 0, sizeof(size_t), H2DcopyStream); CUERR;

    auto d_paddedLengths = thrust::make_transform_iterator(
        d_lengthdata,
        [] __host__ __device__ (const size_t& length){
            return SDIV(length, 4) * 4;
        }
    );

    thrust::inclusive_scan(
        thrust::cuda::par_nosync(thrust_async_allocator<char>(H2DcopyStream)).on(H2DcopyStream),
        d_paddedLengths,
        d_paddedLengths + plan.usedSeq,
        d_offsetdata + 1
    );

};

struct ExecutePinnedCopyCallbackData{
    const DeviceBatchCopyToPinnedPlan* planPtr; 
    char* h_chardata;
    size_t* h_lengthdata;
    size_t* h_offsetdata;
    const std::vector<DBdataView>* dbPartitionsPtr;
};

void executePinnedCopyPlanCallback(void* args){
    ExecutePinnedCopyCallbackData* callbackData = (ExecutePinnedCopyCallbackData*)args;
    const auto& plan = *callbackData->planPtr;
    auto& dbPartitions = *callbackData->dbPartitionsPtr;
    

    executePinnedCopyPlanSerial(
        plan, 
        callbackData->h_chardata,
        callbackData->h_lengthdata,
        callbackData->h_offsetdata,
        dbPartitions
    );

    delete callbackData;
}

void executePinnedCopyPlanWithHostCallback(
    const DeviceBatchCopyToPinnedPlan& plan, 
    char* h_chardata,
    size_t* h_lengthdata,
    size_t* h_offsetdata,
    const std::vector<DBdataView>& dbPartitions, 
    cudaStream_t stream
){
    ExecutePinnedCopyCallbackData* data = new ExecutePinnedCopyCallbackData;

    data->planPtr = &plan;
    data->h_chardata = h_chardata,
    data->h_lengthdata = h_lengthdata,
    data->h_offsetdata = h_offsetdata,
    data->dbPartitionsPtr = &dbPartitions;

    cudaLaunchHostFunc(
        stream,
        executePinnedCopyPlanCallback,
        (void*)data
    ); CUERR;
}








void processQueryOnGpus(
    const std::vector<int>& deviceIds,
    const std::vector<cudaStream_t>& gpuStreams,
    const std::vector<std::unique_ptr<GpuWorkingSet>>& workingSets,
    const std::vector<std::vector<DBdataView>>& dbPartitionsPerGpu,
    const std::vector<std::vector<int>>& /*lengthPartitionIdsPerGpu*/, // dbPartitions[i] belongs to the length partition lengthPartitionIds[i]
    const std::vector<std::vector<DeviceBatchCopyToPinnedPlan>>& batchPlansPerGpu_batched,
    const std::vector<std::vector<DeviceBatchCopyToPinnedPlan>>& batchPlansPerGpu_full,
    const std::vector<size_t>& numberOfSequencesPerGpu,
    const std::vector<size_t>& /*numberOfSequencesPerGpuPrefixSum*/,
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
    // std::vector<size_t> processedSequencesPerGpu(numGpus, 0);

    // std::vector<size_t> processedBatchesPerGpu(numGpus, 0);

    for(int gpu = 0; gpu < numGpus; gpu++){
        cudaSetDevice(deviceIds[gpu]); CUERR;
        auto& ws = *workingSets[gpu];

        cudaMemsetAsync(ws.d_total_overflow_number.data(), 0, sizeof(int), gpuStreams[gpu]);

        cudaMemsetAsync(ws.d_query.data(), 20, SDIV(queryLength, 128) * 128, gpuStreams[gpu]);
        cudaMemcpyAsync(ws.d_query.data(), query, queryLength, cudaMemcpyDefault, gpuStreams[gpu]); CUERR
        NW_convert_protein_single<<<SDIV(queryLength, 128), 128, 0, gpuStreams[gpu]>>>(ws.d_query.data(), queryLength); CUERR

        //cudaMemcpyToSymbolAsync(constantQuery4 ,ws.Fillchar.data(), 512*16, 0, cudaMemcpyDeviceToDevice, gpuStreams[gpu]); CUERR
        cudaMemcpyToSymbolAsync(constantQuery4, ws.d_query.data(), queryLength, 0, cudaMemcpyDeviceToDevice, gpuStreams[gpu]); CUERR

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


    //variables per gpu to keep between loops
    struct Variables{
        int currentBuffer = 0;
        int previousBuffer = 0;
        cudaStream_t H2DcopyStream = cudaStreamLegacy;
        char* h_inputChars = nullptr;
        size_t* h_inputLengths = nullptr;
        size_t* h_inputOffsets = nullptr;
        char* d_inputChars = nullptr;
        size_t* d_inputLengths = nullptr;
        size_t* d_inputOffsets = nullptr;
        int* d_overflow_number = nullptr;
        size_t* d_overflow_positions = nullptr;
        size_t pointerSequencesOffset = 0;
        size_t pointerBytesOffset = 0;
        const std::vector<DeviceBatchCopyToPinnedPlan>* batchPlansPtr;
        size_t processedSequences = 0;
        size_t processedBatches = 0;
    };

    std::vector<Variables> variables_vec(numGpus);
    //init variables
    for(int gpu = 0; gpu < numGpus; gpu++){
        const auto& ws = *workingSets[gpu];
        auto& variables = variables_vec[gpu];
        variables.processedSequences = 0;
        variables.processedBatches = 0;
        if(ws.canStoreFullDB){
            if(ws.fullDBisUploaded){
                assert(batchPlansPerGpu_full[gpu].size() == 1);
                variables.batchPlansPtr = &batchPlansPerGpu_full[gpu];
            }else{
                variables.batchPlansPtr = &batchPlansPerGpu_batched[gpu];
            }
        }else{
            variables.batchPlansPtr = &batchPlansPerGpu_batched[gpu];
        }
    }
    
    while(totalNumberOfProcessedSequences < totalNumberOfSequencesToProcess){
        //set up gpu variables for current iteration
        for(int gpu = 0; gpu < numGpus; gpu++){
            auto& ws = *workingSets[gpu];
            auto& variables = variables_vec[gpu];
            if(variables.processedBatches < variables.batchPlansPtr->size()){
                if(ws.canStoreFullDB){
                    if(ws.fullDBisUploaded){
                        //single buffer, no transfer, 1 batch
                        variables.currentBuffer = 0;
                        variables.previousBuffer = 0;
                        variables.H2DcopyStream = ws.copyStreams[0];
                        variables.h_inputChars = nullptr;
                        variables.h_inputLengths = nullptr;
                        variables.h_inputOffsets = nullptr;
                        variables.d_inputChars = ws.d_fulldb_chardata.data();
                        variables.d_inputLengths = ws.d_fulldb_lengthdata.data();
                        variables.d_inputOffsets = ws.d_fulldb_offsetdata.data();
                        variables.d_overflow_number = ws.d_overflow_number.data() + 0;
                        variables.d_overflow_positions = ws.d_overflow_positions_vec[0].data();
                    }else{
                        //first query, double buffer batched transfer into full db buffer
                        variables.currentBuffer = ws.copyBufferIndex;
                        if(variables.currentBuffer == 0){
                            variables.previousBuffer = ws.numCopyBuffers - 1;
                        }else{
                            variables.previousBuffer = (variables.currentBuffer - 1);
                        } 
                        variables.H2DcopyStream = ws.copyStreams[variables.currentBuffer];
                        variables.h_inputChars = ws.h_chardata_vec[variables.currentBuffer].data();
                        variables.h_inputLengths = ws.h_lengthdata_vec[variables.currentBuffer].data();
                        variables.h_inputOffsets = ws.h_offsetdata_vec[variables.currentBuffer].data();
                        variables.d_inputChars = ws.d_fulldb_chardata.data() + variables.pointerBytesOffset;
                        variables.d_inputLengths = ws.d_fulldb_lengthdata.data() + variables.pointerSequencesOffset;
                        variables.d_inputOffsets = ws.d_fulldb_offsetdata.data() + variables.pointerSequencesOffset;
                        variables.d_overflow_number = ws.d_overflow_number.data() + variables.currentBuffer;
                        variables.d_overflow_positions = ws.d_overflow_positions_vec[variables.currentBuffer].data();
                    }
                }else{
                    //full db not possible, any query, double buffer batched transfer
                    variables.currentBuffer = ws.copyBufferIndex;
                    if(variables.currentBuffer == 0){
                        variables.previousBuffer = ws.numCopyBuffers - 1;
                    }else{
                        variables.previousBuffer = (variables.currentBuffer - 1);
                    } 
                    variables.H2DcopyStream = ws.copyStreams[variables.currentBuffer];
                    variables.h_inputChars = ws.h_chardata_vec[variables.currentBuffer].data();
                    variables.h_inputLengths = ws.h_lengthdata_vec[variables.currentBuffer].data();
                    variables.h_inputOffsets = ws.h_offsetdata_vec[variables.currentBuffer].data();
                    variables.d_inputChars = ws.d_chardata_vec[variables.currentBuffer].data();
                    variables.d_inputLengths = ws.d_lengthdata_vec[variables.currentBuffer].data();
                    variables.d_inputOffsets = ws.d_offsetdata_vec[variables.currentBuffer].data();
                    variables.d_overflow_number = ws.d_overflow_number.data() + variables.currentBuffer;
                    variables.d_overflow_positions = ws.d_overflow_positions_vec[variables.currentBuffer].data();
                }
            }
        }
        //upload batch
        for(int gpu = 0; gpu < numGpus; gpu++){
            auto& ws = *workingSets[gpu];
            auto& variables = variables_vec[gpu];
            if(variables.processedBatches < variables.batchPlansPtr->size()){
                cudaSetDevice(deviceIds[gpu]); CUERR;
                const auto& plan = (*variables.batchPlansPtr)[variables.processedBatches];

                if((ws.canStoreFullDB && !ws.fullDBisUploaded) || !ws.canStoreFullDB){
                    //transfer data
                    //can only overwrite device buffer if it is no longer in use on workstream
                    cudaStreamWaitEvent(variables.H2DcopyStream, ws.deviceBufferEvents[variables.currentBuffer], 0); CUERR;

                    if(useExtraThreadForBatchTransfer){
                        cudaStreamWaitEvent(ws.hostFuncStream, ws.pinnedBufferEvents[variables.currentBuffer]); CUERR;
                        executePinnedCopyPlanWithHostCallback(
                            plan, 
                            variables.h_inputChars,
                            variables.h_inputLengths,
                            variables.h_inputOffsets,
                            dbPartitionsPerGpu[gpu], 
                            ws.hostFuncStream
                        );
                        cudaEventRecord(ws.forkStreamEvent, ws.hostFuncStream); CUERR;
                        cudaStreamWaitEvent(variables.H2DcopyStream, ws.forkStreamEvent, 0);

                        cudaMemcpyAsync(
                            variables.d_inputChars,
                            variables.h_inputChars,
                            plan.usedBytes,
                            H2D,
                            variables.H2DcopyStream
                        ); CUERR;
                        cudaMemcpyAsync(
                            variables.d_inputLengths,
                            variables.h_inputLengths,
                            sizeof(size_t) * plan.usedSeq,
                            H2D,
                            variables.H2DcopyStream
                        ); CUERR;
                        cudaMemcpyAsync(
                            variables.d_inputOffsets,
                            variables.h_inputOffsets,
                            sizeof(size_t) * (plan.usedSeq+1),
                            H2D,
                            variables.H2DcopyStream
                        ); CUERR;
                    }else{
                        //synchronize to avoid overwriting pinned buffer of target before it has been fully transferred
                        cudaEventSynchronize(ws.pinnedBufferEvents[variables.currentBuffer]); CUERR;
                        //executePinnedCopyPlanSerial(plan, ws, currentBuffer, dbPartitionsPerGpu[gpu]);

                        executePinnedCopyPlanSerialAndTransferToGpu(
                            plan, 
                            variables.h_inputChars,
                            variables.h_inputLengths,
                            variables.h_inputOffsets,
                            variables.d_inputChars,
                            variables.d_inputLengths,
                            variables.d_inputOffsets,
                            dbPartitionsPerGpu[gpu], 
                            variables.H2DcopyStream
                        );
                    }
                    
                    cudaEventRecord(ws.pinnedBufferEvents[variables.currentBuffer], variables.H2DcopyStream); CUERR;
                }

                cudaMemsetAsync(variables.d_overflow_number, 0, sizeof(int), variables.H2DcopyStream);
                // cudaMemsetAsync(d_overflow_positions, 0, sizeof(size_t) * 10000, H2DcopyStream);
                
                //all data is ready for alignments. create dependencies for work streams
                cudaEventRecord(ws.forkStreamEvent, variables.H2DcopyStream); CUERR;
                cudaStreamWaitEvent(ws.workStreamForTempUsage, ws.forkStreamEvent, 0); CUERR;
                for(auto& stream : ws.workStreamsWithoutTemp){
                    cudaStreamWaitEvent(stream, ws.forkStreamEvent, 0); CUERR;
                }
                //wait for previous batch to finish
                cudaStreamWaitEvent(ws.workStreamForTempUsage, ws.deviceBufferEvents[variables.previousBuffer], 0); CUERR;
                for(auto& stream : ws.workStreamsWithoutTemp){
                    cudaStreamWaitEvent(stream, ws.deviceBufferEvents[variables.previousBuffer], 0); CUERR;
                }
        //     }
        // }

        //launch alignment kernels

        // for(int gpu = 0; gpu < numGpus; gpu++){
        //     if(processedBatchesPerGpu[gpu] < batchPlanPerGpu[gpu].size()){
        //         cudaSetDevice(deviceIds[gpu]); CUERR;
        //         auto& ws = *workingSets[gpu];
        //         const auto& plan = batchPlanPerGpu[gpu][processedBatchesPerGpu[gpu]];
        //         int currentBuffer = 0;
        //         //int previousBuffer = 0;
        //         //cudaStream_t H2DcopyStream = ws.copyStreams[currentBuffer];
        //         if(!ws.singleBatchDBisOnGpu){
        //             currentBuffer = ws.copyBufferIndex;
        //             //H2DcopyStream = ws.copyStreams[currentBuffer];
        //         }

                // const char* const inputChars = ws.d_chardata_vec[currentBuffer].data();
                // const size_t* const inputOffsets = ws.d_offsetdata_vec[currentBuffer].data();
                // const size_t* const inputLengths = ws.d_lengthdata_vec[currentBuffer].data();
                // int* const d_overflow_number = ws.d_overflow_number.data() + currentBuffer;
                // size_t* const d_overflow_positions = ws.d_overflow_positions_vec[currentBuffer].data();

                const char* const inputChars = variables.d_inputChars;
                const size_t* const inputLengths = variables.d_inputLengths;
                const size_t* const inputOffsets = variables.d_inputOffsets;
                int* const d_overflow_number = variables.d_overflow_number;
                size_t* const d_overflow_positions = variables.d_overflow_positions;

                auto runAlignmentKernels = [&](auto& d_scores, size_t* d_overflow_positions, int* d_overflow_number){
                    const char4* const d_query = reinterpret_cast<char4*>(ws.d_query.data());

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
                        
                        //const size_t* const d_selectedPositions = ws.d_selectedPositions.data() + start;
                        auto d_selectedPositions = thrust::make_counting_iterator<size_t>(start);

                        #if 1
                        if(options.singlePassType == KernelType::Half2){
                            
                            constexpr int sh2bs = 256; // single half2 blocksize 
                            if (partId == 0){call_NW_local_affine_Protein_single_pass_half2_new<sh2bs, 2, 24>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, d_query, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            if (partId == 1){call_NW_local_affine_Protein_single_pass_half2_new<sh2bs, 4, 16>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, d_query, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            if (partId == 2){call_NW_local_affine_Protein_single_pass_half2_new<sh2bs, 8, 10>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, d_query, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            if (partId == 3){call_NW_local_affine_Protein_single_pass_half2_new<sh2bs, 8, 12>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, d_query, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            if (partId == 4){call_NW_local_affine_Protein_single_pass_half2_new<sh2bs, 8, 14>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, d_query, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            if (partId == 5){call_NW_local_affine_Protein_single_pass_half2_new<sh2bs, 8, 16>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, d_query, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            if (partId == 6){call_NW_local_affine_Protein_single_pass_half2_new<sh2bs, 8, 18>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, d_query, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            if (partId == 7){call_NW_local_affine_Protein_single_pass_half2_new<sh2bs, 8, 20>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, d_query, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            if (partId == 8){call_NW_local_affine_Protein_single_pass_half2_new<sh2bs, 8, 22>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, d_query, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            if (partId == 9){call_NW_local_affine_Protein_single_pass_half2_new<sh2bs, 8, 24>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, d_query, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            if (partId == 10){call_NW_local_affine_Protein_single_pass_half2_new<sh2bs, 8, 26>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, d_query, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            if (partId == 11){call_NW_local_affine_Protein_single_pass_half2_new<sh2bs, 8, 28>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, d_query, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            if (partId == 12){call_NW_local_affine_Protein_single_pass_half2_new<sh2bs, 8, 30>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, d_query, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            if (partId == 13){call_NW_local_affine_Protein_single_pass_half2_new<sh2bs, 8, 32>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            if (partId == 14){call_NW_local_affine_Protein_single_pass_half2_new<sh2bs, 16, 18>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            if (partId == 15){call_NW_local_affine_Protein_single_pass_half2_new<sh2bs, 16, 20>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            if (partId == 16){call_NW_local_affine_Protein_single_pass_half2_new<sh2bs, 16, 22>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            if (partId == 17){call_NW_local_affine_Protein_single_pass_half2_new<sh2bs, 16, 24>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            if (partId == 18){call_NW_local_affine_Protein_single_pass_half2_new<sh2bs, 16, 26>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            if (partId == 19){call_NW_local_affine_Protein_single_pass_half2_new<sh2bs, 16, 28>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            if (partId == 20){call_NW_local_affine_Protein_single_pass_half2_new<sh2bs, 16, 30>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            if (partId == 21){call_NW_local_affine_Protein_single_pass_half2_new<sh2bs, 16, 32>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            if (partId == 22){call_NW_local_affine_Protein_single_pass_half2_new<sh2bs, 32, 18>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            if (partId == 23){call_NW_local_affine_Protein_single_pass_half2_new<sh2bs, 32, 20>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            if (partId == 24){call_NW_local_affine_Protein_single_pass_half2_new<sh2bs, 32, 22>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            if (partId == 25){call_NW_local_affine_Protein_single_pass_half2_new<sh2bs, 32, 24>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            if (partId == 26){call_NW_local_affine_Protein_single_pass_half2_new<sh2bs, 32, 26>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            if (partId == 27){call_NW_local_affine_Protein_single_pass_half2_new<sh2bs, 32, 28>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            if (partId == 28){call_NW_local_affine_Protein_single_pass_half2_new<sh2bs, 32, 30>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            if (partId == 29){call_NW_local_affine_Protein_single_pass_half2_new<sh2bs, 32, 32>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            if (partId == 30){call_NW_local_affine_Protein_single_pass_half2_new<sh2bs, 32, 34>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            if (partId == 31){call_NW_local_affine_Protein_single_pass_half2_new<sh2bs, 32, 36>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            if (partId == 32){call_NW_local_affine_Protein_single_pass_half2_new<sh2bs, 32, 38>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            if (partId == 33){call_NW_local_affine_Protein_single_pass_half2_new<sh2bs, 32, 40>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        
                        }else if(options.singlePassType == KernelType::DPXs16){

                            // constexpr int sh2bs = 256; // dpx s16 blocksize 
                            // if (partId == 0){call_NW_local_affine_single_pass_s16_DPX_new<sh2bs, 2, 24>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            // if (partId == 1){call_NW_local_affine_single_pass_s16_DPX_new<sh2bs, 4, 16>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            // if (partId == 2){call_NW_local_affine_single_pass_s16_DPX_new<sh2bs, 8, 10>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            // if (partId == 3){call_NW_local_affine_single_pass_s16_DPX_new<sh2bs, 8, 12>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            // if (partId == 4){call_NW_local_affine_single_pass_s16_DPX_new<sh2bs, 8, 14>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            // if (partId == 5){call_NW_local_affine_single_pass_s16_DPX_new<sh2bs, 8, 16>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            // if (partId == 6){call_NW_local_affine_single_pass_s16_DPX_new<sh2bs, 8, 18>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            // if (partId == 7){call_NW_local_affine_single_pass_s16_DPX_new<sh2bs, 8, 20>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            // if (partId == 8){call_NW_local_affine_single_pass_s16_DPX_new<sh2bs, 8, 22>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            // if (partId == 9){call_NW_local_affine_single_pass_s16_DPX_new<sh2bs, 8, 24>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            // if (partId == 10){call_NW_local_affine_single_pass_s16_DPX_new<sh2bs, 8, 26>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            // if (partId == 11){call_NW_local_affine_single_pass_s16_DPX_new<sh2bs, 8, 28>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            // if (partId == 12){call_NW_local_affine_single_pass_s16_DPX_new<sh2bs, 8, 30>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            // if (partId == 13){call_NW_local_affine_single_pass_s16_DPX_new<sh2bs, 8, 32>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            // if (partId == 14){call_NW_local_affine_single_pass_s16_DPX_new<sh2bs, 16, 18>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            // if (partId == 15){call_NW_local_affine_single_pass_s16_DPX_new<sh2bs, 16, 20>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            // if (partId == 16){call_NW_local_affine_single_pass_s16_DPX_new<sh2bs, 16, 22>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            // if (partId == 17){call_NW_local_affine_single_pass_s16_DPX_new<sh2bs, 16, 24>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            // if (partId == 18){call_NW_local_affine_single_pass_s16_DPX_new<sh2bs, 16, 26>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            // if (partId == 19){call_NW_local_affine_single_pass_s16_DPX_new<sh2bs, 16, 28>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            // if (partId == 20){call_NW_local_affine_single_pass_s16_DPX_new<sh2bs, 16, 30>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            // if (partId == 21){call_NW_local_affine_single_pass_s16_DPX_new<sh2bs, 16, 32>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            // if (partId == 22){call_NW_local_affine_single_pass_s16_DPX_new<sh2bs, 32, 18>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            // if (partId == 23){call_NW_local_affine_single_pass_s16_DPX_new<sh2bs, 32, 20>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            // if (partId == 24){call_NW_local_affine_single_pass_s16_DPX_new<sh2bs, 32, 22>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            // if (partId == 25){call_NW_local_affine_single_pass_s16_DPX_new<sh2bs, 32, 24>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            // if (partId == 26){call_NW_local_affine_single_pass_s16_DPX_new<sh2bs, 32, 26>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            // if (partId == 27){call_NW_local_affine_single_pass_s16_DPX_new<sh2bs, 32, 28>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            // if (partId == 28){call_NW_local_affine_single_pass_s16_DPX_new<sh2bs, 32, 30>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            // if (partId == 29){call_NW_local_affine_single_pass_s16_DPX_new<sh2bs, 32, 32>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            // if (partId == 30){call_NW_local_affine_single_pass_s16_DPX_new<sh2bs, 32, 34>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            // if (partId == 31){call_NW_local_affine_single_pass_s16_DPX_new<sh2bs, 32, 36>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            // if (partId == 32){call_NW_local_affine_single_pass_s16_DPX_new<sh2bs, 32, 38>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                            // if (partId == 33){call_NW_local_affine_single_pass_s16_DPX_new<sh2bs, 32, 40>(options.blosumType,inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                        
                        }
                        #endif


                        if(partId == numLengthPartitions - 2){
                            if(options.manyPassType_small == KernelType::Half2){
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
                                    call_NW_local_affine_Protein_many_pass_half2_new<blocksize, groupsize, 22>(
                                        options.blosumType,
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
                                        d_query,
                                        queryLength, 
                                        gop, 
                                        gex,
                                        ws.workStreamForTempUsage
                                    ); CUERR
                                }
                            }else if(options.manyPassType_small == KernelType::DPXs16){
                                constexpr int blocksize = 32 * 8;
                                constexpr int groupsize = 32;
                                constexpr int groupsPerBlock = blocksize / groupsize;
                                constexpr int alignmentsPerGroup = 2;
                                constexpr int alignmentsPerBlock = groupsPerBlock * alignmentsPerGroup;
                                
                                const size_t tempBytesPerBlockPerBuffer = sizeof(short2) * alignmentsPerBlock * queryLength;

                                const size_t maxNumBlocks = ws.numTempBytes / (tempBytesPerBlockPerBuffer * 2);
                                const size_t maxSubjectsPerIteration = std::min(maxNumBlocks * alignmentsPerBlock, size_t(numSeq));

                                const size_t numBlocksPerIteration = SDIV(maxSubjectsPerIteration, alignmentsPerBlock);
                                const size_t requiredTempBytes = tempBytesPerBlockPerBuffer * 2 * numBlocksPerIteration;

                                short2* d_temp = (short2*)ws.d_tempStorageHE.data();
                                short2* d_tempHcol2 = d_temp;
                                short2* d_tempEcol2 = (short2*)(((char*)d_tempHcol2) + requiredTempBytes / 2);

                                const int numIters =  SDIV(numSeq, maxSubjectsPerIteration);

                                for(int iter = 0; iter < numIters; iter++){
                                    const size_t begin = iter * maxSubjectsPerIteration;
                                    const size_t end = iter < numIters-1 ? (iter+1) * maxSubjectsPerIteration : numSeq;
                                    const size_t num = end - begin;                      

                                    cudaMemsetAsync(d_temp, 0, requiredTempBytes, ws.workStreamForTempUsage); CUERR;
                                    //std::cout << "iter " << iter << " / " << numIters << " gridsize " << SDIV(num, alignmentsPerBlock) << "\n";

                                    //cudaDeviceSynchronize(); CUERR;
                                    
                                    call_NW_local_affine_many_pass_s16_DPX_new<blocksize, groupsize, 22>(
                                        options.blosumType,
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
                                        gex,
                                        ws.workStreamForTempUsage
                                    ); CUERR
                                }
                            }else{
                                assert(false);
                            }
                        }


                        if(partId == numLengthPartitions - 1){
                            if(options.manyPassType_large == KernelType::Float){
                                const size_t tempBytesPerSubjectPerBuffer = sizeof(float2) * SDIV(queryLength,32) * 32;
                                const size_t maxSubjectsPerIteration = std::min(size_t(numSeq), ws.numTempBytes / (tempBytesPerSubjectPerBuffer * 2));

                                float2* d_temp = (float2*)ws.d_tempStorageHE.data();
                                float2* d_tempHcol2 = d_temp;
                                float2* d_tempEcol2 = (float2*)(((char*)d_tempHcol2) + maxSubjectsPerIteration * tempBytesPerSubjectPerBuffer);

                                const int numIters =  SDIV(numSeq, maxSubjectsPerIteration);
                                for(int iter = 0; iter < numIters; iter++){
                                    const size_t begin = iter * maxSubjectsPerIteration;
                                    const size_t end = iter < numIters-1 ? (iter+1) * maxSubjectsPerIteration : numSeq;
                                    const size_t num = end - begin;

                                    cudaMemsetAsync(d_temp, 0, tempBytesPerSubjectPerBuffer * 2 * num, ws.workStreamForTempUsage); CUERR;

                                    //cudaDeviceSynchronize(); CUERR;

                                    //NW_local_affine_read4_float_query_Protein<32, 32><<<num, 32, 0, ws.workStreamForTempUsage>>>(
                                    call_NW_local_affine_read4_float_query_Protein_new<20>(
                                        options.blosumType,
                                        inputChars, 
                                        d_scores, 
                                        d_tempHcol2, 
                                        d_tempEcol2, 
                                        inputOffsets, 
                                        inputLengths, 
                                        d_selectedPositions + begin, 
                                        num,
                                        queryLength, 
                                        gop, 
                                        gex,
                                        ws.workStreamForTempUsage
                                    ); CUERR 
                                }
                            }else{
                                assert(false);
                            }
                        }

                        // std::cout << "partId = " << partId << ",numSeq " << numSeq << "\n";
                        //cudaDeviceSynchronize(); CUERR;

                        //exclPs += numSeq;
                    }
                };

                auto maxReduceArray = ws.getMaxReduceArray(variables.processedSequences);
                //runAlignmentKernels(ws.devAlignmentScoresFloat.data() + variables.processedSequences, d_overflow_positions, d_overflow_number);
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
            auto& ws = *workingSets[gpu];
            const auto& variables = variables_vec[gpu];
            if(variables.processedBatches < variables.batchPlansPtr->size()){
                cudaSetDevice(deviceIds[gpu]); CUERR;

                const char* const inputChars = variables.d_inputChars;
                const size_t* const inputLengths = variables.d_inputLengths;
                const size_t* const inputOffsets = variables.d_inputOffsets;
                int* const d_overflow_number = variables.d_overflow_number;
                size_t* const d_overflow_positions = variables.d_overflow_positions;

                auto maxReduceArray = ws.getMaxReduceArray(variables.processedSequences);

                if(options.overflowType == KernelType::Float){
                    //std::cerr << "overflow processing\n";
                    float2* d_temp = (float2*)ws.d_tempStorageHE.data();

                    //launch_process_overflow_alignments_kernel_NW_local_affine_read4_float_query_Protein<32,32><<<1,1,0, ws.workStreamForTempUsage>>>(
                    if(hostBlosumDim == 21){
                        launch_process_overflow_alignments_kernel_NW_local_affine_read4_float_query_Protein_new<20,21><<<1,1,0, ws.workStreamForTempUsage>>>(
                            d_overflow_number,
                            d_temp, 
                            ws.numTempBytes,
                            inputChars, 
                            //ws.devAlignmentScoresFloat.data() + variables.processedSequences, 
                            maxReduceArray,
                            inputOffsets, 
                            inputLengths, 
                            d_overflow_positions, 
                            queryLength, gop, gex
                        ); CUERR
                    #ifdef CAN_USE_FULL_BLOSUM
                    }else if(hostBlosumDim == 25){
                        launch_process_overflow_alignments_kernel_NW_local_affine_read4_float_query_Protein_new<20,25><<<1,1,0, ws.workStreamForTempUsage>>>(
                            d_overflow_number,
                            d_temp, 
                            ws.numTempBytes,
                            inputChars, 
                            //ws.devAlignmentScoresFloat.data() + variables.processedSequences, 
                            maxReduceArray,
                            inputOffsets, 
                            inputLengths, 
                            d_overflow_positions, 
                            queryLength, gop, gex
                        ); CUERR
                    #endif
                    }else{
                        assert(false);
                    }
                }else{
                    assert(false);
                }

                //update total num overflows for query
                helpers::lambda_kernel<<<1,1,0, ws.workStreamForTempUsage>>>(
                    [
                        d_total_overflow_number = ws.d_total_overflow_number.data(),
                        d_overflow_number = d_overflow_number
                    ]__device__(){
                        *d_total_overflow_number += *d_overflow_number;
                    }
                ); CUERR;

                //after processing overflow alignments, the batch is done and its data can be resused
                cudaEventRecord(ws.deviceBufferEvents[variables.currentBuffer], ws.workStreamForTempUsage); CUERR;

                //let other workstreams depend on temp usage stream
                for(auto& stream : ws.workStreamsWithoutTemp){
                    cudaStreamWaitEvent(ws.workStreamForTempUsage, ws.deviceBufferEvents[variables.currentBuffer], 0); CUERR;    
                }

                ws.copyBufferIndex = (ws.copyBufferIndex+1) % ws.numCopyBuffers;
            }
        }

        //update running numbers
        for(int gpu = 0; gpu < numGpus; gpu++){
            auto& variables = variables_vec[gpu];
            if(variables.processedBatches < variables.batchPlansPtr->size()){
                const auto& plan = (*variables.batchPlansPtr)[variables.processedBatches];
                variables.pointerSequencesOffset += plan.usedSeq;
                variables.pointerBytesOffset += plan.usedBytes;
                variables.processedSequences += plan.usedSeq;
                variables.processedBatches++;

                totalNumberOfProcessedSequences += plan.usedSeq;
            } 
        }

    } //while not done


    for(int gpu = 0; gpu < numGpus; gpu++){
        auto& ws = *workingSets[gpu];
        if(ws.canStoreFullDB && !ws.fullDBisUploaded){
            ws.fullDBisUploaded = true;

            // current offsets in d_fulldb_offsetdata store the offsets for each batch, i.e. for each batch the offsets will start again at 0
            // compute prefix sum of d_fulldb_lengthdata to obtain the single-batch offsets

            cudaMemsetAsync(ws.d_fulldb_offsetdata.data(), 0, sizeof(size_t), ws.workStreamForTempUsage); CUERR;

            auto d_paddedLengths = thrust::make_transform_iterator(
                ws.d_fulldb_lengthdata.data(),
                [] __host__ __device__ (const size_t& length){
                    return SDIV(length, 4) * 4;
                }
            );

            thrust::inclusive_scan(
                thrust::cuda::par_nosync(thrust_async_allocator<char>(ws.workStreamForTempUsage)).on(ws.workStreamForTempUsage),
                d_paddedLengths,
                d_paddedLengths + ws.d_fulldb_lengthdata.size(),
                ws.d_fulldb_offsetdata.data() + 1
            );
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







int main(int argc, char* argv[])
{

    if(argc < 3) {
        std::cout << "Usage:\n  " << argv[0] << " <FASTA filename 1> [dbPrefix]\n";
        return 0;
    }

    CudaSW4Options options;
    bool parseSuccess = parseArgs(argc, argv, options);

    if(!parseSuccess || options.help){
        printHelp(argc, argv);
        return 0;
    }

    printOptions(options);

  



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
	sequence_batch batchOfQueries;
    {
        //batchOfQueries = read_all_sequences_and_headers_from_file(options.queryFile);
        constexpr int ALIGN = 4;
        kseqpp::KseqPP reader(options.queryFile);
        batchOfQueries.offsets.push_back(0);
        while(reader.next() >= 0){
            const std::string& header = reader.getCurrentHeader();
            const std::string& sequence = reader.getCurrentSequence();
            //we ignore quality
            //const std::string& quality = reader.getCurrentQuality();

            batchOfQueries.chars.insert(batchOfQueries.chars.end(), sequence.begin(), sequence.end());
            //padding
            if(batchOfQueries.chars.size() % ALIGN != 0){
                batchOfQueries.chars.insert(batchOfQueries.chars.end(), ALIGN - batchOfQueries.chars.size() % ALIGN, ' ');
            }

            batchOfQueries.offsets.push_back(batchOfQueries.chars.size());
            batchOfQueries.lengths.push_back(sequence.size());
            batchOfQueries.headers.push_back(header);
        }
    }
	std::cout << "Read Protein Query File 1\n";

    // chars   = concatenation of all sequences
    // offsets = starting indices of individual sequences (1st: 0, last: one behind end of 'chars')
    char*   queryChars       = batchOfQueries.chars.data();
    const size_t* queryOffsets     = batchOfQueries.offsets.data();
    const size_t* queryLengths      = batchOfQueries.lengths.data();
    const size_t  totalNumQueryBytes   = batchOfQueries.chars.size();
    int numQueries = batchOfQueries.lengths.size();
    const char* maxNumQueriesString = std::getenv("ALIGNER_MAX_NUM_QUERIES");
    if(maxNumQueriesString != nullptr){
        int maxNumQueries = std::atoi(maxNumQueriesString);
        numQueries = std::min(numQueries, maxNumQueries);
    }


    std::cout << "Number of input sequences Query-File:  " << numQueries<< '\n';
    std::cout << "Number of input characters Query-File: " << totalNumQueryBytes << '\n';

    AnyDBWrapper fullDB;

    if(!options.usePseudoDB){
        std::cout << "Reading Database: \n";
        helpers::CpuTimer timer_read_db("READ_DB");
        constexpr bool writeAccess = false;
        constexpr bool prefetchSeq = true;
        //DB fullDB_tmp = loadDB(options.dbPrefix, writeAccess, prefetchSeq);
        auto fullDB_tmp = std::make_shared<DB>(loadDB(options.dbPrefix, writeAccess, prefetchSeq));
        timer_read_db.print();

        fullDB = AnyDBWrapper(fullDB_tmp);
    }else{
        std::cout << "Generating pseudo db\n";
        helpers::CpuTimer timer_read_db("READ_DB");
        //PseudoDB fullDB_tmp = loadPseudoDB(options.pseudoDBSize, options.pseudoDBLength);
        auto fullDB_tmp = std::make_shared<PseudoDB>(loadPseudoDB(options.pseudoDBSize, options.pseudoDBLength));
        timer_read_db.print();
        
        fullDB = AnyDBWrapper(fullDB_tmp);
    }


    const int numDBChunks = fullDB.getInfo().numChunks;
    std::cout << "Number of DB chunks: " << numDBChunks << "\n";
    assert(numDBChunks == 1);

    auto lengthBoundaries = getLengthPartitionBoundaries();
    const int numLengthPartitions = getLengthPartitionBoundaries().size();


    std::vector<size_t> fullDB_numSequencesPerLengthPartition(numLengthPartitions);
    {
        const auto& dbData = fullDB.getChunks()[0];
        auto partitionBegin = dbData.lengths();
        for(int i = 0; i < numLengthPartitions; i++){
            //length k is in partition i if boundaries[i-1] < k <= boundaries[i]
            int searchFor = lengthBoundaries[i];
            if(searchFor < std::numeric_limits<int>::max()){
                searchFor += 1;
            }
            auto partitionEnd = std::lower_bound(
                partitionBegin, 
                dbData.lengths() + dbData.numSequences(), 
                searchFor
            );
            fullDB_numSequencesPerLengthPartition[i] = std::distance(partitionBegin, partitionEnd);
            partitionBegin = partitionEnd;
        }
    }

    std::cout << "DB chunk " << 0 << ": " << fullDB.getChunks()[0].numSequences() << " sequences, " << fullDB.getChunks()[0].numChars() << " characters\n";
    if(options.printLengthPartitions){
        for(int i = 0; i < numLengthPartitions; i++){
            std::cout << "<= " << lengthBoundaries[i] << ": " << fullDB_numSequencesPerLengthPartition[i] << "\n";
        }
    }


    // for(int i = 0; i < numDBChunks; i++){
    //     const auto& chunkData = fullDB.chunks[i];
    //     const auto& dbMetaData = chunkData.getMetaData();
    //     std::cout << "DB chunk " << i << ": " << chunkData.numSequences() << " sequences, " << chunkData.numChars() << " characters\n";
    //     for(int i = 0; i < int(dbMetaData.lengthBoundaries.size()); i++){
    //         std::cout << "<= " << dbMetaData.lengthBoundaries[i] << ": " << dbMetaData.numSequencesPerLengthPartition[i] << "\n";
    //     }
    // }

    size_t totalNumberOfSequencesInDB = 0;
    size_t maximumNumberOfSequencesInDBChunk = 0;
    for(int i = 0; i < numDBChunks; i++){
        const auto& chunkData = fullDB.getChunks()[i];
        totalNumberOfSequencesInDB += chunkData.numSequences();
        maximumNumberOfSequencesInDBChunk = std::max(maximumNumberOfSequencesInDBChunk, chunkData.numSequences());
    }


    //determine maximal and minimal read lengths
    size_t max_length = 0, min_length = std::numeric_limits<size_t>::max(), avg_length = 0;
    size_t max_length_2 = 0, min_length_2 = std::numeric_limits<size_t>::max(), avg_length_2 = 0;
    for (int i=0; i<numQueries; i++) {
        if (queryLengths[i] > max_length) max_length = queryLengths[i];
        if (queryLengths[i] < min_length) min_length = queryLengths[i];
        avg_length += queryLengths[i];
    }

    for(int i = 0; i < numDBChunks; i++){
        const auto& chunkData = fullDB.getChunks()[i];
        size_t numSeq = chunkData.numSequences();

        for (size_t i=0; i < numSeq; i++) {
            if (chunkData.lengths()[i] > max_length_2) max_length_2 = chunkData.lengths()[i];
            if (chunkData.lengths()[i] < min_length_2) min_length_2 = chunkData.lengths()[i];
            avg_length_2 += chunkData.lengths()[i];
        }
    }




    std::cout << "Max Length 1: " << max_length << ", Max Length 2: " << max_length_2 <<"\n";
    std::cout << "Min Length 1: " << min_length << ", Min Length 2: " << min_length_2 <<"\n";
    std::cout << "Avg Length 1: " << avg_length/numQueries << ", Avg Length 2: " << avg_length_2/totalNumberOfSequencesInDB <<"\n";



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

   

    

    //partition chars of whole DB amongst the gpus
    std::vector<std::vector<size_t>> numSequencesPerLengthPartitionPrefixSum_perDBchunk(numDBChunks);
    std::vector<std::vector<DBdataView>> dbPartitionsByLengthPartitioning_perDBchunk(numDBChunks);
    std::vector<std::vector<std::vector<DBdataView>>> subPartitionsForGpus_perDBchunk(numDBChunks);
    std::vector<std::vector<std::vector<int>>> lengthPartitionIdsForGpus_perDBchunk(numDBChunks);
    std::vector<std::vector<size_t>> numSequencesPerGpu_perDBchunk(numDBChunks);
    std::vector<std::vector<size_t>> numSequencesPerGpuPrefixSum_perDBchunk(numDBChunks);

    for(int chunkId = 0; chunkId < numDBChunks; chunkId++){
        const auto& dbChunk = fullDB.getChunks()[chunkId];

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
            numSequencesPerLengthPartitionPrefixSum[i+1] = numSequencesPerLengthPartitionPrefixSum[i] + fullDB_numSequencesPerLengthPartition[i];
        }

        for(int i = 0; i < numLengthPartitions; i++){
            size_t begin = numSequencesPerLengthPartitionPrefixSum[i];
            size_t end = begin + fullDB_numSequencesPerLengthPartition[i];
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

            // for(int gpu = 0; gpu < int(partitionedByGpu.size()); gpu++){     
            //     subPartitionsForGpus[gpu].push_back(partitionedByGpu[gpu]);
            //     lengthPartitionIdsForGpus[gpu].push_back(lengthPartitionId);
            // }
            for(int gpu = 0; gpu < numGpus; gpu++){
                if(gpu < int(partitionedByGpu.size())){
                    subPartitionsForGpus[gpu].push_back(partitionedByGpu[gpu]);
                    lengthPartitionIdsForGpus[gpu].push_back(lengthPartitionId);
                }else{
                    //add empty partition
                    subPartitionsForGpus[gpu].push_back(DBdataView(dbChunk, 0, 0));
                    lengthPartitionIdsForGpus[gpu].push_back(0);
                }
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






    


    const int results_per_query = std::min(size_t(options.numTopOutputs), totalNumberOfSequencesInDB);
    MyPinnedBuffer<float> alignment_scores_float(numQueries *numDBChunks * results_per_query);
    MyPinnedBuffer<size_t> sorted_indices(numQueries *numDBChunks * results_per_query);
    MyPinnedBuffer<int> resultDbChunkIndices(numQueries *numDBChunks * results_per_query);
    MyPinnedBuffer<int> resultNumOverflows(numQueries *numDBChunks);

    //set up gpus



    std::vector<std::unique_ptr<GpuWorkingSet>> workingSets(numGpus);  


    std::cout << "Allocate Memory: \n";
    //nvtx::push_range("ALLOC_MEM", 0);
	helpers::CpuTimer allocTimer("ALLOC_MEM");

    for(int gpu = 0; gpu < numGpus; gpu++){
        cudaSetDevice(deviceIds[gpu]);

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

        size_t maxGpuMem = options.maxGpuMem;

        size_t freeMem, totalMem;
        cudaMemGetInfo(&freeMem, &totalMem);
        constexpr size_t safety = 256*1024*1024;
        size_t memlimit = std::min(freeMem, maxGpuMem);
        if(memlimit > safety){
            memlimit -= safety;
        }

        std::cout << "gpu " << gpu << " may use " << memlimit << " bytes. ";

        workingSets[gpu] = std::make_unique<GpuWorkingSet>(
            memlimit,
            numQueries,
            totalNumQueryBytes,
            options,
            subPartitionsForGpus_perDBchunk[0][gpu]
        );

        if(workingSets[gpu]->canStoreFullDB){
            std::cout << "It can store its DB in memory\n";
        }else{
            std::cout << "It will process its DB in batches\n";
        }

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
    std::vector<std::vector<std::vector<DeviceBatchCopyToPinnedPlan>>> batchPlans_fulldb_perChunk(numDBChunks);

    MyDeviceBuffer<char> d_fulldb_chardata;
    MyDeviceBuffer<size_t> d_fulldb_lengthdata;
    MyDeviceBuffer<size_t> d_fulldb_offsetdata;
    
    for(int chunkId = 0; chunkId < numDBChunks; chunkId++){
        batchPlans_perChunk[chunkId].resize(numGpus);
        batchPlans_fulldb_perChunk[chunkId].resize(numGpus);

        for(int gpu = 0; gpu < numGpus; gpu++){
            const auto& ws = *workingSets[gpu];
            
            batchPlans_perChunk[chunkId][gpu] = computeDbCopyPlan(
                subPartitionsForGpus_perDBchunk[chunkId][gpu],
                lengthPartitionIdsForGpus_perDBchunk[chunkId][gpu],
                sizeof(char) * ws.h_chardata_vec[0].size(),
                ws.h_lengthdata_vec[0].size()
            );
            std::cout << "Batch plan chunk " << chunkId << ", gpu " << gpu << ": " << batchPlans_perChunk[chunkId][gpu].size() << " batches\n";

            if(ws.canStoreFullDB){
                batchPlans_fulldb_perChunk[chunkId][gpu] = computeDbCopyPlan(
                    subPartitionsForGpus_perDBchunk[chunkId][gpu],
                    lengthPartitionIdsForGpus_perDBchunk[chunkId][gpu],
                    sizeof(char) * ws.d_fulldb_chardata.size(),
                    ws.d_fulldb_lengthdata.size()
                );
                std::cout << "Batch plan fulldb chunk " << chunkId << ", gpu " << gpu << ": " << batchPlans_fulldb_perChunk[chunkId][gpu].size() << " batches\n";
            }else{
                batchPlans_fulldb_perChunk[chunkId][gpu] = batchPlans_perChunk[chunkId][gpu];
            }

            // for(int i = 0; i < std::min(5, int(batchPlans_perChunk[chunkId][gpu].size())); i++){
            //     std::cout << batchPlans_perChunk[chunkId][gpu][i] << "\n";
            // }
        }
    }

    std::vector<size_t> sequencesInPartitions(numGpus * numLengthPartitions);
    for(int gpu = 0; gpu < numGpus; gpu++){
        assert(subPartitionsForGpus_perDBchunk[0][gpu].size() == numLengthPartitions);
        for(int i = 0; i < numLengthPartitions; i++){
            sequencesInPartitions[gpu * numLengthPartitions + i] = subPartitionsForGpus_perDBchunk[0][gpu][i].numSequences();
        }
    }


    HostGpuPartitionOffsets hostGpuPartitionOffsets(numGpus, numLengthPartitions, sequencesInPartitions);
    for(int gpu = 0; gpu < numGpus; gpu++){
        cudaSetDevice(deviceIds[gpu]); CUERR;
        auto& ws = *workingSets[gpu];
        ws.setPartitionOffsets(hostGpuPartitionOffsets);
    }

 




    // for(int gpu = 0; gpu < numGpus; gpu++){
    //     cudaSetDevice(deviceIds[gpu]);
    //     const int chunkId = 0;
    //     if(true && batchPlans_perChunk[chunkId][gpu].size() == 1){
    //         auto& ws = *workingSets[gpu];
    //         const auto& plan = batchPlans_perChunk[chunkId][gpu][0];
    //         std::cout << "Upload single batch DB to gpu " << gpu << "\n";
    //         helpers::CpuTimer copyTimer("copy db");
    //         const int currentBuffer = 0;

    //         cudaStream_t H2DcopyStream = ws.copyStreams[currentBuffer];

    //         executeCopyPlanH2DDirect(
    //             plan,
    //             ws.d_chardata_vec[currentBuffer].data(),
    //             ws.d_lengthdata_vec[currentBuffer].data(),
    //             ws.d_offsetdata_vec[currentBuffer].data(),
    //             subPartitionsForGpus_perDBchunk[chunkId][gpu],
    //             H2DcopyStream
    //         );
            
    //         cudaStreamSynchronize(H2DcopyStream); CUERR;
    //         copyTimer.print();
    //         //ws.singleBatchDBisOnGpu = true;
    //     }
    // }

    //set blosum for new kernels
    setProgramWideBlosum(options.blosumType);

    cudaSetDevice(masterDeviceId);

    MyDeviceBuffer<float> devAllAlignmentScoresFloat(512 * 1024 * numGpus);
    MyDeviceBuffer<size_t> dev_sorted_indices(512 * 1024 * numGpus);
    MyDeviceBuffer<int> d_resultNumOverflows(numGpus);
    


    cudaSetDevice(masterDeviceId);

    std::vector<std::unique_ptr<helpers::GpuTimer>> queryTimers;
    for(int i = 0; i < numQueries; i++){
        queryTimers.emplace_back(std::make_unique<helpers::GpuTimer>(masterStream1, "Query " + std::to_string(i)));
    }

    std::cout << "Starting FULLSCAN_CUDA: \n";
    helpers::GpuTimer fullscanTimer(masterStream1, "FULLSCAN_CUDA");


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
                batchPlans_fulldb_perChunk[chunkId],
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

                //we could sort the maxReduceArray here as well and only send the best results_per_query entries

                if(numGpus > 1){
                    //transform per gpu local sequence indices into global sequence indices
                    helpers::lambda_kernel<<<SDIV(512*1024, 128), 128, 0, gpuStreams[gpu]>>>(
                        [
                            gpu, 
                            //N = numSequencesPerGpu_perDBchunk[chunkId][gpu],
                            N = 512*1024,
                            partitionOffsets = ws.deviceGpuPartitionOffsets.getDeviceView(),
                            d_maxReduceArrayIndices = ws.d_maxReduceArrayIndices.data()
                        ] __device__ (){
                            const int tid = threadIdx.x + blockIdx.x * blockDim.x;

                            if(tid < N){
                                d_maxReduceArrayIndices[tid] = partitionOffsets.getGlobalIndex(gpu, d_maxReduceArrayIndices[tid]);
                            }
                        }
                    ); CUERR;
                }

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
                cudaMemcpyAsync(
                    d_resultNumOverflows.data() + gpu,
                    ws.d_total_overflow_number.data(),
                    sizeof(int),
                    cudaMemcpyDeviceToDevice,
                    gpuStreams[gpu]
                ); CUERR;                

                cudaEventRecord(ws.forkStreamEvent, gpuStreams[gpu]); CUERR;

                cudaSetDevice(masterDeviceId);
                cudaStreamWaitEvent(masterStream1, ws.forkStreamEvent, 0); CUERR;
            }

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

            if(numGpus > 1){
                //sum the overflows per gpu
                helpers::lambda_kernel<<<1,1,0,masterStream1>>>(
                    [
                        numGpus,
                        d_resultNumOverflows = d_resultNumOverflows.data()
                    ]__device__ (){
                        int sum = d_resultNumOverflows[0];
                        for(int gpu = 1; gpu < numGpus; gpu++){
                            sum += d_resultNumOverflows[gpu];
                        }
                        d_resultNumOverflows[0] = sum;
                    }
                ); CUERR;
            }

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
            cudaMemcpyAsync(
                &(resultNumOverflows[query_num*numDBChunks + chunkId]), 
                d_resultNumOverflows.data(), 
                sizeof(int), 
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

    if(options.numTopOutputs > 0 && !options.usePseudoDB){

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

        for (int i=0; i<numQueries; i++) {
            int numOverflows = 0;
            for(int chunkId = 0; chunkId < numDBChunks; chunkId++){
                numOverflows += resultNumOverflows[i*numDBChunks + chunkId];
            }
            std::cout << numOverflows << " total overflow positions \n";
            std::cout << "Query Length:" << queryLengths[i] << " Header: ";
            std::cout << batchOfQueries.headers[i] << '\n';

            for(int j = 0; j < results_per_query; ++j) {
                const int arrayIndex = i*results_per_query+j;
                const size_t sortedIndex = final_sorted_indices[arrayIndex];
                //std::cout << "sortedIndex " << sortedIndex << "\n";
                const int dbChunkIndex = final_resultDbChunkIndices[arrayIndex];
                
                const auto& chunkData = fullDB.getChunks()[dbChunkIndex];

                const char* headerBegin = chunkData.headers() + chunkData.headerOffsets()[sortedIndex];
                const char* headerEnd = chunkData.headers() + chunkData.headerOffsets()[sortedIndex+1];
                std::cout << "Result: "<< j <<", Length: " << chunkData.lengths()[sortedIndex] << " Score: " << final_alignment_scores_float[arrayIndex] << " : ";
                std::copy(headerBegin, headerEnd,std::ostream_iterator<char>{std::cout});
                //cout << "\n";

                std::cout << "\n";
                //std::cout << " dbChunkIndex " << dbChunkIndex << "\n";
                //std::cout << "sortedIndex " << sortedIndex << "\n";

                // std::copy(chunkData.chars() + chunkData.offsets()[sortedIndex], 
                //     chunkData.chars() + chunkData.offsets()[sortedIndex] + chunkData.lengths()[sortedIndex] ,
                //     std::ostream_iterator<char>{cout});
                // std::transform(chunkData.chars() + chunkData.offsets()[sortedIndex], 
                //     chunkData.chars() + chunkData.offsets()[sortedIndex] + chunkData.lengths()[sortedIndex] ,
                //     std::ostream_iterator<char>{cout},
                //     inverse_convert_AA);
                // std::cout << "\n";
            }
        }

        CUERR;

    }

}
