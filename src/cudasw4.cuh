#ifndef CUDASW4_CUH
#define CUDASW4_CUH

#include "hpc_helpers/cuda_raiiwrappers.cuh"
#include "hpc_helpers/all_helpers.cuh"
#include "hpc_helpers/nvtx_markers.cuh"
#include "hpc_helpers/simple_allocation.cuh"

#include "config.hpp"
#include "kseqpp/kseqpp.hpp"
#include "dbdata.hpp"
#include "length_partitions.hpp"
#include "util.cuh"
#include "kernels.cuh"
#include "blosum.hpp"
#include "types.hpp"
#include "dbbatching.cuh"
#include "convert.cuh"

#include "gpudatabaseallocation.cuh"

#include <thrust/binary_search.h>
#include <thrust/sort.h>
#include <thrust/equal.h>
#include <thrust/merge.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/constant_iterator.h>

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <string_view>

namespace cudasw4{

    template<class T, T factor>
    struct RoundToNextMultiple{
        __host__ __device__ 
        T operator()(const T& value){
            return SDIV(value, factor) * factor;
        }        
    };

    __global__
    void addKernel(int* output, const int* input1, const int* input2){
        *output = *input1 + *input2;
    }


    __global__
    void sumNumOverflowsKernel(int* output, const int* input, int numGpus){
        int sum = input[0];
        for(int gpu = 1; gpu < numGpus; gpu++){
            sum += input[gpu];
        }
        output[0] = sum;
    }

    template<class PartitionOffsets, class Indices>
    __global__
    void transformLocalSequenceIndicesToGlobalIndices(
        int gpu,
        int N,
        PartitionOffsets partitionOffsets,
        Indices maxReduceArrayIndices
    ){
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;

        if(tid < N){
            maxReduceArrayIndices[tid] = partitionOffsets.getGlobalIndex(gpu, maxReduceArrayIndices[tid]);
        }
    }

    struct BenchmarkStats{
        int numOverflows{};
        double seconds{};
        double gcups{};
    };

    struct ScanResult{
        std::vector<int> scores{};
        std::vector<ReferenceIdT> referenceIds{};
        BenchmarkStats stats{};
    };

    struct KernelTypeConfig{
        KernelType singlePassType = KernelType::Half2;
        KernelType manyPassType_small = KernelType::Half2;
        KernelType manyPassType_large = KernelType::Float;
        KernelType overflowType = KernelType::Float;
    };

    struct MemoryConfig{
        size_t maxBatchBytes = 128ull * 1024ull * 1024ull;
        size_t maxBatchSequences = 10'000'000;
        size_t maxTempBytes = 4ull * 1024ull * 1024ull * 1024ull;
        size_t maxGpuMem = std::numeric_limits<size_t>::max();
    };

    struct HostGpuPartitionOffsets{
        int numGpus;
        int numLengthPartitions;
        std::vector<size_t> partitionSizes;
        std::vector<size_t> horizontalPS;
        std::vector<size_t> verticalPS;
        std::vector<size_t> totalPerLengthPartitionPS;

        HostGpuPartitionOffsets() = default;

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
        template<class T>
        using MyDeviceBuffer = helpers::SimpleAllocationDevice<T, 0>;

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


    class CudaSW4{
    public:
        template<class T>
        using MyPinnedBuffer = helpers::SimpleAllocationPinnedHost<T, 0>;
        template<class T>
        using MyDeviceBuffer = helpers::SimpleAllocationDevice<T, 0>;

        struct GpuWorkingSet{
        
            GpuWorkingSet(
                size_t gpumemlimit,
                size_t maxBatchBytes,
                size_t maxBatchSequences,
                size_t maxTempBytes,
                const std::vector<DBdataView>& dbPartitions,
                const std::vector<DeviceBatchCopyToPinnedPlan>& dbBatches,
                int maxBatchResultListSize_ = 512 * 1024
            ) : maxBatchResultListSize(maxBatchResultListSize_)
            {
                cudaGetDevice(&deviceId);

                size_t numSubjects = 0;
                size_t numSubjectBytes = 0;
                for(const auto& p : dbPartitions){
                    numSubjects += p.numSequences();
                    numSubjectBytes += p.numChars();
                }
        
                d_query.resize(1024*1024); CUERR

                numTempBytes = std::min(maxTempBytes, gpumemlimit);
                d_tempStorageHE.resize(numTempBytes);
                d_batchResultListScores.resize(numSubjects);
                d_batchResultListRefIds.resize(numSubjects);    

                size_t usedGpuMem = 0;
                usedGpuMem += numTempBytes;
                usedGpuMem += sizeof(float) * numSubjects; // d_batchResultListScores
                usedGpuMem += sizeof(ReferenceIdT) * numSubjects; // d_batchResultListRefIds

                if(usedGpuMem > gpumemlimit){
                    throw std::runtime_error("Out of memory working set");
                }              
        
                //devAlignmentScoresFloat.resize(numSubjects);
        
                forkStreamEvent = CudaEvent{cudaEventDisableTiming}; CUERR;
                numWorkStreamsWithoutTemp = 10;
                workstreamIndex = 0;
                workStreamsWithoutTemp.resize(numWorkStreamsWithoutTemp);

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

                //only transfer buffers are double buffers
                //do not need a double buffer for d_overflow_positions since batches are processed one at a time.
                //d_overflow_number remains double buffer for simpler initialization of counter
                d_overflow_positions_vec.resize(1);

                size_t memoryRequiredForFullDB = 0;
                memoryRequiredForFullDB += numSubjectBytes; // d_fulldb_chardata
                memoryRequiredForFullDB += sizeof(SequenceLengthT) * numSubjects; //d_fulldb_lengthdata
                memoryRequiredForFullDB += sizeof(size_t) * (numSubjects+1); //d_fulldb_offsetdata
                memoryRequiredForFullDB += sizeof(ReferenceIdT) * numSubjects * 1; //d_overflow_positions_vec
        
               

                if(usedGpuMem + memoryRequiredForFullDB <= gpumemlimit){
                    numBatchesInCachedDB = dbBatches.size();
                    charsOfBatches = numSubjectBytes;
                    subjectsOfBatches = numSubjects;
                    d_cacheddb = std::make_shared<GpuDatabaseAllocation>(numSubjectBytes, numSubjects);

                    for(int i = 0; i < numCopyBuffers; i++){
                        h_chardata_vec[i].resize(maxBatchBytes);
                        h_lengthdata_vec[i].resize(maxBatchSequences);
                        h_offsetdata_vec[i].resize(maxBatchSequences+1);               
                        pinnedBufferEvents[i] = CudaEvent{cudaEventDisableTiming}; CUERR;
                        deviceBufferEvents[i] = CudaEvent{cudaEventDisableTiming}; CUERR;
                        // d_overflow_positions_vec[i].resize(numSubjects);
                    }
                    d_overflow_positions_vec[0].resize(numSubjects);
                }else{
                    //allocate a double buffer for batch transfering
                    size_t memoryRequiredForBatchedProcessing = 0;
                    memoryRequiredForBatchedProcessing += maxBatchBytes * 2; // d_chardata_vec
                    memoryRequiredForBatchedProcessing += sizeof(SequenceLengthT) * maxBatchSequences * 2; //d_lengthdata_vec
                    memoryRequiredForBatchedProcessing += sizeof(size_t) * (maxBatchSequences+1) * 2; //d_offsetdata_vec
                    usedGpuMem += memoryRequiredForBatchedProcessing;
                    if(usedGpuMem > gpumemlimit){
                        throw std::runtime_error("Out of memory working set");
                    }
                    
                    for(int i = 0; i < numCopyBuffers; i++){
                        h_chardata_vec[i].resize(maxBatchBytes);
                        h_lengthdata_vec[i].resize(maxBatchSequences);
                        h_offsetdata_vec[i].resize(maxBatchSequences+1);
                        d_chardata_vec[i].resize(maxBatchBytes);
                        d_lengthdata_vec[i].resize(maxBatchSequences);
                        d_offsetdata_vec[i].resize(maxBatchSequences+1);
                        pinnedBufferEvents[i] = CudaEvent{cudaEventDisableTiming}; CUERR;
                        deviceBufferEvents[i] = CudaEvent{cudaEventDisableTiming}; CUERR;
                        // d_overflow_positions_vec[i].resize(maxBatchSequences);
                    }
                    
                    //count how many batches fit into remaining gpu memory
                    
                    numBatchesInCachedDB = 0;
                    charsOfBatches = 0;
                    subjectsOfBatches = 0;
                    size_t totalRequiredMemForBatches = sizeof(size_t);
                    for(; numBatchesInCachedDB < dbBatches.size(); numBatchesInCachedDB++){
                        const auto& batch = dbBatches[numBatchesInCachedDB];
                        const size_t requiredMemForBatch = batch.usedSeq * sizeof(SequenceLengthT) //lengths
                            + batch.usedSeq * sizeof(size_t) //offsets
                            + batch.usedBytes //chars
                            + batch.usedSeq * sizeof(ReferenceIdT); //overflow positions
                        if(usedGpuMem + totalRequiredMemForBatches + requiredMemForBatch <= gpumemlimit){
                            //ok, fits
                            totalRequiredMemForBatches += requiredMemForBatch;
                            charsOfBatches += batch.usedBytes;
                            subjectsOfBatches += batch.usedSeq;
                        }else{
                            //does not fit
                            break;
                        }
                    }
                    assert(numBatchesInCachedDB < dbBatches.size());
                    d_overflow_positions_vec[0].resize(std::max(maxBatchSequences, subjectsOfBatches));

                    //std::cout << "numBatchesInCachedDB " << numBatchesInCachedDB << ", charsOfBatches " << charsOfBatches << ", subjectsOfBatches " << subjectsOfBatches << "\n";

                    assert(usedGpuMem + totalRequiredMemForBatches <= gpumemlimit);
                    d_cacheddb = std::make_shared<GpuDatabaseAllocation>(charsOfBatches, subjectsOfBatches);
                }
            }
        
            BatchResultList getBatchResultList(size_t offset){
                return BatchResultList(
                    d_batchResultListScores.data(), 
                    d_batchResultListRefIds.data(), 
                    offset,
                    maxBatchResultListSize
                );
            }
        
            void resetBatchResultList(cudaStream_t stream){
                thrust::fill(thrust::cuda::par_nosync.on(stream),
                    d_batchResultListScores.data(),
                    d_batchResultListScores.data() + maxBatchResultListSize,
                    -1.f
                );
                cudaMemsetAsync(d_batchResultListRefIds.data(), 0, sizeof(ReferenceIdT) * maxBatchResultListSize, stream);
            }
        
            void setPartitionOffsets(const HostGpuPartitionOffsets& offsets){
                deviceGpuPartitionOffsets = DeviceGpuPartitionOffsets(offsets);
            }
            
            size_t getNumCharsInCachedDB() const{
                return charsOfBatches;
            }

            size_t getNumSequencesInCachedDB() const{
                return subjectsOfBatches;
            }

            size_t getNumBatchesInCachedDB() const{
                return numBatchesInCachedDB;
            }

            void allocateTempStorageHE(){
                d_tempStorageHE.resize(numTempBytes);
            }

            void deallocateTempStorageHE(){
                d_tempStorageHE.destroy();
            }
        

            int deviceId;
            int numCopyBuffers;
            int numWorkStreamsWithoutTemp = 1;
            int workstreamIndex;
            int copyBufferIndex = 0;
            int maxBatchResultListSize = 512 * 1024;
            size_t numTempBytes;
            size_t numBatchesInCachedDB = 0;
            size_t charsOfBatches = 0;
            size_t subjectsOfBatches = 0;
        
            MyDeviceBuffer<float> d_batchResultListScores;
            MyDeviceBuffer<ReferenceIdT> d_batchResultListRefIds;
        
            MyDeviceBuffer<char> d_query;
            MyDeviceBuffer<char> d_tempStorageHE;

            MyDeviceBuffer<int> d_total_overflow_number;
            MyDeviceBuffer<int> d_overflow_number;
            MyPinnedBuffer<int> h_overflow_number;
            CudaStream hostFuncStream;
            CudaStream workStreamForTempUsage;
            CudaEvent forkStreamEvent;
        
            size_t maxNumBatchesInCachedDB = 0;
            std::shared_ptr<GpuDatabaseAllocationBase> d_cacheddb;

            
            std::vector<MyPinnedBuffer<char>> h_chardata_vec;
            std::vector<MyPinnedBuffer<SequenceLengthT>> h_lengthdata_vec;
            std::vector<MyPinnedBuffer<size_t>> h_offsetdata_vec;
            std::vector<MyDeviceBuffer<char>> d_chardata_vec;
            std::vector<MyDeviceBuffer<SequenceLengthT>> d_lengthdata_vec;
            std::vector<MyDeviceBuffer<size_t>> d_offsetdata_vec;
            std::vector<CudaStream> copyStreams;
            std::vector<CudaEvent> pinnedBufferEvents;
            std::vector<CudaEvent> deviceBufferEvents;
            std::vector<CudaStream> workStreamsWithoutTemp;
            std::vector<MyDeviceBuffer<ReferenceIdT>> d_overflow_positions_vec;
        
            DeviceGpuPartitionOffsets deviceGpuPartitionOffsets;
        
        };

        struct SequenceLengthStatistics{
            SequenceLengthT max_length = 0;
            SequenceLengthT min_length = std::numeric_limits<SequenceLengthT>::max();
            size_t sumOfLengths = 0;
        };
    private:
        struct BatchDstInfo{
            bool isUploaded{};
            char* charsPtr{};
            SequenceLengthT* lengthsPtr{};
            size_t* offsetsPtr{};
        };

    public:
        CudaSW4(
            std::vector<int> deviceIds_, 
            int numTop,
            BlosumType blosumType,
            const KernelTypeConfig& kernelTypeConfig,
            const MemoryConfig& memoryConfig,
            bool verbose_
        ) : deviceIds(std::move(deviceIds_)), verbose(verbose_)
        {
            #ifdef CUDASW_DEBUG_CHECK_CORRECTNESS
                blosumType = BlosumType::BLOSUM62_20;
            #endif
            if(deviceIds.size() == 0){ 
                throw std::runtime_error("No device selected");
            
            }
            RevertDeviceId rdi{};

            initializeGpus();

            h_resultNumOverflows.resize(1);

            const int numGpus = deviceIds.size();
            cudaSetDevice(deviceIds[0]);
            
            d_resultNumOverflows.resize(numGpus);
            scanTimer = std::make_unique<helpers::GpuTimer>("Scan");
            totalTimer = std::make_unique<helpers::GpuTimer>("Total");

            setBlosum(blosumType);
            setNumTop(numTop);
            setKernelTypeConfig(kernelTypeConfig);
            setMemoryConfig(memoryConfig);

            dbIsReady = false;
        }

        CudaSW4() = delete;
        CudaSW4(const CudaSW4&) = delete;
        CudaSW4(CudaSW4&&) = default;
        CudaSW4& operator=(const CudaSW4&) = delete;
        CudaSW4& operator=(CudaSW4&&) = default;

        void setGapOpenScore(int score){
            if(score >= 0){
                std::cout << "Warning, gap open score set to non-negative value. Is this intended?\n";
            }
            gop = score;
        }
        void setGapExtendScore(int score){
            if(score >= 0){
                std::cout << "Warning, gap extend score set to non-negative value. Is this intended?\n";
            }
            gex = score;
        }

        void setDatabase(std::shared_ptr<DB> dbPtr){
            RevertDeviceId rdi{};
            fullDB = AnyDBWrapper(dbPtr);
            makeReady();
        }

        void setDatabase(std::shared_ptr<DBWithVectors> dbPtr){
            RevertDeviceId rdi{};
            fullDB = AnyDBWrapper(dbPtr);
            makeReady();
        }

        void setDatabase(std::shared_ptr<PseudoDB> dbPtr){
            RevertDeviceId rdi{};
            fullDB = AnyDBWrapper(dbPtr);
            makeReady();
        }

        void setBlosum(BlosumType blosumType){
            setProgramWideBlosum(blosumType, deviceIds);
        }

        void setNumTop(int value){
            if(value >= 0){
                numTop = value;
                updateNumResultsPerQuery();

                cub::SwitchDevice sd(deviceIds[0]);
                const int numGpus = deviceIds.size();           

                h_finalAlignmentScores.resize(results_per_query);
                h_finalReferenceIds.resize(results_per_query);
                d_finalAlignmentScores_allGpus.resize(results_per_query * numGpus);
                d_finalReferenceIds_allGpus.resize(results_per_query * numGpus);                
            }
        }

        void setKernelTypeConfig(const KernelTypeConfig& val){
            if(!isValidSinglePassType(val.singlePassType)){
                throw std::runtime_error("Invalid singlepass kernel type");
            }

            if(!isValidMultiPassType_small(val.manyPassType_small)){
                throw std::runtime_error("Invalid manyPassType_small kernel type");
            }

            if(!isValidMultiPassType_large(val.manyPassType_large)){
                throw std::runtime_error("Invalid manyPassType_large kernel type");
            }

            if(!isValidOverflowType(val.overflowType)){
                throw std::runtime_error("Invalid overflow kernel type");
            }

            kernelTypeConfig = val;
        }

        void setMemoryConfig(const MemoryConfig& val){
            memoryConfig = val;
        }

        std::string_view getReferenceHeader(ReferenceIdT referenceId) const{
            const auto& data = fullDB.getData();
            const char* const headerBegin = data.headers() + data.headerOffsets()[referenceId];
            const char* const headerEnd = data.headers() + data.headerOffsets()[referenceId+1];
            return std::string_view(headerBegin, std::distance(headerBegin, headerEnd));
        }

        int getReferenceLength(ReferenceIdT referenceId) const{
            const auto& data = fullDB.getData();
            return data.lengths()[referenceId];
        }

        std::string getReferenceSequence(ReferenceIdT referenceId) const{
            const auto& data = fullDB.getData();
            const char* const begin = data.chars() + data.offsets()[referenceId];
            const char* const end = begin + getReferenceLength(referenceId);

            std::string sequence(end - begin, '\0');
            std::transform(
                begin, 
                end,
                sequence.begin(),
                InverseConvertAA_20{}
            );

            return sequence;
        }

        void markCachedDBBatchesAsUploaded(int gpu){
            auto& ws = *workingSets[gpu];
            if(ws.getNumBatchesInCachedDB() > 0){
                batchPlansDstInfoVec_cachedDB[gpu][0].isUploaded = true;
                for(size_t i = 0; i < ws.getNumBatchesInCachedDB(); i++){
                    batchPlansDstInfoVec[gpu][i].isUploaded = true;
                }
            }
        }

        void prefetchDBToGpus(){
            RevertDeviceId rdi{};

            const int numGpus = deviceIds.size();
            std::vector<int> copyIds;

            helpers::CpuTimer copyTimer("transfer DB to GPUs");
            for(int gpu = 0; gpu < numGpus; gpu++){
                cudaSetDevice(deviceIds[gpu]);
                auto& ws = *workingSets[gpu];

                if(ws.getNumBatchesInCachedDB() > 0){
                    const auto& plan = batchPlans_cachedDB[gpu][0];
                    const int currentBuffer = 0;
                    cudaStream_t H2DcopyStream = ws.copyStreams[currentBuffer];

                    executeCopyPlanH2DDirect(
                        plan,
                        ws.d_cacheddb->getCharData(),
                        ws.d_cacheddb->getLengthData(),
                        ws.d_cacheddb->getOffsetData(),
                        subPartitionsForGpus[gpu],
                        H2DcopyStream
                    );
                    
                    copyIds.push_back(gpu);

                    markCachedDBBatchesAsUploaded(gpu);
                }
            }
            for(int gpu = 0; gpu < numGpus; gpu++){
                cudaSetDevice(deviceIds[gpu]);
                cudaDeviceSynchronize(); CUERR;
            }
            copyTimer.stop();
            if(copyIds.size() > 0){
                if(verbose){
                    std::cout << "Transferred DB data in advance to GPU(s) ";
                    for(auto x : copyIds){
                        std::cout << x << " ";
                    }
                    std::cout << "\n";
                    copyTimer.print();
                }
            }
        }

        ScanResult scan(const char* query, SequenceLengthT queryLength){
            if(!dbIsReady){
                throw std::runtime_error("DB not set correctly");
            }
            RevertDeviceId rdi{};

            const int masterDeviceId = deviceIds[0];
            cudaSetDevice(masterDeviceId);

            scanTimer->reset();
            scanTimer->start();

            setQuery(query, queryLength);

            scanDatabaseForQuery();

            scanTimer->stop();

            totalProcessedQueryLengths += queryLength;
            totalNumOverflows += h_resultNumOverflows[0];

            const auto& sequenceLengthStatistics = getSequenceLengthStatistics();

            ScanResult result;
            result.stats = makeBenchmarkStats(
                scanTimer->elapsed() / 1000, 
                sequenceLengthStatistics.sumOfLengths * queryLength, 
                h_resultNumOverflows[0]
            );

            #ifdef CUDASW_DEBUG_CHECK_CORRECTNESS

            std::vector<int> cpuScores = computeAllScoresCPU_blosum62(query, queryLength);
            //bool checkOk = true;
            int numErrors = 0;
            for(size_t i = 0; i < cpuScores.size(); i++){
                const auto refId = h_finalReferenceIds[i];
                const int gpu = h_finalAlignmentScores[i];
                const int cpu = cpuScores[refId];
                if(cpu != gpu){
                    if(numErrors == 0){
                        std::cout << "error. i " << i << ", sequence id " << refId 
                            << ", cpu score " << cpu << ", gpu score " << gpu << ".";
                        std::cout << "Query:\n";
                        std::copy(query, query + queryLength, std::ostream_iterator<char>(std::cout, ""));
                        std::cout << "\n";
                        std::cout << "db sequence:\n";
                        std::cout << getReferenceSequence(refId) << "\n";
                    }
                    numErrors++;
                }
            }
            if(numErrors == 0){
                std::cout << "Check ok, cpu and gpu produced same results\n";
            }else{
                std::cout << "Check not ok!!! " << numErrors << " sequences produced different results\n";
            }

            //#endif
            #else

            result.scores.insert(result.scores.end(), h_finalAlignmentScores.begin(), h_finalAlignmentScores.begin() + results_per_query);
            result.referenceIds.insert(result.referenceIds.end(), h_finalReferenceIds.begin(), h_finalReferenceIds.begin() + results_per_query);
            
            #endif

            return result;
        }

        std::vector<int> computeAllScoresCPU_blosum62(const char* query, SequenceLengthT queryLength){
            const auto& view = fullDB.getData();
            size_t numSequences = view.numSequences();
            std::vector<int> result(numSequences);

            std::vector<char> convertedQuery(queryLength);
            std::transform(
                query,
                query + queryLength,
                convertedQuery.data(),
                ConvertAA_20{}
            );
            #pragma omp parallel for
            for(size_t i = 0; i < numSequences; i++){
                size_t offset = view.offsets()[i];
                int length = view.lengths()[i];
                const char* seq = view.chars() + offset;

                int score = affine_local_DP_host_protein_blosum62_converted(
                    convertedQuery.data(),
                    seq,
                    queryLength,
                    length,
                    gop,
                    gex
                );
                result[i] = score;
            }
            return result;
        }


        void printDBInfo() const{
            const size_t numSequences = fullDB.getData().numSequences();
            std::cout << numSequences << " sequences, " << fullDB.getData().numChars() << " characters\n";

            SequenceLengthStatistics stats = getSequenceLengthStatistics();

            std::cout << "Min length " << stats.min_length << ", max length " << stats.max_length 
                << ", avg length " << stats.sumOfLengths / numSequences << "\n";
        }

        void printDBLengthPartitions() const{
            auto lengthBoundaries = getLengthPartitionBoundaries();
            const int numLengthPartitions = getLengthPartitionBoundaries().size();

            for(int i = 0; i < numLengthPartitions; i++){
                std::cout << "<= " << lengthBoundaries[i] << ": " << fullDB_numSequencesPerLengthPartition[i] << "\n";
            }
        }

        void totalTimerStart(){
            RevertDeviceId rdi{};
            cudaSetDevice(deviceIds[0]);
            totalProcessedQueryLengths = 0;
            totalNumOverflows = 0;
            totalTimer->start();
        }

        BenchmarkStats totalTimerStop(){
            RevertDeviceId rdi{};
            cudaSetDevice(deviceIds[0]);
            totalTimer->stop();

            const auto& sequenceLengthStatistics = getSequenceLengthStatistics();
            BenchmarkStats stats = makeBenchmarkStats(
                totalTimer->elapsed() / 1000,
                totalProcessedQueryLengths * sequenceLengthStatistics.sumOfLengths,
                totalNumOverflows
            );

            return stats;
        }

        bool isValidSinglePassType(KernelType type) const{
            return (type == KernelType::Half2 || type == KernelType::DPXs16 || type == KernelType::DPXs32 || type == KernelType::Float);
        }

        bool isValidMultiPassType_small(KernelType type) const{
            return (type == KernelType::Half2 || type == KernelType::DPXs16);
        }

        bool isValidMultiPassType_large(KernelType type) const{
            return (type == KernelType::Float || type == KernelType::DPXs32);
        }

        bool isValidOverflowType(KernelType type) const{
            return (type == KernelType::Float || type == KernelType::DPXs32);
        }

    private:
        void initializeGpus(){
            const int numGpus = deviceIds.size();

            for(int i = 0; i < numGpus; i++){
                cudaSetDevice(deviceIds[i]); CUERR
                helpers::init_cuda_context(); CUERR
                cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);CUERR
        
                cudaMemPool_t mempool;
                cudaDeviceGetDefaultMemPool(&mempool, deviceIds[i]); CUERR
                uint64_t threshold = UINT64_MAX;
                cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);CUERR
            
                gpuStreams.emplace_back();
                gpuEvents.emplace_back(cudaEventDisableTiming);
            }
        }

        void makeReady(){
            const auto& dbData = fullDB.getData();
            const size_t numDBSequences = dbData.numSequences();
            maxBatchResultListSize = numDBSequences;

            #ifdef CUDASW_DEBUG_CHECK_CORRECTNESS
            if(numDBSequences > size_t(std::numeric_limits<int>::max()))
                throw std::runtime_error("cannot check correctness for this db size");

            results_per_query = maxBatchResultListSize;
            setNumTop(maxBatchResultListSize);
            #endif


            dbSequenceLengthStatistics = nullptr;

            computeTotalNumSequencePerLengthPartition();
            partitionDBAmongstGpus();

            createDBBatchesForGpus();
            allocateGpuWorkingSets();
            assignBatchesToGpuMem();
            
            
            dbIsReady = true;
            updateNumResultsPerQuery();
        }

        void computeTotalNumSequencePerLengthPartition(){
            auto lengthBoundaries = getLengthPartitionBoundaries();
            const int numLengthPartitions = getLengthPartitionBoundaries().size();

            fullDB_numSequencesPerLengthPartition.resize(numLengthPartitions);

            const auto& dbData = fullDB.getData();
            auto partitionBegin = dbData.lengths();
            for(int i = 0; i < numLengthPartitions; i++){
                //length k is in partition i if boundaries[i-1] < k <= boundaries[i]
                SequenceLengthT searchFor = lengthBoundaries[i];
                if(searchFor < std::numeric_limits<SequenceLengthT>::max()){
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

        void partitionDBAmongstGpus(){
            const int numGpus = deviceIds.size();
            const int numLengthPartitions = getLengthPartitionBoundaries().size();

            numSequencesPerLengthPartitionPrefixSum.clear();
            dbPartitionsByLengthPartitioning.clear();
            subPartitionsForGpus.clear();
            lengthPartitionIdsForGpus.clear();
            numSequencesPerGpu.clear();
            numSequencesPerGpuPrefixSum.clear();

            const auto& data = fullDB.getData();
    
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
                dbPartitionsByLengthPartitioning.emplace_back(data, begin, end);        
            }
    
            for(int lengthPartitionId = 0; lengthPartitionId < numLengthPartitions; lengthPartitionId++){
                const auto& lengthPartition = dbPartitionsByLengthPartitioning[lengthPartitionId];        
                const auto partitionedByGpu = partitionDBdata_by_numberOfChars(lengthPartition, lengthPartition.numChars() / numGpus);
        
                assert(int(partitionedByGpu.size()) <= numGpus);
                for(int gpu = 0; gpu < numGpus; gpu++){
                    if(gpu < int(partitionedByGpu.size())){
                        subPartitionsForGpus[gpu].push_back(partitionedByGpu[gpu]);
                        lengthPartitionIdsForGpus[gpu].push_back(lengthPartitionId);
                    }else{
                        //add empty partition
                        subPartitionsForGpus[gpu].push_back(DBdataView(data, 0, 0));
                        lengthPartitionIdsForGpus[gpu].push_back(0);
                    }
                }
            }
        
            for(int i = 0; i < numGpus; i++){
                for(const auto& p : subPartitionsForGpus[i]){
                    numSequencesPerGpu[i] += p.numSequences();
                }
            }
            for(int i = 0; i < numGpus-1; i++){
                numSequencesPerGpuPrefixSum[i+1] = numSequencesPerGpuPrefixSum[i] + numSequencesPerGpu[i];
            }
        
            numSequencesPerGpu_total.resize(numGpus);
            numSequencesPerGpuPrefixSum_total.resize(numGpus);
            numSequencesPerGpuPrefixSum_total[0] = 0;

        
            for(int i = 0; i < numGpus; i++){
                size_t num = numSequencesPerGpu[i];
                numSequencesPerGpu_total[i] = num;
                if(i < numGpus - 1){
                    numSequencesPerGpuPrefixSum_total[i+1] = numSequencesPerGpuPrefixSum_total[i] + num;
                }
            }

            std::vector<size_t> sequencesInPartitions(numGpus * numLengthPartitions);
            for(int gpu = 0; gpu < numGpus; gpu++){
                assert(subPartitionsForGpus[gpu].size() == numLengthPartitions);
                for(int i = 0; i < numLengthPartitions; i++){
                    sequencesInPartitions[gpu * numLengthPartitions + i] = subPartitionsForGpus[gpu][i].numSequences();
                }
            }
            hostGpuPartitionOffsets = HostGpuPartitionOffsets(numGpus, numLengthPartitions, std::move(sequencesInPartitions));
        }

        void allocateGpuWorkingSets(){
            const int numGpus = deviceIds.size();
            workingSets.clear();
            workingSets.resize(numGpus);

            if(verbose){
                std::cout << "Allocate Memory: \n";
            }
            //nvtx::push_range("ALLOC_MEM", 0);
            helpers::CpuTimer allocTimer("ALLOC_MEM");

            for(int gpu = 0; gpu < numGpus; gpu++){
                cudaSetDevice(deviceIds[gpu]);

                size_t freeMem, totalMem;
                cudaMemGetInfo(&freeMem, &totalMem);
                constexpr size_t safety = 256*1024*1024;
                size_t memlimit = std::min(freeMem, memoryConfig.maxGpuMem);
                if(memlimit > safety){
                    memlimit -= safety;
                }

                if(verbose){
                    std::cout << "gpu " << gpu << " may use " << memlimit << " bytes. ";
                }

                workingSets[gpu] = std::make_unique<GpuWorkingSet>(
                    memlimit,
                    memoryConfig.maxBatchBytes,
                    memoryConfig.maxBatchSequences,
                    memoryConfig.maxTempBytes,
                    subPartitionsForGpus[gpu],
                    batchPlans[gpu],
                    maxBatchResultListSize
                );

                if(verbose){
                    std::cout << "Using " << workingSets[gpu]->numTempBytes << " temp bytes. ";
                }
                if(verbose){
                    std::cout << workingSets[gpu]->getNumBatchesInCachedDB() << " out of " << batchPlans[gpu].size() << " DB batches will be cached in gpu memory\n";
                }

                //set gpu partition table
                workingSets[gpu]->setPartitionOffsets(hostGpuPartitionOffsets);

                //spin up the host callback thread
                auto noop = [](void*){};
                cudaLaunchHostFunc(
                    gpuStreams[gpu], 
                    noop, 
                    nullptr
                ); CUERR
            }    

            if(verbose){
                allocTimer.print();
            }
        }

        void createDBBatchesForGpus(){
            const int numGpus = deviceIds.size();

            batchPlans.clear();
            batchPlans.resize(numGpus);
            batchPlans_cachedDB.clear();
            batchPlans_cachedDB.resize(numGpus);
    
            for(int gpu = 0; gpu < numGpus; gpu++){
                batchPlans[gpu] = computeDbCopyPlan(
                    subPartitionsForGpus[gpu],
                    lengthPartitionIdsForGpus[gpu],
                    memoryConfig.maxBatchBytes,
                    memoryConfig.maxBatchSequences
                );
                if(verbose){
                    std::cout << "Batch plan gpu " << gpu << ": " << batchPlans[gpu].size() << " batches\n";
                }
            }
        }

        void assignBatchesToGpuMem(){
            const int numGpus = deviceIds.size();
            batchPlansDstInfoVec.clear();
            batchPlansDstInfoVec.resize(numGpus);
            batchPlansDstInfoVec_cachedDB.clear();
            batchPlansDstInfoVec_cachedDB.resize(numGpus);
    
            for(int gpu = 0; gpu < numGpus; gpu++){
                cudaSetDevice(deviceIds[gpu]);
                auto& ws = *workingSets[gpu];
                if(ws.getNumBatchesInCachedDB() > 0){
                    //can cache full db in gpu mem

                    auto plansForCachedDB = computeDbCopyPlan(
                        subPartitionsForGpus[gpu],
                        lengthPartitionIdsForGpus[gpu],
                        sizeof(char) * ws.getNumCharsInCachedDB(),
                        ws.getNumSequencesInCachedDB()
                    );
                    assert(plansForCachedDB.size() >= 1);
                    plansForCachedDB.erase(plansForCachedDB.begin() + 1, plansForCachedDB.end());
                    batchPlans_cachedDB[gpu] = plansForCachedDB;
                    // if(verbose){
                    //     std::cout << "Cached db single batch plan " << plansForCachedDB[0] << "\n";
                    // }

                    BatchDstInfo dstInfo;
                    dstInfo.isUploaded = false;
                    dstInfo.charsPtr = ws.d_cacheddb->getCharData();
                    dstInfo.lengthsPtr = ws.d_cacheddb->getLengthData();
                    dstInfo.offsetsPtr = ws.d_cacheddb->getOffsetData();
                    batchPlansDstInfoVec_cachedDB[gpu].push_back(dstInfo);
                }

                {
                    BatchDstInfo dstInfo;
                    dstInfo.isUploaded = false;
                    dstInfo.charsPtr = ws.d_cacheddb->getCharData();
                    dstInfo.lengthsPtr = ws.d_cacheddb->getLengthData();
                    dstInfo.offsetsPtr = ws.d_cacheddb->getOffsetData();

                    for(size_t i = 0; i < ws.getNumBatchesInCachedDB(); i++){
                        batchPlansDstInfoVec[gpu].push_back(dstInfo);
                        const auto& plan = batchPlans[gpu][i];
                        dstInfo.charsPtr += plan.usedBytes;
                        dstInfo.lengthsPtr += plan.usedSeq;
                        dstInfo.offsetsPtr += plan.usedSeq;
                    }

                    for(size_t i = ws.getNumBatchesInCachedDB(), buf = 0; i < batchPlans[gpu].size(); i++, buf = (buf+1)%ws.numCopyBuffers){
                        dstInfo.charsPtr = ws.d_chardata_vec[buf].data();
                        dstInfo.lengthsPtr = ws.d_lengthdata_vec[buf].data();
                        dstInfo.offsetsPtr = ws.d_offsetdata_vec[buf].data();
                        batchPlansDstInfoVec[gpu].push_back(dstInfo);
                    }
                }
            }
        }

        void printDBDataView(const DBdataView& view) const{
            std::cout << "Sequences: " << view.numSequences() << "\n";
            std::cout << "Chars: " << view.offsets()[0] << " - " << view.offsets()[view.numSequences()] << " (" << (view.offsets()[view.numSequences()] - view.offsets()[0]) << ")"
                << " " << view.numChars() << "\n";
        }

        void printDBDataViews(const std::vector<DBdataView>& views) const {
            size_t numViews = views.size();
            for(size_t p = 0; p < numViews; p++){
                const DBdataView& view = views[p];
        
                std::cout << "View " << p << "\n";
                printDBDataView(view);
            }
        }

        SequenceLengthStatistics getSequenceLengthStatistics() const{
            if(dbSequenceLengthStatistics == nullptr){
                dbSequenceLengthStatistics = std::make_unique<SequenceLengthStatistics>();
                const auto& data = fullDB.getData();
                size_t numSeq = data.numSequences();

                for (size_t i=0; i < numSeq; i++) {
                    if (data.lengths()[i] > dbSequenceLengthStatistics->max_length) dbSequenceLengthStatistics->max_length = data.lengths()[i];
                    if (data.lengths()[i] < dbSequenceLengthStatistics->min_length) dbSequenceLengthStatistics->min_length = data.lengths()[i];
                    dbSequenceLengthStatistics->sumOfLengths += data.lengths()[i];
                }
            }
            return *dbSequenceLengthStatistics;
        }

        std::vector<DeviceBatchCopyToPinnedPlan> computeDbCopyPlan(
            const std::vector<DBdataView>& dbPartitions,
            const std::vector<int>& lengthPartitionIds,
            size_t MAX_CHARDATA_BYTES,
            size_t MAX_SEQ
        ) const {
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


        void setQuery(const char* query, SequenceLengthT queryLength){
            if(queryLength > MaxSequenceLength::value()){
                std::string msg = "Query length is " + std::to_string(queryLength) 
                    + ", but config allows only lengths <= " + std::to_string(MaxSequenceLength::value());
                throw std::runtime_error(msg);
            }
            
            currentQueryLength = queryLength;

            //pad query to multiple of 4 for char4 access
            //add sizeof(char4) * warpsize for unguarded accesses outside of the DP matrix
            currentQueryLengthWithPadding = SDIV(queryLength, 4) * 4 + sizeof(char4) * 32;

            const int numGpus = deviceIds.size();
            for(int gpu = 0; gpu < numGpus; gpu++){
                cudaSetDevice(deviceIds[gpu]); CUERR;
                auto& ws = *workingSets[gpu];
                ws.d_query.resize(currentQueryLengthWithPadding);
                cudaMemsetAsync(ws.d_query.data() + currentQueryLength, 20, currentQueryLengthWithPadding - currentQueryLength, gpuStreams[gpu]);
                cudaMemcpyAsync(ws.d_query.data(), query, currentQueryLength, cudaMemcpyDefault, gpuStreams[gpu]); CUERR

                thrust::transform(
                    thrust::cuda::par_nosync.on(gpuStreams[gpu]),
                    ws.d_query.data(),
                    ws.d_query.data() + currentQueryLength,
                    ws.d_query.data(),
                    ConvertAA_20{}
                );
                //NW_convert_protein_single<<<SDIV(queryLength, 128), 128, 0, gpuStreams[gpu]>>>(ws.d_query.data(), queryLength); CUERR
            }
        }

        void scanDatabaseForQuery(){
            const int numGpus = deviceIds.size();
            const int masterDeviceId = deviceIds[0];
            const auto& masterStream1 = gpuStreams[0];
            auto& masterevent1 = gpuEvents[0];

            cudaSetDevice(masterDeviceId);
            // scanTimer->reset();
            // scanTimer->start();

            thrust::fill(
                thrust::cuda::par_nosync.on(masterStream1),
                d_finalAlignmentScores_allGpus.begin(),
                d_finalAlignmentScores_allGpus.end(),
                0
            );

            cudaSetDevice(masterDeviceId);           

            cudaEventRecord(masterevent1, masterStream1); CUERR;

            for(int gpu = 0; gpu < numGpus; gpu++){
                cudaSetDevice(deviceIds[gpu]); CUERR;
                cudaStreamWaitEvent(gpuStreams[gpu], masterevent1, 0); CUERR;
            }

            #ifdef CUDASW_DEBUG_CHECK_CORRECTNESS
            for(int gpu = 0; gpu < numGpus; gpu++){
                cudaSetDevice(deviceIds[gpu]); CUERR;
                auto& ws = *workingSets[gpu];
                ws.allocateTempStorageHE();
            }
            #endif
            
            processQueryOnGpus();
            
            #ifdef CUDASW_DEBUG_CHECK_CORRECTNESS
            for(int gpu = 0; gpu < numGpus; gpu++){
                cudaSetDevice(deviceIds[gpu]); CUERR;
                auto& ws = *workingSets[gpu];
                ws.deallocateTempStorageHE();
            }
            #endif
            

            for(int gpu = 0; gpu < numGpus; gpu++){
                cudaSetDevice(deviceIds[gpu]); CUERR;
                auto& ws = *workingSets[gpu];
                cudaStream_t stream = gpuStreams[gpu];

                const auto& dbData = fullDB.getData();
                const size_t numDBSequences = dbData.numSequences();

                //sort the results. since we are only interested in the top "results_per_query" results, but do not need a fully sorted range
                //perform sorting in batches to reduce memory requirements

                const size_t sortBatchSize = 1000000;
                float* d_myTopResults_scores;
                ReferenceIdT* d_myTopResults_refIds;
                float* d_myTopResults_scores_tmp;
                ReferenceIdT* d_myTopResults_refIds_tmp;
                cudaMallocAsync(&d_myTopResults_scores, sizeof(float) * results_per_query, stream); CUERR;
                cudaMallocAsync(&d_myTopResults_refIds, sizeof(ReferenceIdT) * results_per_query, stream); CUERR;
                cudaMallocAsync(&d_myTopResults_scores_tmp, sizeof(float) * (results_per_query + sortBatchSize), stream); CUERR;
                cudaMallocAsync(&d_myTopResults_refIds_tmp, sizeof(ReferenceIdT) * (results_per_query + sortBatchSize), stream); CUERR;
                cudaMemsetAsync(d_myTopResults_scores, 0, sizeof(float) * results_per_query, stream); CUERR;

                for(size_t sortOffset = 0; sortOffset < numDBSequences; sortOffset += sortBatchSize){
                    size_t numInBatch = std::min(numDBSequences - sortOffset, sortBatchSize);
                    thrust::sort_by_key(
                        thrust::cuda::par_nosync(thrust_async_allocator<char>(stream)).on(stream),
                        ws.d_batchResultListScores.data() + sortOffset,
                        ws.d_batchResultListScores.data() + sortOffset + numInBatch,
                        ws.d_batchResultListRefIds.data() + sortOffset,
                        thrust::greater<float>()
                    );

                    thrust::merge_by_key(
                        thrust::cuda::par_nosync(thrust_async_allocator<char>(stream)).on(stream),
                        ws.d_batchResultListScores.data() + sortOffset,
                        ws.d_batchResultListScores.data() + sortOffset + std::min(numInBatch, size_t(results_per_query)),
                        d_myTopResults_scores, 
                        d_myTopResults_scores + results_per_query,
                        ws.d_batchResultListRefIds.data() + sortOffset, 
                        d_myTopResults_refIds,
                        d_myTopResults_scores_tmp, 
                        d_myTopResults_refIds_tmp,
                        thrust::greater<float>()
                    );

                    cudaMemcpyAsync(d_myTopResults_scores, d_myTopResults_scores_tmp, sizeof(float) * results_per_query, cudaMemcpyDeviceToDevice, stream); CUERR;
                    cudaMemcpyAsync(d_myTopResults_refIds, d_myTopResults_refIds_tmp, sizeof(ReferenceIdT) * results_per_query, cudaMemcpyDeviceToDevice, stream); CUERR;
                }

                if(numGpus > 1){
                    //transform per gpu local sequence indices into global sequence indices
                    if(results_per_query > 0){
                        transformLocalSequenceIndicesToGlobalIndices<<<SDIV(results_per_query, 128), 128, 0, stream>>>(
                            gpu,
                            results_per_query,
                            ws.deviceGpuPartitionOffsets.getDeviceView(),
                            d_myTopResults_refIds
                        ); CUERR;
                    }
                }

                cudaMemcpyAsync(
                    d_finalAlignmentScores_allGpus.data() + results_per_query*gpu,
                    d_myTopResults_scores,
                    sizeof(float) * results_per_query,
                    cudaMemcpyDeviceToDevice,
                    stream
                ); CUERR;
                cudaMemcpyAsync(
                    d_finalReferenceIds_allGpus.data() + results_per_query*gpu,
                    d_myTopResults_refIds,
                    sizeof(ReferenceIdT) * results_per_query,
                    cudaMemcpyDeviceToDevice,
                    stream
                ); CUERR;                
                cudaMemcpyAsync(
                    d_resultNumOverflows.data() + gpu,
                    ws.d_total_overflow_number.data(),
                    sizeof(int),
                    cudaMemcpyDeviceToDevice,
                    stream
                ); CUERR;

                cudaFreeAsync(d_myTopResults_scores, stream); CUERR;
                cudaFreeAsync(d_myTopResults_refIds, stream); CUERR;
                cudaFreeAsync(d_myTopResults_scores_tmp, stream); CUERR;
                cudaFreeAsync(d_myTopResults_refIds_tmp, stream); CUERR;

                cudaEventRecord(ws.forkStreamEvent, stream); CUERR;

                cudaSetDevice(masterDeviceId);
                cudaStreamWaitEvent(masterStream1, ws.forkStreamEvent, 0); CUERR;
            }

            cudaSetDevice(masterDeviceId);

            if(numGpus > 1){
                //sort per-gpu top results to find overall top results
                thrust::sort_by_key(
                    thrust::cuda::par_nosync(thrust_async_allocator<char>(masterStream1)).on(masterStream1),
                    d_finalAlignmentScores_allGpus.begin(),
                    d_finalAlignmentScores_allGpus.begin() + results_per_query * numGpus,
                    d_finalReferenceIds_allGpus.begin(),
                    thrust::greater<float>()
                );


                //sum the overflows per gpu
                sumNumOverflowsKernel<<<1,1,0,masterStream1>>>(d_resultNumOverflows.data(), d_resultNumOverflows.data(), numGpus); CUERR;                
            }

            cudaMemcpyAsync(
                h_finalAlignmentScores.data(), 
                d_finalAlignmentScores_allGpus.data(), 
                sizeof(float) * results_per_query, 
                cudaMemcpyDeviceToHost, 
                masterStream1
            );  CUERR
            cudaMemcpyAsync(
                h_finalReferenceIds.data(), 
                d_finalReferenceIds_allGpus.data(), 
                sizeof(ReferenceIdT) * results_per_query, 
                cudaMemcpyDeviceToHost, 
                masterStream1
            );  CUERR
            cudaMemcpyAsync(
                h_resultNumOverflows.data(), 
                d_resultNumOverflows.data(), 
                sizeof(int), 
                cudaMemcpyDeviceToHost, 
                masterStream1
            );  CUERR

            cudaStreamSynchronize(masterStream1); CUERR;
        }

        void processQueryOnGpus(){

            // std::cout << "ProcessQueryOnGpus: dstinfos isUploaded\n";
            // for(size_t i = 0; i < batchPlans[0].size(); i++){
            //     std::cout << batchPlansDstInfoVec[0][i].isUploaded << " ";
            // }
            // std::cout << "\n";

            const std::vector<std::vector<DBdataView>>& dbPartitionsPerGpu = subPartitionsForGpus;
            
            constexpr auto boundaries = getLengthPartitionBoundaries();
            constexpr int numLengthPartitions = boundaries.size();
            const int numGpus = deviceIds.size();
            const bool useExtraThreadForBatchTransfer = numGpus > 1;
        
            size_t totalNumberOfSequencesToProcess = std::reduce(numSequencesPerGpu.begin(), numSequencesPerGpu.end());
            
            size_t totalNumberOfProcessedSequences = 0;
        
            for(int gpu = 0; gpu < numGpus; gpu++){
                cudaSetDevice(deviceIds[gpu]); CUERR;
                auto& ws = *workingSets[gpu];
        
                cudaMemsetAsync(ws.d_total_overflow_number.data(), 0, sizeof(int), gpuStreams[gpu]);
                
                ws.resetBatchResultList(gpuStreams[gpu]);
        
                //create dependency on mainStream
                cudaEventRecord(ws.forkStreamEvent, gpuStreams[gpu]); CUERR;
                cudaStreamWaitEvent(ws.workStreamForTempUsage, ws.forkStreamEvent, 0); CUERR;
                for(auto& stream : ws.workStreamsWithoutTemp){
                    cudaStreamWaitEvent(stream, ws.forkStreamEvent, 0); CUERR;
                }
                cudaStreamWaitEvent(ws.hostFuncStream, ws.forkStreamEvent, 0); CUERR;
            }       
        
            //variables per gpu to keep between loops
            struct Variables{
                int currentBuffer = 0;
                int previousBuffer = 0;
                cudaStream_t H2DcopyStream = cudaStreamLegacy;
                char* h_inputChars = nullptr;
                SequenceLengthT* h_inputLengths = nullptr;
                size_t* h_inputOffsets = nullptr;
                char* d_inputChars = nullptr;
                SequenceLengthT* d_inputLengths = nullptr;
                size_t* d_inputOffsets = nullptr;
                int* d_overflow_number = nullptr;
                ReferenceIdT* d_overflow_positions = nullptr;
                const std::vector<DeviceBatchCopyToPinnedPlan>* batchPlansPtr;
                const std::vector<DeviceBatchCopyToPinnedPlan>* batchPlansCachedDBPtr;
                const DeviceBatchCopyToPinnedPlan* currentPlanPtr;
                size_t processedSequences = 0;
                size_t processedBatches = 0;
            };
        
            std::vector<Variables> variables_vec(numGpus);
            //init variables
            for(int gpu = 0; gpu < numGpus; gpu++){
                cudaSetDevice(deviceIds[gpu]); CUERR;
                const auto& ws = *workingSets[gpu];
                auto& variables = variables_vec[gpu];
                variables.processedSequences = 0;
                variables.processedBatches = 0;
                variables.batchPlansPtr = &batchPlans[gpu];
                variables.batchPlansCachedDBPtr = &batchPlans_cachedDB[gpu];
            }
            
            while(totalNumberOfProcessedSequences < totalNumberOfSequencesToProcess){
                //set up gpu variables for current iteration
                for(int gpu = 0; gpu < numGpus; gpu++){
                    cudaSetDevice(deviceIds[gpu]); CUERR;
                    auto& ws = *workingSets[gpu];
                    auto& variables = variables_vec[gpu];
                    if(variables.processedBatches < variables.batchPlansPtr->size()){

                        if(variables.processedBatches < ws.getNumBatchesInCachedDB()){
                            //will process a batch that could be cached in gpu memory
                            if(batchPlansDstInfoVec[gpu][variables.processedBatches].isUploaded == false){
                                //it is not cached, need upload
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
                                variables.d_inputChars = batchPlansDstInfoVec[gpu][variables.processedBatches].charsPtr;
                                variables.d_inputLengths = batchPlansDstInfoVec[gpu][variables.processedBatches].lengthsPtr;
                                variables.d_inputOffsets = batchPlansDstInfoVec[gpu][variables.processedBatches].offsetsPtr;
                                variables.d_overflow_number = ws.d_overflow_number.data() + variables.currentBuffer;                                
                                variables.d_overflow_positions = ws.d_overflow_positions_vec[0].data();
                            }else{
                                //already uploaded. process all batches for cached db together
                                assert(variables.processedBatches == 0);
                                variables.currentBuffer = 0;
                                variables.previousBuffer = 0;
                                variables.H2DcopyStream = ws.copyStreams[0];
                                variables.h_inputChars = nullptr;
                                variables.h_inputLengths = nullptr;
                                variables.h_inputOffsets = nullptr;
                                variables.d_inputChars = ws.d_cacheddb->getCharData();
                                variables.d_inputLengths = ws.d_cacheddb->getLengthData();
                                variables.d_inputOffsets = ws.d_cacheddb->getOffsetData();
                                variables.d_overflow_number = ws.d_overflow_number.data() + variables.currentBuffer;
                                variables.d_overflow_positions = ws.d_overflow_positions_vec[0].data();
                                
                            }
                        }else{
                            //will process batch that cannot be cached
                            //upload to double buffer
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
                            variables.d_inputChars = batchPlansDstInfoVec[gpu][variables.processedBatches].charsPtr;
                            variables.d_inputLengths = batchPlansDstInfoVec[gpu][variables.processedBatches].lengthsPtr;
                            variables.d_inputOffsets = batchPlansDstInfoVec[gpu][variables.processedBatches].offsetsPtr;
                            variables.d_overflow_number = ws.d_overflow_number.data() + variables.currentBuffer;
                            variables.d_overflow_positions = ws.d_overflow_positions_vec[0].data();
                        }
                    }
                }

                //upload batch
                for(int gpu = 0; gpu < numGpus; gpu++){
                    cudaSetDevice(deviceIds[gpu]); CUERR;
                    auto& ws = *workingSets[gpu];
                    auto& variables = variables_vec[gpu];
                    if(variables.processedBatches < variables.batchPlansPtr->size()){
                        const bool needsUpload = !batchPlansDstInfoVec[gpu][variables.processedBatches].isUploaded;

                        variables.currentPlanPtr = [&](){
                            if(variables.processedBatches < ws.getNumBatchesInCachedDB()){
                                if(!needsUpload){
                                    return &(*variables.batchPlansCachedDBPtr)[0];
                                }else{
                                    return &(*variables.batchPlansPtr)[variables.processedBatches];
                                }
                            }else{
                                return &(*variables.batchPlansPtr)[variables.processedBatches];
                            }
                        }();
                            
        
                        if(needsUpload){
                            //transfer data
                            //can only overwrite device buffer if it is no longer in use on workstream
                            cudaStreamWaitEvent(variables.H2DcopyStream, ws.deviceBufferEvents[variables.currentBuffer], 0); CUERR;
        
                            if(useExtraThreadForBatchTransfer){
                                cudaStreamWaitEvent(ws.hostFuncStream, ws.pinnedBufferEvents[variables.currentBuffer]); CUERR;
                                executePinnedCopyPlanWithHostCallback(
                                    *variables.currentPlanPtr, 
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
                                    variables.currentPlanPtr->usedBytes,
                                    H2D,
                                    variables.H2DcopyStream
                                ); CUERR;
                                cudaMemcpyAsync(
                                    variables.d_inputLengths,
                                    variables.h_inputLengths,
                                    sizeof(SequenceLengthT) * variables.currentPlanPtr->usedSeq,
                                    H2D,
                                    variables.H2DcopyStream
                                ); CUERR;
                                cudaMemcpyAsync(
                                    variables.d_inputOffsets,
                                    variables.h_inputOffsets,
                                    sizeof(size_t) * (variables.currentPlanPtr->usedSeq+1),
                                    H2D,
                                    variables.H2DcopyStream
                                ); CUERR;
                            }else{
                                //synchronize to avoid overwriting pinned buffer of target before it has been fully transferred
                                cudaEventSynchronize(ws.pinnedBufferEvents[variables.currentBuffer]); CUERR;

                                executeCopyPlanH2DDirect(
                                    *variables.currentPlanPtr, 
                                    variables.d_inputChars,
                                    variables.d_inputLengths,
                                    variables.d_inputOffsets,
                                    dbPartitionsPerGpu[gpu], 
                                    variables.H2DcopyStream
                                );
        
                                // executePinnedCopyPlanSerialAndTransferToGpu(
                                //     *variables.currentPlanPtr, 
                                //     variables.h_inputChars,
                                //     variables.h_inputLengths,
                                //     variables.h_inputOffsets,
                                //     variables.d_inputChars,
                                //     variables.d_inputLengths,
                                //     variables.d_inputOffsets,
                                //     dbPartitionsPerGpu[gpu], 
                                //     variables.H2DcopyStream
                                // );
                            }
                            
                            cudaEventRecord(ws.pinnedBufferEvents[variables.currentBuffer], variables.H2DcopyStream); CUERR;
                        }
                    }
                }

                //process batch
                for(int gpu = 0; gpu < numGpus; gpu++){
                    cudaSetDevice(deviceIds[gpu]); CUERR;
                    auto& ws = *workingSets[gpu];
                    auto& variables = variables_vec[gpu];
                    if(variables.processedBatches < variables.batchPlansPtr->size()){
        
                        cudaMemsetAsync(variables.d_overflow_number, 0, sizeof(int), variables.H2DcopyStream);
                        
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
        
                        const char* const inputChars = variables.d_inputChars;
                        const SequenceLengthT* const inputLengths = variables.d_inputLengths;
                        const size_t* const inputOffsets = variables.d_inputOffsets;
                        int* const d_overflow_number = variables.d_overflow_number;
                        ReferenceIdT* const d_overflow_positions = variables.d_overflow_positions;
                        const auto& plan = *variables.currentPlanPtr;
        
                        auto runAlignmentKernels = [&](auto& d_scores, ReferenceIdT* d_overflow_positions, int* d_overflow_number){
                            const char4* const d_query = reinterpret_cast<char4*>(ws.d_query.data());
        
                            auto nextWorkStreamNoTemp = [&](){
                                ws.workstreamIndex = (ws.workstreamIndex + 1) % ws.numWorkStreamsWithoutTemp;
                                return (cudaStream_t)ws.workStreamsWithoutTemp[ws.workstreamIndex];
                            };
                            std::vector<size_t> numPerPartitionPrefixSum(plan.h_numPerPartition.size());
                            for(size_t i = 0; i < plan.h_numPerPartition.size()-1; i++){
                                numPerPartitionPrefixSum[i+1] = numPerPartitionPrefixSum[i] + plan.h_numPerPartition[i];
                            }
                            //size_t exclPs = 0;
                            //for(int lp = 0; lp < int(plan.h_partitionIds.size()); lp++){
                            for(int lp = plan.h_partitionIds.size() - 1; lp >= 0; lp--){
                                const int partId = plan.h_partitionIds[lp];
                                const size_t numSeq = plan.h_numPerPartition[lp];
                                const int start = numPerPartitionPrefixSum[lp];
                                //std::cout << "partId " << partId << " numSeq " << numSeq << "\n";
                                
                                //const size_t* const d_selectedPositions = ws.d_selectedPositions.data() + start;
                                auto d_selectedPositions = thrust::make_counting_iterator<ReferenceIdT>(start);
                                #if 1
                                if(kernelTypeConfig.singlePassType == KernelType::Half2){
                                    
                                    constexpr int sh2bs = 256; // single half2 blocksize 
                                    if (partId == 0){call_NW_local_affine_single_pass_half2<sh2bs, 2, 24>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 1){call_NW_local_affine_single_pass_half2<sh2bs, 4, 16>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 2){call_NW_local_affine_single_pass_half2<sh2bs, 8, 10>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 3){call_NW_local_affine_single_pass_half2<sh2bs, 8, 12>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 4){call_NW_local_affine_single_pass_half2<sh2bs, 8, 14>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 5){call_NW_local_affine_single_pass_half2<sh2bs, 8, 16>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 6){call_NW_local_affine_single_pass_half2<sh2bs, 8, 18>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 7){call_NW_local_affine_single_pass_half2<sh2bs, 8, 20>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 8){call_NW_local_affine_single_pass_half2<sh2bs, 8, 22>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 9){call_NW_local_affine_single_pass_half2<sh2bs, 8, 24>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 10){call_NW_local_affine_single_pass_half2<sh2bs, 8, 26>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 11){call_NW_local_affine_single_pass_half2<sh2bs, 8, 28>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 12){call_NW_local_affine_single_pass_half2<sh2bs, 8, 30>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 13){call_NW_local_affine_single_pass_half2<sh2bs, 8, 32>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 14){call_NW_local_affine_single_pass_half2<sh2bs, 16, 18>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 15){call_NW_local_affine_single_pass_half2<sh2bs, 16, 20>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 16){call_NW_local_affine_single_pass_half2<sh2bs, 16, 22>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 17){call_NW_local_affine_single_pass_half2<sh2bs, 16, 24>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 18){call_NW_local_affine_single_pass_half2<sh2bs, 16, 26>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 19){call_NW_local_affine_single_pass_half2<sh2bs, 16, 28>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 20){call_NW_local_affine_single_pass_half2<sh2bs, 16, 30>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 21){call_NW_local_affine_single_pass_half2<sh2bs, 16, 32>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 22){call_NW_local_affine_single_pass_half2<sh2bs, 32, 18>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 23){call_NW_local_affine_single_pass_half2<sh2bs, 32, 20>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 24){call_NW_local_affine_single_pass_half2<sh2bs, 32, 22>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 25){call_NW_local_affine_single_pass_half2<sh2bs, 32, 24>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 26){call_NW_local_affine_single_pass_half2<sh2bs, 32, 26>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 27){call_NW_local_affine_single_pass_half2<sh2bs, 32, 28>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 28){call_NW_local_affine_single_pass_half2<sh2bs, 32, 30>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 29){call_NW_local_affine_single_pass_half2<sh2bs, 32, 32>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 30){call_NW_local_affine_single_pass_half2<sh2bs, 32, 34>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 31){call_NW_local_affine_single_pass_half2<sh2bs, 32, 36>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 32){call_NW_local_affine_single_pass_half2<sh2bs, 32, 38>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 33){call_NW_local_affine_single_pass_half2<sh2bs, 32, 40>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                
                                }else if(kernelTypeConfig.singlePassType == KernelType::DPXs16){
        
                                    constexpr int sh2bs = 256; // dpx s16 blocksize 
                                    if (partId == 0){call_NW_local_affine_single_pass_dpx_s16<sh2bs, 2, 24>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 1){call_NW_local_affine_single_pass_dpx_s16<sh2bs, 4, 16>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 2){call_NW_local_affine_single_pass_dpx_s16<sh2bs, 8, 10>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 3){call_NW_local_affine_single_pass_dpx_s16<sh2bs, 8, 12>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 4){call_NW_local_affine_single_pass_dpx_s16<sh2bs, 8, 14>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 5){call_NW_local_affine_single_pass_dpx_s16<sh2bs, 8, 16>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 6){call_NW_local_affine_single_pass_dpx_s16<sh2bs, 8, 18>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 7){call_NW_local_affine_single_pass_dpx_s16<sh2bs, 8, 20>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 8){call_NW_local_affine_single_pass_dpx_s16<sh2bs, 8, 22>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 9){call_NW_local_affine_single_pass_dpx_s16<sh2bs, 8, 24>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 10){call_NW_local_affine_single_pass_dpx_s16<sh2bs, 8, 26>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 11){call_NW_local_affine_single_pass_dpx_s16<sh2bs, 8, 28>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 12){call_NW_local_affine_single_pass_dpx_s16<sh2bs, 8, 30>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 0, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 13){call_NW_local_affine_single_pass_dpx_s16<sh2bs, 8, 32>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 14){call_NW_local_affine_single_pass_dpx_s16<sh2bs, 16, 18>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 15){call_NW_local_affine_single_pass_dpx_s16<sh2bs, 16, 20>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 16){call_NW_local_affine_single_pass_dpx_s16<sh2bs, 16, 22>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 17){call_NW_local_affine_single_pass_dpx_s16<sh2bs, 16, 24>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 18){call_NW_local_affine_single_pass_dpx_s16<sh2bs, 16, 26>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 19){call_NW_local_affine_single_pass_dpx_s16<sh2bs, 16, 28>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 20){call_NW_local_affine_single_pass_dpx_s16<sh2bs, 16, 30>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 21){call_NW_local_affine_single_pass_dpx_s16<sh2bs, 16, 32>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 22){call_NW_local_affine_single_pass_dpx_s16<sh2bs, 32, 18>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 23){call_NW_local_affine_single_pass_dpx_s16<sh2bs, 32, 20>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 24){call_NW_local_affine_single_pass_dpx_s16<sh2bs, 32, 22>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 25){call_NW_local_affine_single_pass_dpx_s16<sh2bs, 32, 24>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 26){call_NW_local_affine_single_pass_dpx_s16<sh2bs, 32, 26>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 27){call_NW_local_affine_single_pass_dpx_s16<sh2bs, 32, 28>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 28){call_NW_local_affine_single_pass_dpx_s16<sh2bs, 32, 30>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 29){call_NW_local_affine_single_pass_dpx_s16<sh2bs, 32, 32>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 30){call_NW_local_affine_single_pass_dpx_s16<sh2bs, 32, 34>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 31){call_NW_local_affine_single_pass_dpx_s16<sh2bs, 32, 36>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 32){call_NW_local_affine_single_pass_dpx_s16<sh2bs, 32, 38>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 33){call_NW_local_affine_single_pass_dpx_s16<sh2bs, 32, 40>(blosumType, inputChars, d_scores, inputOffsets , inputLengths, d_selectedPositions, numSeq, d_overflow_positions, d_overflow_number, 1, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }

                                }else if(kernelTypeConfig.singlePassType == KernelType::Float){
                                    //single pass ! must not use temp storage! allowed sequence length <= 32*numregs
                                    if (partId == 0){ call_NW_local_affine_single_pass_float<2>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 1){ call_NW_local_affine_single_pass_float<2>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 2){ call_NW_local_affine_single_pass_float<4>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 3){ call_NW_local_affine_single_pass_float<4>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 4){ call_NW_local_affine_single_pass_float<4>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 5){ call_NW_local_affine_single_pass_float<4>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 6){ call_NW_local_affine_single_pass_float<6>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 7){ call_NW_local_affine_single_pass_float<6>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 8){ call_NW_local_affine_single_pass_float<6>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 9){ call_NW_local_affine_single_pass_float<6>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 10){ call_NW_local_affine_single_pass_float<8>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 11){ call_NW_local_affine_single_pass_float<8>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 12){ call_NW_local_affine_single_pass_float<8>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 13){ call_NW_local_affine_single_pass_float<8>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 14){ call_NW_local_affine_single_pass_float<10>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 15){ call_NW_local_affine_single_pass_float<10>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 16){ call_NW_local_affine_single_pass_float<12>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 17){ call_NW_local_affine_single_pass_float<12>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 18){ call_NW_local_affine_single_pass_float<14>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 19){ call_NW_local_affine_single_pass_float<14>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 20){ call_NW_local_affine_single_pass_float<16>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 21){ call_NW_local_affine_single_pass_float<16>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 22){ call_NW_local_affine_single_pass_float<18>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 23){ call_NW_local_affine_single_pass_float<20>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 24){ call_NW_local_affine_single_pass_float<22>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 25){ call_NW_local_affine_single_pass_float<24>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 26){ call_NW_local_affine_single_pass_float<26>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 27){ call_NW_local_affine_single_pass_float<28>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 28){ call_NW_local_affine_single_pass_float<30>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 29){ call_NW_local_affine_single_pass_float<32>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 30){ call_NW_local_affine_single_pass_float<34>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 31){ call_NW_local_affine_single_pass_float<36>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 32){ call_NW_local_affine_single_pass_float<38>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 33){ call_NW_local_affine_single_pass_float<40>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                
                                }else if(kernelTypeConfig.singlePassType == KernelType::DPXs32){
                                    //single pass ! must not use temp storage! allowed sequence length <= 32*numregs
                                    if (partId == 0){ call_NW_local_affine_single_pass_dpx_s32<2>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 1){ call_NW_local_affine_single_pass_dpx_s32<2>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 2){ call_NW_local_affine_single_pass_dpx_s32<4>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 3){ call_NW_local_affine_single_pass_dpx_s32<4>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 4){ call_NW_local_affine_single_pass_dpx_s32<4>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 5){ call_NW_local_affine_single_pass_dpx_s32<4>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 6){ call_NW_local_affine_single_pass_dpx_s32<6>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 7){ call_NW_local_affine_single_pass_dpx_s32<6>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 8){ call_NW_local_affine_single_pass_dpx_s32<6>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 9){ call_NW_local_affine_single_pass_dpx_s32<6>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 10){ call_NW_local_affine_single_pass_dpx_s32<8>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 11){ call_NW_local_affine_single_pass_dpx_s32<8>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 12){ call_NW_local_affine_single_pass_dpx_s32<8>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 13){ call_NW_local_affine_single_pass_dpx_s32<8>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 14){ call_NW_local_affine_single_pass_dpx_s32<10>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 15){ call_NW_local_affine_single_pass_dpx_s32<10>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 16){ call_NW_local_affine_single_pass_dpx_s32<12>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 17){ call_NW_local_affine_single_pass_dpx_s32<12>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 18){ call_NW_local_affine_single_pass_dpx_s32<14>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 19){ call_NW_local_affine_single_pass_dpx_s32<14>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 20){ call_NW_local_affine_single_pass_dpx_s32<16>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 21){ call_NW_local_affine_single_pass_dpx_s32<16>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 22){ call_NW_local_affine_single_pass_dpx_s32<18>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 23){ call_NW_local_affine_single_pass_dpx_s32<20>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 24){ call_NW_local_affine_single_pass_dpx_s32<22>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 25){ call_NW_local_affine_single_pass_dpx_s32<24>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 26){ call_NW_local_affine_single_pass_dpx_s32<26>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 27){ call_NW_local_affine_single_pass_dpx_s32<28>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 28){ call_NW_local_affine_single_pass_dpx_s32<30>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 29){ call_NW_local_affine_single_pass_dpx_s32<32>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 30){ call_NW_local_affine_single_pass_dpx_s32<34>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 31){ call_NW_local_affine_single_pass_dpx_s32<36>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 32){ call_NW_local_affine_single_pass_dpx_s32<38>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                    if (partId == 33){ call_NW_local_affine_single_pass_dpx_s32<40>(blosumType, inputChars, d_scores, inputOffsets, inputLengths, d_selectedPositions, numSeq, d_query, currentQueryLength, gop, gex, nextWorkStreamNoTemp()); CUERR }
                                
                                    
                                }else{
                                    assert(false);
                                }
                                #endif
        
                                if(partId == numLengthPartitions - 2){
                                    if(kernelTypeConfig.manyPassType_small == KernelType::Half2){
                                        constexpr int blocksize = 32 * 8;
                                        constexpr int groupsize = 32;
                                        constexpr int groupsPerBlock = blocksize / groupsize;
                                        constexpr int alignmentsPerGroup = 2;
                                        constexpr int alignmentsPerBlock = groupsPerBlock * alignmentsPerGroup;
                                        
                                        const size_t tempBytesPerBlockPerBuffer = sizeof(__half2) * groupsPerBlock * currentQueryLengthWithPadding;
        
                                        const size_t maxNumBlocks = ws.numTempBytes / (tempBytesPerBlockPerBuffer * 2);
                                        const size_t maxSubjectsPerIteration = std::min(maxNumBlocks * alignmentsPerBlock, size_t(numSeq));
        
                                        const size_t numBlocksPerIteration = SDIV(maxSubjectsPerIteration, alignmentsPerBlock);
                                        const size_t requiredTempBytes = tempBytesPerBlockPerBuffer * 2 * numBlocksPerIteration;
        
                                        __half2* d_temp = (__half2*)ws.d_tempStorageHE.data();
                                        __half2* d_tempHcol2 = d_temp;
                                        __half2* d_tempEcol2 = (__half2*)(((char*)d_tempHcol2) + requiredTempBytes / 2);
        
                                        const size_t numIters =  SDIV(numSeq, maxSubjectsPerIteration);
        
                                        for(size_t iter = 0; iter < numIters; iter++){
                                            const size_t begin = iter * maxSubjectsPerIteration;
                                            const size_t end = iter < numIters-1 ? (iter+1) * maxSubjectsPerIteration : numSeq;
                                            const size_t num = end - begin;                      
        
                                            cudaMemsetAsync(d_temp, 0, requiredTempBytes, ws.workStreamForTempUsage); CUERR;
                                            
                                            call_NW_local_affine_multi_pass_half2<blocksize, groupsize, 22>(
                                                blosumType,
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
                                                currentQueryLength, 
                                                gop, 
                                                gex,
                                                ws.workStreamForTempUsage
                                            ); CUERR
                                        }
                                    }else if(kernelTypeConfig.manyPassType_small == KernelType::DPXs16){
                                        constexpr int blocksize = 32 * 8;
                                        constexpr int groupsize = 32;
                                        constexpr int groupsPerBlock = blocksize / groupsize;
                                        constexpr int alignmentsPerGroup = 2;
                                        constexpr int alignmentsPerBlock = groupsPerBlock * alignmentsPerGroup;
                                        
                                        const size_t tempBytesPerBlockPerBuffer = sizeof(short2) * groupsPerBlock * currentQueryLengthWithPadding;
        
                                        const size_t maxNumBlocks = ws.numTempBytes / (tempBytesPerBlockPerBuffer * 2);
                                        const size_t maxSubjectsPerIteration = std::min(maxNumBlocks * alignmentsPerBlock, size_t(numSeq));
        
                                        const size_t numBlocksPerIteration = SDIV(maxSubjectsPerIteration, alignmentsPerBlock);
                                        const size_t requiredTempBytes = tempBytesPerBlockPerBuffer * 2 * numBlocksPerIteration;
        
                                        short2* d_temp = (short2*)ws.d_tempStorageHE.data();
                                        short2* d_tempHcol2 = d_temp;
                                        short2* d_tempEcol2 = (short2*)(((char*)d_tempHcol2) + requiredTempBytes / 2);
        
                                        const size_t numIters =  SDIV(numSeq, maxSubjectsPerIteration);
        
                                        for(size_t iter = 0; iter < numIters; iter++){
                                            const size_t begin = iter * maxSubjectsPerIteration;
                                            const size_t end = iter < numIters-1 ? (iter+1) * maxSubjectsPerIteration : numSeq;
                                            const size_t num = end - begin;                      
        
                                            cudaMemsetAsync(d_temp, 0, requiredTempBytes, ws.workStreamForTempUsage); CUERR;
                                            //std::cout << "iter " << iter << " / " << numIters << " gridsize " << SDIV(num, alignmentsPerBlock) << "\n";
        
                                            //cudaDeviceSynchronize(); CUERR;
                                            
                                            call_NW_local_affine_multi_pass_dpx_s16<blocksize, groupsize, 22>(
                                                blosumType,
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
                                                currentQueryLength, 
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
                                    if(kernelTypeConfig.manyPassType_large == KernelType::Float){
                                        const size_t tempBytesPerSubjectPerBuffer = sizeof(float2) * currentQueryLengthWithPadding / 2;
                                        const size_t maxSubjectsPerIteration = std::min(size_t(numSeq), ws.numTempBytes / (tempBytesPerSubjectPerBuffer * 2));
        
                                        float2* d_temp = (float2*)ws.d_tempStorageHE.data();
                                        float2* d_tempHcol2 = d_temp;
                                        float2* d_tempEcol2 = (float2*)(((char*)d_tempHcol2) + maxSubjectsPerIteration * tempBytesPerSubjectPerBuffer);
        
                                        const size_t numIters =  SDIV(numSeq, maxSubjectsPerIteration);
                                        for(size_t iter = 0; iter < numIters; iter++){
                                            const size_t begin = iter * maxSubjectsPerIteration;
                                            const size_t end = iter < numIters-1 ? (iter+1) * maxSubjectsPerIteration : numSeq;
                                            const size_t num = end - begin;
        
                                            cudaMemsetAsync(d_temp, 0, tempBytesPerSubjectPerBuffer * 2 * num, ws.workStreamForTempUsage); CUERR;

                                            call_NW_local_affine_multi_pass_float<20>(
                                                blosumType,
                                                inputChars, 
                                                d_scores, 
                                                d_tempHcol2, 
                                                d_tempEcol2, 
                                                inputOffsets, 
                                                inputLengths, 
                                                d_selectedPositions + begin, 
                                                num,
                                                d_query,
                                                currentQueryLength, 
                                                gop, 
                                                gex,
                                                ws.workStreamForTempUsage
                                            ); CUERR 
                                        }
                                    }else if(kernelTypeConfig.manyPassType_large == KernelType::DPXs32){
                                        const size_t tempBytesPerSubjectPerBuffer = sizeof(int2) * currentQueryLengthWithPadding / 2;
                                        const size_t maxSubjectsPerIteration = std::min(size_t(numSeq), ws.numTempBytes / (tempBytesPerSubjectPerBuffer * 2));
        
                                        int2* d_temp = (int2*)ws.d_tempStorageHE.data();
                                        int2* d_tempHcol2 = d_temp;
                                        int2* d_tempEcol2 = (int2*)(((char*)d_tempHcol2) + maxSubjectsPerIteration * tempBytesPerSubjectPerBuffer);
        
                                        const size_t numIters =  SDIV(numSeq, maxSubjectsPerIteration);
                                        for(size_t iter = 0; iter < numIters; iter++){
                                            const size_t begin = iter * maxSubjectsPerIteration;
                                            const size_t end = iter < numIters-1 ? (iter+1) * maxSubjectsPerIteration : numSeq;
                                            const size_t num = end - begin;
        
                                            cudaMemsetAsync(d_temp, 0, tempBytesPerSubjectPerBuffer * 2 * num, ws.workStreamForTempUsage); CUERR;

                                            call_NW_local_affine_multi_pass_dpx_s32<20>(
                                                blosumType,
                                                inputChars, 
                                                d_scores, 
                                                d_tempHcol2, 
                                                d_tempEcol2, 
                                                inputOffsets, 
                                                inputLengths, 
                                                d_selectedPositions + begin, 
                                                num,
                                                d_query,
                                                currentQueryLength, 
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

                        auto batchResultList = ws.getBatchResultList(variables.processedSequences);

                        runAlignmentKernels(batchResultList, d_overflow_positions, d_overflow_number);

                        //alignments are done in workstreams. now, join all workstreams into workStreamForTempUsage to process overflow alignments
                        for(auto& stream : ws.workStreamsWithoutTemp){
                            cudaEventRecord(ws.forkStreamEvent, stream); CUERR;
                            cudaStreamWaitEvent(ws.workStreamForTempUsage, ws.forkStreamEvent, 0); CUERR;    
                        }
                    }
                }
        
                //process overflow alignments and finish processing of batch
                for(int gpu = 0; gpu < numGpus; gpu++){
                    cudaSetDevice(deviceIds[gpu]); CUERR;
                    auto& ws = *workingSets[gpu];
                    const auto& variables = variables_vec[gpu];
                    if(variables.processedBatches < variables.batchPlansPtr->size()){
        
                        const char* const inputChars = variables.d_inputChars;
                        const SequenceLengthT* const inputLengths = variables.d_inputLengths;
                        const size_t* const inputOffsets = variables.d_inputOffsets;
                        int* const d_overflow_number = variables.d_overflow_number;
                        ReferenceIdT* const d_overflow_positions = variables.d_overflow_positions;
        
                        const char4* const d_query = reinterpret_cast<char4*>(ws.d_query.data());
        
                        auto batchResultList = ws.getBatchResultList(variables.processedSequences);
        
                        if(kernelTypeConfig.overflowType == KernelType::Float){
                            //std::cerr << "overflow processing\n";
                            float2* d_temp = (float2*)ws.d_tempStorageHE.data();
                            call_launch_process_overflow_alignments_kernel_NW_local_affine_multi_pass_float<20>(
                                d_overflow_number,
                                d_temp, 
                                ws.numTempBytes,
                                inputChars, 
                                batchResultList,
                                inputOffsets, 
                                inputLengths, 
                                d_overflow_positions, 
                                d_query,
                                currentQueryLength, 
                                gop, 
                                gex,
                                ws.workStreamForTempUsage
                            );
                        }else if(kernelTypeConfig.overflowType == KernelType::DPXs32){
                            //std::cerr << "overflow processing\n";
                            int2* d_temp = (int2*)ws.d_tempStorageHE.data();
                            call_launch_process_overflow_alignments_kernel_NW_local_affine_multi_pass_dpx_s32<20>(
                                d_overflow_number,
                                d_temp, 
                                ws.numTempBytes,
                                inputChars, 
                                batchResultList,
                                inputOffsets, 
                                inputLengths, 
                                d_overflow_positions, 
                                d_query,
                                currentQueryLength, 
                                gop, 
                                gex,
                                ws.workStreamForTempUsage
                            );
                        }else{
                            assert(false);
                        }
        
                        //update total num overflows for query
                        addKernel<<<1,1,0, ws.workStreamForTempUsage>>>(ws.d_total_overflow_number.data(), ws.d_total_overflow_number.data(), d_overflow_number); CUERR;
        
                        //after processing overflow alignments, the batch is done and its data can be reused
                        cudaEventRecord(ws.deviceBufferEvents[variables.currentBuffer], ws.workStreamForTempUsage); CUERR;
        
                        //let other workstreams depend on temp usage stream
                        for(auto& stream : ws.workStreamsWithoutTemp){
                            cudaStreamWaitEvent(stream, ws.deviceBufferEvents[variables.currentBuffer], 0); CUERR;    
                        }
        
                        ws.copyBufferIndex = (ws.copyBufferIndex+1) % ws.numCopyBuffers;
                    }
                }
        
                //update running numbers
                for(int gpu = 0; gpu < numGpus; gpu++){
                    auto& variables = variables_vec[gpu];
                    if(variables.processedBatches < variables.batchPlansPtr->size()){

                        variables.processedSequences += variables.currentPlanPtr->usedSeq;
                        if(batchPlansDstInfoVec[gpu][variables.processedBatches].isUploaded){
                            variables.processedBatches += workingSets[gpu]->getNumBatchesInCachedDB();
                        }else{
                            variables.processedBatches++;
                        }
                        //std::cout << "variables.processedBatches: " << variables.processedBatches << "\n";
        
                        totalNumberOfProcessedSequences += variables.currentPlanPtr->usedSeq;
                    } 
                }
        
            } //while not done
        
        
            for(int gpu = 0; gpu < numGpus; gpu++){
                cudaSetDevice(deviceIds[gpu]); CUERR;
                auto& ws = *workingSets[gpu];

                if(!batchPlansDstInfoVec[gpu][0].isUploaded){
                    //all batches for cached db are now resident in gpu memory. update the flags
                    if(ws.getNumBatchesInCachedDB() > 0){
                        markCachedDBBatchesAsUploaded(gpu);

                        // current offsets in cached db store the offsets for each batch, i.e. for each batch the offsets will start again at 0
                        // compute prefix sum to obtain the single-batch offsets
            
                        cudaMemsetAsync(ws.d_cacheddb->getOffsetData(), 0, sizeof(size_t), ws.workStreamForTempUsage); CUERR;
            
                        auto d_paddedLengths = thrust::make_transform_iterator(
                            ws.d_cacheddb->getLengthData(),
                            RoundToNextMultiple<size_t, 4>{}
                        );
            
                        thrust::inclusive_scan(
                            thrust::cuda::par_nosync(thrust_async_allocator<char>(ws.workStreamForTempUsage)).on(ws.workStreamForTempUsage),
                            d_paddedLengths,
                            d_paddedLengths + ws.getNumSequencesInCachedDB(),
                            ws.d_cacheddb->getOffsetData() + 1
                        );
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

        BenchmarkStats makeBenchmarkStats(double seconds, double cells, int overflows) const{
            BenchmarkStats stats;
            stats.seconds = seconds;
            stats.gcups = cells / 1000. / 1000. / 1000.;
            stats.gcups = stats.gcups / stats.seconds;
            stats.numOverflows = overflows;
            return stats;
        }

        void updateNumResultsPerQuery(){

            results_per_query = std::min(size_t(numTop), size_t(maxBatchResultListSize));
            if(dbIsReady){
                results_per_query = std::min(size_t(results_per_query), fullDB.getData().numSequences());
            }
        }

        int affine_local_DP_host_protein_blosum62(
            const char* seq1,
            const char* seq2,
            const int length1,
            const int length2,
            const int gap_open,
            const int gap_extend
        ) {
            const int NEGINFINITY = -10000;
            std::vector<int> penalty_H(2*(length2+1));
            std::vector<int> penalty_F(2*(length2+1));

            int E, F, maxi = 0, result;
            penalty_H[0] = 0;
            penalty_F[0] = NEGINFINITY;
            for (int index = 1; index <= length2; index++) {
                penalty_H[index] = 0;
                penalty_F[index] = NEGINFINITY;
            }

            auto convert_AA = cudasw4::ConvertAA_20{};

            auto BLOSUM = cudasw4::BLOSUM62_20::get2D();

            for (int row = 1; row <= length1; row++) {
                char seq1_char = seq1[row-1];
                char seq2_char;

                const int target_row = row & 1;
                const int source_row = !target_row;
                penalty_H[target_row*(length2+1)] = 0; //gap_open + (row-1)*gap_extend;
                penalty_F[target_row*(length2+1)] = gap_open + (row-1)*gap_extend;
                E = NEGINFINITY;
                for (int col = 1; col <= length2; col++) {
                    const int diag = penalty_H[source_row*(length2+1)+col-1];
                    const int abve = penalty_H[source_row*(length2+1)+col+0];
                    const int left = penalty_H[target_row*(length2+1)+col-1];
                    seq2_char = seq2[col-1];
                    const int residue = BLOSUM[convert_AA(seq1_char)][convert_AA(seq2_char)];
                    E = std::max(E+gap_extend, left+gap_open);
                    F = std::max(penalty_F[source_row*(length2+1)+col+0]+gap_extend, abve+gap_open);
                    result = std::max(0, std::max(diag + residue, std::max(E, F)));
                    penalty_H[target_row*(length2+1)+col] = result;
                    if (result > maxi) maxi = result;
                    penalty_F[target_row*(length2+1)+col] = F;
                }
            }
            return maxi;
        }

        int affine_local_DP_host_protein_blosum62_converted(
            const char* seq1,
            const char* seq2,
            const int length1,
            const int length2,
            const int gap_open,
            const int gap_extend
        ) {
            const int NEGINFINITY = -10000;
            std::vector<int> penalty_H(2*(length2+1));
            std::vector<int> penalty_F(2*(length2+1));

            // std::cout << "length1 " << length1 << ", length2 " << length2 << "\n";

            // for(int i = 0; i < length1; i++){
            //     std::cout << int(seq1[i]) << " ";
            // }
            // std::cout << "\n";

            // for(int i = 0; i < length2; i++){
            //     std::cout << int(seq2[i]) << " ";
            // }
            // std::cout << "\n";

            int E, F, maxi = 0, result;
            penalty_H[0] = 0;
            penalty_F[0] = NEGINFINITY;
            for (int index = 1; index <= length2; index++) {
                penalty_H[index] = 0;
                penalty_F[index] = NEGINFINITY;
            }

            auto BLOSUM = cudasw4::BLOSUM62_20::get2D();

            for (int row = 1; row <= length1; row++) {
                int seq1_char = seq1[row-1];
                int seq2_char;

                const int target_row = row & 1;
                const int source_row = !target_row;
                penalty_H[target_row*(length2+1)] = 0; //gap_open + (row-1)*gap_extend;
                penalty_F[target_row*(length2+1)] = gap_open + (row-1)*gap_extend;
                E = NEGINFINITY;
                for (int col = 1; col <= length2; col++) {
                    const int diag = penalty_H[source_row*(length2+1)+col-1];
                    const int abve = penalty_H[source_row*(length2+1)+col+0];
                    const int left = penalty_H[target_row*(length2+1)+col-1];
                    seq2_char = seq2[col-1];
                    const int residue = BLOSUM[seq1_char][seq2_char];
                    E = std::max(E+gap_extend, left+gap_open);
                    F = std::max(penalty_F[source_row*(length2+1)+col+0]+gap_extend, abve+gap_open);
                    result = std::max(0, std::max(diag + residue, std::max(E, F)));
                    penalty_H[target_row*(length2+1)+col] = result;
                    if (result > maxi) maxi = result;
                    penalty_F[target_row*(length2+1)+col] = F;

                    //std::cout << maxi << " ";
                }
                //std::cout << "\n";
            }
            return maxi;
        }

        std::vector<size_t> fullDB_numSequencesPerLengthPartition;
        std::vector<size_t> numSequencesPerGpu_total;
        std::vector<size_t> numSequencesPerGpuPrefixSum_total;

        //partition chars of whole DB amongst the gpus
        std::vector<size_t> numSequencesPerLengthPartitionPrefixSum;
        std::vector<DBdataView> dbPartitionsByLengthPartitioning;
        std::vector<std::vector<DBdataView>> subPartitionsForGpus;
        std::vector<std::vector<int>> lengthPartitionIdsForGpus;
        std::vector<size_t> numSequencesPerGpu;
        std::vector<size_t> numSequencesPerGpuPrefixSum;
        std::vector<CudaStream> gpuStreams;
        std::vector<CudaEvent> gpuEvents;
        std::vector<std::unique_ptr<GpuWorkingSet>> workingSets;

        std::vector<std::vector<DeviceBatchCopyToPinnedPlan>> batchPlans;
        std::vector<std::vector<BatchDstInfo>> batchPlansDstInfoVec;

        std::vector<std::vector<DeviceBatchCopyToPinnedPlan>> batchPlans_cachedDB;
        std::vector<std::vector<BatchDstInfo>> batchPlansDstInfoVec_cachedDB;

        int results_per_query;
        SequenceLengthT currentQueryLength;
        SequenceLengthT currentQueryLengthWithPadding;

        bool dbIsReady{};
        AnyDBWrapper fullDB;

        mutable std::unique_ptr<SequenceLengthStatistics> dbSequenceLengthStatistics;

        //final scan results. device data resides on gpu deviceIds[0]
        MyPinnedBuffer<float> h_finalAlignmentScores;
        MyPinnedBuffer<ReferenceIdT> h_finalReferenceIds;
        MyPinnedBuffer<int> h_resultNumOverflows;
        MyDeviceBuffer<float> d_finalAlignmentScores_allGpus;
        MyDeviceBuffer<ReferenceIdT> d_finalReferenceIds_allGpus;
        MyDeviceBuffer<int> d_resultNumOverflows;
        std::unique_ptr<helpers::GpuTimer> scanTimer;

        size_t totalProcessedQueryLengths{};
        size_t totalNumOverflows{};
        std::unique_ptr<helpers::GpuTimer> totalTimer;

        HostGpuPartitionOffsets hostGpuPartitionOffsets;

        

        //--------------------------------------
        bool verbose = false;
        int gop = -11;
        int gex = -1;
        int numTop = 10;
        BlosumType blosumType = BlosumType::BLOSUM62_20;
        int maxBatchResultListSize = 512 * 1024;

        KernelTypeConfig kernelTypeConfig;
        MemoryConfig memoryConfig;
        
        std::vector<int> deviceIds;

    };


} //namespace cudasw4

#endif


