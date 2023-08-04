#ifndef UTIL_CUH
#define UTIL_CUH

#include "config.hpp"

#include <thrust/device_malloc_allocator.h>

namespace cudasw4{



template <class T>
struct thrust_async_allocator : public thrust::device_malloc_allocator<T> {
public:
    using Base      = thrust::device_malloc_allocator<T>;
    using pointer   = typename Base::pointer;
    using size_type = typename Base::size_type;

    thrust_async_allocator(cudaStream_t stream_) : stream{stream_} {}

    pointer allocate(size_type num){
        //std::cout << "allocate " << num << "\n";
        T* result = nullptr;
        cudaError_t status = cudaMallocAsync(&result, sizeof(T) * num, stream);
        if(status != cudaSuccess){
            throw std::runtime_error("thrust_async_allocator error allocate");
        }
        return thrust::device_pointer_cast(result);
    }

    void deallocate(pointer ptr, size_type /*num*/){
        //std::cout << "deallocate \n";
        cudaError_t status = cudaFreeAsync(thrust::raw_pointer_cast(ptr), stream);
        if(status != cudaSuccess){
            throw std::runtime_error("thrust_async_allocator error deallocate");
        }
    }

private:
    cudaStream_t stream;
};

template <class T>
struct thrust_preallocated_single_allocator : public thrust::device_malloc_allocator<T> {
public:
    using Base      = thrust::device_malloc_allocator<T>;
    using pointer   = typename Base::pointer;
    using size_type = typename Base::size_type;

    thrust_preallocated_single_allocator(void* ptr, size_t size) : preallocated{ptr}, preallocatedSize{size} {}

    pointer allocate(size_type num){
        if(!free){
            throw std::runtime_error("thrust_async_allocator error allocate");
        }else{
            if(sizeof(T) * num <= preallocatedSize){
                T* result = (T*)preallocated;
                free = false;
                return thrust::device_pointer_cast(result);
            }else{
                throw std::runtime_error("thrust_async_allocator error allocate");
            }
        }
    }

    void deallocate(pointer ptr, size_type /*num*/){
        if(free){
            throw std::runtime_error("thrust_async_allocator error deallocate");
        }else{
            T* result = thrust::raw_pointer_cast(ptr);
            if((void*) result != preallocated){
                throw std::runtime_error("thrust_async_allocator error deallocate");
            }
            free = true;
        }
    }

private:
    bool free = true;
    void* preallocated;
    size_t preallocatedSize;
    cudaStream_t stream;
};

//Call cudaSetDevice on destruction
struct RevertDeviceId{
    RevertDeviceId(){
        cudaGetDevice(&id);
    }
    RevertDeviceId(int id_) : id(id_){}
    ~RevertDeviceId(){
        cudaSetDevice(id);
    }
    int id;
};


//template<size_t size>
struct TopNMaximaArray{
    struct Ref{
        size_t index;
        size_t indexOffset;
        int* d_locks;
        volatile float* d_scores;
        ReferenceIdT* d_indices;
        size_t size;

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

    TopNMaximaArray(float* d_scores_, size_t* d_indices_, int* d_locks_, size_t offset, size_t size_)
        : indexOffset(offset), d_locks(d_locks_), d_scores(d_scores_), d_indices(d_indices_), size(size_){}

    template<class Index>
    __device__
    Ref operator[](Index index) const{
        Ref r;
        r.index = index;
        r.indexOffset = indexOffset;
        r.d_locks = d_locks;
        r.d_scores = d_scores;
        r.d_indices = d_indices;
        r.size = size;
        return r;
    }

    size_t indexOffset = 0;
    int* d_locks;
    volatile float* d_scores;
    ReferenceIdT* d_indices;
    size_t size;
};

}

#endif