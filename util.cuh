#ifndef UTIL_CUH
#define UTIL_CUH

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

}

#endif