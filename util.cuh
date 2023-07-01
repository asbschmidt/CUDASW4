#include <thrust/mr/memory_resource.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/device_malloc_allocator.h>


using pinned_mr = thrust::system::cuda::universal_host_pinned_memory_resource;
template<class T>
using pinned_allocator = thrust::mr::stateless_resource_allocator<T, pinned_mr>;

template <class T>
using pinned_vector = thrust::host_vector<T, pinned_allocator<T>>;


template <class T>
struct thrust_async_allocator : public thrust::device_malloc_allocator<T> {
public:
    using Base      = thrust::device_malloc_allocator<T>;
    using pointer   = typename Base::pointer;
    using size_type = typename Base::size_type;

    thrust_async_allocator(cudaStream_t stream_) : stream{stream_} {}

    pointer allocate(size_type num){
        T* result = nullptr;
        cudaError_t status = cudaMallocAsync(&result, sizeof(T) * num, stream);
        if(status != cudaSuccess){
            throw std::runtime_error("thrust_async_allocator error allocate");
        }
        return thrust::device_pointer_cast(result);
    }

    void deallocate(pointer ptr, size_type /*num*/){
        cudaError_t status = cudaFreeAsync(thrust::raw_pointer_cast(ptr), stream);
        if(status != cudaSuccess){
            throw std::runtime_error("thrust_async_allocator error deallocate");
        }
    }

private:
    cudaStream_t stream;
};