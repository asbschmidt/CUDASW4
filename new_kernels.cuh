#ifndef NEW_KERNELS_CUH
#define NEW_KERNELS_CUH

#define MAX_ACC_HALF2 2048.0 // 1024 ??
#define MAX_ACC_SHORT 25000 //TODO which value here ???

#include <thrust/iterator/counting_iterator.h>

#include "kernelhelpers.cuh" //must be included before the kernels

#include "half2_kernels.cuh"
#include "dpx_s16_kernels.cuh"
#include "dpx_s32_kernels.cuh"
#include "float_kernels.cuh"

namespace cudasw4{

//extern template instantiations

//ScoreOutputIterator
using ScoreIterInst = TopNMaximaArray; 
//PositionsIterator
using PosIterInst = decltype(thrust::make_counting_iterator<size_t>(0));

//half2_kernel_instantiations.cu
extern template void call_NW_local_affine_Protein_single_pass_half2_new<256, 2, 24, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const size_t* const, PosIterInst const, const int, size_t* const, int* const, const bool, const char4*, const int, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_Protein_single_pass_half2_new<256, 4, 16, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const size_t* const, PosIterInst const, const int, size_t* const, int* const, const bool, const char4*, const int, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_Protein_single_pass_half2_new<256, 8, 10, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const size_t* const, PosIterInst const, const int, size_t* const, int* const, const bool, const char4*, const int, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_Protein_single_pass_half2_new<256, 8, 12, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const size_t* const, PosIterInst const, const int, size_t* const, int* const, const bool, const char4*, const int, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_Protein_single_pass_half2_new<256, 8, 14, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const size_t* const, PosIterInst const, const int, size_t* const, int* const, const bool, const char4*, const int, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_Protein_single_pass_half2_new<256, 8, 16, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const size_t* const, PosIterInst const, const int, size_t* const, int* const, const bool, const char4*, const int, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_Protein_single_pass_half2_new<256, 8, 18, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const size_t* const, PosIterInst const, const int, size_t* const, int* const, const bool, const char4*, const int, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_Protein_single_pass_half2_new<256, 8, 20, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const size_t* const, PosIterInst const, const int, size_t* const, int* const, const bool, const char4*, const int, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_Protein_single_pass_half2_new<256, 8, 22, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const size_t* const, PosIterInst const, const int, size_t* const, int* const, const bool, const char4*, const int, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_Protein_single_pass_half2_new<256, 8, 24, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const size_t* const, PosIterInst const, const int, size_t* const, int* const, const bool, const char4*, const int, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_Protein_single_pass_half2_new<256, 8, 26, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const size_t* const, PosIterInst const, const int, size_t* const, int* const, const bool, const char4*, const int, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_Protein_single_pass_half2_new<256, 8, 28, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const size_t* const, PosIterInst const, const int, size_t* const, int* const, const bool, const char4*, const int, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_Protein_single_pass_half2_new<256, 8, 30, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const size_t* const, PosIterInst const, const int, size_t* const, int* const, const bool, const char4*, const int, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_Protein_single_pass_half2_new<256, 8, 32, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const size_t* const, PosIterInst const, const int, size_t* const, int* const, const bool, const char4*, const int, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 18, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const size_t* const, PosIterInst const, const int, size_t* const, int* const, const bool, const char4*, const int, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 20, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const size_t* const, PosIterInst const, const int, size_t* const, int* const, const bool, const char4*, const int, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 22, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const size_t* const, PosIterInst const, const int, size_t* const, int* const, const bool, const char4*, const int, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 24, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const size_t* const, PosIterInst const, const int, size_t* const, int* const, const bool, const char4*, const int, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 26, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const size_t* const, PosIterInst const, const int, size_t* const, int* const, const bool, const char4*, const int, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 28, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const size_t* const, PosIterInst const, const int, size_t* const, int* const, const bool, const char4*, const int, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 30, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const size_t* const, PosIterInst const, const int, size_t* const, int* const, const bool, const char4*, const int, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_Protein_single_pass_half2_new<256, 16, 32, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const size_t* const, PosIterInst const, const int, size_t* const, int* const, const bool, const char4*, const int, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 18, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const size_t* const, PosIterInst const, const int, size_t* const, int* const, const bool, const char4*, const int, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 20, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const size_t* const, PosIterInst const, const int, size_t* const, int* const, const bool, const char4*, const int, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 22, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const size_t* const, PosIterInst const, const int, size_t* const, int* const, const bool, const char4*, const int, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 24, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const size_t* const, PosIterInst const, const int, size_t* const, int* const, const bool, const char4*, const int, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 26, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const size_t* const, PosIterInst const, const int, size_t* const, int* const, const bool, const char4*, const int, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 28, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const size_t* const, PosIterInst const, const int, size_t* const, int* const, const bool, const char4*, const int, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 30, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const size_t* const, PosIterInst const, const int, size_t* const, int* const, const bool, const char4*, const int, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 32, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const size_t* const, PosIterInst const, const int, size_t* const, int* const, const bool, const char4*, const int, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 34, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const size_t* const, PosIterInst const, const int, size_t* const, int* const, const bool, const char4*, const int, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 36, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const size_t* const, PosIterInst const, const int, size_t* const, int* const, const bool, const char4*, const int, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 38, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const size_t* const, PosIterInst const, const int, size_t* const, int* const, const bool, const char4*, const int, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_Protein_single_pass_half2_new<256, 32, 40, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const size_t* const, PosIterInst const, const int, size_t* const, int* const, const bool, const char4*, const int, const float, const float, cudaStream_t);


} //namespace cudasw4

#endif