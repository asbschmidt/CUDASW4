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

#include "util.cuh"
#include "config.hpp"

namespace cudasw4{

//extern template instantiations

//ScoreOutputIterator
// using ScoreIterInst = TopNMaximaArray; 
using ScoreIterInst = BatchResultList;
//using ScoreIterInst = float*; 
//PositionsIterator
using PosIterInst = decltype(thrust::make_counting_iterator<ReferenceIdT>(0));

//half2_kernel_instantiations.cu
extern template void call_NW_local_affine_single_pass_half2<256, 2, 24, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_half2<256, 4, 16, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_half2<256, 8, 10, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_half2<256, 8, 12, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_half2<256, 8, 14, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_half2<256, 8, 16, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_half2<256, 8, 18, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_half2<256, 8, 20, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_half2<256, 8, 22, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_half2<256, 8, 24, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_half2<256, 8, 26, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_half2<256, 8, 28, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_half2<256, 8, 30, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_half2<256, 8, 32, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_half2<256, 16, 18, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_half2<256, 16, 20, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_half2<256, 16, 22, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_half2<256, 16, 24, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_half2<256, 16, 26, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_half2<256, 16, 28, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_half2<256, 16, 30, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_half2<256, 16, 32, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_half2<256, 32, 18, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_half2<256, 32, 20, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_half2<256, 32, 22, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_half2<256, 32, 24, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_half2<256, 32, 26, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_half2<256, 32, 28, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_half2<256, 32, 30, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_half2<256, 32, 32, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_half2<256, 32, 34, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_half2<256, 32, 36, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_half2<256, 32, 38, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_half2<256, 32, 40, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);

extern template void call_NW_local_affine_multi_pass_half2<256, 32, 22, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, __half2*, __half2*, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);



//float_kernel_instantiations.cu

extern template void call_NW_local_affine_single_pass_float<2, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_float<4, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_float<6, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_float<8, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_float<10, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_float<12, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_float<14, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_float<16, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_float<18, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_float<20, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_float<22, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_float<24, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_float<26, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_float<28, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_float<30, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_float<32, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_float<34, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_float<36, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_float<38, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);
extern template void call_NW_local_affine_single_pass_float<40, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);

extern template void call_NW_local_affine_multi_pass_float<20, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, float2 * const, float2 * const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const float, const float, cudaStream_t);

extern template void call_launch_process_overflow_alignments_kernel_NW_local_affine_multi_pass_float<20, ScoreIterInst, ReferenceIdT*>(const int* const, float2* const, const size_t, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, ReferenceIdT* const, const char4* const, const SequenceLengthT, const float, const float, cudaStream_t);



//dpx_s32_kernel_instantiations.cu

extern template void call_NW_local_affine_single_pass_dpx_s32<2, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s32<4, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s32<6, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s32<8, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s32<10, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s32<12, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s32<14, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s32<16, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s32<18, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s32<20, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s32<22, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s32<24, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s32<26, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s32<28, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s32<30, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s32<32, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s32<34, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s32<36, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s32<38, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s32<40, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);

extern template void call_NW_local_affine_multi_pass_dpx_s32<20, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, int2 * const, int2 * const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);

extern template void call_launch_process_overflow_alignments_kernel_NW_local_affine_multi_pass_dpx_s32<20, ScoreIterInst, ReferenceIdT*>(const int* const, int2* const, const size_t, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, ReferenceIdT* const, const char4* const, const SequenceLengthT, const int, const int, cudaStream_t);



//dpx_s16_kernel_instantiations.cu
extern template void call_NW_local_affine_single_pass_dpx_s16<256, 2, 24, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s16<256, 4, 16, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s16<256, 8, 10, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s16<256, 8, 12, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s16<256, 8, 14, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s16<256, 8, 16, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s16<256, 8, 18, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s16<256, 8, 20, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s16<256, 8, 22, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s16<256, 8, 24, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s16<256, 8, 26, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s16<256, 8, 28, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s16<256, 8, 30, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s16<256, 8, 32, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s16<256, 16, 18, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s16<256, 16, 20, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s16<256, 16, 22, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s16<256, 16, 24, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s16<256, 16, 26, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s16<256, 16, 28, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s16<256, 16, 30, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s16<256, 16, 32, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s16<256, 32, 18, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s16<256, 32, 20, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s16<256, 32, 22, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s16<256, 32, 24, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s16<256, 32, 26, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s16<256, 32, 28, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s16<256, 32, 30, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s16<256, 32, 32, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s16<256, 32, 34, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s16<256, 32, 36, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s16<256, 32, 38, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
extern template void call_NW_local_affine_single_pass_dpx_s16<256, 32, 40, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);

extern template void call_NW_local_affine_multi_pass_dpx_s16<256, 32, 22, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, short2*, short2*, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, ReferenceIdT* const, int* const, const bool, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);


} //namespace cudasw4

#endif