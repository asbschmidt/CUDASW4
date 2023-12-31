//#include "util.cuh"
#include "kernels.cuh"

//#include "thrust/iterator/counting_iterator.h"

namespace cudasw4{

// using ScoreIterInst = TopNMaximaArray;
// using PosIterInst = decltype(thrust::make_counting_iterator<size_t>(0));


template void call_NW_local_affine_single_pass_dpx_s32<2, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
template void call_NW_local_affine_single_pass_dpx_s32<4, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
template void call_NW_local_affine_single_pass_dpx_s32<6, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
template void call_NW_local_affine_single_pass_dpx_s32<8, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
template void call_NW_local_affine_single_pass_dpx_s32<10, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
template void call_NW_local_affine_single_pass_dpx_s32<12, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
template void call_NW_local_affine_single_pass_dpx_s32<14, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
template void call_NW_local_affine_single_pass_dpx_s32<16, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
template void call_NW_local_affine_single_pass_dpx_s32<18, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
template void call_NW_local_affine_single_pass_dpx_s32<20, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
template void call_NW_local_affine_single_pass_dpx_s32<22, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
template void call_NW_local_affine_single_pass_dpx_s32<24, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
template void call_NW_local_affine_single_pass_dpx_s32<26, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
template void call_NW_local_affine_single_pass_dpx_s32<28, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
template void call_NW_local_affine_single_pass_dpx_s32<30, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
template void call_NW_local_affine_single_pass_dpx_s32<32, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
template void call_NW_local_affine_single_pass_dpx_s32<34, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
template void call_NW_local_affine_single_pass_dpx_s32<36, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
template void call_NW_local_affine_single_pass_dpx_s32<38, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);
template void call_NW_local_affine_single_pass_dpx_s32<40, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);

template void call_NW_local_affine_multi_pass_dpx_s32<20, ScoreIterInst, PosIterInst>(BlosumType, const char * const, ScoreIterInst const, int2 * const, int2 * const, const size_t* const, const SequenceLengthT* const, PosIterInst const, const int, const char4*, const SequenceLengthT, const int, const int, cudaStream_t);

template void call_launch_process_overflow_alignments_kernel_NW_local_affine_multi_pass_dpx_s32<20, ScoreIterInst, ReferenceIdT*>(const int* const, int2* const, const size_t, const char * const, ScoreIterInst const, const size_t* const, const SequenceLengthT* const, ReferenceIdT* const, const char4* const, const SequenceLengthT, const int, const int, cudaStream_t);

} // namespace cudasw4