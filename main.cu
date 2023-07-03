
#include <iostream>
#include <algorithm>
#include <iterator>
#include <string>
#include <cuda_fp16.h>
#include <future>
#include <cstdlib>
#include <numeric>

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

#include "sequence_io.h"
#include "cuda_helpers.cuh"
#include <omp.h>
#include "dbdata.hpp"
#include "length_partitions.hpp"
#include "util.cuh"
#include "convert.cuh"

constexpr int ALIGN = 4;

typedef union
{
	int32_t   i;
    uint32_t  u;
    short2   s2;
} data_pack;

typedef short2	  score_type;
typedef data_pack score_pack;

template<class T>
__device__
T warp_max_reduce_broadcast(unsigned int mask, T val){
    #if __CUDA_ARCH__ >= 800
        return __reduce_max_sync(mask, val);
    #else
        for (int offset = 16; offset > 0; offset /= 2){
            T tmp = __shfl_down_sync(mask, val, offset);
            val = tmp > val ? tmp : val;
        }
        return __shfl_sync(mask, val, 0);
    #endif
}

inline __device__
uint32_t hset2_fp16x2(uint32_t a, uint32_t b) {
    uint32_t d;
    asm("set.eq.u32.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}

inline __device__ short2 shfl_up_2xint16(const uint32_t bitmap, const short2 value, const int lane, const int group_size) {
	score_pack v, res;
	v.s2 = value;
	res.u =__shfl_up_sync(bitmap, v.u, lane, group_size);
	return(res.s2);
}

inline __device__ short2 shfl_down_2xint16(const uint32_t bitmap, const short2 value, const int lane, const int group_size) {
	score_pack v, res;
	v.s2 = value;
	res.u =__shfl_down_sync(bitmap, v.u, lane, group_size);
	return(res.s2);
}

inline __device__ short2 shfl_snyc_2xint16(const uint32_t bitmap, const short2 value, const int lane, const int group_size) {
	score_pack v, res;
	v.s2 = value;
	res.u =__shfl_sync(bitmap, v.u, lane, group_size);
	return(res.s2);
}

inline __device__ uint32_t __hbeq2_emu(const short2 a, const short2 b) {
	score_pack a_in, b_in;
    a_in.s2 = a;
    b_in.s2 = b;
	return(hset2_fp16x2(a_in.u, b_in.u));
}

inline __device__ short2 eq_cmp(const short2 r_j, const short2 q_i,
								const short2 MATCH_SCORE, const short2 MISMATCH_SCORE) {
	score_pack query, refer, eq, w_ij, missm, match;

	refer.s2 = r_j;
	query.s2 = q_i;
	missm.s2 = MISMATCH_SCORE;
	match.s2 = MATCH_SCORE;

	eq.u = hset2_fp16x2(refer.u, query.u);
	w_ij.u   = (~eq.u & missm.u) | (eq.u & match.u);
	return(w_ij.s2);
}

inline __device__ short2 viadd(const short2 a_in, const short2 b_in) {
	score_pack a, b, d;
	a.s2 = a_in;
	b.s2 = b_in;
	d.u = __vadd2(a.u, b.u);
	return(d.s2);
}

inline __device__ short2 vimax(const short2 a_in, const short2 b_in) {
	score_pack a, b, d;
	a.s2 = a_in;
	b.s2 = b_in;
	d.u = __vmaxs2(a.u, b.u);
	return(d.s2);
}

inline __device__ short2 vimax3(const short2 a_in, const short2 b_in, const short2 c_in) {
	score_pack a, b, c, d;
	a.s2 = a_in;
	b.s2 = b_in;
	c.s2 = c_in;
	d.u = __vimax3_s16x2_relu(a.u, b.u, c.u);
	return(d.s2);
}

inline __device__ short2 viaddmax(const short2 a_in, const short2 b_in, const short2 c_in) {
	score_pack a, b, c, d;
	a.s2 = a_in;
	b.s2 = b_in;
	c.s2 = c_in;
	d.u = __viaddmax_s16x2(a.u, b.u, c.u);
	return(d.s2);
}

inline __device__ short2 vimax3_non_dpx(const short2 a_in, const short2 b_in, const short2 c_in, const short2 d_in) {
	score_pack a, b, c, d, e;
	a.s2 = a_in;
	b.s2 = b_in;
	c.s2 = c_in;
    d.s2 = d_in;
	e.u = __vmaxs2(__vmaxs2(__vmaxs2(a.u, b.u),c.u),d.u);//   __vimax3_s16x2_relu(a.u, b.u, c.u);
	return(e.s2);
}

inline __device__ short2 viaddmax_non_dpx(const short2 a_in, const short2 b_in, const short2 c_in) {
	score_pack a, b, c, d;
	a.s2 = a_in;
	b.s2 = b_in;
	c.s2 = c_in;
	d.u = __vmaxs2(__vadd2(a.u, b.u),c.u); // __viaddmax_s16x2(a.u, b.u, c.u);
	return(d.s2);
}

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

                
#define MAX_ACC_HALF2 2048.0 // 1024 ??

using std::cout;
using std::copy;


//__constant__ char cQuery[8*1024];
__constant__ char4 constantQuery4[2048];
__constant__ char cBLOSUM62_dev[21*21];
//__constant__ char2 cQuery2[4*1024];

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


GLOBALQUALIFIER
void test(char* chars, size_t* offsets, size_t nseq)
{
    printf("Number of sequences: %lu\n", nseq);

    for(size_t i = 0; i < nseq; ++i) {
        printf("Sequence %lu: ", i);
        for(size_t j = offsets[i]; j < offsets[i+1]; ++j) {
            printf("%c", chars[j]);
        }
        printf("\n");
    }
}






// Needleman-Wunsch (NW): global alignment with linear gap penalty
// numRegs values per thread
// uses a single warp per CUDA thread block;
// every groupsize threads computes an alignmen score
template <int group_size, int numRegs, class PositionsIterator> __global__
void NW_local_affine_many_pass_s16_DPX(
    const char * devChars,
    float * devAlignmentScores,
    short2 * devTempHcol2,
    short2 * devTempEcol2,
    const size_t* devOffsets,
    size_t* devLengths,
    const size_t* d_positions_of_selected_lengths,
    const int numSelected,
    const int length_2,
    const int32_t gap_open,
    const int32_t gap_extend
) {

    __shared__ score_type shared_BLOSUM62[21][21*21];
    int subject[numRegs];

    const score_type ZERO_in      = {0, 0};
    const score_type NEGINFINITY2 = {-1000, -1000};
    const score_type gap_open2    = {(short)gap_open, (short) gap_open};
    const score_type gap_extend2  = {(short)gap_extend, (short)gap_extend};

    const int blid = blockIdx.x;
    const int thid = threadIdx.x;
    const int group_id = thid%group_size;
	int offset = group_id + group_size;

    int check_last = blockDim.x/group_size;
    int check_last2 = 0;
    if (blid == gridDim.x-1) {
        if (numSelected % (2*blockDim.x/group_size)) {
            check_last = (numSelected/2) % (blockDim.x/group_size);
            check_last2 = numSelected%2;
            check_last = check_last + check_last2;
        }
    }
    check_last = check_last * group_size;

    const int length_S0 = devLengths[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)]];
    const int base_S0 = devOffsets[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)]];

    int length_S1 = devLengths[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)+1]];
    int base_S1 = devOffsets[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)+1]];

    if (blid == gridDim.x-1)
        if (check_last2)
            if (((thid%check_last) >= check_last-group_size) && ((thid%check_last) < check_last)) {
                length_S1 = length_S0;
                base_S1 = base_S0;
            }

    const int length = max(length_S0, length_S1);
    //const int length = warp_max_reduce_broadcast(0xFFFFFFFF, temp_length);

    //const int lane_2 = length_2+1;
    short2 H_temp_out, H_temp_in;
    short2 E_temp_out, E_temp_in;

    score_type penalty_temp0, penalty_temp1; // penalty_temp2;
    score_type penalty_left, penalty_diag;
    score_type penalty_here31;
    score_type penalty_here_array[numRegs];
    score_type F_here_array[numRegs];
    score_type E = NEGINFINITY2;
    //const int base_3 = d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)]*length_2; // blid*length_2;
	const int base_3 = (2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size))*length_2; // blid*length_2;
    score_type maximum = ZERO_in;
    short2 * devTempHcol = (short2*)(&devTempHcol2[base_3]);
    short2 * devTempEcol = (short2*)(&devTempEcol2[base_3]);


    auto init_penalties_local = [&](const auto& value) {
        //ZERO = NEGINFINITY2;
        penalty_left = NEGINFINITY2;
        penalty_diag = NEGINFINITY2;
        E = NEGINFINITY2;
        #pragma unroll
        for (int i=0; i<numRegs; i++) penalty_here_array[i] = NEGINFINITY2;
        for (int i=0; i<numRegs; i++) F_here_array[i] = NEGINFINITY2;
        if (thid % group_size == 0) {
            penalty_left = ZERO_in;
            penalty_diag = ZERO_in;
            #pragma unroll
            for (int i=0; i<numRegs; i++) penalty_here_array[i] = ZERO_in;
        }
        if (thid % group_size== 1) {
            penalty_left = ZERO_in;
        }
    };

    score_type temp0; // temp1, E_temp_float, H_temp_float;

    auto init_local_score_profile_BLOSUM62 = [&](const auto& offset_isc) {
        if (!offset_isc) {
		for (int i=thid; i<21*21; i+=blockDim.x) {
            temp0.x = cBLOSUM62_dev[21*(i/21)+(i%21)];
            for (int j=0; j<21; j++) {
                temp0.y = cBLOSUM62_dev[21*(i/21)+j];
                shared_BLOSUM62[i/21][21*(i%21)+j]=temp0;
            }
        }
        __syncthreads();
	   }

       for (int i=0; i<numRegs; i++) {

           if (offset_isc+numRegs*(thid%group_size)+i >= length_S0) subject[i] = 20;
           else subject[i] = devChars[offset_isc+base_S0+numRegs*(thid%group_size)+i];

           if (offset_isc+numRegs*(thid%group_size)+i >= length_S1) subject[i] += 20*21;
           else subject[i] += 21*devChars[offset_isc+base_S1+numRegs*(thid%group_size)+i];
       }
    };

    H_temp_out.x = -1000; H_temp_out.y = -1000;
    E_temp_out.x = -1000; E_temp_out.y = -1000;
    int counter = 1;
    //int counter2 = 1;
    char query_letter = 20;
    char4 new_query_letter4 = constantQuery4[thid%group_size];
    if (thid % group_size== 0) query_letter = new_query_letter4.x;

    score_type score2;
    score_type *sbt_row;

    auto initial_calc32_local_affine_float = [&](const auto& value){
        sbt_row = shared_BLOSUM62[query_letter];

        score2 = sbt_row[subject[0]];
        penalty_temp0 = penalty_here_array[0];
        if (!value || (thid%group_size)) E = viaddmax(E, gap_extend2,viadd(penalty_left,gap_open2));
        F_here_array[0] = viaddmax(F_here_array[0],gap_extend2,viadd(penalty_here_array[0],gap_open2));
        penalty_here_array[0] = vimax3(viadd(penalty_diag,score2),E, F_here_array[0]);
        //maximum = vimax(temp0,maximum);

        score2 = sbt_row[subject[1]];
        penalty_temp1 = penalty_here_array[1];
        E = viaddmax(E,gap_extend2,viadd(penalty_here_array[0],gap_open2));
        F_here_array[1] = viaddmax(F_here_array[1],gap_extend2,viadd(penalty_here_array[1],gap_open2));
        penalty_here_array[1] = vimax3(viadd(penalty_temp0,score2),E, F_here_array[1]);
        //maximum = vimax(temp0,maximum);
		maximum = vimax3(maximum,penalty_here_array[0],penalty_here_array[1]);

        #pragma unroll
        for (int i=1; i<numRegs/2; i++) {
            score2 = sbt_row[subject[2*i]];
            penalty_temp0 = penalty_here_array[2*i];
            E = vimax(viadd(E,gap_extend2), viadd(penalty_here_array[2*i-1],gap_open2));
            F_here_array[2*i] = vimax(viadd(F_here_array[2*i],gap_extend2), viadd(penalty_here_array[2*i],gap_open2));
            penalty_here_array[2*i] = vimax3(viadd(penalty_temp1,score2), E, F_here_array[2*i]);
            //maximum = vimax(temp0,maximum);

            score2 = sbt_row[subject[2*i+1]];
            penalty_temp1 = penalty_here_array[2*i+1];
            E = vimax(viadd(E,gap_extend2), viadd(penalty_here_array[2*i],gap_open2));
            F_here_array[2*i+1] = vimax(viadd(F_here_array[2*i+1],gap_extend2), viadd(penalty_here_array[2*i+1],gap_open2));
            penalty_here_array[2*i+1] = vimax3(viadd(penalty_temp0,score2), E, F_here_array[2*i+1]);
            //maximum = vimax(temp0,maximum);
			maximum = vimax3(maximum,penalty_here_array[2*i],penalty_here_array[2*i+1]);

        }

        penalty_here31 = penalty_here_array[numRegs-1];
        E = vimax(viadd(E, gap_extend2), viadd(penalty_here31, gap_open2));
        for (int i=0; i<numRegs; i++) F_here_array[i] = vimax(viadd(F_here_array[i],gap_extend2), viadd(penalty_here_array[i],gap_open2));
    };


    auto calc32_local_affine_float = [&](){
        sbt_row = shared_BLOSUM62[query_letter];

        score2 = sbt_row[subject[0]];
        penalty_temp0 = penalty_here_array[0];
        penalty_here_array[0] = vimax3(viadd(penalty_diag,score2), E, F_here_array[0]);
        penalty_temp1 = viadd(penalty_here_array[0],gap_open2);
        E = viaddmax(E,gap_extend2, penalty_temp1);
        F_here_array[0] = viaddmax(F_here_array[0],gap_extend2, penalty_temp1);

        score2 = sbt_row[subject[1]];
        penalty_temp1 = penalty_here_array[1];
        penalty_here_array[1] = vimax3(viadd(penalty_temp0,score2), E, F_here_array[1]);
        penalty_temp0 = viadd(penalty_here_array[1],gap_open2);
        E = viaddmax(E,gap_extend2, penalty_temp0);
        F_here_array[1] = viaddmax(F_here_array[1],gap_extend2,penalty_temp0);
        maximum = vimax3(maximum,penalty_here_array[0],penalty_here_array[1]);

        #pragma unroll
        for (int i=1; i<numRegs/2; i++) {
            score2 = sbt_row[subject[2*i]];
            penalty_temp0 = penalty_here_array[2*i];
            penalty_here_array[2*i] = vimax3(viadd(penalty_temp1,score2), E, F_here_array[2*i]);
            penalty_temp1 = viadd(penalty_here_array[2*i],gap_open2);
            E = viaddmax(E,gap_extend2, penalty_temp1);
            F_here_array[2*i] = viaddmax(F_here_array[2*i],gap_extend2, penalty_temp1);

            score2 = sbt_row[subject[2*i+1]];
            penalty_temp1 = penalty_here_array[2*i+1];

            penalty_here_array[2*i+1] = vimax3(viadd(penalty_temp0,score2), E, F_here_array[2*i+1]);
            penalty_temp0 = viadd(penalty_here_array[2*i+1],gap_open2);
            E = viaddmax(E,gap_extend2, penalty_temp0);
            F_here_array[2*i+1] = viaddmax(F_here_array[2*i+1],gap_extend2,penalty_temp0);
            maximum = vimax3(maximum,penalty_here_array[2*i],penalty_here_array[2*i+1]);
        }
        //for (int i=0; i<numRegs/4; i++)
        //    maximum = vimax3(maximum,vimax(penalty_here_array[4*i],penalty_here_array[4*i+1]),vimax(penalty_here_array[4*i+2],penalty_here_array[4*i+3]));

        penalty_here31 = penalty_here_array[numRegs-1];
    };

    //const int passes_S0 = ceil((1.0*length_S0)/(group_size*numRegs));
    //const int passes_S1 = ceil((1.0*length_S1)/(group_size*numRegs));

    const int passes = ceil((1.0*length)/(group_size*numRegs));

    int offset_out = group_id;
    int offset_in = group_id;

    auto shuffle_query = [&](const auto& new_letter) {
        query_letter = __shfl_up_sync(0xFFFFFFFF, query_letter, 1, 32);
        if (!group_id) query_letter = new_letter;
    };

    auto shuffle_affine_penalty = [&](const auto& new_penalty_left, const auto& new_E_left) {
        penalty_diag = penalty_left;
        penalty_left = shfl_up_2xint16(0xFFFFFFFF, penalty_here31, 1, 32);
        E = shfl_up_2xint16(0xFFFFFFFF, E, 1, 32);
        if (!group_id) {
            penalty_left = new_penalty_left;
            E = new_E_left;
        }
    };

    uint32_t temp;

    auto shuffle_H_E_temp_out = [&]() {
        temp = __shfl_down_sync(0xFFFFFFFF, *((int*)(&H_temp_out)), 1, 32);
        H_temp_out = *((short2*)(&temp));
        temp = __shfl_down_sync(0xFFFFFFFF, *((int*)(&E_temp_out)), 1, 32);
        E_temp_out = *((short2*)(&temp));
    };

    auto shuffle_H_E_temp_in = [&]() {
        temp = __shfl_down_sync(0xFFFFFFFF, *((int*)(&H_temp_in)), 1, 32);
        H_temp_in = *((short2*)(&temp));
        temp = __shfl_down_sync(0xFFFFFFFF, *((int*)(&E_temp_in)), 1, 32);
        E_temp_in = *((short2*)(&temp));

    };

    auto shuffle_new_query = [&]() {
        temp = __shfl_down_sync(0xFFFFFFFF, *((int*)(&new_query_letter4)), 1, 32);
        new_query_letter4 = *((char4*)(&temp));
    };

    auto set_H_E_temp_out = [&]() {
        if (thid % group_size == 31) {
            H_temp_out = penalty_here31;
            E_temp_out = E;
        }
    };

    int k;

    const uint32_t thread_result = ((length-1)%(group_size*numRegs))/numRegs;              // 0, 1, ..., or 31

    // FIRST PASS (of many passes)
    // Note first pass has always full seqeunce length
    init_penalties_local(0);
    init_local_score_profile_BLOSUM62(0);
    initial_calc32_local_affine_float(0);
    shuffle_query(new_query_letter4.y);
    shuffle_affine_penalty(ZERO_in, NEGINFINITY2);

    //shuffle_max();
    calc32_local_affine_float();
    shuffle_query(new_query_letter4.z);
    shuffle_affine_penalty(ZERO_in, NEGINFINITY2);

    //shuffle_max();
    calc32_local_affine_float();
    shuffle_query(new_query_letter4.w);
    shuffle_affine_penalty(ZERO_in, NEGINFINITY2);
    shuffle_new_query();
    counter++;

    for (k = 4; k <= length_2+28; k+=4) {
        //shuffle_max();
        calc32_local_affine_float();
        set_H_E_temp_out();
        shuffle_H_E_temp_out();

        shuffle_query(new_query_letter4.x);
        shuffle_affine_penalty(ZERO_in,NEGINFINITY2);

        //shuffle_max();
        calc32_local_affine_float();
        set_H_E_temp_out();
        shuffle_H_E_temp_out();

        shuffle_query(new_query_letter4.y);
        shuffle_affine_penalty(ZERO_in,NEGINFINITY2);

        //shuffle_max();
        calc32_local_affine_float();
        set_H_E_temp_out();
        shuffle_H_E_temp_out();

        shuffle_query(new_query_letter4.z);
        shuffle_affine_penalty(ZERO_in,NEGINFINITY2);

        //shuffle_max();
        calc32_local_affine_float();

        set_H_E_temp_out();
        if (counter%8 == 0 && counter > 8) {
            devTempHcol[offset_out]=H_temp_out;
            devTempEcol[offset_out]=E_temp_out;
            offset_out += group_size;
        }
        shuffle_H_E_temp_out();
        shuffle_query(new_query_letter4.w);
        shuffle_affine_penalty(ZERO_in,NEGINFINITY2);
        shuffle_new_query();
        if (counter%group_size == 0) {
            new_query_letter4 = constantQuery4[offset];
            offset += group_size;
        }
        counter++;
    }
    if (length_2 % 4 == 0) {
        temp = __shfl_up_sync(0xFFFFFFFF, *((int*)(&H_temp_out)), 1, 32);
        H_temp_out = *((short2*)(&temp));
        temp = __shfl_up_sync(0xFFFFFFFF, *((int*)(&E_temp_out)), 1, 32);
        E_temp_out = *((short2*)(&temp));
    }
    if (length_2%4 == 1) {
        //shuffle_max();
        calc32_local_affine_float();
        set_H_E_temp_out();
    }

    if (length_2%4 == 2) {
        //shuffle_max();
        calc32_local_affine_float();
        set_H_E_temp_out();
        shuffle_H_E_temp_out();
        shuffle_query(new_query_letter4.x);
        shuffle_affine_penalty(ZERO_in,NEGINFINITY2);
        //shuffle_max();
        calc32_local_affine_float();
        set_H_E_temp_out();
    }
    if (length_2%4 == 3) {
        //shuffle_max();
        calc32_local_affine_float();
        set_H_E_temp_out();
        shuffle_H_E_temp_out();
        shuffle_query(new_query_letter4.x);
        shuffle_affine_penalty(ZERO_in,NEGINFINITY2);
        //shuffle_max();
        calc32_local_affine_float();
        set_H_E_temp_out();
        shuffle_H_E_temp_out();
        shuffle_query(new_query_letter4.y);
        shuffle_affine_penalty(ZERO_in,NEGINFINITY2);
        //shuffle_max();
        calc32_local_affine_float();
        set_H_E_temp_out();
    }
    int final_out = length_2 % 32;
    int from_thread_id = 32 - final_out;

    if (thid>=from_thread_id) {
        devTempHcol[offset_out-from_thread_id]=H_temp_out;
        devTempEcol[offset_out-from_thread_id]=E_temp_out;
    }

   //middle passes
   for (int pass = 1; pass < passes-1; pass++) {
       //maximum = __shfl_sync(0xFFFFFFFF, maximum, 31, 32);
       counter = 1;
       //counter2 = 1;
       query_letter = 20;
       new_query_letter4 = constantQuery4[thid%group_size];
       if (thid % group_size== 0) query_letter = new_query_letter4.x;

       offset = group_id + group_size;
       offset_out = group_id;
       offset_in = group_id;
       H_temp_in = devTempHcol[offset_in];
       E_temp_in = devTempEcol[offset_in];
       offset_in += group_size;

      init_penalties_local(0);
      init_local_score_profile_BLOSUM62(pass*(32*numRegs));

      if (!group_id) {
          penalty_left = H_temp_in;
          E = E_temp_in;
      }
      shuffle_H_E_temp_in();

      initial_calc32_local_affine_float(1);
      shuffle_query(new_query_letter4.y);
      shuffle_affine_penalty(H_temp_in,E_temp_in);
      shuffle_H_E_temp_in();

      //shuffle_max();
      calc32_local_affine_float();
      shuffle_query(new_query_letter4.z);
      shuffle_affine_penalty(H_temp_in,E_temp_in);
      shuffle_H_E_temp_in();

      //shuffle_max();
      calc32_local_affine_float();
      shuffle_query(new_query_letter4.w);
      shuffle_affine_penalty(H_temp_in,E_temp_in);
      shuffle_H_E_temp_in();
      shuffle_new_query();
      counter++;

      for (k = 4; k <= length_2+28; k+=4) {
              //shuffle_max();
              calc32_local_affine_float();
              set_H_E_temp_out();
              shuffle_H_E_temp_out();

              shuffle_query(new_query_letter4.x);
              shuffle_affine_penalty(H_temp_in,E_temp_in);
              shuffle_H_E_temp_in();

              //shuffle_max();
              calc32_local_affine_float();
              set_H_E_temp_out();
              shuffle_H_E_temp_out();

              shuffle_query(new_query_letter4.y);
              shuffle_affine_penalty(H_temp_in,E_temp_in);
              shuffle_H_E_temp_in();

              //shuffle_max();
              calc32_local_affine_float();
              set_H_E_temp_out();
              shuffle_H_E_temp_out();

              shuffle_query(new_query_letter4.z);
              shuffle_affine_penalty(H_temp_in,E_temp_in);
              shuffle_H_E_temp_in();

              //shuffle_max();
              calc32_local_affine_float();
              set_H_E_temp_out();
              if (counter%8 == 0 && counter > 8) {
                  devTempHcol[offset_out]=H_temp_out;
                  devTempEcol[offset_out]=E_temp_out;
                  offset_out += group_size;
              }
              shuffle_H_E_temp_out();
              shuffle_query(new_query_letter4.w);
              shuffle_affine_penalty(H_temp_in,E_temp_in);
              shuffle_new_query();
              if (counter%group_size == 0) {
                  new_query_letter4 = constantQuery4[offset];
                  offset += group_size;
              }
              shuffle_H_E_temp_in();
              if (counter%8 == 0) {
                  H_temp_in = devTempHcol[offset_in];
                  E_temp_in = devTempEcol[offset_in];
                  offset_in += group_size;
              }
              counter++;
      }
      if (length_2 % 4 == 0) {
          temp = __shfl_up_sync(0xFFFFFFFF, *((int*)(&H_temp_out)), 1, 32);
          H_temp_out = *((short2*)(&temp));
          temp = __shfl_up_sync(0xFFFFFFFF, *((int*)(&E_temp_out)), 1, 32);
          E_temp_out = *((short2*)(&temp));
      }
      if (length_2%4 == 1) {
          //shuffle_max();
          calc32_local_affine_float();
          set_H_E_temp_out();
      }

      if (length_2%4 == 2) {
          //shuffle_max();
          calc32_local_affine_float();
          set_H_E_temp_out();
          shuffle_H_E_temp_out();
          shuffle_query(new_query_letter4.x);
          shuffle_affine_penalty(H_temp_in,E_temp_in);
          shuffle_H_E_temp_in();
          //shuffle_max();
          calc32_local_affine_float();
          set_H_E_temp_out();
      }
      if (length_2%4 == 3) {
          //shuffle_max();
          calc32_local_affine_float();
          set_H_E_temp_out();
          shuffle_H_E_temp_out();
          shuffle_query(new_query_letter4.x);
          shuffle_affine_penalty(H_temp_in,E_temp_in);
          shuffle_H_E_temp_in();
          //shuffle_max();
          calc32_local_affine_float();
          set_H_E_temp_out();
          shuffle_H_E_temp_out();
          shuffle_query(new_query_letter4.y);
          shuffle_affine_penalty(H_temp_in,E_temp_in);
          shuffle_H_E_temp_in();
          //shuffle_max();
          calc32_local_affine_float();
          set_H_E_temp_out();
      }
      int final_out = length_2 % 32;
      int from_thread_id = 32 - final_out;

      if (thid>=from_thread_id) {
          devTempHcol[offset_out-from_thread_id]=H_temp_out;
          devTempEcol[offset_out-from_thread_id]=E_temp_out;
      }

      //if (thid == 31) {
    //      max_results_S0[pass] = maximum.x;
    //      max_results_S1[pass] = maximum.y;
     // }
   }

   // Final pass
   //maximum = __shfl_sync(0xFFFFFFFF, maximum, 31, 32);
   counter = 1;
   //counter2 = 1;
   query_letter = 20;
   new_query_letter4 = constantQuery4[thid%group_size];
   if (thid % group_size== 0) query_letter = new_query_letter4.x;

   offset = group_id + group_size;
   offset_in = group_id;
   H_temp_in = devTempHcol[offset_in];
   E_temp_in = devTempEcol[offset_in];
   offset_in += group_size;

   init_penalties_local(0);
   init_local_score_profile_BLOSUM62((passes-1)*(32*numRegs));
   //copy_H_E_temp_in();
   if (!group_id) {
       penalty_left = H_temp_in;
       E = E_temp_in;
   }
   shuffle_H_E_temp_in();

   initial_calc32_local_affine_float(1);
   shuffle_query(new_query_letter4.y);
   shuffle_affine_penalty(H_temp_in,E_temp_in);
   shuffle_H_E_temp_in();
  if (length_2+thread_result >=2) {
      //shuffle_max();
      calc32_local_affine_float();
      shuffle_query(new_query_letter4.z);
      //copy_H_E_temp_in();
      shuffle_affine_penalty(H_temp_in,E_temp_in);
      shuffle_H_E_temp_in();
  }

   if (length_2+thread_result >=3) {
       //shuffle_max();
       calc32_local_affine_float();
       shuffle_query(new_query_letter4.w);
       //copy_H_E_temp_in();
       shuffle_affine_penalty(H_temp_in,E_temp_in);
       shuffle_H_E_temp_in();
       shuffle_new_query();
       counter++;
   }
   if (length_2+thread_result >=4) {
   //for (k = 5; k < lane_2+thread_result-2; k+=4) {
   for (k = 4; k <= length_2+(thread_result-3); k+=4) {
           //shuffle_max();
           calc32_local_affine_float();

           shuffle_query(new_query_letter4.x);
           //copy_H_E_temp_in();
           shuffle_affine_penalty(H_temp_in,E_temp_in);
           shuffle_H_E_temp_in();

           //shuffle_max();
           calc32_local_affine_float();
           shuffle_query(new_query_letter4.y);
           //copy_H_E_temp_in();
           shuffle_affine_penalty(H_temp_in,E_temp_in);
           shuffle_H_E_temp_in();

           //shuffle_max();
           calc32_local_affine_float();
           shuffle_query(new_query_letter4.z);
           //copy_H_E_temp_in();
           shuffle_affine_penalty(H_temp_in,E_temp_in);
           shuffle_H_E_temp_in();

           //shuffle_max();
           calc32_local_affine_float();
           shuffle_query(new_query_letter4.w);
           //copy_H_E_temp_in();
           shuffle_affine_penalty(H_temp_in,E_temp_in);
           shuffle_new_query();
           if (counter%group_size == 0) {
               new_query_letter4 = constantQuery4[offset];
               offset += group_size;
           }
           shuffle_H_E_temp_in();
           if (counter%8 == 0) {
               H_temp_in = devTempHcol[offset_in];
               E_temp_in = devTempEcol[offset_in];
               offset_in += group_size;
           }
           counter++;
       }
   //    if (blid == 0) {
   //        if (thid % group_size == thread_result)
   //            printf("Result in Thread: %d, Register: %d, Value: %f, #passes: %d\n", thid, (length-1)%numRegs, penalty_here_array[(length-1)%numRegs], passes);
   //    }
//      if ((length_2+thread_result+1)%4 > 0) {
    //if (counter2-(length_2+thread_result) > 0) {

        if ((k-1)-(length_2+thread_result) > 0) {
            //shuffle_max();
            calc32_local_affine_float();
            shuffle_query(new_query_letter4.x);
            shuffle_affine_penalty(H_temp_in,E_temp_in);
            shuffle_H_E_temp_in();
            k++;
        }


       if ((k-1)-(length_2+thread_result) > 0) {
           //shuffle_max();
           calc32_local_affine_float();
           shuffle_query(new_query_letter4.y);
           shuffle_affine_penalty(H_temp_in,E_temp_in);
           shuffle_H_E_temp_in();
           k++;
       }

       if ((k-1)-(length_2+thread_result) > 0) {
           //shuffle_max();
           calc32_local_affine_float();
       }

   //    if (blid == 0) {
   //        if (thid % group_size == thread_result)
   //            printf("Result in Thread: %d, Register: %d, Value: %f, #passes: %d\n", thid, (length-1)%numRegs, penalty_here_array[(length-1)%numRegs], passes);
   //    }
   }

   for (int offset=group_size/2; offset>0; offset/=2)  maximum = vimax(maximum,shfl_down_2xint16(0xFFFFFFFF,maximum,offset,group_size));

  // if (thid % group_size == thread_result)
    //  printf("Result in Block: %d, in Thread: %d, Register: %d, Value: %f, #passes: %d\n", blid, thid, (length-1)%numRegs, penalty_here_array[(length-1)%numRegs], passes);
   if (!group_id) {
       if (blid < gridDim.x-1) {
           devAlignmentScores[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*(thid/group_size)]] =  maximum.y; // lane_2+thread_result+1-length_2%4; penalty_here_array[(length-1)%numRegs];
           devAlignmentScores[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*(thid/group_size)+1]] =  maximum.x; // lane_2+thread_result+1-length_2%4; penalty_here_array[(length-1)%numRegs];
       } else {
           devAlignmentScores[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)]] =  maximum.y; // lane_2+thread_result+1-length_2%4; penalty_here_array[(length-1)%numRegs];
           if (!check_last2 || (thid%check_last) < check_last-group_size) devAlignmentScores[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)+1]] =  maximum.x; // lane_2+thread_result+1-length_2%4; penalty_here_array[(length-1)%numRegs];
       }
   }
}




// Needleman-Wunsch (NW): global alignment with linear gap penalty
// numRegs values per thread
// uses a single warp per CUDA thread block;
// every groupsize threads computes an alignmen score
template <int group_size, int numRegs, class PositionsIterator> __global__
void NW_local_affine_Protein_many_pass_half2(
    const char * devChars,
    float * devAlignmentScores,
    __half2 * devTempHcol2,
    __half2 * devTempEcol2,
    const size_t* devOffsets,
    size_t* devLengths,
    PositionsIterator d_positions_of_selected_lengths,
    const int numSelected,
	size_t* d_overflow_positions,
	int *d_overflow_number,
	const bool overflow_check,
    const int length_2,
    const float gap_open,
    const float gap_extend
) {

    __shared__ __half2 shared_BLOSUM62[21][21*21];
    int subject[numRegs];


    //__shared__ __half2 devTempHcol[1024];
    //__shared__ __half2 devTempEcol[1024];

    const __half2 NEGINFINITY2 = __float2half2_rn(-1000.0);
    const __half2 gap_open2 = __float2half2_rn(gap_open);
    const __half2 gap_extend2 = __float2half2_rn(gap_extend);

    const int blid = blockIdx.x;
    const int thid = threadIdx.x;
    const int group_id = thid%group_size;
	int offset = group_id + group_size;

    int check_last = blockDim.x/group_size;
    int check_last2 = 0;
    if (blid == gridDim.x-1) {
        if (numSelected % (2*blockDim.x/group_size)) {
            check_last = (numSelected/2) % (blockDim.x/group_size);
            check_last2 = numSelected%2;
            check_last = check_last + check_last2;
        }
    }
    check_last = check_last * group_size;

    const int length_S0 = devLengths[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)]];
    const int base_S0 = devOffsets[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)]]-devOffsets[0];

	int length_S1 = devLengths[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)+1]];
	int base_S1 = devOffsets[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)+1]]-devOffsets[0];


    //int check_blid = 0;
	//const int seqID_check = 269765;
	//if (!group_id)
	//	if (d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)] == seqID_check) {
	//		printf("Sequence %d in Block: %d, Thread: %d, Length_S0: %d, Length_S1: %d, as S0\n", seqID_check, blid, thid, length_S0, length_S1);
	//		check_blid = 1;
		//}

	//if (!group_id)
		//if (d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)+1] == seqID_check) {
		//	printf("Sequence %d in Block: %d, Thread: %d, Length_S0: %d, Length_S1: %d, as S1\n", seqID_check, blid, thid, length_S0, length_S1);
		//	check_blid = 1;
	//	}

	//if (blid == gridDim.x-1)
	 //  if (thid%group_size == 0) {
	//	   int pos1 = d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)];
	//	   int pos2 = d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)+1];
	//	   printf("Blid: %d, Thid: %d, SeqID_S0: %d, SeqID_S1: %d, L0: %d, L1: %d, check_last: %d \n", blid, thid, pos1, pos2, length_S0, length_S1, check_last);
	//   }


    if (blid == gridDim.x-1)
        if (check_last2)
            if (((thid%check_last) >= check_last-group_size) && ((thid%check_last) < check_last)) {
                length_S1 = length_S0;
                base_S1 = base_S0;
            }

    //    if (blid+1 < 50) {
    //         int p1 = d_positions_of_selected_lengths[2*blid];
    //         int p2 = d_positions_of_selected_lengths[2*blid+1];
    //         if (thid == 0)
    //            printf("Blid: %d, Position0: %d, Position1: %d, L0: %d, L1: %d\n", blid, p1, p2, length_S0, length_S1);
    //    }

    //if ((length_S0 < min_length) || (length_S0 > max_length) || (length_S1 < min_length) || (length_S1 > max_length)) return;


    const int length = max(length_S0, length_S1);
    //const int length = warp_max_reduce_broadcast(0xFFFFFFFF, temp_length);

    //const int lane_2 = length_2+1;
    __half2 H_temp_out, H_temp_in;
    __half2 E_temp_out, E_temp_in;

    __half2 penalty_temp0, penalty_temp1; // penalty_temp2;
    __half2 penalty_left, penalty_diag;
    __half2 penalty_here31;
    __half2 penalty_here_array[numRegs];
    __half2 F_here_array[numRegs];
    __half2 E = NEGINFINITY2;

	const int base_3 = (2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size))*length_2;
    // if(blockIdx.x == 0){
    //     printf("tid %d, base_3 %d, ql %d\n", threadIdx.x, base_3, length_2);
    // }
	//const int base_3 = (2*(blockDim.x/group_size)*blid+2*(thid/group_size))*(SDIV(length_2, group_size) * group_size);
    __half2 maximum = __float2half2_rn(0.0);
    const __half2 ZERO = __float2half2_rn(0.0);
    __half2 * devTempHcol = (half2*)(&devTempHcol2[base_3]);
    __half2 * devTempEcol = (half2*)(&devTempEcol2[base_3]);

    auto checkHEindex = [&](auto x){
        // assert(x >= 0); //positive index
        // assert(2*(blockDim.x/group_size)*blid * length_2 <= base_3 + x);
        // assert(base_3+x < 2*(blockDim.x/group_size)*(blid+1) * length_2);
    };


    auto init_penalties_local = [&](const auto& value) {
        //ZERO = NEGINFINITY2;
        penalty_left = NEGINFINITY2;
        penalty_diag = NEGINFINITY2;
        E = NEGINFINITY2;
        #pragma unroll
        for (int i=0; i<numRegs; i++) penalty_here_array[i] = NEGINFINITY2;
        for (int i=0; i<numRegs; i++) F_here_array[i] = NEGINFINITY2;
        if (thid % group_size == 0) {
            penalty_left = __floats2half2_rn(0,0);
            penalty_diag = __floats2half2_rn(0,0);
            #pragma unroll
            for (int i=0; i<numRegs; i++) penalty_here_array[i] = __floats2half2_rn(0,0);
        }
        if (thid % group_size == 1) {
            penalty_left = __floats2half2_rn(0,0);
        }
    };

    __half2 temp0; // temp1, E_temp_float, H_temp_float;

    auto init_local_score_profile_BLOSUM62 = [&](const auto& offset_isc) {
        if (!offset_isc) {
		for (int i=thid; i<21*21; i+=blockDim.x) {
            temp0.x = cBLOSUM62_dev[21*(i/21)+(i%21)];
            for (int j=0; j<21; j++) {
                temp0.y = cBLOSUM62_dev[21*(i/21)+j];
                shared_BLOSUM62[i/21][21*(i%21)+j]=temp0;
            }
        }
        __syncthreads();
	   }
       for (int i=0; i<numRegs; i++) {

           if (offset_isc+numRegs*(thid%group_size)+i >= length_S0) subject[i] = 1; // 20;
           else subject[i] = devChars[offset_isc+base_S0+numRegs*(thid%group_size)+i];

           if (offset_isc+numRegs*(thid%group_size)+i >= length_S1) subject[i] += 1*21; // 20*21;
           else subject[i] += 21*devChars[offset_isc+base_S1+numRegs*(thid%group_size)+i];
       }
    };

    H_temp_out.x = -1000; H_temp_out.y = -1000;
    E_temp_out.x = -1000; E_temp_out.y = -1000;
    int counter = 1;
    //int counter2 = 1;
    char query_letter = 20;
    char4 new_query_letter4 = constantQuery4[thid%group_size];
    if (thid % group_size== 0) query_letter = new_query_letter4.x;

    __half2 score2;
    __half2 *sbt_row;

    auto initial_calc32_local_affine_float = [&](const auto& value){
        sbt_row = shared_BLOSUM62[query_letter];

        score2 = sbt_row[subject[0]];
        //score2.y = sbt_row[subject1[0].x];
        penalty_temp0 = penalty_here_array[0];
        if (!value || (thid%group_size)) E = __hmax2(__hadd2(E,gap_extend2), __hadd2(penalty_left,gap_open2));
        F_here_array[0] = __hmax2(__hadd2(F_here_array[0],gap_extend2), __hadd2(penalty_here_array[0],gap_open2));
        penalty_here_array[0] = temp0 = __hmax2(__hmax2(__hadd2(penalty_diag,score2), __hmax2(E, F_here_array[0])),ZERO);
        maximum = __hmax2(temp0,maximum);

        score2 = sbt_row[subject[1]];
        //score2.y = sbt_row[subject1[0].y];
        penalty_temp1 = penalty_here_array[1];
        E = __hmax2(__hadd2(E,gap_extend2), __hadd2(penalty_here_array[0],gap_open2));
        F_here_array[1] = __hmax2(__hadd2(F_here_array[1],gap_extend2), __hadd2(penalty_here_array[1],gap_open2));
        temp0 = penalty_here_array[1] = __hmax2(__hmax2(__hadd2(penalty_temp0,score2), __hmax2(E, F_here_array[1])),ZERO);
        maximum = __hmax2(temp0,maximum);

        #pragma unroll
        for (int i=1; i<numRegs/2; i++) {
            score2 = sbt_row[subject[2*i]];
            //score2.y = sbt_row[subject1[i].x];
            penalty_temp0 = penalty_here_array[2*i];
            E = __hmax2(__hadd2(E,gap_extend2), __hadd2(penalty_here_array[2*i-1],gap_open2));
            F_here_array[2*i] = __hmax2(__hadd2(F_here_array[2*i],gap_extend2), __hadd2(penalty_here_array[2*i],gap_open2));
            temp0 = penalty_here_array[2*i] = __hmax2(__hmax2(__hadd2(penalty_temp1,score2), __hmax2(E, F_here_array[2*i])),ZERO);
            maximum = __hmax2(temp0,maximum);

            score2 = sbt_row[subject[2*i+1]];
            //score2.y = sbt_row[subject1[i].y];
            penalty_temp1 = penalty_here_array[2*i+1];
            E = __hmax2(__hadd2(E,gap_extend2), __hadd2(penalty_here_array[2*i],gap_open2));
            F_here_array[2*i+1] = __hmax2(__hadd2(F_here_array[2*i+1],gap_extend2), __hadd2(penalty_here_array[2*i+1],gap_open2));
            temp0 = penalty_here_array[2*i+1] = __hmax2(__hmax2(__hadd2(penalty_temp0,score2), __hmax2(E, F_here_array[2*i+1])),ZERO);
            maximum = __hmax2(temp0,maximum);
        }

        penalty_here31 = penalty_here_array[numRegs-1];
        E = __hmax2(E+gap_extend2, penalty_here31+gap_open2);
        for (int i=0; i<numRegs; i++) F_here_array[i] = __hmax2(__hadd2(F_here_array[i],gap_extend2), __hadd2(penalty_here_array[i],gap_open2));
    };


    auto calc32_local_affine_float = [&](){
        sbt_row = shared_BLOSUM62[query_letter];

        score2 = sbt_row[subject[0]];
        //score2.y = sbt_row[subject1[0].x];
        penalty_temp0 = penalty_here_array[0];
        penalty_here_array[0] = __hmax2(__hmax2(__hadd2(penalty_diag,score2), __hmax2(E, F_here_array[0])),ZERO);
        //maximum = __hmax2(penalty_here_array[0],maximum);
        penalty_temp1 = __hadd2(penalty_here_array[0],gap_open2);
        E = __hmax2(__hadd2(E,gap_extend2), penalty_temp1);
        F_here_array[0] = __hmax2(__hadd2(F_here_array[0],gap_extend2), penalty_temp1);

        score2 = sbt_row[subject[1]];
        //score2.y = sbt_row[subject1[0].y];
        penalty_temp1 = penalty_here_array[1];
        penalty_here_array[1] = __hmax2(__hmax2(__hadd2(penalty_temp0,score2), __hmax2(E, F_here_array[1])),ZERO);
        //maximum = __hmax2(penalty_here_array[1],maximum);
        penalty_temp0 = __hadd2(penalty_here_array[1],gap_open2);
        E = __hmax2(__hadd2(E,gap_extend2), penalty_temp0);
        F_here_array[1] = __hmax2(__hadd2(F_here_array[1],gap_extend2), penalty_temp0);
		maximum = __hmax2(maximum, __hmax2(penalty_here_array[1],penalty_here_array[0]));

        #pragma unroll
        for (int i=1; i<numRegs/2; i++) {
            score2 = sbt_row[subject[2*i]];
            penalty_temp0 = penalty_here_array[2*i];
            penalty_here_array[2*i] = __hmax2(__hmax2(__hadd2(penalty_temp1,score2), __hmax2(E, F_here_array[2*i])),ZERO);
            //maximum = __hmax2(penalty_here_array[2*i],maximum);
            penalty_temp1 = __hadd2(penalty_here_array[2*i],gap_open2);
            E = __hmax2(__hadd2(E,gap_extend2), penalty_temp1);
            F_here_array[2*i] = __hmax2(__hadd2(F_here_array[2*i],gap_extend2), penalty_temp1);

            score2 = sbt_row[subject[2*i+1]];
            penalty_temp1 = penalty_here_array[2*i+1];
            penalty_here_array[2*i+1] = __hmax2(__hmax2(__hadd2(penalty_temp0,score2), __hmax2(E, F_here_array[2*i+1])),ZERO);
            //maximum = __hmax2(penalty_here_array[2*i+1],maximum);
            penalty_temp0 = __hadd2(penalty_here_array[2*i+1],gap_open2);
            E = __hmax2(__hadd2(E,gap_extend2), penalty_temp0);
            F_here_array[2*i+1] = __hmax2(__hadd2(F_here_array[2*i+1],gap_extend2), penalty_temp0);
			maximum = __hmax2(maximum,__hmax2(penalty_here_array[2*i+1],penalty_here_array[2*i]));
        }

        //for (int i=0; i<numRegs/4; i++)
		 //   maximum = __hmax2(maximum,__hmax2(__hmax2(penalty_here_array[4*i],penalty_here_array[4*i+1]),__hmax2(penalty_here_array[4*i+2],penalty_here_array[4*i+3])));
        penalty_here31 = penalty_here_array[numRegs-1];
    };

    //const int passes_S0 = ceil((1.0*length_S0)/(group_size*numRegs));
    //const int passes_S1 = ceil((1.0*length_S1)/(group_size*numRegs));

    const int passes = ceil((1.0*length)/(group_size*numRegs));
    //if (passes < 3) {
    //    if (thid==0) printf("Passes: %d, Length: %d \n", passes, length);
    //    return;
    //}

    int offset_out = group_id;
    int offset_in = group_id;

    auto shuffle_query = [&](const auto& new_letter) {
        query_letter = __shfl_up_sync(0xFFFFFFFF, query_letter, 1, 32);
        if (!group_id) query_letter = new_letter;
    };

    auto shuffle_affine_penalty = [&](const auto& new_penalty_left, const auto& new_E_left) {
        penalty_diag = penalty_left;
        penalty_left = __shfl_up_sync(0xFFFFFFFF, penalty_here31, 1, 32);
        E = __shfl_up_sync(0xFFFFFFFF, E, 1, 32);
        if (!group_id) {
            penalty_left = new_penalty_left;
            E = new_E_left;
        }
    };

    uint32_t temp;

    auto shuffle_H_E_temp_out = [&]() {
        temp = __shfl_down_sync(0xFFFFFFFF, *((int*)(&H_temp_out)), 1, 32);
        H_temp_out = *((half2*)(&temp));
        temp = __shfl_down_sync(0xFFFFFFFF, *((int*)(&E_temp_out)), 1, 32);
        E_temp_out = *((half2*)(&temp));
    };

    auto shuffle_H_E_temp_in = [&]() {
        temp = __shfl_down_sync(0xFFFFFFFF, *((int*)(&H_temp_in)), 1, 32);
        H_temp_in = *((half2*)(&temp));
        temp = __shfl_down_sync(0xFFFFFFFF, *((int*)(&E_temp_in)), 1, 32);
        E_temp_in = *((half2*)(&temp));

    };

    auto shuffle_new_query = [&]() {
        temp = __shfl_down_sync(0xFFFFFFFF, *((int*)(&new_query_letter4)), 1, 32);
        new_query_letter4 = *((char4*)(&temp));
    };

    auto set_H_E_temp_out = [&]() {
        if (thid % group_size == 31) {
            H_temp_out = penalty_here31;
            E_temp_out = E;
        }
    };

    int k;

    const uint32_t thread_result = ((length-1)%(group_size*numRegs))/numRegs;              // 0, 1, ..., or 31

    // FIRST PASS (of many passes)
    // Note first pass has always full seqeunce length
    init_penalties_local(0);
    init_local_score_profile_BLOSUM62(0);
    initial_calc32_local_affine_float(0);
    shuffle_query(new_query_letter4.y);
    shuffle_affine_penalty(ZERO, NEGINFINITY2);

    //shuffle_max();
    calc32_local_affine_float();
    shuffle_query(new_query_letter4.z);
    shuffle_affine_penalty(ZERO, NEGINFINITY2);

    //shuffle_max();
    calc32_local_affine_float();
    shuffle_query(new_query_letter4.w);
    shuffle_affine_penalty(ZERO, NEGINFINITY2);
    shuffle_new_query();
    counter++;

    for (k = 4; k <= length_2+28; k+=4) {
        //shuffle_max();
        calc32_local_affine_float();
        set_H_E_temp_out();
        shuffle_H_E_temp_out();

        shuffle_query(new_query_letter4.x);
        shuffle_affine_penalty(ZERO,NEGINFINITY2);

        //shuffle_max();
        calc32_local_affine_float();
        set_H_E_temp_out();
        shuffle_H_E_temp_out();

        shuffle_query(new_query_letter4.y);
        shuffle_affine_penalty(ZERO,NEGINFINITY2);

        //shuffle_max();
        calc32_local_affine_float();
        set_H_E_temp_out();
        shuffle_H_E_temp_out();

        shuffle_query(new_query_letter4.z);
        shuffle_affine_penalty(ZERO,NEGINFINITY2);

        //shuffle_max();
        calc32_local_affine_float();

        set_H_E_temp_out();
        if (counter%8 == 0 && counter > 8) {
            checkHEindex(offset_out);
            devTempHcol[offset_out]=H_temp_out;
            devTempEcol[offset_out]=E_temp_out;
            offset_out += group_size;
        }
        if (k != length_2+28) shuffle_H_E_temp_out();
        shuffle_query(new_query_letter4.w);
        shuffle_affine_penalty(ZERO,NEGINFINITY2);
        shuffle_new_query();
        if (counter%group_size == 0) {
            new_query_letter4 = constantQuery4[offset];
            offset += group_size;
        }
        counter++;
    }
    //if (length_2 % 4 == 0) {
        //temp = __shfl_up_sync(0xFFFFFFFF, *((int*)(&H_temp_out)), 1, 32);
        //H_temp_out = *((half2*)(&temp));
        //temp = __shfl_up_sync(0xFFFFFFFF, *((int*)(&E_temp_out)), 1, 32);
        //E_temp_out = *((half2*)(&temp));
    //}
    if (length_2%4 == 1) {
        //shuffle_max();
        calc32_local_affine_float();
        set_H_E_temp_out();
    }

    if (length_2%4 == 2) {
        //shuffle_max();
        calc32_local_affine_float();
        set_H_E_temp_out();
        shuffle_H_E_temp_out();
        shuffle_query(new_query_letter4.x);
        shuffle_affine_penalty(ZERO,NEGINFINITY2);
        //shuffle_max();
        calc32_local_affine_float();
        set_H_E_temp_out();
    }
    if (length_2%4 == 3) {
        //shuffle_max();
        calc32_local_affine_float();
        set_H_E_temp_out();
        shuffle_H_E_temp_out();
        shuffle_query(new_query_letter4.x);
        shuffle_affine_penalty(ZERO,NEGINFINITY2);
        //shuffle_max();
        calc32_local_affine_float();
        set_H_E_temp_out();
        shuffle_H_E_temp_out();
        shuffle_query(new_query_letter4.y);
        shuffle_affine_penalty(ZERO,NEGINFINITY2);
        //shuffle_max();
        calc32_local_affine_float();
        set_H_E_temp_out();
    }
    int final_out = length_2 % 32;
    int from_thread_id = 32 - final_out;

    if (thid>=from_thread_id) {
        checkHEindex(offset_out-from_thread_id);
        devTempHcol[offset_out-from_thread_id]=H_temp_out;
        devTempEcol[offset_out-from_thread_id]=E_temp_out;
    }

    //middle passes
    for (int pass = 1; pass < passes-1; pass++) {

        //only for checking
        //   for (int offset=group_size/2; offset>0; offset/=2)
            //	  maximum = __hmax2(maximum,__shfl_down_sync(0xFFFFFFFF,maximum,offset,group_size));
            //   if (!group_id)
            //	   if (check_blid){
        // 		   float2 tempf2 = __half22float2(maximum);
        // 		   printf("Maximum before MIDDLE PASS %d, in Block; %d, Thread: %d, Max_S0: %f, Max_S1: %f, Length_2: %d, thread_result: %d\n", pass, blid, thid, tempf2.y, tempf2.x, length_2, thread_result);
        // 	   }


        //maximum = __shfl_sync(0xFFFFFFFF, maximum, 31, 32);
        counter = 1;
        //counter2 = 1;
        query_letter = 20;
        new_query_letter4 = constantQuery4[thid%group_size];
        if (thid % group_size== 0) query_letter = new_query_letter4.x;

        offset = group_id + group_size;
        offset_out = group_id;
        offset_in = group_id;
        checkHEindex(offset_in);
        H_temp_in = devTempHcol[offset_in];
        E_temp_in = devTempEcol[offset_in];
        offset_in += group_size;

        init_penalties_local(0);
        init_local_score_profile_BLOSUM62(pass*(32*numRegs));

        if (!group_id) {
            penalty_left = H_temp_in;
            E = E_temp_in;
        }
        shuffle_H_E_temp_in();

        initial_calc32_local_affine_float(1);
        shuffle_query(new_query_letter4.y);
        shuffle_affine_penalty(H_temp_in,E_temp_in);
        shuffle_H_E_temp_in();

        //shuffle_max();
        calc32_local_affine_float();
        shuffle_query(new_query_letter4.z);
        shuffle_affine_penalty(H_temp_in,E_temp_in);
        shuffle_H_E_temp_in();

        //shuffle_max();
        calc32_local_affine_float();
        shuffle_query(new_query_letter4.w);
        shuffle_affine_penalty(H_temp_in,E_temp_in);
        shuffle_H_E_temp_in();
        shuffle_new_query();
        counter++;

        for (k = 4; k <= length_2+28; k+=4) {
            //shuffle_max();
            calc32_local_affine_float();
            set_H_E_temp_out();
            shuffle_H_E_temp_out();

            shuffle_query(new_query_letter4.x);
            shuffle_affine_penalty(H_temp_in,E_temp_in);
            shuffle_H_E_temp_in();

            //shuffle_max();
            calc32_local_affine_float();
            set_H_E_temp_out();
            shuffle_H_E_temp_out();

            shuffle_query(new_query_letter4.y);
            shuffle_affine_penalty(H_temp_in,E_temp_in);
            shuffle_H_E_temp_in();

            //shuffle_max();
            calc32_local_affine_float();
            set_H_E_temp_out();
            shuffle_H_E_temp_out();

            shuffle_query(new_query_letter4.z);
            shuffle_affine_penalty(H_temp_in,E_temp_in);
            shuffle_H_E_temp_in();

            //shuffle_max();
            calc32_local_affine_float();
            set_H_E_temp_out();
            if (counter%8 == 0 && counter > 8) {
                checkHEindex(offset_out);
                devTempHcol[offset_out]=H_temp_out;
                devTempEcol[offset_out]=E_temp_out;
                offset_out += group_size;
            }
            if (k != length_2+28) shuffle_H_E_temp_out();
            shuffle_query(new_query_letter4.w);
            shuffle_affine_penalty(H_temp_in,E_temp_in);
            shuffle_new_query();
            if (counter%group_size == 0) {
                new_query_letter4 = constantQuery4[offset];
                offset += group_size;
            }
            shuffle_H_E_temp_in();
            if (counter%8 == 0) {
                checkHEindex(offset_in);
                H_temp_in = devTempHcol[offset_in];
                E_temp_in = devTempEcol[offset_in];
                offset_in += group_size;
            }
            counter++;
        }
        //if (length_2 % 4 == 0) {
            //temp = __shfl_up_sync(0xFFFFFFFF, *((int*)(&H_temp_out)), 1, 32);
            //H_temp_out = *((half2*)(&temp));
            //temp = __shfl_up_sync(0xFFFFFFFF, *((int*)(&E_temp_out)), 1, 32);
            //E_temp_out = *((half2*)(&temp));
        //}
        if (length_2%4 == 1) {
            //shuffle_max();
            calc32_local_affine_float();
            set_H_E_temp_out();
        }

        if (length_2%4 == 2) {
            //shuffle_max();
            calc32_local_affine_float();
            set_H_E_temp_out();
            shuffle_H_E_temp_out();
            shuffle_query(new_query_letter4.x);
            shuffle_affine_penalty(H_temp_in,E_temp_in);
            shuffle_H_E_temp_in();
            //shuffle_max();
            calc32_local_affine_float();
            set_H_E_temp_out();
        }
        if (length_2%4 == 3) {
            //shuffle_max();
            calc32_local_affine_float();
            set_H_E_temp_out();
            shuffle_H_E_temp_out();
            shuffle_query(new_query_letter4.x);
            shuffle_affine_penalty(H_temp_in,E_temp_in);
            shuffle_H_E_temp_in();
            //shuffle_max();
            calc32_local_affine_float();
            set_H_E_temp_out();
            shuffle_H_E_temp_out();
            shuffle_query(new_query_letter4.y);
            shuffle_affine_penalty(H_temp_in,E_temp_in);
            shuffle_H_E_temp_in();
            //shuffle_max();
            calc32_local_affine_float();
            set_H_E_temp_out();
        }
        int final_out = length_2 % 32;
        int from_thread_id = 32 - final_out;

        if (thid>=from_thread_id) {
            checkHEindex(offset_out-from_thread_id);
            devTempHcol[offset_out-from_thread_id]=H_temp_out;
            devTempEcol[offset_out-from_thread_id]=E_temp_out;
        }

        //if (thid == 31) {
        //      max_results_S0[pass] = maximum.x;
        //      max_results_S1[pass] = maximum.y;
        // }
    }

    // if(blockIdx.x == 0){
    //     printf("tid %d, final offset_out %d\n", threadIdx.x, offset_out);
    // }

    //only for checking
    //for (int offset=group_size/2; offset>0; offset/=2)
        //  maximum = __hmax2(maximum,__shfl_down_sync(0xFFFFFFFF,maximum,offset,group_size));
        //if (!group_id)
        //   if (check_blid){
    //		   float2 tempf2 = __half22float2(maximum);
        //	   printf("Maximum before FINAL PASS in Block; %d, Thread: %d, Max_S0: %f, Max_S1: %f, Length_2: %d, thread_result: %d\n", blid, thid, tempf2.y, tempf2.x, length_2, thread_result);
        //   }


    // Final pass
    //maximum = __shfl_sync(0xFFFFFFFF, maximum, 31, 32);
    counter = 1;
    //counter2 = 1;
    query_letter = 20;
    new_query_letter4 = constantQuery4[thid%group_size];
    if (thid % group_size== 0) query_letter = new_query_letter4.x;

    offset = group_id + group_size;
    offset_in = group_id;
    checkHEindex(offset_in);
    H_temp_in = devTempHcol[offset_in];
    E_temp_in = devTempEcol[offset_in];
    offset_in += group_size;

    init_penalties_local(0);
    init_local_score_profile_BLOSUM62((passes-1)*(32*numRegs));
    //copy_H_E_temp_in();
    if (!group_id) {
        penalty_left = H_temp_in;
        E = E_temp_in;
    }
    shuffle_H_E_temp_in();

    initial_calc32_local_affine_float(1);
    shuffle_query(new_query_letter4.y);
    shuffle_affine_penalty(H_temp_in,E_temp_in);
    shuffle_H_E_temp_in();
    if (length_2+thread_result >=2) {
        //shuffle_max();
        calc32_local_affine_float();
        shuffle_query(new_query_letter4.z);
        //copy_H_E_temp_in();
        shuffle_affine_penalty(H_temp_in,E_temp_in);
        shuffle_H_E_temp_in();
    }

    if (length_2+thread_result >=3) {
        //shuffle_max();
        calc32_local_affine_float();
        shuffle_query(new_query_letter4.w);
        //copy_H_E_temp_in();
        shuffle_affine_penalty(H_temp_in,E_temp_in);
        shuffle_H_E_temp_in();
        shuffle_new_query();
        counter++;
    }
    if (length_2+thread_result >=4) {
        //for (k = 5; k < lane_2+thread_result-2; k+=4) {
        for (k = 4; k <= length_2+(thread_result-3); k+=4) {
            //shuffle_max();
            calc32_local_affine_float();

            shuffle_query(new_query_letter4.x);
            //copy_H_E_temp_in();
            shuffle_affine_penalty(H_temp_in,E_temp_in);
            shuffle_H_E_temp_in();

            //shuffle_max();
            calc32_local_affine_float();
            shuffle_query(new_query_letter4.y);
            //copy_H_E_temp_in();
            shuffle_affine_penalty(H_temp_in,E_temp_in);
            shuffle_H_E_temp_in();

            //shuffle_max();
            calc32_local_affine_float();
            shuffle_query(new_query_letter4.z);
            //copy_H_E_temp_in();
            shuffle_affine_penalty(H_temp_in,E_temp_in);
            shuffle_H_E_temp_in();

            //shuffle_max();
            calc32_local_affine_float();
            shuffle_query(new_query_letter4.w);
            //copy_H_E_temp_in();
            shuffle_affine_penalty(H_temp_in,E_temp_in);
            shuffle_new_query();
            if (counter%group_size == 0) {
                new_query_letter4 = constantQuery4[offset];
                offset += group_size;
            }
            shuffle_H_E_temp_in();
            if (counter%8 == 0) {
                checkHEindex(offset_in);
                H_temp_in = devTempHcol[offset_in];
                E_temp_in = devTempEcol[offset_in];
                offset_in += group_size;
            }
            counter++;
        }
        //    if (blid == 0) {
        //        if (thid % group_size == thread_result)
        //            printf("Result in Thread: %d, Register: %d, Value: %f, #passes: %d\n", thid, (length-1)%numRegs, penalty_here_array[(length-1)%numRegs], passes);
        //    }
        //      if ((length_2+thread_result+1)%4 > 0) {
            //if (counter2-(length_2+thread_result) > 0) {

        if ((k-1)-(length_2+thread_result) > 0) {
            //shuffle_max();
            calc32_local_affine_float();
            shuffle_query(new_query_letter4.x);
            shuffle_affine_penalty(H_temp_in,E_temp_in);
            shuffle_H_E_temp_in();
            k++;
        }


        if ((k-1)-(length_2+thread_result) > 0) {
            //shuffle_max();
            calc32_local_affine_float();
            shuffle_query(new_query_letter4.y);
            shuffle_affine_penalty(H_temp_in,E_temp_in);
            shuffle_H_E_temp_in();
            k++;
        }

        if ((k-1)-(length_2+thread_result) > 0) {
            //shuffle_max();
            calc32_local_affine_float();
        }

    //    if (blid == 0) {
    //        if (thid % group_size == thread_result)
    //            printf("Result in Thread: %d, Register: %d, Value: %f, #passes: %d\n", thid, (length-1)%numRegs, penalty_here_array[(length-1)%numRegs], passes);
    //    }
    }

    for (int offset=group_size/2; offset>0; offset/=2)
        maximum = __hmax2(maximum,__shfl_down_sync(0xFFFFFFFF,maximum,offset,group_size));

//	if (!group_id)
//		if (check_blid){
	//		float2 tempf2 = __half22float2(maximum);
	//		printf("Result in Block; %d, Thread: %d, Max_S0: %f, Max_S1: %f, Length_2: %d, thread_result: %d\n", blid, thid, tempf2.y, tempf2.x, length_2, thread_result);
//		}


  // if (thid % group_size == thread_result)
    //  printf("Result in Block: %d, in Thread: %d, Register: %d, Value: %f, #passes: %d\n", blid, thid, (length-1)%numRegs, penalty_here_array[(length-1)%numRegs], passes);
    if (!group_id) {
        if (blid < gridDim.x-1) {
            devAlignmentScores[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*(thid/group_size)]] =  maximum.y; // lane_2+thread_result+1-length_2%4; penalty_here_array[(length-1)%numRegs];
            devAlignmentScores[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*(thid/group_size)+1]] =  maximum.x; // lane_2+thread_result+1-length_2%4; penalty_here_array[(length-1)%numRegs];
        } else {
            devAlignmentScores[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)]] =  maximum.y; // lane_2+thread_result+1-length_2%4; penalty_here_array[(length-1)%numRegs];
            if (!check_last2 || (thid%check_last) < check_last-group_size) devAlignmentScores[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)+1]] =  maximum.x; // lane_2+thread_result+1-length_2%4; penalty_here_array[(length-1)%numRegs];
        }

        // check for overflow
        if (overflow_check){
            half max_half2 = __float2half_rn(MAX_ACC_HALF2);
            if (maximum.y >= max_half2) {
                int pos_overflow = atomicAdd(d_overflow_number,1);
                int pos = d_overflow_positions[pos_overflow] = d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)];
                //printf("Overflow_S0 %d, SeqID: %d, Length: %d, score: %f\n", pos_overflow, pos, length_S0, __half2float(maximum.y));
            }
            if (maximum.x >= max_half2) {
                int pos_overflow = atomicAdd(d_overflow_number,1);
                int pos = d_overflow_positions[pos_overflow] = d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)+1];
                //printf("Overflow_S1 %d, SeqID: %d, Length: %d, score: %f\n", pos_overflow, pos, length_S1, __half2float(maximum.x));
            }
        }
    }
}


// Needleman-Wunsch (NW): global alignment with linear gap penalty
// numRegs values per thread
// uses a single warp per CUDA thread block;
// every groupsize threads computes an alignmen score
template <int group_size, int numRegs, class PositionsIterator> 
__launch_bounds__(512)
__global__
void NW_local_affine_Protein_single_pass_half2(
    const char * devChars,
    float * devAlignmentScores,
    const size_t* devOffsets,
    const size_t* devLengths,
    PositionsIterator d_positions_of_selected_lengths,
    const int numSelected,
	size_t* d_overflow_positions,
	int *d_overflow_number,
	const int overflow_check,
    const int length_2,
    const float gap_open,
    const float gap_extend
) {

    __shared__ __half2 shared_BLOSUM62[21][21*21];
    int subject[numRegs];

    const float NEGINFINITY = -1000.0;
    const __half2 NEGINFINITY2 = __float2half2_rn(-1000.0);
    const __half2 gap_open2 = __float2half2_rn(gap_open);
    const __half2 gap_extend2 = __float2half2_rn(gap_extend);

    const int blid = blockIdx.x;
    const int thid = threadIdx.x;
    const int group_id = thid%group_size;
	int offset = group_id + group_size;

    int check_last = blockDim.x/group_size;
    int check_last2 = 0;
    if (blid == gridDim.x-1) {
        if (numSelected % (2*blockDim.x/group_size)) {
            check_last = (numSelected/2) % (blockDim.x/group_size);
            check_last2 = numSelected%2;
            check_last = check_last + check_last2;
        }
    }
    check_last = check_last * group_size;

    //if ((blid == 2) && (!thid)) printf("Values in Block: %d, in Thread: %d, numSelected: %d, check_last: %d, check_last2 = %d\n", blid, thid, numSelected, check_last, check_last2);

    const size_t alignmentId_checklast_0 = d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)];
    const size_t alignmentId_checklast_1 = d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)+1];
    const size_t alignmentId_0 = d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*(thid/group_size)];
    const size_t alignmentId_1 = d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*(thid/group_size)+1];

    //if we can assume that d_positions_of_selected_lengths is the sequence 0...N, we do not need it at all
    // const size_t alignmentId_checklast_0 = 2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size);
    // const size_t alignmentId_checklast_1 = 2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)+1;
    // const size_t alignmentId_0 = 2*(blockDim.x/group_size)*blid+2*(thid/group_size);
    // const size_t alignmentId_1 = 2*(blockDim.x/group_size)*blid+2*(thid/group_size)+1;

 

    const int length_S0 = devLengths[alignmentId_checklast_0];
    const int base_S0 = devOffsets[alignmentId_checklast_0]-devOffsets[0];

    //if ((blid == 2) && (!thid)) printf("Values in Block: %d, in Thread: %d, length_S0: %d, base_S0: %d\n", blid, thid, length_S0, base_S0);

	int length_S1 = length_S0;
	int base_S1 = base_S0;
	if ((blid < gridDim.x-1) || (!check_last2) || ((thid%check_last) < check_last-group_size) || ((thid%check_last) >= check_last)) {
		length_S1 = devLengths[alignmentId_checklast_1];
	    base_S1 = devOffsets[alignmentId_checklast_1]-devOffsets[0];
	}

	//if (blid == gridDim.x-1)
    //    if (check_last2)
    //        if (((thid%check_last) >= check_last-group_size) && ((thid%check_last) < check_last)) {
    //            length_S1 = length_S0;
    //            base_S1 = base_S0;
    //        }

	//if ((blid == 2) && (!thid)) printf("Values in Block: %d, in Thread: %d, length_S1: %d, base_S1: %d\n", blid, thid, length_S1, base_S1);


    //if ((blid == gridDim.x-1) && (!thid)) printf("Values in Block: %d, in Thread: %d, numSelected: %d, check_last: %d, check_last2: %d\n", blid, thid, numSelected, check_last, check_last2);

    //if ((length_S0 < min_length) || (length_S0 > max_length) || (length_S1 < min_length) || (length_S1 > max_length)) return;

    int temp_length = max(length_S0, length_S1);
    const int length = warp_max_reduce_broadcast(0xFFFFFFFF, temp_length);

    __half2 penalty_temp0, penalty_temp1; // penalty_temp2;
    __half2 penalty_left, penalty_diag;
    __half2 penalty_here31;
    __half2 penalty_here_array[numRegs];
    __half2 F_here_array[numRegs];
    __half2 E = NEGINFINITY2;
    //const int base_3 = blid*length_2; // too large???
    __half2 maximum = __float2half2_rn(0.0);
    const __half2 ZERO = __float2half2_rn(0.0);;

    auto init_penalties_local = [&](const auto& value) {
        //ZERO = NEGINFINITY2;
        penalty_left = NEGINFINITY2;
        penalty_diag = NEGINFINITY2;
        E = NEGINFINITY2;
        #pragma unroll
        for (int i=0; i<numRegs; i++) penalty_here_array[i] = NEGINFINITY2;
        for (int i=0; i<numRegs; i++) F_here_array[i] = NEGINFINITY2;
        if (thid % group_size == 0) {
            penalty_left = __floats2half2_rn(0,0);
            penalty_diag = __floats2half2_rn(0,0);
            #pragma unroll
            for (int i=0; i<numRegs; i++) penalty_here_array[i] = __floats2half2_rn(0,0);
        }
        if (thid % group_size== 1) {
            penalty_left = __floats2half2_rn(0,0);
        }
    };

    __half2 temp0; // temp1, E_temp_float, H_temp_float;

    auto init_local_score_profile_BLOSUM62 = [&](const auto& offset_isc) {
        for (int i=thid; i<21*21; i+=blockDim.x) {
            temp0.x = cBLOSUM62_dev[21*(i/21)+(i%21)];
            for (int j=0; j<21; j++) {
                temp0.y = cBLOSUM62_dev[21*(i/21)+j];
                shared_BLOSUM62[i/21][21*(i%21)+j]=temp0;
            }
        }
        __syncthreads();

       for (int i=0; i<numRegs; i++) {

           if (offset_isc+numRegs*(thid%group_size)+i >= length_S0) subject[i] = 1; // 20;
           else subject[i] = devChars[offset_isc+base_S0+numRegs*(thid%group_size)+i];

           if (offset_isc+numRegs*(thid%group_size)+i >= length_S1) subject[i] += 1*21; // 20*21;
           else subject[i] += 21*devChars[offset_isc+base_S1+numRegs*(thid%group_size)+i];
       }

    };

    int counter = 1;
    //int counter2 = 1;
    char query_letter = 20;
    char4 new_query_letter4 = constantQuery4[thid%group_size];
    if (thid % group_size== 0) query_letter = new_query_letter4.x;

    __half2 score2;
    __half2 *sbt_row;

    auto initial_calc32_local_affine_float = [&](const auto& value){
        sbt_row = shared_BLOSUM62[query_letter];

        score2 = sbt_row[subject[0]];
        //score2.y = sbt_row[subject1[0].x];
        penalty_temp0 = penalty_here_array[0];
        if (!value || (thid%group_size)) E = __hmax2(__hadd2(E,gap_extend2), __hadd2(penalty_left,gap_open2));
        F_here_array[0] = __hmax2(__hadd2(F_here_array[0],gap_extend2), __hadd2(penalty_here_array[0],gap_open2));
        penalty_here_array[0] = temp0 = __hmax2(__hmax2(__hadd2(penalty_diag,score2), __hmax2(E, F_here_array[0])),ZERO);
        maximum = __hmax2(temp0,maximum);

        score2 = sbt_row[subject[1]];
        //score2.y = sbt_row[subject1[0].y];
        penalty_temp1 = penalty_here_array[1];
        E = __hmax2(__hadd2(E,gap_extend2), __hadd2(penalty_here_array[0],gap_open2));
        F_here_array[1] = __hmax2(__hadd2(F_here_array[1],gap_extend2), __hadd2(penalty_here_array[1],gap_open2));
        temp0 = penalty_here_array[1] = __hmax2(__hmax2(__hadd2(penalty_temp0,score2), __hmax2(E, F_here_array[1])),ZERO);
        maximum = __hmax2(temp0,maximum);

        #pragma unroll
        for (int i=1; i<numRegs/2; i++) {
            score2 = sbt_row[subject[2*i]];
            //score2.y = sbt_row[subject1[i].x];
            penalty_temp0 = penalty_here_array[2*i];
            E = __hmax2(__hadd2(E,gap_extend2), __hadd2(penalty_here_array[2*i-1],gap_open2));
            F_here_array[2*i] = __hmax2(__hadd2(F_here_array[2*i],gap_extend2), __hadd2(penalty_here_array[2*i],gap_open2));
            temp0 = penalty_here_array[2*i] = __hmax2(__hmax2(__hadd2(penalty_temp1,score2), __hmax2(E, F_here_array[2*i])),ZERO);
            maximum = __hmax2(temp0,maximum);

            score2 = sbt_row[subject[2*i+1]];
            //score2.y = sbt_row[subject1[i].y];
            penalty_temp1 = penalty_here_array[2*i+1];
            E = __hmax2(__hadd2(E,gap_extend2), __hadd2(penalty_here_array[2*i],gap_open2));
            F_here_array[2*i+1] = __hmax2(__hadd2(F_here_array[2*i+1],gap_extend2), __hadd2(penalty_here_array[2*i+1],gap_open2));
            temp0 = penalty_here_array[2*i+1] = __hmax2(__hmax2(__hadd2(penalty_temp0,score2), __hmax2(E, F_here_array[2*i+1])),ZERO);
            maximum = __hmax2(temp0,maximum);
        }

		penalty_here31 = penalty_here_array[numRegs-1];
        E = __hmax2(E+gap_extend2, penalty_here31+gap_open2);
        for (int i=0; i<numRegs; i++) F_here_array[i] = __hmax2(__hadd2(F_here_array[i],gap_extend2), __hadd2(penalty_here_array[i],gap_open2));
    };

    auto calc32_local_affine_float = [&](){
        sbt_row = shared_BLOSUM62[query_letter];

        score2 = sbt_row[subject[0]];
        //score2.y = sbt_row[subject1[0].x];
        penalty_temp0 = penalty_here_array[0];
        penalty_here_array[0] = __hmax2(__hmax2(__hadd2(penalty_diag,score2), __hmax2(E, F_here_array[0])),ZERO);
        //maximum = __hmax2(penalty_here_array[0],maximum);
        penalty_temp1 = __hadd2(penalty_here_array[0],gap_open2);
        E = __hmax2(__hadd2(E,gap_extend2), penalty_temp1);
        F_here_array[0] = __hmax2(__hadd2(F_here_array[0],gap_extend2), penalty_temp1);

        score2 = sbt_row[subject[1]];
        //score2.y = sbt_row[subject1[0].y];
        penalty_temp1 = penalty_here_array[1];
        penalty_here_array[1] = __hmax2(__hmax2(__hadd2(penalty_temp0,score2), __hmax2(E, F_here_array[1])),ZERO);
        //maximum = __hmax2(penalty_here_array[1],maximum);
        penalty_temp0 = __hadd2(penalty_here_array[1],gap_open2);
        E = __hmax2(__hadd2(E,gap_extend2), penalty_temp0);
        F_here_array[1] = __hmax2(__hadd2(F_here_array[1],gap_extend2), penalty_temp0);
		maximum = __hmax2(maximum, __hmax2(penalty_here_array[1],penalty_here_array[0]));

        #pragma unroll
        for (int i=1; i<numRegs/2; i++) {
            score2 = sbt_row[subject[2*i]];
            penalty_temp0 = penalty_here_array[2*i];
            penalty_here_array[2*i] = __hmax2(__hmax2(__hadd2(penalty_temp1,score2), __hmax2(E, F_here_array[2*i])),ZERO);
            //maximum = __hmax2(penalty_here_array[2*i],maximum);
            penalty_temp1 = __hadd2(penalty_here_array[2*i],gap_open2);
            E = __hmax2(__hadd2(E,gap_extend2), penalty_temp1);
            F_here_array[2*i] = __hmax2(__hadd2(F_here_array[2*i],gap_extend2), penalty_temp1);

            score2 = sbt_row[subject[2*i+1]];
            penalty_temp1 = penalty_here_array[2*i+1];
            penalty_here_array[2*i+1] = __hmax2(__hmax2(__hadd2(penalty_temp0,score2), __hmax2(E, F_here_array[2*i+1])),ZERO);
            //maximum = __hmax2(penalty_here_array[2*i+1],maximum);
            penalty_temp0 = __hadd2(penalty_here_array[2*i+1],gap_open2);
            E = __hmax2(__hadd2(E,gap_extend2), penalty_temp0);
            F_here_array[2*i+1] = __hmax2(__hadd2(F_here_array[2*i+1],gap_extend2), penalty_temp0);
			maximum = __hmax2(maximum,__hmax2(penalty_here_array[2*i+1],penalty_here_array[2*i]));
        }

        //for (int i=0; i<numRegs/4; i++)
		 //   maximum = __hmax2(maximum,__hmax2(__hmax2(penalty_here_array[4*i],penalty_here_array[4*i+1]),__hmax2(penalty_here_array[4*i+2],penalty_here_array[4*i+3])));
        penalty_here31 = penalty_here_array[numRegs-1];
    };

    const int passes = ceil((1.0*length)/(group_size*numRegs));

    auto shuffle_query = [&](const auto& new_letter) {
        query_letter = __shfl_up_sync(0xFFFFFFFF, query_letter, 1, 32);
        if (!group_id) query_letter = new_letter;
    };

    //auto shuffle_max = [&]() {
    //    temp0 = __shfl_up_sync(0xFFFFFFFF, maximum, 1, 32);
    //    if (group_id) {
    //        maximum = __hmax2(temp0,maximum);
    //    }
    //};

    auto shuffle_affine_penalty = [&](const auto& new_penalty_left, const auto& new_E_left) {
        penalty_diag = penalty_left;
        penalty_left = __shfl_up_sync(0xFFFFFFFF, penalty_here31, 1, 32);
        E = __shfl_up_sync(0xFFFFFFFF, E, 1, 32);
        if (!group_id) {
            penalty_left = new_penalty_left;
            E = new_E_left;
        }
    };

    uint32_t temp;

    auto shuffle_new_query = [&]() {
        temp = __shfl_down_sync(0xFFFFFFFF, *((int*)(&new_query_letter4)), 1, 32);
        new_query_letter4 = *((char4*)(&temp));
    };

    int k;

    const uint32_t thread_result = ((length-1)%(group_size*numRegs))/numRegs;              // 0, 1, ..., or 31

    if (passes == 1) {
        // Single pass
    //    for (int k = 5; k < lane_2+group_size-length_2%4; k+=4) {
    //    for (int k = 5; k < lane_2+thread_result+1-thread_result%4; k+=4) {
    init_penalties_local(0);
    init_local_score_profile_BLOSUM62(0);

    initial_calc32_local_affine_float(0);
    shuffle_query(new_query_letter4.y);
    shuffle_affine_penalty(ZERO, NEGINFINITY2);

    if (length_2+thread_result >=2) {
        //shuffle_max();
        calc32_local_affine_float();
        shuffle_query(new_query_letter4.z);
        shuffle_affine_penalty(ZERO, NEGINFINITY2);
    }

    if (length_2+thread_result >=3) {
        //shuffle_max();
        calc32_local_affine_float();
        shuffle_query(new_query_letter4.w);
        shuffle_affine_penalty(ZERO, NEGINFINITY2);
        shuffle_new_query();
        counter++;

    }
    if (length_2+thread_result >=4) {
    //for (k = 5; k < lane_2+thread_result-2; k+=4) {
    for (k = 4; k <= length_2+(thread_result-3); k+=4) {
            //shuffle_max();
            calc32_local_affine_float();

            shuffle_query(new_query_letter4.x);
            shuffle_affine_penalty(ZERO,NEGINFINITY2);

            //shuffle_max();
            calc32_local_affine_float();
            shuffle_query(new_query_letter4.y);
            shuffle_affine_penalty(ZERO,NEGINFINITY2);

            //shuffle_max();
            calc32_local_affine_float();
            shuffle_query(new_query_letter4.z);
            shuffle_affine_penalty(ZERO,NEGINFINITY2);

            //shuffle_max();
            calc32_local_affine_float();
            shuffle_query(new_query_letter4.w);
            shuffle_affine_penalty(ZERO,NEGINFINITY2);
            shuffle_new_query();
            if (counter%group_size == 0) {
                new_query_letter4 = constantQuery4[offset];
                offset += group_size;
            }
            counter++;
        }

        if ((k-1)-(length_2+thread_result) > 0) {
            //shuffle_max();
            calc32_local_affine_float();
            shuffle_query(new_query_letter4.x);
            shuffle_affine_penalty(ZERO,NEGINFINITY2);
            k++;
        }

        if ((k-1)-(length_2+thread_result) > 0) {
            //shuffle_max();
            calc32_local_affine_float();
            shuffle_query(new_query_letter4.y);
            shuffle_affine_penalty(ZERO,NEGINFINITY2);
            k++;
        }

        if ((k-1)-(length_2+thread_result) > 0) {
            //shuffle_max();
            calc32_local_affine_float();
        }

    //    if (blid == 0) {
    //        if (thid % group_size == thread_result)
    //            printf("Result in Thread: %d, Register: %d, Value: %f, #passes: %d\n", thid, (length-1)%numRegs, penalty_here_array[(length-1)%numRegs], passes);
    //    }
    }
    }
    for (int offset=group_size/2; offset>0; offset/=2)
	    maximum = __hmax2(maximum,__shfl_down_sync(0xFFFFFFFF,maximum,offset,group_size));

  // if (thid % group_size == thread_result)
    //  printf("Result in Block: %d, in Thread: %d, Register: %d, Value: %f, #passes: %d\n", blid, thid, (length-1)%numRegs, penalty_here_array[(length-1)%numRegs], passes);
   if (!group_id) {

      if (blid < gridDim.x-1) {
          devAlignmentScores[alignmentId_0] =  maximum.y; // lane_2+thread_result+1-length_2%4; penalty_here_array[(length-1)%numRegs];
          devAlignmentScores[alignmentId_1] =  maximum.x; // lane_2+thread_result+1-length_2%4; penalty_here_array[(length-1)%numRegs];
      } else {
          devAlignmentScores[alignmentId_checklast_0] =  maximum.y; // lane_2+thread_result+1-length_2%4; penalty_here_array[(length-1)%numRegs];
          if (!check_last2 || (thid%check_last) < check_last-group_size) devAlignmentScores[alignmentId_checklast_1] =  maximum.x; // lane_2+thread_result+1-length_2%4; penalty_here_array[(length-1)%numRegs];
      }

      float2 temp_temp = __half22float2(maximum);
      //if ((blid == 2) && (!thid)) printf("Results in Block: %d, in Thread: %d, max.x: %f, max.y: %f\n", blid, thid, temp_temp.x, temp_temp.y);

	  // check for overflow
	  if (overflow_check) {
		  half max_half2 = __float2half_rn(MAX_ACC_HALF2);
		  if (maximum.y >= max_half2) {
			  int pos_overflow = atomicAdd(d_overflow_number,1);
			  int pos = d_overflow_positions[pos_overflow] = alignmentId_checklast_0;
		  }
		  if (maximum.x >= max_half2) {
			  int pos_overflow = atomicAdd(d_overflow_number,1);
			  int pos = d_overflow_positions[pos_overflow] = alignmentId_checklast_1;
		  }
	  }
  }
// devAlignmentScores[(32/group_size)*blid+thid/group_size] =  penalty_here_array[(length-1)%numRegs]; // lane_2+thread_result+1-length_2%4; penalty_here_array[(length-1)%numRegs];
    // if (thid % group_size == thread_result)
    //     printf("Result in Block: %d, in Thread: %d, Register: %d, Value: %f, #passes: %d\n", blid, thid, (length-1)%numRegs, penalty_here_array[(length-1)%numRegs], passes);

}


// Needleman-Wunsch (NW): global alignment with linear gap penalty
// numRegs values per thread
// uses a single warp per CUDA thread block;
// every groupsize threads computes an alignmen score
template <int group_size, int numRegs> __global__
void NW_local_affine_single_pass_s16_DPX(
    const char * devChars,
    float * devAlignmentScores,
    const size_t* devOffsets,
    const size_t* devLengths,
    const size_t* d_positions_of_selected_lengths,
    const int numSelected,
    const int length_2,
    const int32_t gap_open,
    const int32_t gap_extend
) {

    __shared__ score_type shared_BLOSUM62[21][21*21];
    int subject[numRegs];

    const score_type ZERO_in      = {0, 0};
    const score_type NEGINFINITY2 = {-1000, -1000};
    const score_type gap_open2    = {(short)gap_open, (short) gap_open};
    const score_type gap_extend2  = {(short)gap_extend, (short)gap_extend};

    const int blid = blockIdx.x;
    const int thid = threadIdx.x;
    const int group_id = thid%group_size;
	int offset = group_id + group_size;

    int check_last = blockDim.x/group_size;
    int check_last2 = 0;
    if (blid == gridDim.x-1) {
        if (numSelected % (2*blockDim.x/group_size)) {
            check_last = (numSelected/2) % (blockDim.x/group_size);
            check_last2 = numSelected%2;
            check_last = check_last + check_last2;
        }
    }
    check_last = check_last * group_size;

    const int length_S0 = devLengths[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)]];
    const int base_S0 = devOffsets[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)]];

    int length_S1 = devLengths[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)+1]];
    int base_S1 = devOffsets[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)+1]];

    if (blid == gridDim.x-1)
        if (check_last2)
            if (((thid%check_last) >= check_last-group_size) && ((thid%check_last) < check_last)) {
                length_S1 = length_S0;
                base_S1 = base_S0;
            }

    //if ((blid == gridDim.x-1) && (!thid)) printf("Values in Block: %d, in Thread: %d, numSelected: %d, check_last: %d, check_last2: %d\n", blid, thid, numSelected, check_last, check_last2);

    //if ((length_S0 < min_length) || (length_S0 > max_length) || (length_S1 < min_length) || (length_S1 > max_length)) return;

    int temp_length = max(length_S0, length_S1);
    const int length = warp_max_reduce_broadcast(0xFFFFFFFF, temp_length);

    score_type penalty_temp0, penalty_temp1; // penalty_temp2;
    score_type penalty_left, penalty_diag;
    score_type penalty_here31;
    score_type penalty_here_array[numRegs];
    score_type F_here_array[numRegs];
    score_type E = NEGINFINITY2;
    //const int base_3 = blid*length_2; // too large???
    score_type maximum = ZERO_in;
    //const __half2 ZERO = __float2half2_rn(0.0);;

    auto init_penalties_local = [&](const auto& value) {
        //ZERO = NEGINFINITY2;
        penalty_left = NEGINFINITY2;
        penalty_diag = NEGINFINITY2;
        E = NEGINFINITY2;
        #pragma unroll
        for (int i=0; i<numRegs; i++) penalty_here_array[i] = NEGINFINITY2;
        for (int i=0; i<numRegs; i++) F_here_array[i] = NEGINFINITY2;
        if (thid % group_size == 0) {
            penalty_left = ZERO_in;
            penalty_diag = ZERO_in;
            #pragma unroll
            for (int i=0; i<numRegs; i++) penalty_here_array[i] = ZERO_in;
        }
        if (thid % group_size== 1) {
            penalty_left = ZERO_in;
        }
    };

    score_type temp0; // temp1, E_temp_float, H_temp_float;

    auto init_local_score_profile_BLOSUM62 = [&](const auto& offset_isc) {
        for (int i=thid; i<21*21; i+=blockDim.x) {
            temp0.x = cBLOSUM62_dev[21*(i/21)+(i%21)];
            for (int j=0; j<21; j++) {
                temp0.y = cBLOSUM62_dev[21*(i/21)+j];
                shared_BLOSUM62[i/21][21*(i%21)+j]=temp0;
            }
        }
        __syncthreads();

       for (int i=0; i<numRegs; i++) {

           if (offset_isc+numRegs*(thid%group_size)+i >= length_S0) subject[i] = 20;
           else subject[i] = devChars[offset_isc+base_S0+numRegs*(thid%group_size)+i];

           if (offset_isc+numRegs*(thid%group_size)+i >= length_S1) subject[i] += 20*21;
           else subject[i] += 21*devChars[offset_isc+base_S1+numRegs*(thid%group_size)+i];
       }

    };

    int counter = 1;
    //int counter2 = 1;
    char query_letter = 20;
    char4 new_query_letter4 = constantQuery4[thid%group_size];
    if (thid % group_size== 0) query_letter = new_query_letter4.x;

    score_type score2;
    score_type *sbt_row;

    auto initial_calc32_local_affine_float = [&](const auto& value){
        sbt_row = shared_BLOSUM62[query_letter];

        score2 = sbt_row[subject[0]];
        penalty_temp0 = penalty_here_array[0];
        if (!value || (thid%group_size)) E = viaddmax(E, gap_extend2,viadd(penalty_left,gap_open2));
        F_here_array[0] = viaddmax(F_here_array[0],gap_extend2,viadd(penalty_here_array[0],gap_open2));
        penalty_here_array[0] = vimax3(viadd(penalty_diag,score2),E, F_here_array[0]);
        //maximum = vimax(temp0,maximum);

        score2 = sbt_row[subject[1]];
        penalty_temp1 = penalty_here_array[1];
        E = viaddmax(E,gap_extend2,viadd(penalty_here_array[0],gap_open2));
        F_here_array[1] = viaddmax(F_here_array[1],gap_extend2,viadd(penalty_here_array[1],gap_open2));
        penalty_here_array[1] = vimax3(viadd(penalty_temp0,score2),E, F_here_array[1]);
        //maximum = vimax(temp0,maximum);
		maximum = vimax3(maximum,penalty_here_array[0],penalty_here_array[1]);

        #pragma unroll
        for (int i=1; i<numRegs/2; i++) {
            score2 = sbt_row[subject[2*i]];
            penalty_temp0 = penalty_here_array[2*i];
            E = vimax(viadd(E,gap_extend2), viadd(penalty_here_array[2*i-1],gap_open2));
            F_here_array[2*i] = vimax(viadd(F_here_array[2*i],gap_extend2), viadd(penalty_here_array[2*i],gap_open2));
            penalty_here_array[2*i] = vimax3(viadd(penalty_temp1,score2), E, F_here_array[2*i]);
            //maximum = vimax(temp0,maximum);

            score2 = sbt_row[subject[2*i+1]];
            penalty_temp1 = penalty_here_array[2*i+1];
            E = vimax(viadd(E,gap_extend2), viadd(penalty_here_array[2*i],gap_open2));
            F_here_array[2*i+1] = vimax(viadd(F_here_array[2*i+1],gap_extend2), viadd(penalty_here_array[2*i+1],gap_open2));
            penalty_here_array[2*i+1] = vimax3(viadd(penalty_temp0,score2), E, F_here_array[2*i+1]);
            //maximum = vimax(temp0,maximum);
			maximum = vimax3(maximum,penalty_here_array[2*i],penalty_here_array[2*i+1]);

        }

        penalty_here31 = penalty_here_array[numRegs-1];
        E = vimax(viadd(E, gap_extend2), viadd(penalty_here31, gap_open2));
        for (int i=0; i<numRegs; i++) F_here_array[i] = vimax(viadd(F_here_array[i],gap_extend2), viadd(penalty_here_array[i],gap_open2));
    };

    auto calc32_local_affine_float = [&](){
        sbt_row = shared_BLOSUM62[query_letter];

        score2 = sbt_row[subject[0]];
        penalty_temp0 = penalty_here_array[0];
        penalty_here_array[0] = vimax3(viadd(penalty_diag,score2), E, F_here_array[0]);
        penalty_temp1 = viadd(penalty_here_array[0],gap_open2);
        E = viaddmax(E,gap_extend2, penalty_temp1);
        F_here_array[0] = viaddmax(F_here_array[0],gap_extend2, penalty_temp1);

        score2 = sbt_row[subject[1]];
        penalty_temp1 = penalty_here_array[1];
        penalty_here_array[1] = vimax3(viadd(penalty_temp0,score2), E, F_here_array[1]);
        penalty_temp0 = viadd(penalty_here_array[1],gap_open2);
        E = viaddmax(E,gap_extend2, penalty_temp0);
        F_here_array[1] = viaddmax(F_here_array[1],gap_extend2,penalty_temp0);
        maximum = vimax3(maximum,penalty_here_array[0],penalty_here_array[1]);

        #pragma unroll
        for (int i=1; i<numRegs/2; i++) {
            score2 = sbt_row[subject[2*i]];
            penalty_temp0 = penalty_here_array[2*i];
            penalty_here_array[2*i] = vimax3(viadd(penalty_temp1,score2), E, F_here_array[2*i]);
            penalty_temp1 = viadd(penalty_here_array[2*i],gap_open2);
            E = viaddmax(E,gap_extend2, penalty_temp1);
            F_here_array[2*i] = viaddmax(F_here_array[2*i],gap_extend2, penalty_temp1);

            score2 = sbt_row[subject[2*i+1]];
            penalty_temp1 = penalty_here_array[2*i+1];

            penalty_here_array[2*i+1] = vimax3(viadd(penalty_temp0,score2), E, F_here_array[2*i+1]);
            penalty_temp0 = viadd(penalty_here_array[2*i+1],gap_open2);
            E = viaddmax(E,gap_extend2, penalty_temp0);
            F_here_array[2*i+1] = viaddmax(F_here_array[2*i+1],gap_extend2,penalty_temp0);
            maximum = vimax3(maximum,penalty_here_array[2*i],penalty_here_array[2*i+1]);
        }
		//for (int i=0; i<numRegs/4; i++)
		//    maximum = vimax3(maximum,vimax(penalty_here_array[4*i],penalty_here_array[4*i+1]),vimax(penalty_here_array[4*i+2],penalty_here_array[4*i+3]));

        penalty_here31 = penalty_here_array[numRegs-1];
    };

    const int passes = ceil((1.0*length)/(group_size*numRegs));

    auto shuffle_query = [&](const auto& new_letter) {
        query_letter = __shfl_up_sync(0xFFFFFFFF, query_letter, 1, 32);
        if (!group_id) query_letter = new_letter;
    };

    auto shuffle_affine_penalty = [&](const auto& new_penalty_left, const auto& new_E_left) {
        penalty_diag = penalty_left;
        penalty_left = shfl_up_2xint16(0xFFFFFFFF, penalty_here31, 1, 32);
        E = shfl_up_2xint16(0xFFFFFFFF, E, 1, 32);
        if (!group_id) {
            penalty_left = new_penalty_left;
            E = new_E_left;
        }
    };

    uint32_t temp;

    auto shuffle_new_query = [&]() {
        temp = __shfl_down_sync(0xFFFFFFFF, *((int*)(&new_query_letter4)), 1, 32);
        new_query_letter4 = *((char4*)(&temp));
    };

    int k;

    const uint32_t thread_result = ((length-1)%(group_size*numRegs))/numRegs;              // 0, 1, ..., or 31

    if (passes == 1) {
        // Single pass
    //    for (int k = 5; k < lane_2+group_size-length_2%4; k+=4) {
    //    for (int k = 5; k < lane_2+thread_result+1-thread_result%4; k+=4) {
    init_penalties_local(0);
    init_local_score_profile_BLOSUM62(0);

    initial_calc32_local_affine_float(0);
    shuffle_query(new_query_letter4.y);
    shuffle_affine_penalty(ZERO_in, NEGINFINITY2);

    if (length_2+thread_result >=2) {
        //shuffle_max();
        calc32_local_affine_float();
        shuffle_query(new_query_letter4.z);
        shuffle_affine_penalty(ZERO_in, NEGINFINITY2);
    }

    if (length_2+thread_result >=3) {
        //shuffle_max();
        calc32_local_affine_float();
        shuffle_query(new_query_letter4.w);
        shuffle_affine_penalty(ZERO_in, NEGINFINITY2);
        shuffle_new_query();
        counter++;

    }
    if (length_2+thread_result >=4) {
    //for (k = 5; k < lane_2+thread_result-2; k+=4) {
    for (k = 4; k <= length_2+(thread_result-3); k+=4) {
            //shuffle_max();
            calc32_local_affine_float();

            shuffle_query(new_query_letter4.x);
            shuffle_affine_penalty(ZERO_in,NEGINFINITY2);

            //shuffle_max();
            calc32_local_affine_float();
            shuffle_query(new_query_letter4.y);
            shuffle_affine_penalty(ZERO_in,NEGINFINITY2);

            //shuffle_max();
            calc32_local_affine_float();
            shuffle_query(new_query_letter4.z);
            shuffle_affine_penalty(ZERO_in,NEGINFINITY2);

            //shuffle_max();
            calc32_local_affine_float();
            shuffle_query(new_query_letter4.w);
            shuffle_affine_penalty(ZERO_in,NEGINFINITY2);
            shuffle_new_query();
            if (counter%group_size == 0) {
                new_query_letter4 = constantQuery4[offset];
                offset += group_size;
            }
            counter++;
        }

        if ((k-1)-(length_2+thread_result) > 0) {
            //shuffle_max();
            calc32_local_affine_float();
            shuffle_query(new_query_letter4.x);
            shuffle_affine_penalty(ZERO_in,NEGINFINITY2);
            k++;
        }

        if ((k-1)-(length_2+thread_result) > 0) {
            //shuffle_max();
            calc32_local_affine_float();
            shuffle_query(new_query_letter4.y);
            shuffle_affine_penalty(ZERO_in,NEGINFINITY2);
            k++;
        }

        if ((k-1)-(length_2+thread_result) > 0) {
            //shuffle_max();
            calc32_local_affine_float();
        }

    }
    }
    for (int offset=group_size/2; offset>0; offset/=2)  maximum = vimax(maximum,shfl_down_2xint16(0xFFFFFFFF,maximum,offset,group_size));

    if (!group_id) {
      if (blid < gridDim.x-1) {
          devAlignmentScores[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*(thid/group_size)]] =  maximum.y; // lane_2+thread_result+1-length_2%4; penalty_here_array[(length-1)%numRegs];
          devAlignmentScores[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*(thid/group_size)+1]] =  maximum.x; // lane_2+thread_result+1-length_2%4; penalty_here_array[(length-1)%numRegs];
      } else {
          devAlignmentScores[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)]] =  maximum.y; // lane_2+thread_result+1-length_2%4; penalty_here_array[(length-1)%numRegs];
          if (!check_last2 || (thid%check_last) < check_last-group_size) devAlignmentScores[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)+1]] =  maximum.x; // lane_2+thread_result+1-length_2%4; penalty_here_array[(length-1)%numRegs];
      }
  }
}


// Needleman-Wunsch (NW): global alignment with linear gap penalty
// numRegs values per thread
// uses a single warp per CUDA thread block;
// every groupsize threads computes an alignmen score
template <int group_size, int numRegs> __global__
void NW_local_affine_Protein_single_pass(
    const char * devChars,
    float * devAlignmentScores,
    const size_t* devOffsets,
    const size_t* devLengths,
    const size_t* d_positions_of_selected_lengths,
    const int numSelected,
    const int length_2,
    const float gap_open,
    const float gap_extend
) {

    //__shared__ char4 score_profile[numRegs/4][20][32];
    __shared__ float shared_BLOSUM62[21][21];
    int subject[numRegs];


    const float NEGINFINITY = -10000.0;
    const int blid = blockIdx.x;
    const int thid = threadIdx.x;
    const int group_id = thid%group_size;
    int offset = group_id + group_size;

    int check_last = 32/group_size;
    if ((blid == gridDim.x-1) && (group_size < 32)) {
        if (numSelected % (32/group_size)) check_last = numSelected % (32/group_size);
    }
    check_last = check_last * group_size;
    const int length_S0 = devLengths[d_positions_of_selected_lengths[(32/group_size)*blid+(thid%check_last)/group_size]]; // devLengths[d_positions_of_selected_lengths[blid]]; //devLengths[(32/group_size)*blid+thid/group_size];
    const int base = devOffsets[d_positions_of_selected_lengths[(32/group_size)*blid+(thid%check_last)/group_size]]-devOffsets[0]; //[(32/group_size)*blid+thid/group_size];
    const int length = warp_max_reduce_broadcast(0xFFFFFFFF, length_S0);

    float penalty_temp0, penalty_temp1; // penalty_temp2;
    float penalty_left, penalty_diag;
    float penalty_here31;
    float penalty_here_array[numRegs];
    float F_here_array[numRegs];
    float E = NEGINFINITY;
    //const int base_3 = blid*length_2; // too large???
    float maximum = 0;
    const float ZERO = 0;

    //if ((length < min_length) || (length > max_length)) return;

    //printf("Base_2: %d, %d, %d\n", base_2, blid, thid);
    //printf("Seq1: length: %d, base: %d; Seq2: length: %d, base: %d, blid: %d, thid: %d \n", length, base, length_2, base_2, blid, thid);

//    char4 * cQuery4 = (char4*)(&devChars_2[0]);

    auto init_penalties_local = [&](const auto& value) {
        //ZERO = NEGINFINITY;
        penalty_left = NEGINFINITY;
        penalty_diag = NEGINFINITY;
        E = NEGINFINITY;
        #pragma unroll
        for (int i=0; i<numRegs; i++) penalty_here_array[i] = NEGINFINITY;
        for (int i=0; i<numRegs; i++) F_here_array[i] = NEGINFINITY;
        //if (value==0) penalty_temp0 = gap_open; else penalty_temp0 = gap_extend;
        if (thid % group_size == 0) {
            //if (value==0) penalty_left = value+gap_open; else penalty_left = value+gap_extend;
            //penalty_diag = value;
            penalty_left = 0;
            penalty_diag = 0;
            #pragma unroll
            for (int i=0; i<numRegs; i++) penalty_here_array[i] = 0;
        }
        if (thid % group_size == 1) {
            //penalty_left = value+penalty_temp0+(numRegs-1)*gap_extend;
            //E = value+penalty_temp0+(numRegs-1)*gap_extend;
            penalty_left = 0;
        }
    };

    float temp0; // temp1, E_temp_float, H_temp_float;

    auto init_local_score_profile_BLOSUM62 = [&](const auto& offset_isc) {

        for (int i=thid; i<21*21; i+=32) shared_BLOSUM62[i/21][i%21]=cBLOSUM62_dev[i];
        __syncwarp();

        for (int i=0; i<numRegs; i++) {
            if (offset_isc+numRegs*(thid%group_size)+i >= length_S0) subject[i] = 20;
            else subject[i] = devChars[offset_isc+base+numRegs*(thid%group_size)+i];
        }
    };

    int counter = 1;
    //int counter2 = 1;
    char query_letter = 20;
    char4 new_query_letter4 = constantQuery4[thid%group_size];
    if (thid % group_size== 0) query_letter = new_query_letter4.x;

    float score_temp;
    float *sbt_row;

    auto initial_calc32_local_affine_float = [&](const auto& value){
        sbt_row = shared_BLOSUM62[query_letter];

        score_temp = sbt_row[subject[0]];
        penalty_temp0 = penalty_here_array[0];
        if (!value || (thid%group_size)) E = max(E+gap_extend, penalty_left+gap_open);
        F_here_array[0] = max(F_here_array[0]+gap_extend, penalty_here_array[0]+gap_open);
        penalty_here_array[0] = temp0 = max(max(penalty_diag + score_temp, max(E, F_here_array[0])),ZERO);
        if (temp0 > maximum) maximum = temp0;

        score_temp = sbt_row[subject[1]];
        penalty_temp1 = penalty_here_array[1];
        E = max(E+gap_extend, penalty_here_array[0]+gap_open);
        F_here_array[1] = max(F_here_array[1]+gap_extend, penalty_here_array[1]+gap_open);
        temp0 = penalty_here_array[1] = max(max(penalty_temp0 + score_temp, max(E, F_here_array[1])),ZERO);
        if (temp0 > maximum) maximum = temp0;

        #pragma unroll
        for (int i=1; i<numRegs/2; i++) {
            score_temp = sbt_row[subject[2*i]];
            penalty_temp0 = penalty_here_array[2*i];
            E = max(E+gap_extend, penalty_here_array[2*i-1]+gap_open);
            F_here_array[2*i] = max(F_here_array[2*i]+gap_extend, penalty_here_array[2*i]+gap_open);
            temp0 = penalty_here_array[2*i] = max(max(penalty_temp1 + score_temp, max(E, F_here_array[2*i])),ZERO);
            if (temp0 > maximum) maximum = temp0;

            score_temp = sbt_row[subject[2*i+1]];
            penalty_temp1 = penalty_here_array[2*i+1];
            E = max(E+gap_extend, penalty_here_array[2*i]+gap_open);
            F_here_array[2*i+1] = max(F_here_array[2*i+1]+gap_extend, penalty_here_array[2*i+1]+gap_open);
            temp0 = penalty_here_array[2*i+1] = max(max(penalty_temp0 + score_temp, max(E, F_here_array[2*i+1])),ZERO);
            if (temp0 > maximum) maximum = temp0;

        }

        penalty_here31 = penalty_here_array[numRegs-1];
        E = max(E+gap_extend, penalty_here31+gap_open);
        for (int i=0; i<numRegs; i++) F_here_array[i] = max(F_here_array[i]+gap_extend, penalty_here_array[i]+gap_open);
    };


    auto calc32_local_affine_float = [&](){
        sbt_row = shared_BLOSUM62[query_letter];

        score_temp = sbt_row[subject[0]];
        penalty_temp0 = penalty_here_array[0];
        penalty_here_array[0] = max(max(penalty_diag + score_temp, max(E, F_here_array[0])),ZERO);
        //if (penalty_here_array[0] > maximum) maximum = penalty_here_array[0];
        penalty_temp1 = penalty_here_array[0]+gap_open;
        E = max(E+gap_extend, penalty_temp1);
        F_here_array[0] = max(F_here_array[0]+gap_extend, penalty_temp1);

        score_temp = sbt_row[subject[1]];
        penalty_temp1 = penalty_here_array[1];
        penalty_here_array[1] = max(max(penalty_temp0 + score_temp, max(E, F_here_array[1])),ZERO);
        //if (penalty_here_array[1]  > maximum) maximum = penalty_here_array[1] ;
        penalty_temp0 = penalty_here_array[1]+gap_open;
        E = max(E+gap_extend, penalty_temp0);
        F_here_array[1] = max(F_here_array[1]+gap_extend, penalty_temp0);
		maximum = max(maximum,max(penalty_here_array[0],penalty_here_array[1]));

        #pragma unroll
        for (int i=1; i<numRegs/2; i++) {
            score_temp = sbt_row[subject[2*i]];
            penalty_temp0 = penalty_here_array[2*i];
            penalty_here_array[2*i] = max(max(penalty_temp1 + score_temp, max(E, F_here_array[2*i])),ZERO);
            //if (penalty_here_array[2*i] > maximum) maximum = penalty_here_array[2*i];
            penalty_temp1 = penalty_here_array[2*i]+gap_open;
            E = max(E+gap_extend, penalty_temp1);
            F_here_array[2*i] = max(F_here_array[2*i]+gap_extend, penalty_temp1);

            score_temp = sbt_row[subject[2*i+1]];
            penalty_temp1 = penalty_here_array[2*i+1];
            penalty_here_array[2*i+1] = max(max(penalty_temp0 + score_temp, max(E, F_here_array[2*i+1])),ZERO);
            //if (penalty_here_array[2*i+1] > maximum) maximum = penalty_here_array[2*i+1];
            penalty_temp0 = penalty_here_array[2*i+1]+gap_open;
            E = max(E+gap_extend, penalty_temp0);
            F_here_array[2*i+1] = max(F_here_array[2*i+1]+gap_extend, penalty_temp0);
			maximum = max(maximum,max(penalty_here_array[2*i],penalty_here_array[2*i+1]));
        }
		//for (int i=0; i<numRegs/4; i++)
		//	maximum = max(maximum,max(max(penalty_here_array[4*i],penalty_here_array[4*i+1]), max(penalty_here_array[4*i+2],penalty_here_array[4*i+3])));
        penalty_here31 = penalty_here_array[numRegs-1];
    };

    const int passes = ceil((1.0*length)/(group_size*numRegs));
    //printf("%d, %d, Passes: %d, lenght1: %d, length_2: %d\n", blid, thid, passes, length, length_2);

    auto shuffle_query = [&](const auto& new_letter) {
        query_letter = __shfl_up_sync(0xFFFFFFFF, query_letter, 1, 32);
        if (!group_id) query_letter = new_letter;
    };

    //auto shuffle_max = [&]() {
    //    temp0 = __shfl_up_sync(0xFFFFFFFF, maximum, 1, 32);
    //    if (group_id) {
    //        if (temp0 > maximum) maximum = temp0;
    //    }
    //};

    auto shuffle_affine_penalty = [&](const auto& new_penalty_left, const auto& new_E_left) {
        penalty_diag = penalty_left;
        penalty_left = __shfl_up_sync(0xFFFFFFFF, penalty_here31, 1, 32);
        E = __shfl_up_sync(0xFFFFFFFF, E, 1, 32);
        if (!group_id) {
            penalty_left = new_penalty_left;
            E = new_E_left;
        }
    };

    uint32_t temp;

    auto shuffle_new_query = [&]() {
        temp = __shfl_down_sync(0xFFFFFFFF, *((int*)(&new_query_letter4)), 1, 32);
        new_query_letter4 = *((char4*)(&temp));
    };

    int k;

    const uint32_t thread_result = ((length-1)%(group_size*numRegs))/numRegs;

    if (passes == 1) {
        // Single pass
    //    for (int k = 5; k < lane_2+group_size-length_2%4; k+=4) {
    //    for (int k = 5; k < lane_2+thread_result+1-thread_result%4; k+=4) {
    init_penalties_local(0);
    init_local_score_profile_BLOSUM62(0);

    initial_calc32_local_affine_float(0);
    shuffle_query(new_query_letter4.y);
    shuffle_affine_penalty(0, NEGINFINITY);

    if (length_2+thread_result >=2) {
        //shuffle_max();
        calc32_local_affine_float();
        shuffle_query(new_query_letter4.z);
        shuffle_affine_penalty(0, NEGINFINITY);
    }

    if (length_2+thread_result >=3) {
        //shuffle_max();
        calc32_local_affine_float();
        shuffle_query(new_query_letter4.w);
        shuffle_affine_penalty(0, NEGINFINITY);
        shuffle_new_query();
        counter++;

    }
    if (length_2+thread_result >=4) {
    for (k = 4; k <= length_2+(thread_result-3); k+=4) {
    //for (k = 5; k < lane_2+thread_result-2; k+=4) {
            //shuffle_max();
            calc32_local_affine_float();

            shuffle_query(new_query_letter4.x);
            shuffle_affine_penalty(0,NEGINFINITY);

            //shuffle_max();
            calc32_local_affine_float();
            shuffle_query(new_query_letter4.y);
            shuffle_affine_penalty(0,NEGINFINITY);

            //shuffle_max();
            calc32_local_affine_float();
            shuffle_query(new_query_letter4.z);
            shuffle_affine_penalty(0,NEGINFINITY);

            //shuffle_max();
            calc32_local_affine_float();
            shuffle_query(new_query_letter4.w);
            shuffle_affine_penalty(0,NEGINFINITY);
            shuffle_new_query();
            if (counter%group_size == 0) {
                new_query_letter4 = constantQuery4[offset];
                offset += group_size;
            }
            counter++;
        }

        //if (counter2-(length_2+thread_result) > 0) {
        if ((k-1)-(length_2+thread_result) > 0) {
            //shuffle_max();
            calc32_local_affine_float();
            shuffle_query(new_query_letter4.x);
            shuffle_affine_penalty(0,NEGINFINITY);
            k++;
        }

        //if (counter2-(length_2+thread_result) > 0) {
        if ((k-1)-(length_2+thread_result) > 0) {
            //shuffle_max();
            calc32_local_affine_float();
            shuffle_query(new_query_letter4.y);
            shuffle_affine_penalty(0,NEGINFINITY);
            k++;
        }

        //if (counter2-(length_2+thread_result) > 0) {
        if ((k-1)-(length_2+thread_result) > 0) {
            //shuffle_max();
            calc32_local_affine_float();
        }

    }
    }
    for (int offset=group_size/2; offset>0; offset/=2)
           maximum = max(maximum,__shfl_down_sync(0xFFFFFFFF,maximum,offset,group_size));
  // if (thid % group_size == thread_result)
    //  printf("Result in Block: %d, in Thread: %d, Register: %d, Value: %f, #passes: %d\n", blid, thid, (length-1)%numRegs, penalty_here_array[(length-1)%numRegs], passes);
   if (!group_id)
      //devAlignmentScores[(32/group_size)*blid+thid/group_size] =  maximum;
      devAlignmentScores[d_positions_of_selected_lengths[(32/group_size)*blid+(thid%check_last)/group_size]] =  maximum;
// devAlignmentScores[(32/group_size)*blid+thid/group_size] =  penalty_here_array[(length-1)%numRegs]; // lane_2+thread_result+1-length_2%4; penalty_here_array[(length-1)%numRegs];
    // if (thid % group_size == thread_result)
    //     printf("Result in Block: %d, in Thread: %d, Register: %d, Value: %f, #passes: %d\n", blid, thid, (length-1)%numRegs, penalty_here_array[(length-1)%numRegs], passes);

}


// Needleman-Wunsch (NW): global alignment with linear gap penalty
// numRegs values per thread
// uses a single warp per CUDA thread block;
// every groupsize threads computes an alignmen score
template <int group_size, int numRegs> __global__
void NW_local_affine_single_pass_s32_DPX(
    const char * devChars,
    float * devAlignmentScores,
    const size_t* devOffsets,
    const size_t* devLengths,
    const size_t* d_positions_of_selected_lengths,
    const int numSelected,
    const int length_2,
    const int gap_open,
    const int gap_extend
) {

    //__shared__ char4 score_profile[numRegs/4][20][32];
    __shared__ int shared_BLOSUM62[21][21];
    int subject[numRegs];


    const int NEGINFINITY = -10000;
    const int blid = blockIdx.x;
    const int thid = threadIdx.x;
    const int group_id = thid%group_size;
    int offset = group_id + group_size;

    int check_last = 32/group_size;
    if ((blid == gridDim.x-1) && (group_size < 32)) {
        if (numSelected % (32/group_size)) check_last = numSelected % (32/group_size);
    }
    check_last = check_last * group_size;
    const int length_S0 = devLengths[d_positions_of_selected_lengths[(32/group_size)*blid+(thid%check_last)/group_size]]; // devLengths[d_positions_of_selected_lengths[blid]]; //devLengths[(32/group_size)*blid+thid/group_size];
    const int base = devOffsets[d_positions_of_selected_lengths[(32/group_size)*blid+(thid%check_last)/group_size]]; //[(32/group_size)*blid+thid/group_size];
    const int length = warp_max_reduce_broadcast(0xFFFFFFFF, length_S0);

    int penalty_temp0, penalty_temp1; // penalty_temp2;
    int penalty_left, penalty_diag;
    int penalty_here31;
    int penalty_here_array[numRegs];
    int F_here_array[numRegs];
    int E = NEGINFINITY;
    //const int base_3 = blid*length_2; // too large???
    int maximum = 0;
    //const float ZERO = 0;

    auto init_penalties_local = [&](const auto& value) {
        //ZERO = NEGINFINITY;
        penalty_left = NEGINFINITY;
        penalty_diag = NEGINFINITY;
        E = NEGINFINITY;
        #pragma unroll
        for (int i=0; i<numRegs; i++) penalty_here_array[i] = NEGINFINITY;
        for (int i=0; i<numRegs; i++) F_here_array[i] = NEGINFINITY;
        if (thid % group_size == 0) {
            penalty_left = 0;
            penalty_diag = 0;
            #pragma unroll
            for (int i=0; i<numRegs; i++) penalty_here_array[i] = 0;
        }
        if (thid % group_size == 1) {
            penalty_left = 0;
        }
    };

    int temp0; // temp1, E_temp_float, H_temp_float;

    auto init_local_score_profile_BLOSUM62 = [&](const auto& offset_isc) {

        for (int i=thid; i<21*21; i+=32) shared_BLOSUM62[i/21][i%21]=cBLOSUM62_dev[i];
        __syncwarp();

        for (int i=0; i<numRegs; i++) {
            if (offset_isc+numRegs*(thid%group_size)+i >= length_S0) subject[i] = 20;
            else subject[i] = devChars[offset_isc+base+numRegs*(thid%group_size)+i];
        }
    };

    int counter = 1;
    //int counter2 = 1;
    char query_letter = 20;
    char4 new_query_letter4 = constantQuery4[thid%group_size];
    if (thid % group_size== 0) query_letter = new_query_letter4.x;

    int score_temp;
    int *sbt_row;

    auto initial_calc32_local_affine_float = [&](const auto& value){
        sbt_row = shared_BLOSUM62[query_letter];

        score_temp = sbt_row[subject[0]];
        penalty_temp0 = penalty_here_array[0];

        if (!value || (thid%group_size)) E = max(E+gap_extend, penalty_left+gap_open);
        F_here_array[0] = max(F_here_array[0]+gap_extend, penalty_here_array[0]+gap_open);
        penalty_here_array[0] = __vimax3_s32_relu(penalty_diag + score_temp,E,F_here_array[0]);

        score_temp = sbt_row[subject[1]];
        penalty_temp1 = penalty_here_array[1];
        E = max(E+gap_extend, penalty_here_array[0]+gap_open);
        F_here_array[1] = max(F_here_array[1]+gap_extend, penalty_here_array[1]+gap_open);
        penalty_here_array[1] = __vimax3_s32_relu(penalty_temp0+score_temp,E,F_here_array[1]);
		maximum = __vimax3_s32(maximum,penalty_here_array[0],penalty_here_array[1]);


        #pragma unroll
        for (int i=1; i<numRegs/2; i++) {
            score_temp = sbt_row[subject[2*i]];
            penalty_temp0 = penalty_here_array[2*i];
            E = max(E+gap_extend, penalty_here_array[2*i-1]+gap_open);
            F_here_array[2*i] = max(F_here_array[2*i]+gap_extend, penalty_here_array[2*i]+gap_open);
            penalty_here_array[2*i] = __vimax3_s32_relu(penalty_temp1+score_temp,E,F_here_array[2*i]);

            score_temp = sbt_row[subject[2*i+1]];
            penalty_temp1 = penalty_here_array[2*i+1];
            E = max(E+gap_extend, penalty_here_array[2*i]+gap_open);
            F_here_array[2*i+1] = max(F_here_array[2*i+1]+gap_extend, penalty_here_array[2*i+1]+gap_open);
            penalty_here_array[2*i+1] = __vimax3_s32_relu(penalty_temp0+score_temp,E,F_here_array[2*i+1]);
			maximum = __vimax3_s32(maximum,penalty_here_array[2*i],penalty_here_array[2*i+1]);
        }

        penalty_here31 = penalty_here_array[numRegs-1];
        E = max(E+gap_extend, penalty_here31+gap_open);
        for (int i=0; i<numRegs; i++) F_here_array[i] = max(F_here_array[i]+gap_extend, penalty_here_array[i]+gap_open);
    };


    auto calc32_local_affine_float = [&](){
        sbt_row = shared_BLOSUM62[query_letter];

        score_temp = sbt_row[subject[0]];
        penalty_temp0 = penalty_here_array[0];

        penalty_here_array[0] = __vimax3_s32_relu(penalty_diag + score_temp,E,F_here_array[0]); // max(max(penalty_diag + score_temp, max(E, F_here_array[0])),ZERO);
        penalty_temp1 = penalty_here_array[0]+gap_open;
        E = __viaddmax_s32(E,gap_extend,penalty_temp1); //E = max(E+gap_extend, penalty_temp1);
        F_here_array[0] = __viaddmax_s32(F_here_array[0],gap_extend,penalty_temp1);

        score_temp = sbt_row[subject[1]];
        penalty_temp1 = penalty_here_array[1];
        penalty_here_array[1] = __vimax3_s32_relu(penalty_temp0+score_temp,E,F_here_array[1]);
        penalty_temp0 = penalty_here_array[1]+gap_open;
        E = __viaddmax_s32(E,gap_extend,penalty_temp0);
        F_here_array[1] = __viaddmax_s32(F_here_array[1],gap_extend,penalty_temp0);
        maximum = __vimax3_s32(maximum,penalty_here_array[0],penalty_here_array[1]);

        #pragma unroll
        for (int i=1; i<numRegs/2; i++) {
            score_temp = sbt_row[subject[2*i]];
            penalty_temp0 = penalty_here_array[2*i];
            penalty_here_array[2*i] = __vimax3_s32_relu(penalty_temp1+score_temp,E,F_here_array[2*i]); // max(max(penalty_diag + score_temp, max(E, F_here_array[0])),ZERO);
            penalty_temp1 = penalty_here_array[2*i]+gap_open;
            E = __viaddmax_s32(E,gap_extend,penalty_temp1); //E = max(E+gap_extend, penalty_temp1);
            F_here_array[2*i] = __viaddmax_s32(F_here_array[2*i],gap_extend,penalty_temp1);// max(F_here_array[0]+gap_extend, penalty_temp1);

            score_temp = sbt_row[subject[2*i+1]];
            penalty_temp1 = penalty_here_array[2*i+1];
            penalty_here_array[2*i+1] = __vimax3_s32_relu(penalty_temp0+score_temp,E,F_here_array[2*i+1]);
            penalty_temp0 = penalty_here_array[2*i+1]+gap_open;
            E = __viaddmax_s32(E,gap_extend,penalty_temp0);
            F_here_array[2*i+1] = __viaddmax_s32(F_here_array[2*i+1],gap_extend,penalty_temp0);
            maximum = __vimax3_s32(maximum,penalty_here_array[2*i],penalty_here_array[2*i+1]);

        }
        //for (int i=0; i<numRegs/4; i++)
        //    maximum = __vimax3_s32(maximum,max(penalty_here_array[4*i],penalty_here_array[4*i+1]),max(penalty_here_array[4*i+2],penalty_here_array[4*i+3]));
        penalty_here31 = penalty_here_array[numRegs-1];
    };

    const int passes = ceil((1.0*length)/(group_size*numRegs));
    //printf("%d, %d, Passes: %d, lenght1: %d, length_2: %d\n", blid, thid, passes, length, length_2);

    auto shuffle_query = [&](const auto& new_letter) {
        query_letter = __shfl_up_sync(0xFFFFFFFF, query_letter, 1, 32);
        if (!group_id) query_letter = new_letter;
    };

    auto shuffle_affine_penalty = [&](const auto& new_penalty_left, const auto& new_E_left) {
        penalty_diag = penalty_left;
        penalty_left = __shfl_up_sync(0xFFFFFFFF, penalty_here31, 1, 32);
        E = __shfl_up_sync(0xFFFFFFFF, E, 1, 32);
        if (!group_id) {
            penalty_left = new_penalty_left;
            E = new_E_left;
        }
    };

    uint32_t temp;

    auto shuffle_new_query = [&]() {
        temp = __shfl_down_sync(0xFFFFFFFF, *((int*)(&new_query_letter4)), 1, 32);
        new_query_letter4 = *((char4*)(&temp));
    };

    int k;

    const uint32_t thread_result = ((length-1)%(group_size*numRegs))/numRegs;

    if (passes == 1) {
        // Single pass
    //    for (int k = 5; k < lane_2+group_size-length_2%4; k+=4) {
    //    for (int k = 5; k < lane_2+thread_result+1-thread_result%4; k+=4) {
    init_penalties_local(0);
    init_local_score_profile_BLOSUM62(0);

    initial_calc32_local_affine_float(0);
    shuffle_query(new_query_letter4.y);
    shuffle_affine_penalty(0, NEGINFINITY);

    if (length_2+thread_result >=2) {
        //shuffle_max();
        calc32_local_affine_float();
        shuffle_query(new_query_letter4.z);
        shuffle_affine_penalty(0, NEGINFINITY);
    }

    if (length_2+thread_result >=3) {
        //shuffle_max();
        calc32_local_affine_float();
        shuffle_query(new_query_letter4.w);
        shuffle_affine_penalty(0, NEGINFINITY);
        shuffle_new_query();
        counter++;

    }
    if (length_2+thread_result >=4) {
    for (k = 4; k <= length_2+(thread_result-3); k+=4) {
    //for (k = 5; k < lane_2+thread_result-2; k+=4) {
            //shuffle_max();
            calc32_local_affine_float();

            shuffle_query(new_query_letter4.x);
            shuffle_affine_penalty(0,NEGINFINITY);

            //shuffle_max();
            calc32_local_affine_float();
            shuffle_query(new_query_letter4.y);
            shuffle_affine_penalty(0,NEGINFINITY);

            //shuffle_max();
            calc32_local_affine_float();
            shuffle_query(new_query_letter4.z);
            shuffle_affine_penalty(0,NEGINFINITY);

            //shuffle_max();
            calc32_local_affine_float();
            shuffle_query(new_query_letter4.w);
            shuffle_affine_penalty(0,NEGINFINITY);
            shuffle_new_query();
            if (counter%group_size == 0) {
                new_query_letter4 = constantQuery4[offset];
                offset += group_size;
            }
            counter++;
        }

        //if (counter2-(length_2+thread_result) > 0) {
        if ((k-1)-(length_2+thread_result) > 0) {
            //shuffle_max();
            calc32_local_affine_float();
            shuffle_query(new_query_letter4.x);
            shuffle_affine_penalty(0,NEGINFINITY);
            k++;
        }

        //if (counter2-(length_2+thread_result) > 0) {
        if ((k-1)-(length_2+thread_result) > 0) {
            //shuffle_max();
            calc32_local_affine_float();
            shuffle_query(new_query_letter4.y);
            shuffle_affine_penalty(0,NEGINFINITY);
            k++;
        }

        //if (counter2-(length_2+thread_result) > 0) {
        if ((k-1)-(length_2+thread_result) > 0) {
            //shuffle_max();
            calc32_local_affine_float();
        }

    }
    }
    for (int offset=group_size/2; offset>0; offset/=2)
           maximum = max(maximum,__shfl_down_sync(0xFFFFFFFF,maximum,offset,group_size));
  // if (thid % group_size == thread_result)
    //  printf("Result in Block: %d, in Thread: %d, Register: %d, Value: %f, #passes: %d\n", blid, thid, (length-1)%numRegs, penalty_here_array[(length-1)%numRegs], passes);
   if (!group_id)
      //devAlignmentScores[(32/group_size)*blid+thid/group_size] =  maximum;
      devAlignmentScores[d_positions_of_selected_lengths[(32/group_size)*blid+(thid%check_last)/group_size]] =  maximum;
// devAlignmentScores[(32/group_size)*blid+thid/group_size] =  penalty_here_array[(length-1)%numRegs]; // lane_2+thread_result+1-length_2%4; penalty_here_array[(length-1)%numRegs];
    // if (thid % group_size == thread_result)
    //     printf("Result in Block: %d, in Thread: %d, Register: %d, Value: %f, #passes: %d\n", blid, thid, (length-1)%numRegs, penalty_here_array[(length-1)%numRegs], passes);

}



// Needleman-Wunsch (NW): global alignment with linear gap penalty
// numRegs values per thread
// uses a single warp per CUDA thread block;
// every groupsize threads computes an alignmen score
template <int group_size, int numRegs, class PositionsIterator> __global__
void NW_local_affine_read4_float_query_Protein(
    const char * devChars,
    float * devAlignmentScores,
    short2 * devTempHcol2,
    short2 * devTempEcol2,
    const size_t* devOffsets,
    const size_t* devLengths,
    PositionsIterator d_positions_of_selected_lengths,
    const int length_2,
    const float gap_open,
    const float gap_extend
) {

    __shared__ float shared_BLOSUM62[21][21];
    int subject[numRegs];

    const float NEGINFINITY = -10000.0;
    const int blid = blockIdx.x;
    const int thid = threadIdx.x;
    const int group_id = thid%group_size;
    int offset = group_id + group_size;

    int test_length =  devLengths[d_positions_of_selected_lengths[blid]];//devLengths[(32/group_size)*blid+thid/group_size];
    const int length = abs(test_length);
    const int base = devOffsets[d_positions_of_selected_lengths[blid]]-devOffsets[0]; // devOffsets[(32/group_size)*blid+thid/group_size];

    short2 H_temp_out, H_temp_in;
    short2 E_temp_out, E_temp_in;
    float penalty_temp0, penalty_temp1; // penalty_temp2;
    float penalty_left, penalty_diag;
    float penalty_here31;
    //constexpr int numRegs = 32;
    float penalty_here_array[numRegs];
    float F_here_array[numRegs];
    float E = NEGINFINITY;

    //const int base_3 = d_positions_of_selected_lengths[blid]*length_2; // blid*length_2;
	const int base_3 = blid*length_2; // blid*length_2;
    short2 * devTempHcol = (short2*)(&devTempHcol2[base_3]);
    short2 * devTempEcol = (short2*)(&devTempEcol2[base_3]);
    float maximum = 0;
    const float ZERO = 0;

    auto checkHEindex = [&](auto x){
        // assert(x >= 0); //positive index
        // assert(blid*length_2 <= base_3 + x);
        // assert(base_3+x < (blid+1)*length_2);
    };

    //if (test_length>0 && ((length < min_length) || (length > max_length))) return;

    //printf("Base_2: %d, %d, %d\n", base_2, blid, thid);
    //printf("Seq1: length: %d, base: %d; Seq2: length: %d, base: %d, blid: %d, thid: %d \n", length, base, length_2, base_2, blid, thid);

//    char4 * cQuery4 = (char4*)(&devChars_2[0]);



    auto init_penalties_local = [&](const auto& value) {
        //ZERO = NEGINFINITY;
        penalty_left = NEGINFINITY;
        penalty_diag = NEGINFINITY;
        E = NEGINFINITY;
        #pragma unroll
        for (int i=0; i<numRegs; i++) penalty_here_array[i] = NEGINFINITY;
        for (int i=0; i<numRegs; i++) F_here_array[i] = NEGINFINITY;
        //if (value==0) penalty_temp0 = gap_open; else penalty_temp0 = gap_extend;
        if (!thid) {
            //if (value==0) penalty_left = value+gap_open; else penalty_left = value+gap_extend;
            //penalty_diag = value;
            penalty_left = 0;
            penalty_diag = 0;
            #pragma unroll
            for (int i=0; i<numRegs; i++) penalty_here_array[i] = 0;
        }
        if (thid == 1) {
            //penalty_left = value+penalty_temp0+(numRegs-1)*gap_extend;
            //E = value+penalty_temp0+(numRegs-1)*gap_extend;
            penalty_left = 0;
        }
    };


    float temp0; // temp1, E_temp_float, H_temp_float;
    auto init_local_score_profile_BLOSUM62 = [&](const auto& offset_isc) {

        if (offset_isc == 0) {
            for (int i=thid; i<21*21; i+=32) shared_BLOSUM62[i/21][i%21]=cBLOSUM62_dev[i];
            __syncwarp();
        }

        for (int i=0; i<numRegs; i++) {
            if (offset_isc+numRegs*(thid%group_size)+i >= length) subject[i] = 1; // 20;
            else subject[i] = devChars[offset_isc+base+numRegs*(thid%group_size)+i];
        }
    };


    //H_temp_out = __floats2half2_rn(NEGINFINITY,NEGINFINITY);
    H_temp_out.x = -30000; H_temp_out.y = -30000;
    E_temp_out.x = -30000; E_temp_out.y = -30000;
    int counter = 1;
    //int counter2 = 1;
    char query_letter = 20;
    char4 new_query_letter4 = constantQuery4[thid%group_size];
    if (!thid) query_letter = new_query_letter4.x;

    float score_temp;
    float *sbt_row;

    auto initial_calc32_local_affine_float = [&](const auto& value){
        sbt_row = shared_BLOSUM62[query_letter];

        score_temp = sbt_row[subject[0]];
        penalty_temp0 = penalty_here_array[0];
        if (!value || (thid%group_size)) E = max(E+gap_extend, penalty_left+gap_open);
        F_here_array[0] = max(F_here_array[0]+gap_extend, penalty_here_array[0]+gap_open);
        penalty_here_array[0] = temp0 = max(max(penalty_diag + score_temp, max(E, F_here_array[0])),ZERO);
        if (temp0 > maximum) maximum = temp0;

        score_temp = sbt_row[subject[1]];
        penalty_temp1 = penalty_here_array[1];
        E = max(E+gap_extend, penalty_here_array[0]+gap_open);
        F_here_array[1] = max(F_here_array[1]+gap_extend, penalty_here_array[1]+gap_open);
        temp0 = penalty_here_array[1] = max(max(penalty_temp0 + score_temp, max(E, F_here_array[1])),ZERO);
        if (temp0 > maximum) maximum = temp0;

        #pragma unroll
        for (int i=1; i<numRegs/2; i++) {
            score_temp = sbt_row[subject[2*i]];
            penalty_temp0 = penalty_here_array[2*i];
            E = max(E+gap_extend, penalty_here_array[2*i-1]+gap_open);
            F_here_array[2*i] = max(F_here_array[2*i]+gap_extend, penalty_here_array[2*i]+gap_open);
            temp0 = penalty_here_array[2*i] = max(max(penalty_temp1 + score_temp, max(E, F_here_array[2*i])),ZERO);
            if (temp0 > maximum) maximum = temp0;

            score_temp = sbt_row[subject[2*i+1]];
            penalty_temp1 = penalty_here_array[2*i+1];
            E = max(E+gap_extend, penalty_here_array[2*i]+gap_open);
            F_here_array[2*i+1] = max(F_here_array[2*i+1]+gap_extend, penalty_here_array[2*i+1]+gap_open);
            temp0 = penalty_here_array[2*i+1] = max(max(penalty_temp0 + score_temp, max(E, F_here_array[2*i+1])),ZERO);
            if (temp0 > maximum) maximum = temp0;

        }

        penalty_here31 = penalty_here_array[numRegs-1];
        E = max(E+gap_extend, penalty_here31+gap_open);
        for (int i=0; i<numRegs; i++) F_here_array[i] = max(F_here_array[i]+gap_extend, penalty_here_array[i]+gap_open);

    };


    auto calc32_local_affine_float = [&](){
        sbt_row = shared_BLOSUM62[query_letter];

        score_temp = sbt_row[subject[0]];
        penalty_temp0 = penalty_here_array[0];
        penalty_here_array[0] = max(max(penalty_diag + score_temp, max(E, F_here_array[0])),ZERO);
        //if (penalty_here_array[0] > maximum) maximum = penalty_here_array[0];
        penalty_temp1 = penalty_here_array[0]+gap_open;
        E = max(E+gap_extend, penalty_temp1);
        F_here_array[0] = max(F_here_array[0]+gap_extend, penalty_temp1);

        score_temp = sbt_row[subject[1]];
        penalty_temp1 = penalty_here_array[1];
        penalty_here_array[1] = max(max(penalty_temp0 + score_temp, max(E, F_here_array[1])),ZERO);
        //if (penalty_here_array[1]  > maximum) maximum = penalty_here_array[1] ;
        penalty_temp0 = penalty_here_array[1]+gap_open;
        E = max(E+gap_extend, penalty_temp0);
        F_here_array[1] = max(F_here_array[1]+gap_extend, penalty_temp0);
		maximum = max(maximum,max(penalty_here_array[0],penalty_here_array[1]));

        #pragma unroll
        for (int i=1; i<numRegs/2; i++) {
            score_temp = sbt_row[subject[2*i]];
            penalty_temp0 = penalty_here_array[2*i];
            penalty_here_array[2*i] = max(max(penalty_temp1 + score_temp, max(E, F_here_array[2*i])),ZERO);
            //if (penalty_here_array[2*i] > maximum) maximum = penalty_here_array[2*i];
            penalty_temp1 = penalty_here_array[2*i]+gap_open;
            E = max(E+gap_extend, penalty_temp1);
            F_here_array[2*i] = max(F_here_array[2*i]+gap_extend, penalty_temp1);

            score_temp = sbt_row[subject[2*i+1]];
            penalty_temp1 = penalty_here_array[2*i+1];
            penalty_here_array[2*i+1] = max(max(penalty_temp0 + score_temp, max(E, F_here_array[2*i+1])),ZERO);
            //if (penalty_here_array[2*i+1] > maximum) maximum = penalty_here_array[2*i+1];
            penalty_temp0 = penalty_here_array[2*i+1]+gap_open;
            E = max(E+gap_extend, penalty_temp0);
            F_here_array[2*i+1] = max(F_here_array[2*i+1]+gap_extend, penalty_temp0);
			maximum = max(maximum,max(penalty_here_array[2*i],penalty_here_array[2*i+1]));
        }
		//for (int i=0; i<numRegs/4; i++)
		//	maximum = max(maximum,max(max(penalty_here_array[4*i],penalty_here_array[4*i+1]), max(penalty_here_array[4*i+2],penalty_here_array[4*i+3])));
        penalty_here31 = penalty_here_array[numRegs-1];
    };

    const int passes = ceil((1.0*length)/(32*numRegs));
    int offset_out = group_id;
    int offset_in = group_id;

    //printf("%d, %d, Passes: %d, lenght1: %d, length_2: %d\n", blid, thid, passes, length, length_2);

    auto shuffle_query = [&](const auto& new_letter) {
        query_letter = __shfl_up_sync(0xFFFFFFFF, query_letter, 1, 32);
        if (!thid) query_letter = new_letter;
    };

    //auto shuffle_max = [&]() {
    //    temp0 = __shfl_up_sync(0xFFFFFFFF, maximum, 1, 32);
    //    if (thid) {
    //        if (temp0 > maximum) maximum = temp0;
    //    }
    //};

    auto shuffle_affine_penalty = [&](const auto& new_penalty_left, const auto& new_E_left) {
        penalty_diag = penalty_left;
        penalty_left = __shfl_up_sync(0xFFFFFFFF, penalty_here31, 1, 32);
        E = __shfl_up_sync(0xFFFFFFFF, E, 1, 32);
        if (!thid) {
            penalty_left = new_penalty_left;
            E = new_E_left;
        }
    };

    uint32_t temp;
    auto shuffle_H_E_temp_out = [&]() {
        temp = __shfl_down_sync(0xFFFFFFFF, *((int*)(&H_temp_out)), 1, 32);
        H_temp_out = *((short2*)(&temp));
        temp = __shfl_down_sync(0xFFFFFFFF, *((int*)(&E_temp_out)), 1, 32);
        E_temp_out = *((short2*)(&temp));
    };

    auto shuffle_H_E_temp_in = [&]() {
        temp = __shfl_down_sync(0xFFFFFFFF, *((int*)(&H_temp_in)), 1, 32);
        H_temp_in = *((short2*)(&temp));
        temp = __shfl_down_sync(0xFFFFFFFF, *((int*)(&E_temp_in)), 1, 32);
        E_temp_in = *((short2*)(&temp));
    };

    auto shuffle_new_query = [&]() {
        temp = __shfl_down_sync(0xFFFFFFFF, *((int*)(&new_query_letter4)), 1, 32);
        new_query_letter4 = *((char4*)(&temp));
    };

    auto set_H_E_temp_out_x = [&]() {
        if (thid == 31) {
            H_temp_out.x = penalty_here31;
            E_temp_out.x = E;
        }
    };

    auto set_H_E_temp_out_y = [&]() {
        if (thid  == 31) {
            H_temp_out.y = penalty_here31;
            E_temp_out.y = E;
        }
    };

    int k;

    const uint32_t thread_result = ((length-1)%(32*numRegs))/numRegs;


    if (passes == 1) {
        // Single pass
        //    for (int k = 5; k < lane_2+group_size-length_2%4; k+=4) {
        //    for (int k = 5; k < lane_2+thread_result+1-thread_result%4; k+=4) {
        init_penalties_local(0);
        init_local_score_profile_BLOSUM62(0);

        initial_calc32_local_affine_float(0);
        shuffle_query(new_query_letter4.y);
        shuffle_affine_penalty(0, NEGINFINITY);

        if (length_2+thread_result >=2) {
            //shuffle_max();
            calc32_local_affine_float();
            shuffle_query(new_query_letter4.z);
            shuffle_affine_penalty(0, NEGINFINITY);
        }

        if (length_2+thread_result >=3) {
            //shuffle_max();
            calc32_local_affine_float();
            shuffle_query(new_query_letter4.w);
            shuffle_affine_penalty(0, NEGINFINITY);
            shuffle_new_query();
            counter++;

        }
        if (length_2+thread_result >=4) {
        //for (k = 5; k < lane_2+thread_result-2; k+=4) {
        for (k = 4; k <= length_2+(thread_result-3); k+=4) {
                //shuffle_max();
                calc32_local_affine_float();

                shuffle_query(new_query_letter4.x);
                shuffle_affine_penalty(0,NEGINFINITY);

                //shuffle_max();
                calc32_local_affine_float();
                shuffle_query(new_query_letter4.y);
                shuffle_affine_penalty(0,NEGINFINITY);

                //shuffle_max();
                calc32_local_affine_float();
                shuffle_query(new_query_letter4.z);
                shuffle_affine_penalty(0,NEGINFINITY);

                //shuffle_max();
                calc32_local_affine_float();
                shuffle_query(new_query_letter4.w);
                shuffle_affine_penalty(0,NEGINFINITY);
                shuffle_new_query();
                if (counter%group_size == 0) {
                    new_query_letter4 = constantQuery4[offset];
                    offset += group_size;
                }
                counter++;
            }

        //if (counter2-(length_2+thread_result) > 0) {
        if ((k-1)-(length_2+thread_result) > 0) {
                //shuffle_max();
                calc32_local_affine_float();
                shuffle_query(new_query_letter4.x);
                shuffle_affine_penalty(0,NEGINFINITY);
                k++;
            }

            //if (counter2-(length_2+thread_result) > 0) {
            if ((k-1)-(length_2+thread_result) > 0) {
                //shuffle_max();
                calc32_local_affine_float();
                shuffle_query(new_query_letter4.y);
                shuffle_affine_penalty(0,NEGINFINITY);
                k++;
            }

            //if (counter2-(length_2+thread_result) > 0) {
            if ((k-1)-(length_2+thread_result) > 0) {
                //shuffle_max();
                calc32_local_affine_float();
                //shuffle_query(new_query_letter4.z);
                //shuffle_affine_penalty(0,NEGINFINITY);
            }

        //    if (blid == 0) {
        //        if (thid % group_size == thread_result)
        //            printf("Result in Thread: %d, Register: %d, Value: %f, #passes: %d\n", thid, (length-1)%numRegs, penalty_here_array[(length-1)%numRegs], passes);
        //    }
        }
    }
    else {

        //if (blid == 12455) {
        //    if (thid == 0)
        //    printf("Block: %d, Thread: %d, Query-length: %d, Database length: %d, passes: %d, thread_result: %d\n", blid, thid, length_2, length, passes, thread_result);
        //}

        // first pass (of multiple passes)
        init_penalties_local(0);
        init_local_score_profile_BLOSUM62(0);
        initial_calc32_local_affine_float(0);
        shuffle_query(new_query_letter4.y);
        shuffle_affine_penalty(0, NEGINFINITY);

        //shuffle_max();
        calc32_local_affine_float();
        shuffle_query(new_query_letter4.z);
        shuffle_affine_penalty(0, NEGINFINITY);

        //shuffle_max();
        calc32_local_affine_float();
        shuffle_query(new_query_letter4.w);
        shuffle_affine_penalty(0, NEGINFINITY);
        shuffle_new_query();
        counter++;
        for (k = 4; k <= 28; k+=4) {
            calc32_local_affine_float();
            shuffle_query(new_query_letter4.x);
            shuffle_affine_penalty(0,NEGINFINITY);

            calc32_local_affine_float();
            shuffle_query(new_query_letter4.y);
            shuffle_affine_penalty(0,NEGINFINITY);

            calc32_local_affine_float();
            shuffle_query(new_query_letter4.z);
            shuffle_affine_penalty(0,NEGINFINITY);

            calc32_local_affine_float();
            shuffle_query(new_query_letter4.w);
            shuffle_affine_penalty(0,NEGINFINITY);
            shuffle_new_query();
            counter++;
        }
        //for (k = 4; k <= length_2+28; k+=4) {
        for (k = 32; k <= length_2+28; k+=4) {
            //shuffle_max();
            calc32_local_affine_float();
            set_H_E_temp_out_x();
            shuffle_query(new_query_letter4.x);
            shuffle_affine_penalty(0,NEGINFINITY);

            //shuffle_max();
            calc32_local_affine_float();

            set_H_E_temp_out_y();
            shuffle_H_E_temp_out();
            shuffle_query(new_query_letter4.y);
            shuffle_affine_penalty(0,NEGINFINITY);

            //shuffle_max();
            calc32_local_affine_float();

            set_H_E_temp_out_x();
            shuffle_query(new_query_letter4.z);
            shuffle_affine_penalty(0,NEGINFINITY);

            //shuffle_max();
            calc32_local_affine_float();

            set_H_E_temp_out_y();

            if ((counter+8)%16 == 0 && counter > 8) {
                checkHEindex(offset_out);
                devTempHcol[offset_out]=H_temp_out;
                devTempEcol[offset_out]=E_temp_out;
                offset_out += group_size;

            }
            shuffle_H_E_temp_out();
            shuffle_query(new_query_letter4.w);
            shuffle_affine_penalty(0,NEGINFINITY);
            shuffle_new_query();
            if (counter%group_size == 0) {
                new_query_letter4 = constantQuery4[offset];
                offset += group_size;
            }
            counter++;
        }
        if (length_2 % 4 == 0) {
            temp = __shfl_up_sync(0xFFFFFFFF, *((int*)(&H_temp_out)), 1, 32);
            H_temp_out = *((short2*)(&temp));
            temp = __shfl_up_sync(0xFFFFFFFF, *((int*)(&E_temp_out)), 1, 32);
            E_temp_out = *((short2*)(&temp));
        }
        if (length_2 % 4 == 1) {
            //shuffle_max();
            calc32_local_affine_float();
            set_H_E_temp_out_x();
            set_H_E_temp_out_y();
        }
        if (length_2 % 4 == 2) {
            //shuffle_max();
            calc32_local_affine_float();
            set_H_E_temp_out_x();
            shuffle_query(new_query_letter4.x);
            shuffle_affine_penalty(0,NEGINFINITY);
            //shuffle_max();
            calc32_local_affine_float();
            set_H_E_temp_out_y();
        }
        if (length_2 % 4 == 3) {
            //shuffle_max();
            calc32_local_affine_float();
            set_H_E_temp_out_x();
            shuffle_query(new_query_letter4.x);
            shuffle_affine_penalty(0,NEGINFINITY);
            //shuffle_max();
            calc32_local_affine_float();
            set_H_E_temp_out_y();
            shuffle_H_E_temp_out();
            shuffle_query(new_query_letter4.y);
            shuffle_affine_penalty(0,NEGINFINITY);
            //shuffle_max();
            calc32_local_affine_float();
            set_H_E_temp_out_x();
            set_H_E_temp_out_y();
        }
        int final_out = length_2 % 64;
        int from_thread_id = 32 - ((final_out+1)/2);

        //if (blid == 0) {
        //    float2 temp_temp = __half22float2(H_temp_out);
        //    if (thid == 0) printf("Counter: %d, k = %d, length_2 = %d, length_2+group_size = %d, from_thread_id = %d\n", counter,k,lane_2-1, length_2+group_size, from_thread_id);
        //    if (thid>=from_thread_id) printf("Thid: %d, Values: %f, %f, offset_out - from_thread_id: %d\n", thid, temp_temp.x, temp_temp.y, offset_out - from_thread_id);
        //}
        if (thid>=from_thread_id) {
            checkHEindex(offset_out-from_thread_id);
            devTempHcol[offset_out-from_thread_id]=H_temp_out;
            devTempEcol[offset_out-from_thread_id]=E_temp_out;
        }

        //    if (32-thid <= (counter+8)%(group_size/2))
        //         devTempHcol[offset_out-(thid-(32-((counter+8)%(group_size/2))))]=H_temp_out;
            //if (blid == 0) {
        //        float2 temp_temp = __half22float2(devTempHcol[thid]);
            //    printf("Thid: %d, Values: %f, %f\n", thid, temp_temp.x, temp_temp.y);
            //    printf("Thid: %d, Values: %f, %f\n", thid, H_temp_out.x, H_temp_out.y);
            //}
        // Middle passes
        //float2 penalty_left2;

        for (int pass = 1; pass < passes-1; pass++) {
            //maximum = __shfl_sync(0xFFFFFFFF, maximum, 31, 32);
            counter = 1;
            query_letter = 20;
            new_query_letter4 = constantQuery4[thid%group_size];
            if (!thid) query_letter = new_query_letter4.x;

            offset = group_id + group_size;
            offset_out = group_id;
            offset_in = group_id;
            checkHEindex(offset_in);
            H_temp_in = devTempHcol[offset_in];
            E_temp_in = devTempEcol[offset_in];
            offset_in += group_size;

            init_penalties_local(gap_open+(pass*32*numRegs-1)*gap_extend);  // CONTINUE HERE!!!!
            init_local_score_profile_BLOSUM62(pass*(32*numRegs));

            if (!group_id) {
                penalty_left = H_temp_in.x; // penalty_left2.x;
                E = E_temp_in.x; // E_2.x;
            }
            initial_calc32_local_affine_float(1);

            shuffle_query(new_query_letter4.y);
            shuffle_affine_penalty(H_temp_in.y,E_temp_in.y);
            shuffle_H_E_temp_in();
            //shuffle_max();
            calc32_local_affine_float();

            shuffle_query(new_query_letter4.z);
            shuffle_affine_penalty(H_temp_in.x,E_temp_in.x);
            //shuffle_max();
            calc32_local_affine_float();

            shuffle_query(new_query_letter4.w);
            shuffle_affine_penalty(H_temp_in.y,E_temp_in.y);
            shuffle_new_query();
            counter++;
            shuffle_H_E_temp_in();

            for (k = 4; k <= 28; k+=4) {
                calc32_local_affine_float();
                shuffle_query(new_query_letter4.x);
                shuffle_affine_penalty(H_temp_in.x,E_temp_in.x);

                calc32_local_affine_float();
                shuffle_query(new_query_letter4.y);
                shuffle_affine_penalty(H_temp_in.y,E_temp_in.y);
                shuffle_H_E_temp_in();

                calc32_local_affine_float();
                shuffle_query(new_query_letter4.z);
                shuffle_affine_penalty(H_temp_in.x,E_temp_in.x);

                calc32_local_affine_float();
                shuffle_query(new_query_letter4.w);
                shuffle_affine_penalty(H_temp_in.y,E_temp_in.y);
                shuffle_new_query();
                shuffle_H_E_temp_in();
                counter++;
            }
            //   for (k = 5; k < length_2+group_size-2; k+=4) {
            for (k = 32; k <= length_2+28; k+=4) {
                //shuffle_max();
                calc32_local_affine_float();
                set_H_E_temp_out_x();
                shuffle_query(new_query_letter4.x);
                shuffle_affine_penalty(H_temp_in.x,E_temp_in.x);

                //shuffle_max();
                calc32_local_affine_float();
                set_H_E_temp_out_y();
                shuffle_H_E_temp_out();
                shuffle_query(new_query_letter4.y);
                shuffle_affine_penalty(H_temp_in.y,E_temp_in.y);
                shuffle_H_E_temp_in();

                //shuffle_max();
                calc32_local_affine_float();
                set_H_E_temp_out_x();
                shuffle_query(new_query_letter4.z);
                shuffle_affine_penalty(H_temp_in.x,E_temp_in.x);

                //shuffle_max();
                calc32_local_affine_float();
                set_H_E_temp_out_y();

                if ((counter+8)%16 == 0 && counter > 8) {
                    checkHEindex(offset_out);
                    devTempHcol[offset_out]=H_temp_out;
                    devTempEcol[offset_out]=E_temp_out;
                    offset_out += group_size;
                }
                shuffle_H_E_temp_out();
                shuffle_query(new_query_letter4.w);
                //shuffle_affine_penalty(penalty_left2.y,E_2.y);
                shuffle_affine_penalty(H_temp_in.y,E_temp_in.y);
                shuffle_new_query();
                if (counter%group_size == 0) {
                    new_query_letter4 = constantQuery4[offset];
                    offset += group_size;
                }
                shuffle_H_E_temp_in();
                if (counter%16 == 0) {
                    checkHEindex(offset_in);
                    H_temp_in = devTempHcol[offset_in];
                    E_temp_in = devTempEcol[offset_in];
                    offset_in += group_size;
                }
                counter++;
            }
            if (length_2 % 4 == 0) {
                temp = __shfl_up_sync(0xFFFFFFFF, *((int*)(&H_temp_out)), 1, 32);
                H_temp_out = *((short2*)(&temp));
                temp = __shfl_up_sync(0xFFFFFFFF, *((int*)(&E_temp_out)), 1, 32);
                E_temp_out = *((short2*)(&temp));
            }
            if (length_2 % 4 == 1) {
                //shuffle_max();
                calc32_local_affine_float();
                set_H_E_temp_out_x();
                set_H_E_temp_out_y();
            }
            if (length_2 % 4 == 2) {
                //shuffle_max();
                // if (blid == 12455 && pass == (passes-2)) {
                //       if (thid == 31) printf("Before calc32, pass: %d, Counter2: %d\n", pass, counter2);
                    //   if (thid == 31) {
                    //       printf("Thread: %d, Maximum: %f\n", thid, maximum);
                    //       for (int i=0; i<numRegs; i++) printf("Reg: %i: %f\n", thid, penalty_here_array[i]);
                    //       printf("pen_left: %f, E: %f\n", penalty_left, E);

                    //   }
                //}
                calc32_local_affine_float();
                set_H_E_temp_out_x();
                shuffle_query(new_query_letter4.x);
                shuffle_affine_penalty(H_temp_in.x,E_temp_in.x);
                //shuffle_max();
                calc32_local_affine_float();

                set_H_E_temp_out_y();

            }
            if (length_2 % 4 == 3) {
                //shuffle_max();
                calc32_local_affine_float();
                set_H_E_temp_out_x();
                shuffle_query(new_query_letter4.x);
                shuffle_affine_penalty(H_temp_in.x,E_temp_in.x);
                //shuffle_max();
                calc32_local_affine_float();
                set_H_E_temp_out_y();
                shuffle_H_E_temp_out();
                shuffle_query(new_query_letter4.y);
                shuffle_affine_penalty(H_temp_in.y,E_temp_in.y);
                //shuffle_H_E_temp_in();
                //shuffle_max();
                calc32_local_affine_float();
                set_H_E_temp_out_x();
                set_H_E_temp_out_y();
            }
            int final_out = length_2 % 64;
            int from_thread_id = 32 - ((final_out+1)/2);

            if (thid>=from_thread_id) {
                checkHEindex(offset_out-from_thread_id);
                devTempHcol[offset_out-from_thread_id]=H_temp_out;
                devTempEcol[offset_out-from_thread_id]=E_temp_out;
            }
        }
        // Final pass
        //if (passes > 1) {

        //maximum = __shfl_sync(0xFFFFFFFF, maximum, 31, 32);
        counter = 1;
        //counter2 = 1;
        query_letter = 20;
        new_query_letter4 = constantQuery4[thid%group_size];
        if (thid % group_size== 0) query_letter = new_query_letter4.x;

        offset = group_id + group_size;
        //offset_in = group_id + passes*(32*numRegs/2)*blid;
        offset_in = group_id;
        checkHEindex(offset_in);
        H_temp_in = devTempHcol[offset_in];
        E_temp_in = devTempEcol[offset_in];
        offset_in += group_size;

        init_penalties_local(gap_open+((passes-1)*32*numRegs-1)*gap_extend);  // CONTINUE HERE!!!!
        init_local_score_profile_BLOSUM62((passes-1)*(32*numRegs));
        // if (thid % group_size == 0) {
        //       penalty_left2 = __half22float2(H_temp_in);
        //       penalty_left = penalty_left2.x;
        //  }
        //copy_H_E_temp_in();
        if (!group_id) {
            penalty_left = H_temp_in.x; //penalty_left2.x;
            E = E_temp_in.x; //E_2.x;
        }

        initial_calc32_local_affine_float(1);

        shuffle_query(new_query_letter4.y);
        //shuffle_affine_penalty(penalty_left2.y,E_2.y);
        shuffle_affine_penalty(H_temp_in.y,E_temp_in.y);
        shuffle_H_E_temp_in();

        if (length_2+thread_result >=2) {
            //shuffle_max();
            calc32_local_affine_float();
            shuffle_query(new_query_letter4.z);
            shuffle_affine_penalty(H_temp_in.x,E_temp_in.x);
            }

        if (length_2+thread_result >=3) {
                //shuffle_max();
                calc32_local_affine_float();
                shuffle_query(new_query_letter4.w);
                shuffle_affine_penalty(H_temp_in.y,E_temp_in.y);
                shuffle_new_query();
                counter++;
                shuffle_H_E_temp_in();
            }
        if (length_2+thread_result >=4) {
            for (k = 4; k <= length_2+(thread_result-3); k+=4) {
            //for (k = 5; k < lane_2+thread_result-2; k+=4) {
                //shuffle_max();
                calc32_local_affine_float();
                shuffle_query(new_query_letter4.x);
                shuffle_affine_penalty(H_temp_in.x,E_temp_in.x);

                //shuffle_max();
                //if (blid == 12455) {
                    //   if (thid == 0) printf("After shuffle, Counter2: %d\n", counter2);
                    //   printf("Thread: %d, Maximum: %f\n", thid, maximum);
                //}

                calc32_local_affine_float();

                shuffle_query(new_query_letter4.y);
                shuffle_affine_penalty(H_temp_in.y,E_temp_in.y);
                shuffle_H_E_temp_in();
                //shuffle_max();
                calc32_local_affine_float();

                shuffle_query(new_query_letter4.z);
                shuffle_affine_penalty(H_temp_in.x,E_temp_in.x);
                //shuffle_max();
                //if (blid == 520971 && thid == 0 && counter2==258) {
                    //   printf("Before 4th calc32, Counter2: %d\n", counter2);
                    //   printf("Thread: %d, Maximum: %f\n", thid, maximum);
                    //   for (int i=0; i<numRegs; i++) printf("Reg: %i: %f\n", thid, penalty_here_array[i]);
                    //   printf("pen_left: %f, E: %f\n", penalty_left, E);
                //}

                calc32_local_affine_float();
                shuffle_query(new_query_letter4.w);
                shuffle_affine_penalty(H_temp_in.y,E_temp_in.y);
                shuffle_new_query();

                if (counter%group_size == 0) {
                    new_query_letter4 = constantQuery4[offset];
                    offset += group_size;
                }
                shuffle_H_E_temp_in();
                if (counter%16 == 0) {
                    checkHEindex(offset_in);
                    H_temp_in = devTempHcol[offset_in];
                    E_temp_in = devTempEcol[offset_in];
                    offset_in += group_size;
                }
                counter++;
            }

            //if (counter2-(length_2+thread_result) > 0) {
            if ((k-1)-(length_2+thread_result) > 0) {
                //shuffle_max();
                calc32_local_affine_float();
                shuffle_query(new_query_letter4.x);
                shuffle_affine_penalty(H_temp_in.x,E_temp_in.x);
                k++;
            }

            //if (counter2-(length_2+thread_result) > 0) {
            if ((k-1)-(length_2+thread_result) > 0) {
                //shuffle_max();
                calc32_local_affine_float();
                shuffle_query(new_query_letter4.y);
                shuffle_affine_penalty(H_temp_in.y,E_temp_in.y);
                shuffle_H_E_temp_in();
                k++;
            }

            //if (counter2-(length_2+thread_result) > 0) {
            if ((k-1)-(length_2+thread_result) > 0) {
                //shuffle_max();
                calc32_local_affine_float();
                //shuffle_query(new_query_letter4.z);
                //shuffle_affine_penalty(H_temp_in.x,E_temp_in.x);
            }

        }
        //max_all_reduce();
    }
      // if (blid == 0) {
    //       if (thid % group_size == thread_result)
    //           printf("Result in Thread: %d, Register: %d, Value: %f, #passes: %d\n", thid, (length-1)%numRegs, penalty_here_array[(length-1)%numRegs], passes);
     //  }


  // if (thid % group_size == thread_result)
    //  printf("Result in Block: %d, in Thread: %d, Register: %d, Value: %f, #passes: %d\n", blid, thid, (length-1)%numRegs, penalty_here_array[(length-1)%numRegs], passes);
    for (int offset=group_size/2; offset>0; offset/=2)
           maximum = max(maximum,__shfl_down_sync(0xFFFFFFFF,maximum,offset,group_size));
   if (!group_id)
      //devAlignmentScores[(32/group_size)*blid+thid/group_size] =  maximum;
      devAlignmentScores[d_positions_of_selected_lengths[blid]] =  maximum;
}


// Needleman-Wunsch (NW): global alignment with linear gap penalty
// numRegs values per thread
// uses a single warp per CUDA thread block;
// every groupsize threads computes an alignmen score
template <int group_size, int numRegs> __global__
void NW_local_affine_multi_pass_s32_DPX(
    const char * devChars,
    float * devAlignmentScores,
    short2 * devTempHcol2,
    short2 * devTempEcol2,
    const size_t* devOffsets,
    const size_t* devLengths,
    const size_t* d_positions_of_selected_lengths,
    const int length_2,
    const int gap_open,
    const int gap_extend
) {

    __shared__ int shared_BLOSUM62[21][21];
    int subject[numRegs];

    const int NEGINFINITY = -10000;
    const int blid = blockIdx.x;
    const int thid = threadIdx.x;
    const int group_id = thid%group_size;
    int offset = group_id + group_size;

    int test_length =  devLengths[d_positions_of_selected_lengths[blid]];//devLengths[(32/group_size)*blid+thid/group_size];
    const int length = abs(test_length);
    const int base = devOffsets[d_positions_of_selected_lengths[blid]]; // devOffsets[(32/group_size)*blid+thid/group_size];

    short2 H_temp_out, H_temp_in;
    short2 E_temp_out, E_temp_in;
    int penalty_temp0, penalty_temp1; // penalty_temp2;
    int penalty_left, penalty_diag;
    int penalty_here31;
    //constexpr int numRegs = 32;
    int penalty_here_array[numRegs];
    int F_here_array[numRegs];
    int E = NEGINFINITY;

    //const int base_3 = d_positions_of_selected_lengths[blid]*length_2; // blid*length_2;
	const int base_3 = blid*length_2; // blid*length_2;

    short2 * devTempHcol = (short2*)(&devTempHcol2[base_3]);
    short2 * devTempEcol = (short2*)(&devTempEcol2[base_3]);
    int maximum = 0;
    //const float ZERO = 0;

    auto init_penalties_local = [&](const auto& value) {
        //ZERO = NEGINFINITY;
        penalty_left = NEGINFINITY;
        penalty_diag = NEGINFINITY;
        E = NEGINFINITY;
        #pragma unroll
        for (int i=0; i<numRegs; i++) penalty_here_array[i] = NEGINFINITY;
        for (int i=0; i<numRegs; i++) F_here_array[i] = NEGINFINITY;
        if (!thid) {
            penalty_left = 0;
            penalty_diag = 0;
            #pragma unroll
            for (int i=0; i<numRegs; i++) penalty_here_array[i] = 0;
        }
        if (thid % group_size == 1) {
            penalty_left = 0;
        }
    };

    int temp0; // temp1, E_temp_float, H_temp_float;

    auto init_local_score_profile_BLOSUM62 = [&](const auto& offset_isc) {

        if (offset_isc == 0) {
            for (int i=thid; i<21*21; i+=32) shared_BLOSUM62[i/21][i%21]=cBLOSUM62_dev[i];
            __syncwarp();
        }

        for (int i=0; i<numRegs; i++) {
            if (offset_isc+numRegs*(thid%group_size)+i >= length) subject[i] = 20;
            else subject[i] = devChars[offset_isc+base+numRegs*(thid%group_size)+i];
        }
    };


    //H_temp_out = __floats2half2_rn(NEGINFINITY,NEGINFINITY);
    H_temp_out.x = -30000; H_temp_out.y = -30000;
    E_temp_out.x = -30000; E_temp_out.y = -30000;
    int counter = 1;
    //int counter2 = 1;
    char query_letter = 20;
    char4 new_query_letter4 = constantQuery4[thid%group_size];
    if (!thid) query_letter = new_query_letter4.x;

    int score_temp;
    int *sbt_row;

    auto initial_calc32_local_affine_float = [&](const auto& value){
        sbt_row = shared_BLOSUM62[query_letter];

        score_temp = sbt_row[subject[0]];
        penalty_temp0 = penalty_here_array[0];

        if (!value || (thid%group_size)) E = max(E+gap_extend, penalty_left+gap_open);
        F_here_array[0] = max(F_here_array[0]+gap_extend, penalty_here_array[0]+gap_open);
        penalty_here_array[0] = __vimax3_s32_relu(penalty_diag + score_temp,E,F_here_array[0]);

        score_temp = sbt_row[subject[1]];
        penalty_temp1 = penalty_here_array[1];
        E = max(E+gap_extend, penalty_here_array[0]+gap_open);
        F_here_array[1] = max(F_here_array[1]+gap_extend, penalty_here_array[1]+gap_open);
        penalty_here_array[1] = __vimax3_s32_relu(penalty_temp0+score_temp,E,F_here_array[1]);
		maximum = __vimax3_s32(maximum,penalty_here_array[0],penalty_here_array[1]);


        #pragma unroll
        for (int i=1; i<numRegs/2; i++) {
            score_temp = sbt_row[subject[2*i]];
            penalty_temp0 = penalty_here_array[2*i];
            E = max(E+gap_extend, penalty_here_array[2*i-1]+gap_open);
            F_here_array[2*i] = max(F_here_array[2*i]+gap_extend, penalty_here_array[2*i]+gap_open);
            penalty_here_array[2*i] = __vimax3_s32_relu(penalty_temp1+score_temp,E,F_here_array[2*i]);

            score_temp = sbt_row[subject[2*i+1]];
            penalty_temp1 = penalty_here_array[2*i+1];
            E = max(E+gap_extend, penalty_here_array[2*i]+gap_open);
            F_here_array[2*i+1] = max(F_here_array[2*i+1]+gap_extend, penalty_here_array[2*i+1]+gap_open);
            penalty_here_array[2*i+1] = __vimax3_s32_relu(penalty_temp0+score_temp,E,F_here_array[2*i+1]);
			maximum = __vimax3_s32(maximum,penalty_here_array[2*i],penalty_here_array[2*i+1]);
        }

        penalty_here31 = penalty_here_array[numRegs-1];
        E = max(E+gap_extend, penalty_here31+gap_open);
        for (int i=0; i<numRegs; i++) F_here_array[i] = max(F_here_array[i]+gap_extend, penalty_here_array[i]+gap_open);
    };


    auto calc32_local_affine_float = [&](){
        sbt_row = shared_BLOSUM62[query_letter];

        score_temp = sbt_row[subject[0]];
        penalty_temp0 = penalty_here_array[0];

        penalty_here_array[0] = __vimax3_s32_relu(penalty_diag + score_temp,E,F_here_array[0]); // max(max(penalty_diag + score_temp, max(E, F_here_array[0])),ZERO);
        penalty_temp1 = penalty_here_array[0]+gap_open;
        E = __viaddmax_s32(E,gap_extend,penalty_temp1); //E = max(E+gap_extend, penalty_temp1);
        F_here_array[0] = __viaddmax_s32(F_here_array[0],gap_extend,penalty_temp1);

        score_temp = sbt_row[subject[1]];
        penalty_temp1 = penalty_here_array[1];
        penalty_here_array[1] = __vimax3_s32_relu(penalty_temp0+score_temp,E,F_here_array[1]);
        penalty_temp0 = penalty_here_array[1]+gap_open;
        E = __viaddmax_s32(E,gap_extend,penalty_temp0);
        F_here_array[1] = __viaddmax_s32(F_here_array[1],gap_extend,penalty_temp0);
        maximum = __vimax3_s32(maximum,penalty_here_array[0],penalty_here_array[1]);

        #pragma unroll
        for (int i=1; i<numRegs/2; i++) {
            score_temp = sbt_row[subject[2*i]];
            penalty_temp0 = penalty_here_array[2*i];
            penalty_here_array[2*i] = __vimax3_s32_relu(penalty_temp1+score_temp,E,F_here_array[2*i]); // max(max(penalty_diag + score_temp, max(E, F_here_array[0])),ZERO);
            penalty_temp1 = penalty_here_array[2*i]+gap_open;
            E = __viaddmax_s32(E,gap_extend,penalty_temp1); //E = max(E+gap_extend, penalty_temp1);
            F_here_array[2*i] = __viaddmax_s32(F_here_array[2*i],gap_extend,penalty_temp1);// max(F_here_array[0]+gap_extend, penalty_temp1);

            score_temp = sbt_row[subject[2*i+1]];
            penalty_temp1 = penalty_here_array[2*i+1];
            penalty_here_array[2*i+1] = __vimax3_s32_relu(penalty_temp0+score_temp,E,F_here_array[2*i+1]);
            penalty_temp0 = penalty_here_array[2*i+1]+gap_open;
            E = __viaddmax_s32(E,gap_extend,penalty_temp0);
            F_here_array[2*i+1] = __viaddmax_s32(F_here_array[2*i+1],gap_extend,penalty_temp0);
            maximum = __vimax3_s32(maximum,penalty_here_array[2*i],penalty_here_array[2*i+1]);

        }
        //for (int i=0; i<numRegs/4; i++)
        //    maximum = __vimax3_s32(maximum,max(penalty_here_array[4*i],penalty_here_array[4*i+1]),max(penalty_here_array[4*i+2],penalty_here_array[4*i+3]));
        penalty_here31 = penalty_here_array[numRegs-1];
    };


    const int passes = ceil((1.0*length)/(group_size*numRegs));
    int offset_out = group_id;
    int offset_in = group_id;

    //printf("%d, %d, Passes: %d, lenght1: %d, length_2: %d\n", blid, thid, passes, length, length_2);

    auto shuffle_query = [&](const auto& new_letter) {
        query_letter = __shfl_up_sync(0xFFFFFFFF, query_letter, 1, 32);
        if (!thid) query_letter = new_letter;
    };

    auto shuffle_affine_penalty = [&](const auto& new_penalty_left, const auto& new_E_left) {
        penalty_diag = penalty_left;
        penalty_left = __shfl_up_sync(0xFFFFFFFF, penalty_here31, 1, 32);
        E = __shfl_up_sync(0xFFFFFFFF, E, 1, 32);
        if (!thid) {
            penalty_left = new_penalty_left;
            E = new_E_left;
        }
    };

    uint32_t temp;
    auto shuffle_H_E_temp_out = [&]() {
        temp = __shfl_down_sync(0xFFFFFFFF, *((int*)(&H_temp_out)), 1, 32);
        H_temp_out = *((short2*)(&temp));
        temp = __shfl_down_sync(0xFFFFFFFF, *((int*)(&E_temp_out)), 1, 32);
        E_temp_out = *((short2*)(&temp));
    };

    auto shuffle_H_E_temp_in = [&]() {
        temp = __shfl_down_sync(0xFFFFFFFF, *((int*)(&H_temp_in)), 1, 32);
        H_temp_in = *((short2*)(&temp));
        temp = __shfl_down_sync(0xFFFFFFFF, *((int*)(&E_temp_in)), 1, 32);
        E_temp_in = *((short2*)(&temp));
    };

    auto shuffle_new_query = [&]() {
        temp = __shfl_down_sync(0xFFFFFFFF, *((int*)(&new_query_letter4)), 1, 32);
        new_query_letter4 = *((char4*)(&temp));
    };

    auto set_H_E_temp_out_x = [&]() {
        if (thid == 31) {
            H_temp_out.x = penalty_here31;
            E_temp_out.x = E;
        }
    };

    auto set_H_E_temp_out_y = [&]() {
        if (thid  == 31) {
            H_temp_out.y = penalty_here31;
            E_temp_out.y = E;
        }
    };

    int k;

    const uint32_t thread_result = ((length-1)%(group_size*numRegs))/numRegs;


    if (passes == 1) {
        // Single pass
    //    for (int k = 5; k < lane_2+group_size-length_2%4; k+=4) {
    //    for (int k = 5; k < lane_2+thread_result+1-thread_result%4; k+=4) {
    init_penalties_local(0);
    init_local_score_profile_BLOSUM62(0);

    initial_calc32_local_affine_float(0);
    shuffle_query(new_query_letter4.y);
    shuffle_affine_penalty(0, NEGINFINITY);

    if (length_2+thread_result >=2) {
        //shuffle_max();
        calc32_local_affine_float();
        shuffle_query(new_query_letter4.z);
        shuffle_affine_penalty(0, NEGINFINITY);
    }

    if (length_2+thread_result >=3) {
        //shuffle_max();
        calc32_local_affine_float();
        shuffle_query(new_query_letter4.w);
        shuffle_affine_penalty(0, NEGINFINITY);
        shuffle_new_query();
        counter++;

    }
    if (length_2+thread_result >=4) {
    //for (k = 5; k < lane_2+thread_result-2; k+=4) {
    for (k = 4; k <= length_2+(thread_result-3); k+=4) {
            //shuffle_max();
            calc32_local_affine_float();

            shuffle_query(new_query_letter4.x);
            shuffle_affine_penalty(0,NEGINFINITY);

            //shuffle_max();
            calc32_local_affine_float();
            shuffle_query(new_query_letter4.y);
            shuffle_affine_penalty(0,NEGINFINITY);

            //shuffle_max();
            calc32_local_affine_float();
            shuffle_query(new_query_letter4.z);
            shuffle_affine_penalty(0,NEGINFINITY);

            //shuffle_max();
            calc32_local_affine_float();
            shuffle_query(new_query_letter4.w);
            shuffle_affine_penalty(0,NEGINFINITY);
            shuffle_new_query();
            if (counter%group_size == 0) {
                new_query_letter4 = constantQuery4[offset];
                offset += group_size;
            }
            counter++;
        }

       //if (counter2-(length_2+thread_result) > 0) {
       if ((k-1)-(length_2+thread_result) > 0) {
            //shuffle_max();
            calc32_local_affine_float();
            shuffle_query(new_query_letter4.x);
            shuffle_affine_penalty(0,NEGINFINITY);
            k++;
        }

        //if (counter2-(length_2+thread_result) > 0) {
        if ((k-1)-(length_2+thread_result) > 0) {
            //shuffle_max();
            calc32_local_affine_float();
            shuffle_query(new_query_letter4.y);
            shuffle_affine_penalty(0,NEGINFINITY);
            k++;
        }

        //if (counter2-(length_2+thread_result) > 0) {
        if ((k-1)-(length_2+thread_result) > 0) {
            //shuffle_max();
            calc32_local_affine_float();
            //shuffle_query(new_query_letter4.z);
            //shuffle_affine_penalty(0,NEGINFINITY);
        }

    }
    }
    else {

    // first pass (of multiple passes)
    init_penalties_local(0);
    init_local_score_profile_BLOSUM62(0);
    initial_calc32_local_affine_float(0);
    shuffle_query(new_query_letter4.y);
    shuffle_affine_penalty(0, NEGINFINITY);

    //shuffle_max();
    calc32_local_affine_float();
    shuffle_query(new_query_letter4.z);
    shuffle_affine_penalty(0, NEGINFINITY);

    //shuffle_max();
    calc32_local_affine_float();
    shuffle_query(new_query_letter4.w);
    shuffle_affine_penalty(0, NEGINFINITY);
    shuffle_new_query();
    counter++;
    for (k = 4; k <= 28; k+=4) {
        calc32_local_affine_float();
        shuffle_query(new_query_letter4.x);
        shuffle_affine_penalty(0,NEGINFINITY);

        calc32_local_affine_float();
        shuffle_query(new_query_letter4.y);
        shuffle_affine_penalty(0,NEGINFINITY);

        calc32_local_affine_float();
        shuffle_query(new_query_letter4.z);
        shuffle_affine_penalty(0,NEGINFINITY);

        calc32_local_affine_float();
        shuffle_query(new_query_letter4.w);
        shuffle_affine_penalty(0,NEGINFINITY);
        shuffle_new_query();
        counter++;
    }
    //for (k = 4; k <= length_2+28; k+=4) {
    for (k = 32; k <= length_2+28; k+=4) {
        //shuffle_max();
        calc32_local_affine_float();
        set_H_E_temp_out_x();
        shuffle_query(new_query_letter4.x);
        shuffle_affine_penalty(0,NEGINFINITY);

        //shuffle_max();
        calc32_local_affine_float();

        set_H_E_temp_out_y();
        shuffle_H_E_temp_out();
        shuffle_query(new_query_letter4.y);
        shuffle_affine_penalty(0,NEGINFINITY);

        //shuffle_max();
        calc32_local_affine_float();

        set_H_E_temp_out_x();
        shuffle_query(new_query_letter4.z);
        shuffle_affine_penalty(0,NEGINFINITY);

        //shuffle_max();
        calc32_local_affine_float();

        set_H_E_temp_out_y();

        if ((counter+8)%16 == 0 && counter > 8) {
            devTempHcol[offset_out]=H_temp_out;
            devTempEcol[offset_out]=E_temp_out;
            offset_out += group_size;

        }
        shuffle_H_E_temp_out();
        shuffle_query(new_query_letter4.w);
        shuffle_affine_penalty(0,NEGINFINITY);
        shuffle_new_query();
        if (counter%group_size == 0) {
            new_query_letter4 = constantQuery4[offset];
            offset += group_size;
        }
        counter++;
    }
    if (length_2 % 4 == 0) {
        temp = __shfl_up_sync(0xFFFFFFFF, *((int*)(&H_temp_out)), 1, 32);
        H_temp_out = *((short2*)(&temp));
        temp = __shfl_up_sync(0xFFFFFFFF, *((int*)(&E_temp_out)), 1, 32);
        E_temp_out = *((short2*)(&temp));
    }
    if (length_2 % 4 == 1) {
        //shuffle_max();
        calc32_local_affine_float();
        set_H_E_temp_out_x();
        set_H_E_temp_out_y();
    }
    if (length_2 % 4 == 2) {
        //shuffle_max();
        calc32_local_affine_float();
        set_H_E_temp_out_x();
        shuffle_query(new_query_letter4.x);
        shuffle_affine_penalty(0,NEGINFINITY);
        //shuffle_max();
        calc32_local_affine_float();
        set_H_E_temp_out_y();
    }
    if (length_2 % 4 == 3) {
        //shuffle_max();
        calc32_local_affine_float();
        set_H_E_temp_out_x();
        shuffle_query(new_query_letter4.x);
        shuffle_affine_penalty(0,NEGINFINITY);
        //shuffle_max();
        calc32_local_affine_float();
        set_H_E_temp_out_y();
        shuffle_H_E_temp_out();
        shuffle_query(new_query_letter4.y);
        shuffle_affine_penalty(0,NEGINFINITY);
        //shuffle_max();
        calc32_local_affine_float();
        set_H_E_temp_out_x();
        set_H_E_temp_out_y();
    }
    int final_out = length_2 % 64;
    int from_thread_id = 32 - ((final_out+1)/2);

    if (thid>=from_thread_id) {
        devTempHcol[offset_out-from_thread_id]=H_temp_out;
        devTempEcol[offset_out-from_thread_id]=E_temp_out;
    }

   for (int pass = 1; pass < passes-1; pass++) {
        //maximum = __shfl_sync(0xFFFFFFFF, maximum, 31, 32);
        counter = 1;
        query_letter = 20;
        new_query_letter4 = constantQuery4[thid%group_size];
        if (!thid) query_letter = new_query_letter4.x;

        offset = group_id + group_size;
        offset_out = group_id;
        offset_in = group_id;
        H_temp_in = devTempHcol[offset_in];
        E_temp_in = devTempEcol[offset_in];
        offset_in += group_size;

       init_penalties_local(gap_open+(pass*32*numRegs-1)*gap_extend);  // CONTINUE HERE!!!!
       init_local_score_profile_BLOSUM62(pass*(32*numRegs));

       if (!group_id) {
           penalty_left = H_temp_in.x; // penalty_left2.x;
           E = E_temp_in.x; // E_2.x;
       }
       initial_calc32_local_affine_float(1);

       shuffle_query(new_query_letter4.y);
       shuffle_affine_penalty(H_temp_in.y,E_temp_in.y);
       shuffle_H_E_temp_in();
       //shuffle_max();
       calc32_local_affine_float();

       shuffle_query(new_query_letter4.z);
       shuffle_affine_penalty(H_temp_in.x,E_temp_in.x);
       //shuffle_max();
       calc32_local_affine_float();

       shuffle_query(new_query_letter4.w);
       shuffle_affine_penalty(H_temp_in.y,E_temp_in.y);
       shuffle_new_query();
       counter++;
       shuffle_H_E_temp_in();

       for (k = 4; k <= 28; k+=4) {
              calc32_local_affine_float();
              shuffle_query(new_query_letter4.x);
              shuffle_affine_penalty(H_temp_in.x,E_temp_in.x);

              calc32_local_affine_float();
              shuffle_query(new_query_letter4.y);
              shuffle_affine_penalty(H_temp_in.y,E_temp_in.y);
              shuffle_H_E_temp_in();

              calc32_local_affine_float();
              shuffle_query(new_query_letter4.z);
              shuffle_affine_penalty(H_temp_in.x,E_temp_in.x);

              calc32_local_affine_float();
              shuffle_query(new_query_letter4.w);
              shuffle_affine_penalty(H_temp_in.y,E_temp_in.y);
              shuffle_new_query();
              shuffle_H_E_temp_in();
              counter++;
          }
    //   for (k = 5; k < length_2+group_size-2; k+=4) {
    for (k = 32; k <= length_2+28; k+=4) {
           //shuffle_max();
           calc32_local_affine_float();
           set_H_E_temp_out_x();
           shuffle_query(new_query_letter4.x);
           shuffle_affine_penalty(H_temp_in.x,E_temp_in.x);

           //shuffle_max();
           calc32_local_affine_float();
           set_H_E_temp_out_y();
           shuffle_H_E_temp_out();
           shuffle_query(new_query_letter4.y);
           shuffle_affine_penalty(H_temp_in.y,E_temp_in.y);
           shuffle_H_E_temp_in();

           //shuffle_max();
           calc32_local_affine_float();
           set_H_E_temp_out_x();
           shuffle_query(new_query_letter4.z);
           shuffle_affine_penalty(H_temp_in.x,E_temp_in.x);

           //shuffle_max();
           calc32_local_affine_float();
           set_H_E_temp_out_y();

           if ((counter+8)%16 == 0 && counter > 8) {
               devTempHcol[offset_out]=H_temp_out;
               devTempEcol[offset_out]=E_temp_out;
               offset_out += group_size;
           }
           shuffle_H_E_temp_out();
           shuffle_query(new_query_letter4.w);
           //shuffle_affine_penalty(penalty_left2.y,E_2.y);
           shuffle_affine_penalty(H_temp_in.y,E_temp_in.y);
           shuffle_new_query();
           if (counter%group_size == 0) {
               new_query_letter4 = constantQuery4[offset];
               offset += group_size;
           }
           shuffle_H_E_temp_in();
           if (counter%16 == 0) {
               H_temp_in = devTempHcol[offset_in];
               E_temp_in = devTempEcol[offset_in];
               offset_in += group_size;
           }
           counter++;
       }
       if (length_2 % 4 == 0) {
           temp = __shfl_up_sync(0xFFFFFFFF, *((int*)(&H_temp_out)), 1, 32);
           H_temp_out = *((short2*)(&temp));
           temp = __shfl_up_sync(0xFFFFFFFF, *((int*)(&E_temp_out)), 1, 32);
           E_temp_out = *((short2*)(&temp));
       }
       if (length_2 % 4 == 1) {
           //shuffle_max();
           calc32_local_affine_float();
           set_H_E_temp_out_x();
           set_H_E_temp_out_y();
       }
       if (length_2 % 4 == 2) {
           calc32_local_affine_float();
           set_H_E_temp_out_x();
           shuffle_query(new_query_letter4.x);
           shuffle_affine_penalty(H_temp_in.x,E_temp_in.x);
           //shuffle_max();
           calc32_local_affine_float();

           set_H_E_temp_out_y();

       }
       if (length_2 % 4 == 3) {
           //shuffle_max();
           calc32_local_affine_float();
           set_H_E_temp_out_x();
           shuffle_query(new_query_letter4.x);
           shuffle_affine_penalty(H_temp_in.x,E_temp_in.x);
           //shuffle_max();
           calc32_local_affine_float();
           set_H_E_temp_out_y();
           shuffle_H_E_temp_out();
           shuffle_query(new_query_letter4.y);
           shuffle_affine_penalty(H_temp_in.y,E_temp_in.y);
           //shuffle_H_E_temp_in();
           //shuffle_max();
           calc32_local_affine_float();
           set_H_E_temp_out_x();
           set_H_E_temp_out_y();
       }
       int final_out = length_2 % 64;
       int from_thread_id = 32 - ((final_out+1)/2);

       if (thid>=from_thread_id) {
           devTempHcol[offset_out-from_thread_id]=H_temp_out;
           devTempEcol[offset_out-from_thread_id]=E_temp_out;
       }
   }
   // Final pass
   //if (passes > 1) {

       //maximum = __shfl_sync(0xFFFFFFFF, maximum, 31, 32);
       counter = 1;
       //counter2 = 1;
       query_letter = 20;
       new_query_letter4 = constantQuery4[thid%group_size];
       if (thid % group_size== 0) query_letter = new_query_letter4.x;

       offset = group_id + group_size;
       //offset_in = group_id + passes*(32*numRegs/2)*blid;
       offset_in = group_id;
       H_temp_in = devTempHcol[offset_in];
       E_temp_in = devTempEcol[offset_in];
       offset_in += group_size;

       init_penalties_local(gap_open+((passes-1)*32*numRegs-1)*gap_extend);  // CONTINUE HERE!!!!
       init_local_score_profile_BLOSUM62((passes-1)*(32*numRegs));

       if (!group_id) {
           penalty_left = H_temp_in.x; //penalty_left2.x;
           E = E_temp_in.x; //E_2.x;
       }

       initial_calc32_local_affine_float(1);

       shuffle_query(new_query_letter4.y);
       //shuffle_affine_penalty(penalty_left2.y,E_2.y);
       shuffle_affine_penalty(H_temp_in.y,E_temp_in.y);
       shuffle_H_E_temp_in();

       if (length_2+thread_result >=2) {
           //shuffle_max();
           calc32_local_affine_float();
           shuffle_query(new_query_letter4.z);
           shuffle_affine_penalty(H_temp_in.x,E_temp_in.x);
        }

       if (length_2+thread_result >=3) {
            //shuffle_max();
            calc32_local_affine_float();
            shuffle_query(new_query_letter4.w);
            shuffle_affine_penalty(H_temp_in.y,E_temp_in.y);
            shuffle_new_query();
            counter++;
            shuffle_H_E_temp_in();
        }
       if (length_2+thread_result >=4) {
       for (k = 4; k <= length_2+(thread_result-3); k+=4) {
       //for (k = 5; k < lane_2+thread_result-2; k+=4) {
           //shuffle_max();
           calc32_local_affine_float();
           shuffle_query(new_query_letter4.x);
           shuffle_affine_penalty(H_temp_in.x,E_temp_in.x);

           calc32_local_affine_float();

           shuffle_query(new_query_letter4.y);
           shuffle_affine_penalty(H_temp_in.y,E_temp_in.y);
           shuffle_H_E_temp_in();
           //shuffle_max();
           calc32_local_affine_float();

           shuffle_query(new_query_letter4.z);
           shuffle_affine_penalty(H_temp_in.x,E_temp_in.x);

           calc32_local_affine_float();
           shuffle_query(new_query_letter4.w);
           shuffle_affine_penalty(H_temp_in.y,E_temp_in.y);
           shuffle_new_query();

           if (counter%group_size == 0) {
               new_query_letter4 = constantQuery4[offset];
               offset += group_size;
           }
           shuffle_H_E_temp_in();
           if (counter%16 == 0) {
               H_temp_in = devTempHcol[offset_in];
               E_temp_in = devTempEcol[offset_in];
               offset_in += group_size;
           }
           counter++;
       }

      //if (counter2-(length_2+thread_result) > 0) {
      if ((k-1)-(length_2+thread_result) > 0) {
           //shuffle_max();
           calc32_local_affine_float();
           shuffle_query(new_query_letter4.x);
           shuffle_affine_penalty(H_temp_in.x,E_temp_in.x);
           k++;
       }

       //if (counter2-(length_2+thread_result) > 0) {
       if ((k-1)-(length_2+thread_result) > 0) {
           //shuffle_max();
           calc32_local_affine_float();
           shuffle_query(new_query_letter4.y);
           shuffle_affine_penalty(H_temp_in.y,E_temp_in.y);
           shuffle_H_E_temp_in();
           k++;
       }

      //if (counter2-(length_2+thread_result) > 0) {
      if ((k-1)-(length_2+thread_result) > 0) {
           //shuffle_max();
           calc32_local_affine_float();
           //shuffle_query(new_query_letter4.z);
           //shuffle_affine_penalty(H_temp_in.x,E_temp_in.x);
       }

   }
   //max_all_reduce();
   }
      // if (blid == 0) {
    //       if (thid % group_size == thread_result)
    //           printf("Result in Thread: %d, Register: %d, Value: %f, #passes: %d\n", thid, (length-1)%numRegs, penalty_here_array[(length-1)%numRegs], passes);
     //  }


  // if (thid % group_size == thread_result)
    //  printf("Result in Block: %d, in Thread: %d, Register: %d, Value: %f, #passes: %d\n", blid, thid, (length-1)%numRegs, penalty_here_array[(length-1)%numRegs], passes);
    for (int offset=group_size/2; offset>0; offset/=2)
           maximum = max(maximum,__shfl_down_sync(0xFFFFFFFF,maximum,offset,group_size));
   if (!group_id)
      //devAlignmentScores[(32/group_size)*blid+thid/group_size] =  maximum;
      devAlignmentScores[d_positions_of_selected_lengths[blid]] =  maximum;
}


char AA2number(char AA) {
    // ORDER of AminoAcids (NCBI): A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
    if (AA == 'A') return 0;
    if (AA == 'R') return 1;
    if (AA == 'N') return 2;
    if (AA == 'D') return 3;
    if (AA == 'C') return 4;
    if (AA == 'Q') return 5;
    if (AA == 'E') return 6;
    if (AA == 'G') return 7;
    if (AA == 'H') return 8;
    if (AA == 'I') return 9;
    if (AA == 'L') return 10;
    if (AA == 'K') return 11;
    if (AA == 'M') return 12;
    if (AA == 'F') return 13;
    if (AA == 'P') return 14;
    if (AA == 'S') return 15;
    if (AA == 'T') return 16;
    if (AA == 'W') return 17;
    if (AA == 'Y') return 18;
    if (AA == 'V') return 19;
    return 20;
}

__global__
void NW_convert_protein(
    char * devChars,
    const size_t* devOffsets) {

    const int blid = blockIdx.x;
    const int thid = threadIdx.x;
    const int base = devOffsets[blid];
    const int length = devOffsets[blid+1] - devOffsets[blid];

    for (int i = thid; i<length; i+=blockDim.x) {
        if (i < length) {
            char AA = devChars[base+i];
            // ORDER of AminoAcids (NCBI): A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
            if (AA == 'A') AA = 0;
            else if (AA == 'R') AA = 1;
            else if (AA == 'N') AA = 2;
            else if (AA == 'D') AA = 3;
            else if (AA == 'C') AA = 4;
            else if (AA == 'Q') AA = 5;
            else if (AA == 'E') AA = 6;
            else if (AA == 'G') AA = 7;
            else if (AA == 'H') AA = 8;
            else if (AA == 'I') AA = 9;
            else if (AA == 'L') AA = 10;
            else if (AA == 'K') AA = 11;
            else if (AA == 'M') AA = 12;
            else if (AA == 'F') AA = 13;
            else if (AA == 'P') AA = 14;
            else if (AA == 'S') AA = 15;
            else if (AA == 'T') AA = 16;
            else if (AA == 'W') AA = 17;
            else if (AA == 'Y') AA = 18;
            else if (AA == 'V') AA = 19;
            else AA = 20;
            devChars[base+i] = AA;
        }
    }
}

void perumte_columns_BLOSUM(
	const char BLOSUM_in[],
	const int alphabet_size,
	const int permutation[],
	char BLOSUM_out[]) {
	for (int col=0; col<alphabet_size; col++)
	    for (int row=0; row<alphabet_size; row++)
			BLOSUM_out[row*alphabet_size+col] = BLOSUM_in[row*alphabet_size+permutation[col]];
}


int affine_local_DP_host_protein(
    const char* seq1,
    const char* seq2,
    const int length1,
    const int length2,
    const int gap_open,
    const int gap_extend,
    const int BLOSUM[][21]) {

    //cout << "Align Seq 1: ";
    //copy(seq1, seq1+length1, std::ostream_iterator<char>(cout, ""));
    //cout << "\n";
    //cout << "to Seq 2: ";
    //copy(seq2, seq2+length2, std::ostream_iterator<char>(cout, ""));
    //cout << "\n";

    const int NEGINFINITY = -10000;
    int *penalty_H = new int[2*(length2+1)];
    int *penalty_F = new int[2*(length2+1)];
    int E, F, maxi = 0, result;
    penalty_H[0] = 0;
    penalty_F[0] = NEGINFINITY;
    for (int index = 1; index <= length2; index++) {
        penalty_H[index] = 0;
        penalty_F[index] = NEGINFINITY;
    }

    //cout << "Row 0: ";
    //copy(penalty, penalty + length2 + 1, std::ostream_iterator<int>(cout, " "));
    //cout << "\n";

    for (int row = 1; row <= length1; row++) {
        char seq1_char = seq1[row-1];
        char seq2_char;
        // if (seq1_char == 'N') seq1_char = 'T';  // special N-letter treatment to match CUDA code
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
            //if (seq2_char == 'N') seq2_char = 'T';  // special N-letter treatment to match CUDA code
            //const int residue = (seq1_char == seq2_char)? match : mismatch;
            const int residue = BLOSUM[AA2number(seq1_char)][AA2number(seq2_char)];
            E = std::max(E+gap_extend, left+gap_open);
            F = std::max(penalty_F[source_row*(length2+1)+col+0]+gap_extend, abve+gap_open);
            result = std::max(0, std::max(diag + residue, std::max(E, F)));
            penalty_H[target_row*(length2+1)+col] = result;
            if (result > maxi) maxi = result;
            penalty_F[target_row*(length2+1)+col] = F;
        }
        //cout << "Row " << row << ": ";
        //copy(penalty + target_row*(length2+1), penalty + target_row*(length2+1) + length2 + 1, std::ostream_iterator<int>(cout, " "));
        //cout << "\n";
    }
    //const int last_row = length1 & 1;
    //const int result = penalty_H[last_row*(length2+1)+length2];
    delete [] penalty_F;
    delete [] penalty_H;
    return maxi;
}


#include <omp.h>







struct GpuWorkingSet{
    int deviceId;
    char* devChars = nullptr;
    size_t* devOffsets = nullptr;
    size_t* devLengths = nullptr;
    char* devChars_2[2];
    size_t* devOffsets_2[2];
    size_t* devLengths_2[2];
    char* buf_host_Chars_2[2];
    size_t* buf_host_Offsets_2[2];
    size_t* buf_host_Lengths_2[2];
    cudaEvent_t pinnedBufferEvents[2];
    cudaEvent_t deviceBufferEvents[2];
    cudaStream_t dblBufferStreams[2];

    size_t maxTempBytes;
    void* d_tempStorageHE;

    float* devAlignmentScoresFloat = nullptr;
    size_t* d_selectedPositions = nullptr;
    size_t* d_overflow_positions = nullptr;
    int* d_overflow_number = nullptr;
    char* Fillchar = nullptr;
    cudaStream_t hostFuncStream;
    cudaStream_t workStreamForTempUsage;
    cudaEvent_t forkStreamEvent;

    std::vector<cudaStream_t> workStreams;




    int* h_numSelectedPerPartition = nullptr;
    //size_t* dev_sorted_indices = nullptr;

    size_t max_batch_num_sequences;
};



template <int group_size, int numRegs> 
__global__
void launch_process_overflow_alignments_kernel_NW_local_affine_read4_float_query_Protein(
    const int* d_overflow_number,
    short2* d_temp,
    size_t maxTempBytes,
    const char * devChars,
    float * devAlignmentScores,
    const size_t* devOffsets,
    const size_t* devLengths,
    const size_t* d_positions_of_selected_lengths,
    const int queryLength,
    const float gap_open,
    const float gap_extend
){
    const int numOverflow = *d_overflow_number;
    if(numOverflow > 0){
        const size_t tempBytesPerSubjectPerBuffer = sizeof(short2) * queryLength;
        const size_t maxSubjectsPerIteration = std::min(size_t(numOverflow), maxTempBytes / (tempBytesPerSubjectPerBuffer * 2));

        short2* d_tempHcol2 = d_temp;
        short2* d_tempEcol2 = (short2*)(((char*)d_tempHcol2) + maxSubjectsPerIteration * tempBytesPerSubjectPerBuffer);

        const int numIters =  SDIV(numOverflow, maxSubjectsPerIteration);
        for(int iter = 0; iter < numIters; iter++){
            const size_t begin = iter * maxSubjectsPerIteration;
            const size_t end = iter < numIters-1 ? (iter+1) * maxSubjectsPerIteration : numOverflow;
            const size_t num = end - begin;

            cudaMemsetAsync(d_temp, 0, tempBytesPerSubjectPerBuffer * 2 * num, 0);

            // cudaMemsetAsync(d_tempHcol2, 0, tempBytesPerSubjectPerBuffer * num, 0);
            // cudaMemsetAsync(d_tempEcol2, 0, tempBytesPerSubjectPerBuffer * num, 0);

            NW_local_affine_read4_float_query_Protein<32, 12><<<num, 32>>>(
                devChars, 
                devAlignmentScores,
                d_tempHcol2, 
                d_tempEcol2, 
                devOffsets, 
                devLengths, 
                d_positions_of_selected_lengths + begin, 
                queryLength, 
                gap_open, 
                gap_extend
            );
        }
    }
}

struct DoMinus_size_t{
    size_t val;
    __host__ __device__
    DoMinus_size_t(size_t v) : val(v){}

    __host__ __device__
    void operator()(size_t& i){
        i -= val;
    }
};

void processQueryOnGpu(
    GpuWorkingSet& ws,
    const std::vector<DBdataView>& dbPartitions,
    const std::vector<int>& lengthPartitionIds, // dbPartitions[i] belongs to the length partition lengthPartitionIds[i]
    const char* d_query,
    const int queryLength,
    bool isFirstQuery,
    int query_num,
    int64_t avg_length_2,
    int select_datatype,
    int select_dpx,
    bool useExtraThreadForBatchTransfer,
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
    auto copyBatchToDevice = [&](const auto& partition, int doubleBufferIndex){
        cudaStream_t copyToPinnedStream = ws.hostFuncStream;
        cudaStream_t H2DcopyStream = ws.dblBufferStreams[doubleBufferIndex];

        if(!useExtraThreadForBatchTransfer){
            //can only overwrite device buffer if it is no longer in use on workstream
            cudaStreamWaitEvent(H2DcopyStream, ws.deviceBufferEvents[doubleBufferIndex], 0); CUERR;
            //synchronize to avoid overwriting pinned buffer of target before it has been fully transferred
            cudaEventSynchronize(ws.pinnedBufferEvents[doubleBufferIndex]); CUERR;

            std::copy(
                partition.chars() + partition.offsets()[0],
                partition.chars() + partition.offsets()[0] + partition.numChars(),
                ws.buf_host_Chars_2[doubleBufferIndex]);
            cudaMemcpyAsync(
                ws.devChars_2[doubleBufferIndex], 
                ws.buf_host_Chars_2[doubleBufferIndex], 
                partition.numChars(), 
                cudaMemcpyHostToDevice, 
                H2DcopyStream); CUERR
            std::copy(
                partition.offsets(),
                partition.offsets() + partition.numSequences(), 
                ws.buf_host_Offsets_2[doubleBufferIndex]);
            cudaMemcpyAsync(
                ws.devOffsets_2[doubleBufferIndex], 
                ws.buf_host_Offsets_2[doubleBufferIndex], 
                sizeof(size_t) * partition.numSequences(), 
                cudaMemcpyHostToDevice, 
                H2DcopyStream); CUERR
            std::copy(
                partition.lengths(),
                partition.lengths() + partition.numSequences(),
                ws.buf_host_Lengths_2[doubleBufferIndex]);
            cudaMemcpyAsync(
                ws.devLengths_2[doubleBufferIndex], 
                ws.buf_host_Lengths_2[doubleBufferIndex], 
                sizeof(size_t) * partition.numSequences(), 
                cudaMemcpyHostToDevice, 
                H2DcopyStream); CUERR

            //after the last H2D transfer finished, pinned buffer may be reused
            cudaEventRecord(ws.pinnedBufferEvents[doubleBufferIndex], H2DcopyStream);
        }else{
            //can only overwrite device buffer if it is no longer in use on workstream
            cudaStreamWaitEvent(H2DcopyStream, ws.deviceBufferEvents[doubleBufferIndex], 0); CUERR;
            //can only overwrite pinned buffer if previous H2D transfer is complete
            cudaStreamWaitEvent(copyToPinnedStream, ws.pinnedBufferEvents[doubleBufferIndex], 0); CUERR;
            
            CopyParams* copyParams0 = new CopyParams;
            copyParams0->numBuffers = 1;
            copyParams0->src[0] = (const void*)(partition.chars() + partition.offsets()[0]);
            copyParams0->dst[0] = (void*)(ws.buf_host_Chars_2[doubleBufferIndex]);
            copyParams0->bytes[0] = partition.numChars();

            cudaLaunchHostFunc(
                copyToPinnedStream, 
                copyBuffersFunc, 
                copyParams0
            ); CUERR

            //transfer chars to gpu in different stream to overlap d2h transfer with copy to pinned of the other buffers
            cudaEventRecord(ws.forkStreamEvent, copyToPinnedStream);
            cudaStreamWaitEvent(H2DcopyStream, ws.forkStreamEvent, 0);
            cudaMemcpyAsync(
                ws.devChars_2[doubleBufferIndex], 
                ws.buf_host_Chars_2[doubleBufferIndex], 
                partition.numChars(), 
                cudaMemcpyHostToDevice, 
                H2DcopyStream); CUERR

            CopyParams* copyParams1 = new CopyParams;
            copyParams1->numBuffers = 2;
            copyParams1->src[0] = (const void*)(partition.offsets());
            copyParams1->dst[0] = (void*)(ws.buf_host_Offsets_2[doubleBufferIndex]);
            copyParams1->bytes[0] = sizeof(size_t) * partition.numSequences();

            copyParams1->src[1] = (const void*)(partition.lengths());
            copyParams1->dst[1] = (void*)(ws.buf_host_Lengths_2[doubleBufferIndex]);
            copyParams1->bytes[1] = sizeof(size_t) * partition.numSequences();

            cudaLaunchHostFunc(
                copyToPinnedStream,
                copyBuffersFunc, 
                copyParams1
            ); CUERR
            
            cudaEventRecord(ws.forkStreamEvent, copyToPinnedStream);
            cudaStreamWaitEvent(H2DcopyStream, ws.forkStreamEvent, 0);

            cudaMemcpyAsync(
                ws.devOffsets_2[doubleBufferIndex], 
                ws.buf_host_Offsets_2[doubleBufferIndex], 
                sizeof(size_t) * partition.numSequences(), 
                cudaMemcpyHostToDevice, 
                H2DcopyStream); CUERR
            cudaMemcpyAsync(
                ws.devLengths_2[doubleBufferIndex], 
                ws.buf_host_Lengths_2[doubleBufferIndex], 
                sizeof(size_t) * partition.numSequences(), 
                cudaMemcpyHostToDevice, 
                H2DcopyStream); CUERR
            //after the last H2D transfer finished, pinned buffer may be reused
            cudaEventRecord(ws.pinnedBufferEvents[doubleBufferIndex], H2DcopyStream);
        }

        //convert from offsets in dbPartition to offsets in ws.devChars_2[doubleBufferIndex]
        //offsets[i] = offsets[i] - offsets[0]
        thrust::for_each(
            thrust::cuda::par_nosync.on(H2DcopyStream),
            ws.devOffsets_2[doubleBufferIndex],
            ws.devOffsets_2[doubleBufferIndex] + partition.numSequences(),
            DoMinus_size_t{partition.offsets()[0]}
        );
    };

    cudaSetDevice(ws.deviceId); CUERR;

    //create dependency on mainStream
    cudaEventRecord(ws.forkStreamEvent, mainStream); CUERR;
    cudaStreamWaitEvent(ws.workStreamForTempUsage, ws.forkStreamEvent, 0); CUERR;
    cudaStreamWaitEvent(ws.hostFuncStream, ws.forkStreamEvent, 0); CUERR;
    cudaStreamWaitEvent(ws.dblBufferStreams[0], ws.forkStreamEvent, 0); CUERR;
    cudaStreamWaitEvent(ws.dblBufferStreams[1], ws.forkStreamEvent, 0); CUERR;

    int64_t dp_cells = 0;
    int totalOverFlowNumber = 0;

    size_t globalSequenceOffsetOfBatch = 0;

    cudaMemcpyToSymbolAsync(constantQuery4,ws.Fillchar,512*16, 0, cudaMemcpyDeviceToDevice, ws.dblBufferStreams[0]); CUERR

    cudaMemcpyToSymbolAsync(constantQuery4, d_query, queryLength, 0, cudaMemcpyDeviceToDevice, ws.dblBufferStreams[0]); CUERR

    // transfer for first batch
    if ((dbPartitions.size() > 1) || isFirstQuery) {
        copyBatchToDevice(dbPartitions[0], 0);
    }


    dp_cells = avg_length_2 * queryLength;
    //TIMERSTART_CUDA_STREAM(NW_local_affine_half2_query_Protein, mainStream)

    //thrust::device_vector<float> fooscores(54465398, 0);


    //std::cout << "int(dbPartitions.size()) " << int(dbPartitions.size()) << "\n";
    
    //for (int batch=0; batch<int(dbPartitions.size()); batch++) {
    for (int batch=0; batch<int(dbPartitions.size()); batch++) {
        // cout << "Query " << query_num << " Batch " << batch 
        //     << " globalSequenceOffsetOfBatch " << globalSequenceOffsetOfBatch << "\n";
        const int source = batch & 1;
        const int target = 1-source;

        const auto& currentPartition = dbPartitions[batch];
        const int lengthPartitionId = lengthPartitionIds[batch];

        std::fill(ws.h_numSelectedPerPartition, ws.h_numSelectedPerPartition + numLengthPartitions, 0);
        ws.h_numSelectedPerPartition[lengthPartitionId] = currentPartition.numSequences();
        // if(lengthPartitionId == 13){
        //     ws.h_numSelectedPerPartition[lengthPartitionId] = currentPartition.numSequences();
        // }


        assert(currentPartition.numSequences() <= ws.max_batch_num_sequences);

        



        //cout << "Starting NW_local_affine_half2: \n";
        //TIMERSTART_CUDA(NW_local_affine_half2_query_Protein)

        if (!select_datatype) {  // HALF2

            cudaStream_t workStream = ws.workStreamForTempUsage;

            cudaEventRecord(ws.forkStreamEvent, ws.dblBufferStreams[source]); CUERR;
            cudaStreamWaitEvent(workStream, ws.forkStreamEvent, 0); CUERR;

            cudaMemsetAsync(ws.d_overflow_number,0,sizeof(int), workStream);

            //all sequences of the batch belong to the same length partition. use a single index array with counts from 0 to N-1
            auto d_selectedPositions = ws.d_selectedPositions;
            
            //&(ws.d_selectedPositions[0*ws.max_batch_num_sequences]);
            // thrust::sequence(
            //     thrust::cuda::par_nosync.on(workStream),
            //     d_selectedPositions,
            //     d_selectedPositions + currentPartition.numSequences(),
            //     0
            // );
            // auto d_selectedPositions = thrust::make_counting_iterator<size_t>(0);


            if (!select_dpx) {
                //cout << "Starting NW_local_affine_half2: Query " << query_num << " Batch " << batch << "\n";

                //cout << "Starting NW_local_affine_half2: \n";
                //TIMERSTART_CUDA(NW_local_affine_half2_query_Protein)

                const float gop = -11.0;
                const float gex = -1.0;

                auto runAlignmentKernels = [&](float* d_scores, size_t* d_overflow_positions, int* d_overflow_number){

                    if (ws.h_numSelectedPerPartition[0]){NW_local_affine_Protein_single_pass_half2<4, 16><<<(ws.h_numSelectedPerPartition[0]+255)/(2*8*4*4), 32*8*2, 0, workStream>>>(ws.devChars_2[source], d_scores, ws.devOffsets_2[source] , ws.devLengths_2[source], d_selectedPositions, ws.h_numSelectedPerPartition[0], d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex); CUERR }
                    if (ws.h_numSelectedPerPartition[1]){NW_local_affine_Protein_single_pass_half2<8, 16><<<(ws.h_numSelectedPerPartition[1]+127)/(2*8*4*2), 32*8*2, 0, workStream>>>(ws.devChars_2[source], d_scores, ws.devOffsets_2[source] , ws.devLengths_2[source], d_selectedPositions, ws.h_numSelectedPerPartition[1], d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex); CUERR }
                    if (ws.h_numSelectedPerPartition[2]){NW_local_affine_Protein_single_pass_half2<8, 24><<<(ws.h_numSelectedPerPartition[2]+63)/(2*8*4), 32*8, 0, workStream>>>(ws.devChars_2[source], d_scores, ws.devOffsets_2[source] , ws.devLengths_2[source], d_selectedPositions, ws.h_numSelectedPerPartition[2], d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex); CUERR }
                    if (ws.h_numSelectedPerPartition[3]){NW_local_affine_Protein_single_pass_half2<16, 16><<<(ws.h_numSelectedPerPartition[3]+31)/(2*8*2), 32*8, 0, workStream>>>(ws.devChars_2[source], d_scores, ws.devOffsets_2[source] , ws.devLengths_2[source], d_selectedPositions, ws.h_numSelectedPerPartition[3], d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex); CUERR }
                    if (ws.h_numSelectedPerPartition[4]){NW_local_affine_Protein_single_pass_half2<16, 20><<<(ws.h_numSelectedPerPartition[4]+31)/(2*8*2), 32*8, 0, workStream>>>(ws.devChars_2[source], d_scores, ws.devOffsets_2[source] , ws.devLengths_2[source], d_selectedPositions, ws.h_numSelectedPerPartition[4], d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex); CUERR }
                    if (ws.h_numSelectedPerPartition[5]){NW_local_affine_Protein_single_pass_half2<16, 24><<<(ws.h_numSelectedPerPartition[5]+31)/(2*8*2), 32*8, 0, workStream>>>(ws.devChars_2[source], d_scores, ws.devOffsets_2[source] , ws.devLengths_2[source], d_selectedPositions, ws.h_numSelectedPerPartition[5], d_overflow_positions, d_overflow_number, 0, queryLength, gop, gex); CUERR }
                    if (ws.h_numSelectedPerPartition[6]){NW_local_affine_Protein_single_pass_half2<16, 28><<<(ws.h_numSelectedPerPartition[6]+31)/(2*8*2), 32*8, 0, workStream>>>(ws.devChars_2[source], d_scores, ws.devOffsets_2[source] , ws.devLengths_2[source], d_selectedPositions, ws.h_numSelectedPerPartition[6], d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex); CUERR }
                    if (ws.h_numSelectedPerPartition[7]){NW_local_affine_Protein_single_pass_half2<32, 16><<<(ws.h_numSelectedPerPartition[7]+15)/(2*8), 32*8, 0, workStream>>>(ws.devChars_2[source], d_scores, ws.devOffsets_2[source] , ws.devLengths_2[source], d_selectedPositions, ws.h_numSelectedPerPartition[7], d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex); CUERR }
                    if (ws.h_numSelectedPerPartition[8]){NW_local_affine_Protein_single_pass_half2<32, 18><<<(ws.h_numSelectedPerPartition[8]+15)/(2*8), 32*8, 0, workStream>>>(ws.devChars_2[source], d_scores, ws.devOffsets_2[source] , ws.devLengths_2[source], d_selectedPositions, ws.h_numSelectedPerPartition[8], d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex); CUERR }
                    if (ws.h_numSelectedPerPartition[9]){NW_local_affine_Protein_single_pass_half2<32, 20><<<(ws.h_numSelectedPerPartition[9]+15)/(2*8), 32*8, 0, workStream>>>(ws.devChars_2[source], d_scores, ws.devOffsets_2[source] , ws.devLengths_2[source], d_selectedPositions, ws.h_numSelectedPerPartition[9], d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex); CUERR }
                    if (ws.h_numSelectedPerPartition[10]){NW_local_affine_Protein_single_pass_half2<32, 24><<<(ws.h_numSelectedPerPartition[10]+15)/(2*8), 32*8, 0, workStream>>>(ws.devChars_2[source], d_scores, ws.devOffsets_2[source] , ws.devLengths_2[source], d_selectedPositions, ws.h_numSelectedPerPartition[10], d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex); CUERR }
                    if (ws.h_numSelectedPerPartition[11]){NW_local_affine_Protein_single_pass_half2<32, 28><<<(ws.h_numSelectedPerPartition[11]+15)/(2*8), 32*8, 0, workStream>>>(ws.devChars_2[source], d_scores, ws.devOffsets_2[source] , ws.devLengths_2[source], d_selectedPositions, ws.h_numSelectedPerPartition[11], d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex); CUERR }
                    if (ws.h_numSelectedPerPartition[12]){NW_local_affine_Protein_single_pass_half2<32, 32><<<(ws.h_numSelectedPerPartition[12]+15)/(2*8), 32*8, 0, workStream>>>(ws.devChars_2[source], d_scores, ws.devOffsets_2[source] , ws.devLengths_2[source], d_selectedPositions, ws.h_numSelectedPerPartition[12], d_overflow_positions, d_overflow_number, 1, queryLength, gop, gex); CUERR }

                    if (ws.h_numSelectedPerPartition[13]){
                        constexpr int blocksize = 32 * 8;
                        constexpr int groupsize = 32;
                        constexpr int groupsPerBlock = blocksize / groupsize;
                        constexpr int alignmentsPerGroup = 2;
                        constexpr int alignmentsPerBlock = groupsPerBlock * alignmentsPerGroup;
                        
                        const size_t tempBytesPerBlockPerBuffer = sizeof(__half2) * alignmentsPerBlock * queryLength;

                        const size_t maxNumBlocks = ws.maxTempBytes / (tempBytesPerBlockPerBuffer * 2);
                        const size_t maxSubjectsPerIteration = std::min(maxNumBlocks * alignmentsPerBlock, size_t(ws.h_numSelectedPerPartition[13]));

                        const size_t numBlocksPerIteration = SDIV(maxSubjectsPerIteration, alignmentsPerBlock);
                        const size_t requiredTempBytes = tempBytesPerBlockPerBuffer * 2 * numBlocksPerIteration;

                        __half2* d_temp = (__half2*)ws.d_tempStorageHE;
                        __half2* d_tempHcol2 = d_temp;
                        __half2* d_tempEcol2 = (__half2*)(((char*)d_tempHcol2) + requiredTempBytes / 2);

                        const int numIters =  SDIV(ws.h_numSelectedPerPartition[13], maxSubjectsPerIteration);

                        for(int iter = 0; iter < numIters; iter++){
                            const size_t begin = iter * maxSubjectsPerIteration;
                            const size_t end = iter < numIters-1 ? (iter+1) * maxSubjectsPerIteration : ws.h_numSelectedPerPartition[13];
                            const size_t num = end - begin;                      

                            cudaMemsetAsync(d_temp, 0, requiredTempBytes, workStream); CUERR;

                            NW_local_affine_Protein_many_pass_half2<groupsize, 12><<<SDIV(num, alignmentsPerBlock), blocksize, 0, workStream>>>(
                                ws.devChars_2[source], 
                                d_scores, 
                                d_tempHcol2, 
                                d_tempEcol2, 
                                ws.devOffsets_2[source], 
                                ws.devLengths_2[source], 
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

                    if (ws.h_numSelectedPerPartition[14]){
                        const size_t tempBytesPerSubjectPerBuffer = sizeof(short2) * SDIV(queryLength,32) * 32;
                        const size_t maxSubjectsPerIteration = std::min(size_t(ws.h_numSelectedPerPartition[14]), ws.maxTempBytes / (tempBytesPerSubjectPerBuffer * 2));

                        short2* d_temp = (short2*)ws.d_tempStorageHE;
                        short2* d_tempHcol2 = d_temp;
                        short2* d_tempEcol2 = (short2*)(((char*)d_tempHcol2) + maxSubjectsPerIteration * tempBytesPerSubjectPerBuffer);

                        const int numIters =  SDIV(ws.h_numSelectedPerPartition[14], maxSubjectsPerIteration);
                        for(int iter = 0; iter < numIters; iter++){
                            const size_t begin = iter * maxSubjectsPerIteration;
                            const size_t end = iter < numIters-1 ? (iter+1) * maxSubjectsPerIteration : ws.h_numSelectedPerPartition[14];
                            const size_t num = end - begin;

                            cudaMemsetAsync(d_temp, 0, tempBytesPerSubjectPerBuffer * 2 * num, workStream); CUERR;

                            NW_local_affine_read4_float_query_Protein<32, 12><<<num, 32, 0, workStream>>>(
                                ws.devChars_2[source], 
                                d_scores, 
                                d_tempHcol2, 
                                d_tempEcol2, 
                                ws.devOffsets_2[source], 
                                ws.devLengths_2[source], 
                                d_selectedPositions + begin, 
                                queryLength, 
                                gop, 
                                gex
                            ); CUERR 
                        }
                    }
                };

                runAlignmentKernels(&(ws.devAlignmentScoresFloat[globalSequenceOffsetOfBatch]), ws.d_overflow_positions, ws.d_overflow_number);


                // test the kernels
                #if 0
                    constexpr float invalidscore = 123456.0f;
                    constexpr int numRepeats = 20;
                    
                    thrust::device_vector<float> fooscores1(currentPartition.numSequences(), invalidscore);
                    thrust::device_vector<size_t> d_overflow_positions1(currentPartition.numSequences(), 99999999);
                    thrust::device_vector<int> d_overflow_number1(1, 0);

                    runAlignmentKernels(fooscores1.data().get(), d_overflow_positions1.data().get(), d_overflow_number1.data().get());

                    thrust::sort(
                        d_overflow_positions1.begin(),
                        d_overflow_positions1.end()
                    );
                    thrust::scatter(
                        thrust::make_constant_iterator<float>(invalidscore),
                        thrust::make_constant_iterator<float>(invalidscore) + d_overflow_number1[0],
                        d_overflow_positions1.begin(),
                        fooscores1.begin()
                    );

                    for(int repeat = 0; repeat < numRepeats; repeat++){
                        thrust::device_vector<float> fooscores2(currentPartition.numSequences(), invalidscore);
                        thrust::device_vector<size_t> d_overflow_positions2(currentPartition.numSequences(), 99999999);
                        thrust::device_vector<int> d_overflow_number2(1, 0);

                        runAlignmentKernels(fooscores2.data().get(), d_overflow_positions2.data().get(), d_overflow_number2.data().get());

                        
                        thrust::sort(
                            d_overflow_positions2.begin(),
                            d_overflow_positions2.end()
                        );                        

                        thrust::scatter(
                            thrust::make_constant_iterator<float>(invalidscore),
                            thrust::make_constant_iterator<float>(invalidscore) + d_overflow_number2[0],
                            d_overflow_positions2.begin(),
                            fooscores2.begin()
                        );

                        if(d_overflow_number1 != d_overflow_number2){
                           std::cout << " Batch: " << batch << ", numseqs " << dbPartitions[batch].numSequences() 
                                << ", lengthPartitionId " << lengthPartitionId << ", d_overflow_number1 " << d_overflow_number1[0] << ", d_overflow_number2 " << d_overflow_number2[0] << "\n";
                        }
                        if(d_overflow_positions1 != d_overflow_positions2){
                            std::cout << " Batch: " << batch << ", numseqs " << dbPartitions[batch].numSequences() 
                                << ", lengthPartitionId " << lengthPartitionId << ", different overflow positions\n";
                        }
                        if(fooscores1 != fooscores2){                            
                            thrust::host_vector<float> h_fooscores1 = fooscores1;
                            thrust::host_vector<float> h_fooscores2 = fooscores2;

                            for(size_t i = 0; i < currentPartition.numSequences(); i++){
                                if(h_fooscores1[i] != h_fooscores2[i]){
                                    std::cout << " Batch: " << batch << ", numseqs " << dbPartitions[batch].numSequences() 
                                        << ", lengthPartitionId " << lengthPartitionId << ", repeat = " << repeat << ", different scores\n";
                                    std::cout << "error i " << i << ", got " << h_fooscores2[i] << " expected " << h_fooscores1[i] << "\n";
                                    std::cout << "expected: ";
                                    for(int k = i; k < std::min(currentPartition.numSequences(), i+30); k++){
                                        std::cout << h_fooscores1[k] << " ";
                                    }
                                    std::cout << "\n";
                                    std::cout << "got     : ";
                                    for(int k = i; k < std::min(currentPartition.numSequences(), i+30); k++){
                                        std::cout << h_fooscores2[k] << " ";
                                    }
                                    std::cout << "\n";
                                    //std::exit(0);
                                }
                            }
                        }
                    }
                #endif

                #if 0
                    if (ws.h_numSelectedPerPartition[13]){
                        constexpr int blocksize = 32 * 8;
                        constexpr int groupsize = 32;
                        constexpr int groupsPerBlock = blocksize / groupsize;
                        constexpr int alignmentsPerGroup = 2;
                        constexpr int alignmentsPerBlock = groupsPerBlock * alignmentsPerGroup;
                        
                        const size_t tempBytesPerBlockPerBuffer = sizeof(__half2) * alignmentsPerBlock * queryLength;

                        const size_t maxNumBlocks = ws.maxTempBytes / (tempBytesPerBlockPerBuffer * 2);
                        const size_t maxSubjectsPerIteration = std::min(maxNumBlocks * alignmentsPerBlock, size_t(ws.h_numSelectedPerPartition[13]));

                        const size_t numBlocksPerIteration = SDIV(maxSubjectsPerIteration, alignmentsPerBlock);
                        const size_t requiredTempBytes = tempBytesPerBlockPerBuffer * 2 * numBlocksPerIteration;

                        __half2* d_temp = (__half2*)ws.d_tempStorageHE;
                        __half2* d_tempHcol2 = d_temp;
                        __half2* d_tempEcol2 = (__half2*)(((char*)d_tempHcol2) + requiredTempBytes / 2);

                        const int numIters =  SDIV(ws.h_numSelectedPerPartition[13], maxSubjectsPerIteration);

                        constexpr float invalidscore = 123456.0f;

                        thrust::device_vector<float> fooscores1(54465398, invalidscore);
                        thrust::device_vector<size_t> d_overflow_positions1(54465398, 99999999);
                        thrust::device_vector<int> d_overflow_number1(1, 0);                    

                        for(int iter = 0; iter < numIters; iter++){
                            const size_t begin = iter * maxSubjectsPerIteration;
                            const size_t end = iter < numIters-1 ? (iter+1) * maxSubjectsPerIteration : ws.h_numSelectedPerPartition[13];
                            const size_t num = end - begin;                      

                            cudaMemsetAsync(d_temp, 0, requiredTempBytes, workStream); CUERR;

                            NW_local_affine_Protein_many_pass_half2<groupsize, 12><<<SDIV(num, alignmentsPerBlock), blocksize, 0, workStream>>>(
                                ws.devChars_2[source], 
                                fooscores1.data().get(), 
                                d_tempHcol2, 
                                d_tempEcol2, 
                                ws.devOffsets_2[source], 
                                ws.devLengths_2[source], 
                                d_selectedPositions + begin, 
                                num, 
                                d_overflow_positions1.data().get(), 
                                d_overflow_number1.data().get(), 
                                1, 
                                queryLength, 
                                gop, 
                                gex
                            ); CUERR
                        }

                        for(int repeat = 0; repeat < 20; repeat++){
                            thrust::device_vector<float> fooscores2(54465398, invalidscore);
                            thrust::device_vector<size_t> d_overflow_positions2(54465398, 99999999);
                            thrust::device_vector<int> d_overflow_number2(1, 0);

                            for(int iter = 0; iter < numIters; iter++){
                                const size_t begin = iter * maxSubjectsPerIteration;
                                const size_t end = iter < numIters-1 ? (iter+1) * maxSubjectsPerIteration : ws.h_numSelectedPerPartition[13];
                                const size_t num = end - begin;                      

                                cudaMemsetAsync(d_temp, 0, requiredTempBytes, workStream); CUERR;

                                NW_local_affine_Protein_many_pass_half2<groupsize, 12><<<SDIV(num, alignmentsPerBlock), blocksize, 0, workStream>>>(
                                    ws.devChars_2[source], 
                                    fooscores2.data().get(), 
                                    d_tempHcol2, 
                                    d_tempEcol2, 
                                    ws.devOffsets_2[source], 
                                    ws.devLengths_2[source], 
                                    d_selectedPositions + begin, 
                                    num, 
                                    d_overflow_positions2.data().get(), 
                                    d_overflow_number2.data().get(), 
                                    1, 
                                    queryLength, 
                                    gop, 
                                    gex
                                ); CUERR
                            }

                            thrust::sort(
                                d_overflow_positions1.begin(),
                                d_overflow_positions1.end()
                            );
                            thrust::sort(
                                d_overflow_positions2.begin(),
                                d_overflow_positions2.end()
                            );

                            thrust::scatter(
                                thrust::make_constant_iterator<float>(invalidscore),
                                thrust::make_constant_iterator<float>(invalidscore) + d_overflow_number1[0],
                                d_overflow_positions1.begin(),
                                fooscores1.begin()
                            );

                            thrust::scatter(
                                thrust::make_constant_iterator<float>(invalidscore),
                                thrust::make_constant_iterator<float>(invalidscore) + d_overflow_number2[0],
                                d_overflow_positions2.begin(),
                                fooscores2.begin()
                            );

                            if(d_overflow_number1 != d_overflow_number2){
                                std::cout << "repeat = " << repeat << ", d_overflow_number1 " << d_overflow_number1[0] << ", d_overflow_number2 " << d_overflow_number2[0] << "\n";
                            }
                            if(d_overflow_positions1 != d_overflow_positions2){
                                std::cout << "repeat = " << repeat << ", different overflow positions\n";
                            }
                            if(fooscores1 != fooscores2){                            
                                thrust::host_vector<float> h_fooscores1 = fooscores1;
                                thrust::host_vector<float> h_fooscores2 = fooscores2;

                                for(int i = 0; i < 54465398; i++){
                                    if(h_fooscores1[i] != h_fooscores2[i]){
                                        std::cout << " Batch: " << batch << ", numseqs " << dbPartitions[batch].numSequences() << ", repeat = " << repeat << ", different scores\n";
                                        std::cout << "error i " << i << ", got " << h_fooscores2[i] << " expected " << h_fooscores1[i] << "\n";
                                        std::cout << "expected: ";
                                        for(int k = i; k < std::min(54465398, i+30); k++){
                                            std::cout << h_fooscores1[k] << " ";
                                        }
                                        std::cout << "\n";
                                        std::cout << "got     : ";
                                        for(int k = i; k < std::min(54465398, i+30); k++){
                                            std::cout << h_fooscores2[k] << " ";
                                        }
                                        std::cout << "\n";
                                        //std::exit(0);
                                    }
                                }
                            }
                        }
                    }
                #endif


                if (batch < int(dbPartitions.size())-1)  {   // data transfer for next batch
                    const auto& nextPartition = dbPartitions[batch+1];
                    copyBatchToDevice(nextPartition, target);
                }

                {
                    short2* d_temp = (short2*)ws.d_tempStorageHE;

                    launch_process_overflow_alignments_kernel_NW_local_affine_read4_float_query_Protein<32, 12><<<1,1,0, workStream>>>(
                        ws.d_overflow_number,
                        d_temp, 
                        ws.maxTempBytes,
                        ws.devChars_2[source], 
                        &(ws.devAlignmentScoresFloat[globalSequenceOffsetOfBatch]), 
                        ws.devOffsets_2[source] , 
                        ws.devLengths_2[source], 
                        ws.d_overflow_positions, 
                        queryLength, gop, gex
                    ); CUERR
                }
            }

            //after all work is done on workStream, the batch data on the device for source can be overwritten
            cudaEventRecord(ws.deviceBufferEvents[source], workStream); CUERR;
        }

        globalSequenceOffsetOfBatch += currentPartition.numSequences();
    }

    //create dependency for mainStream
    cudaEventRecord(ws.forkStreamEvent, ws.workStreamForTempUsage); CUERR;
    cudaStreamWaitEvent(mainStream, ws.forkStreamEvent, 0); CUERR;


    cudaEventRecord(ws.forkStreamEvent, ws.hostFuncStream); CUERR;
    cudaStreamWaitEvent(mainStream, ws.forkStreamEvent, 0); CUERR;

    cudaEventRecord(ws.forkStreamEvent, ws.dblBufferStreams[0]); CUERR;
    cudaStreamWaitEvent(mainStream, ws.forkStreamEvent, 0); CUERR;

    cudaEventRecord(ws.forkStreamEvent,ws.dblBufferStreams[1]); CUERR;
    cudaStreamWaitEvent(mainStream, ws.forkStreamEvent, 0); CUERR;

    //TIMERSTOP_CUDA_STREAM(NW_local_affine_half2_query_Protein, mainStream)

    //cudaStreamSynchronize(mainStream); CUERR;
}










int main(int argc, char* argv[])
{
    //struct char8 {
    //    char x0, x1, x2, x3, x4, x5, x6, x7;
    //};

    if(argc < 3) {
        cout << "Usage:\n  " << argv[0] << " <FASTA filename 1> [dbPrefix]\n";
        return 0;
    }

    constexpr size_t maxTempBytes = 4ull * 1024ull * 1024ull * 1024ull;

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
        init_cuda_context(); CUERR
        cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);CUERR
        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte); CUERR

        cudaMemPool_t mempool;
        cudaDeviceGetDefaultMemPool(&mempool, deviceIds[i]); CUERR
        uint64_t threshold = UINT64_MAX;
        cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);CUERR
    }
    std::vector<cudaStream_t> gpuStreams(numGpus);
    for(int i = 0; i < numGpus; i++){
        cudaSetDevice(deviceIds[i]); CUERR
        cudaStreamCreate(&gpuStreams[i]); CUERR;
    }

    cudaSetDevice(masterDeviceId); CUERR
    cudaEvent_t masterevent1;
    cudaEventCreate(&masterevent1, cudaEventDisableTiming);
    cudaStream_t masterStream1;
    cudaStreamCreate(&masterStream1);






    int select_datatype; // 0 = 2x16-bit, 1 = 32-bit
    if (argc>=4) select_datatype = std::stoi(argv[3]); else select_datatype = 0; // default = 16-bit
    int select_dpx; // 0 = 2x16-bit, 1 = 32-bit
    if (argc>=5) select_dpx = std::stoi(argv[4]); else select_dpx = 0; // default = none
    if (select_datatype == 0)
         if (select_dpx == 0) cout << "Selected datatype: HALF2\n"; else cout << "Selected datatype: s16x2 (DPX)\n";
    else
        if (select_dpx == 0) cout << "Selected datatype: FLOAT\n"; else cout << "Selected datatype: s32 (DPX)\n";

	// read all sequences from FASTA or FASTQ file: query file
	sequence_batch batch = read_all_sequences_and_headers_from_file(argv[1]);
	cout << "Read Protein Query File 1\n";

    // chars   = concatenation of all sequences
    // offsets = starting indices of individual sequences (1st: 0, last: one behind end of 'chars')
    char*   chars       = batch.chars.data();
    const size_t* offsets     = batch.offsets.data();
    const size_t* lengths      = batch.lengths.data();
    const size_t  charBytes   = batch.chars.size();
    const size_t  offsetBytes = batch.offsets.size() * sizeof(size_t);
    int numQueries = batch.offsets.size() - 1;
    const char* maxNumQueriesString = std::getenv("ALIGNER_MAX_NUM_QUERIES");
    if(maxNumQueriesString != nullptr){
        int maxNumQueries = std::atoi(maxNumQueriesString);
        numQueries = std::min(numQueries, maxNumQueries);
    }


    cout << "Number of input sequences Query-File:  " << numQueries<< '\n';
    cout << "Number of input characters Query-File: " << charBytes << '\n';
    int64_t dp_cells = 0;

	cout << "Reading Database: \n";
	TIMERSTART_CUDA(READ_DB)
    constexpr bool writeAccess = false;
    constexpr bool prefetchSeq = true;
    DB fullDB = loadDB(argv[2], writeAccess, prefetchSeq);
	TIMERSTOP_CUDA(READ_DB)

    


    cout << "Read Protein DB Files\n";
    const int numDBChunks = fullDB.info.numChunks;
    std::cout << "Number of DB chunks: " << numDBChunks << "\n";
    for(int i = 0; i < numDBChunks; i++){
        const DBdata& chunkData = fullDB.chunks[i];
        const DBdata::MetaData& dbMetaData = chunkData.getMetaData();
        std::cout << "DB chunk " << i << ": " << chunkData.numSequences() << " sequences, " << chunkData.numChars() << " characters\n";
        for(int i = 0; i < int(dbMetaData.lengthBoundaries.size()); i++){
            std::cout << "<= " << dbMetaData.lengthBoundaries[i] << ": " << dbMetaData.numSequencesPerLengthPartition[i] << "\n";
        }
    }

    size_t totalNumberOfSequencesInDB = 0;
    size_t maximumNumberOfSequencesInDBChunk = 0;
    for(int i = 0; i < numDBChunks; i++){
        const DBdata& chunkData = fullDB.chunks[i];
        totalNumberOfSequencesInDB += chunkData.numSequences();
        maximumNumberOfSequencesInDBChunk = std::max(maximumNumberOfSequencesInDBChunk, chunkData.numSequences());
    }



    uint MAX_Num_Seq = 1000000;
	uint batch_size = 200000;
    if (maximumNumberOfSequencesInDBChunk <= MAX_Num_Seq) {
        batch_size = maximumNumberOfSequencesInDBChunk;
    }else {
	 	batch_size = MAX_Num_Seq;
	}



   //cout << "Read lengths: \n";
    //for (int i=9000000; i<9001000; i++) cout << " " << lengths_2_dbData[i];
    //cout << "\n";

    //determine maximal and minimal read lengths
    int64_t max_length = 0, min_length = 10000, avg_length = 0;
    int64_t max_length_2 = 0, min_length_2 = 10000, avg_length_2 = 0;
    for (int i=0; i<numQueries; i++) {
        if (lengths[i] > max_length) max_length = lengths[i];
        if (lengths[i] < min_length) min_length = lengths[i];
        avg_length += lengths[i];
    }

    for(int i = 0; i < numDBChunks; i++){
        const DBdata& chunkData = fullDB.chunks[i];
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
    cout << "Batch Size: " << batch_size <<"\n";



    for(int i = 0; i < 0; ++i) {
            cout << "Query: "<< i <<" , " << lengths[i] << " : ";
			cout << batch.headers[i] << '\n';
			//cout << i <<" - "<< offsets[i] << " "<< (offsets[i] % ALIGN) <<" "<< lengths[i] <<" ";
			const auto first = chars + offsets[i];
			std::copy(first, first + lengths[i], std::ostream_iterator<char>{cout});
			cout << '\n';
            //for (int j=0; j<lengths[i]; j++) cout << *(chars+offsets[i]+j);
            //cout << '\n';
    }


    // Partition dbData
    auto printPartition = [](const auto& view){
        std::cout << "Sequences: " << view.numSequences() << "\n";
        std::cout << "Chars: " << view.offsets()[0] << " - " << view.offsets()[view.numSequences()] << " (" << (view.offsets()[view.numSequences()] - view.offsets()[0]) << ")"
            << " " << view.numChars() << "\n";
    };
    auto printPartitions = [printPartition](const auto& dbPartitions){
        size_t numPartitions = dbPartitions.size();
        for(size_t p = 0; p < numPartitions; p++){
            const DBdataView& view = dbPartitions[p];
    
            std::cout << "Partition " << p << "\n";
            printPartition(view);
        }
    };

   

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

        
        for(int lengthPartitionId = 0; lengthPartitionId < numLengthPartitions; lengthPartitionId++){
            const auto& lengthPartition = dbPartitionsByLengthPartitioning[lengthPartitionId];
    
            // std::cout << "length partition " << i << "\n";
            // printPartition(lengthPartition);
    
            const auto partitionedByGpu = partitionDBdata_by_numberOfChars(lengthPartition, lengthPartition.numChars() / numGpus);
            // std::cout << "partitionedByGpu \n";
            // printPartitions(partitionedByGpu);
    
            assert(int(partitionedByGpu.size()) <= numGpus);
    
            for(int gpu = 0; gpu < int(partitionedByGpu.size()); gpu++){
                const auto partitionedBySeq = partitionDBdata_by_numberOfSequences(partitionedByGpu[gpu], batch_size);
    
                // std::cout << "partitionedBySeq \n";
                // printPartitions(partitionedBySeq);
    
                subPartitionsForGpus[gpu].insert(subPartitionsForGpus[gpu].end(), partitionedBySeq.begin(), partitionedBySeq.end());
    
                for(size_t x = 0; x < partitionedBySeq.size(); x++){
                    lengthPartitionIdsForGpus[gpu].push_back(lengthPartitionId);
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






    auto createGpuWorkingSet = [&](
        int deviceId, 
        GpuWorkingSet& ws, 
        size_t maxTempBytes,
        size_t max_batch_char_bytes, 
        size_t max_batch_offset_bytes,
        size_t max_batch_num_sequences,
        size_t num_sequences
    ){
        int oldId;
        cudaGetDevice(&oldId);
        cudaSetDevice(deviceId); CUERR;

        ws.deviceId = deviceId;
        ws.maxTempBytes = maxTempBytes;
        ws.max_batch_num_sequences = max_batch_num_sequences;

        cudaMalloc(&ws.devChars, charBytes); CUERR
        cudaMalloc(&ws.devOffsets, offsetBytes); CUERR
        cudaMalloc(&ws.devLengths, offsetBytes); CUERR

        for(int i = 0; i < 2; i++){
            cudaMalloc(&ws.devChars_2[i], max_batch_char_bytes); CUERR
            cudaMalloc(&ws.devOffsets_2[i], max_batch_offset_bytes); CUERR
            cudaMalloc(&ws.devLengths_2[i], max_batch_offset_bytes); CUERR
            cudaMallocHost(&ws.buf_host_Chars_2[i], max_batch_char_bytes); CUERR
            cudaMallocHost(&ws.buf_host_Offsets_2[i], max_batch_offset_bytes); CUERR
            cudaMallocHost(&ws.buf_host_Lengths_2[i], max_batch_offset_bytes); CUERR
            cudaEventCreate(&ws.pinnedBufferEvents[i], cudaEventDisableTiming); CUERR;
            cudaEventCreate(&ws.deviceBufferEvents[i], cudaEventDisableTiming); CUERR;
            cudaStreamCreate(&ws.dblBufferStreams[i]); CUERR;
        }



        cudaMalloc(&ws.devAlignmentScoresFloat, sizeof(float)*num_sequences);

        cudaMalloc(&ws.d_tempStorageHE, ws.maxTempBytes);

        
        cudaMallocHost(&ws.h_numSelectedPerPartition, sizeof(int) * numLengthPartitions);
        cudaMalloc(&ws.d_selectedPositions, sizeof(size_t)*(max_batch_num_sequences)); CUERR
        thrust::sequence(
            thrust::device,
            ws.d_selectedPositions,
            ws.d_selectedPositions + max_batch_num_sequences,
            size_t(0)
        );

        cudaMalloc(&ws.d_overflow_positions, max_batch_num_sequences * sizeof(size_t)); CUERR
        cudaMalloc(&ws.d_overflow_number, 1 * sizeof(int)); CUERR

        cudaMalloc(&ws.Fillchar, 16*512);
        cudaMemset(ws.Fillchar, 20, 16*512);

        cudaStreamCreate(&ws.workStreamForTempUsage); CUERR

        cudaStreamCreate(&ws.hostFuncStream); CUERR

        const int numWorkStreams = 3;
        ws.workStreams.resize(numWorkStreams);
        for(int i = 0; i <numWorkStreams; i++){
            cudaStreamCreate(&ws.workStreams[i]); CUERR;
        }
        

        cudaEventCreate(&ws.forkStreamEvent, cudaEventDisableTiming); CUERR
        

        cudaSetDevice(oldId);
    };

    auto destroyGpuWorkingSet = [](GpuWorkingSet& ws){
        int oldId;
        cudaGetDevice(&oldId);
        cudaSetDevice(ws.deviceId); CUERR;

        cudaFree(ws.devChars); CUERR
        cudaFree(ws.devOffsets); CUERR
        cudaFree(ws.devLengths); CUERR

        for(int i = 0; i < 2; i++){
            cudaFree(ws.devChars_2[i]); CUERR
            cudaFree(ws.devOffsets_2[i]); CUERR
            cudaFree(ws.devLengths_2[i]); CUERR
            cudaFreeHost(ws.buf_host_Chars_2[i]); CUERR
            cudaFreeHost(ws.buf_host_Offsets_2[i]); CUERR
            cudaFreeHost(ws.buf_host_Lengths_2[i]); CUERR
            cudaEventDestroy(ws.pinnedBufferEvents[i]); CUERR;
            cudaEventDestroy(ws.deviceBufferEvents[i]); CUERR;
            cudaStreamDestroy(ws.dblBufferStreams[i]); CUERR;
        }

        cudaFree(ws.devAlignmentScoresFloat); CUERR;
        cudaFree(ws.d_tempStorageHE); CUERR;
        
        cudaFreeHost(ws.h_numSelectedPerPartition); CUERR;
        cudaFree(ws.d_selectedPositions); CUERR

        cudaFree(ws.d_overflow_positions); CUERR
        cudaFree(ws.d_overflow_number); CUERR

        cudaFree(ws.Fillchar);

        cudaStreamDestroy(ws.workStreamForTempUsage); CUERR

        cudaStreamDestroy(ws.hostFuncStream); CUERR
        cudaEventDestroy(ws.forkStreamEvent); CUERR

        for(int i = 0; i < int(ws.workStreams.size()); i++){
            cudaStreamDestroy(ws.workStreams[i]); CUERR;
        }

        cudaSetDevice(oldId);
    };

    const uint results_per_query = 100;
    // std::vector<float> alignment_scores_float(numQueries *numDBChunks * results_per_query);
    // std::vector<size_t> sorted_indices(numQueries *numDBChunks * results_per_query);
    // std::vector<int> resultDbChunkIndices(numQueries *numDBChunks * results_per_query);
    pinned_vector<float> alignment_scores_float(numQueries *numDBChunks * results_per_query);
    pinned_vector<size_t> sorted_indices(numQueries *numDBChunks * results_per_query);
    pinned_vector<int> resultDbChunkIndices(numQueries *numDBChunks * results_per_query);

    //set up gpus


    std::vector<GpuWorkingSet> workingSets(numGpus);

    cout << "Allocate Memory: \n";
	TIMERSTART_CUDA(ALLOC_MEM)

    for(int i = 0; i < numGpus; i++){
        cudaSetDevice(deviceIds[i]);

        size_t max_batch_char_bytes = 0;
        size_t max_batch_num_sequences = 0;

        for(int chunkId = 0; chunkId < numDBChunks; chunkId++){
            const size_t max_batch_char_bytes_chunk = std::max_element(subPartitionsForGpus_perDBchunk[chunkId][i].begin(), subPartitionsForGpus_perDBchunk[chunkId][i].end(),
                [](const auto& l, const auto& r){ return l.numChars() < r.numChars(); }
            )->numChars();

            max_batch_char_bytes = std::max(max_batch_char_bytes, max_batch_char_bytes_chunk);

            const size_t max_batch_num_sequences_chunk = std::max_element(subPartitionsForGpus_perDBchunk[chunkId][i].begin(), subPartitionsForGpus_perDBchunk[chunkId][i].end(),
                [](const auto& l, const auto& r){ return l.numSequences() < r.numSequences(); }
            )->numSequences();

            max_batch_num_sequences = std::max(max_batch_num_sequences, max_batch_num_sequences_chunk);
        }

        const size_t max_batch_offset_bytes = sizeof(size_t) * max_batch_num_sequences;

        std::cout << "max_batch_char_bytes " << max_batch_char_bytes << "\n";
        std::cout << "max_batch_num_sequences " << max_batch_num_sequences << "\n";
        std::cout << "max_batch_offset_bytes " << max_batch_offset_bytes << "\n";

        
        createGpuWorkingSet(
            deviceIds[i], 
            workingSets[i],
            maxTempBytes,
            max_batch_char_bytes, 
            max_batch_offset_bytes,
            max_batch_num_sequences,
            numSequencesPerGpu_total[i]
        );

        //spin up the host callback thread
        auto noop = [](void*){};
        cudaLaunchHostFunc(
            cudaStreamLegacy, 
            noop, 
            nullptr
        ); CUERR
    }
    

	TIMERSTOP_CUDA(ALLOC_MEM)


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
    thrust::device_vector<float> devAllAlignmentScoresFloat(totalNumberOfSequencesInDB);
    thrust::device_vector<size_t> dev_sorted_indices(totalNumberOfSequencesInDB);




    cout << "Starting FULLSCAN_CUDA: \n";
    cudaSetDevice(masterDeviceId);
    TIMERSTART_CUDA(FULLSCAN_CUDA)

    for(int i = 0; i < numGpus; i++){
        cudaSetDevice(deviceIds[i]);
        auto& ws = workingSets[i];
        cudaMemcpyAsync(ws.devChars, chars, charBytes, cudaMemcpyHostToDevice, gpuStreams[i]); CUERR
        cudaMemcpyAsync(ws.devOffsets, offsets, offsetBytes, cudaMemcpyHostToDevice, gpuStreams[i]); CUERR
        cudaMemcpyAsync(ws.devLengths, lengths, offsetBytes, cudaMemcpyHostToDevice, gpuStreams[i]); CUERR
        NW_convert_protein<<<numQueries, 128, 0, gpuStreams[i]>>>(ws.devChars, ws.devOffsets); CUERR
    }



	
	const int FIRST_QUERY_NUM = 0;


	for(int query_num = FIRST_QUERY_NUM; query_num < numQueries; ++query_num) {

        dp_cells = avg_length_2 * lengths[query_num];

        const bool useExtraThreadForBatchTransfer = numGpus > 1;

        cudaSetDevice(masterDeviceId);
        thrust::fill(
            thrust::cuda::par_nosync.on(masterStream1),
            devAllAlignmentScoresFloat.begin(),
            devAllAlignmentScoresFloat.end(),
            0
        );

        std::cout << "Starting NW_local_affine_half2 for Query " << query_num << "\n";
        TIMERSTART_CUDA_STREAM(NW_local_affine_half2_query_Protein, masterStream1)

        for(int chunkId = 0; chunkId < numDBChunks; chunkId++){
            cudaSetDevice(masterDeviceId);
            


            cudaEventRecord(masterevent1, masterStream1); CUERR;

            for(int gpu = 0; gpu < numGpus; gpu++){
                cudaSetDevice(deviceIds[gpu]);

                auto& ws = workingSets[gpu];
                assert(ws.deviceId == deviceIds[gpu]);

                cudaStreamWaitEvent(gpuStreams[gpu], masterevent1, 0); CUERR;

                processQueryOnGpu(
                    ws,
                    subPartitionsForGpus_perDBchunk[chunkId][gpu],
                    lengthPartitionIdsForGpus_perDBchunk[chunkId][gpu],
                    &(ws.devChars[offsets[query_num]]),
                    lengths[query_num],
                    (query_num == FIRST_QUERY_NUM),
                    query_num,
                    avg_length_2,
                    select_datatype,
                    select_dpx,
                    useExtraThreadForBatchTransfer,
                    gpuStreams[gpu]
                );

                cudaMemcpyAsync(
                    devAllAlignmentScoresFloat.data().get() + numSequencesPerGpuPrefixSum_perDBchunk[chunkId][gpu],
                    ws.devAlignmentScoresFloat,
                    //sizeof(float) * dbPartitionsForGpus[i].numSequences(),
                    sizeof(float) * numSequencesPerGpu_perDBchunk[chunkId][gpu],
                    cudaMemcpyDeviceToDevice,
                    gpuStreams[gpu]
                );

                cudaEventRecord(ws.forkStreamEvent, gpuStreams[gpu]); CUERR;
                cudaStreamWaitEvent(masterStream1, ws.forkStreamEvent, 0); CUERR;
            }

            cudaSetDevice(masterDeviceId);

            thrust::sequence(
                thrust::cuda::par_nosync.on(masterStream1),
                dev_sorted_indices.begin(), 
                dev_sorted_indices.end(),
                0
            );
            thrust::sort_by_key(
                thrust::cuda::par_nosync(thrust_async_allocator<char>(masterStream1)).on(masterStream1),
                devAllAlignmentScoresFloat.begin(),
                devAllAlignmentScoresFloat.end(),
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
                devAllAlignmentScoresFloat.data().get(), 
                results_per_query*sizeof(float), 
                cudaMemcpyDeviceToHost, 
                masterStream1
            );  CUERR
            cudaMemcpyAsync(
                &(sorted_indices[query_num*numDBChunks*results_per_query + chunkId * results_per_query]), 
                dev_sorted_indices.data().get(), 
                results_per_query*sizeof(size_t), 
                cudaMemcpyDeviceToHost, 
                masterStream1
            );  CUERR

        }

        cudaSetDevice(masterDeviceId);
        TIMERSTOP_CUDA_STREAM(NW_local_affine_half2_query_Protein, masterStream1)

    }
    cudaSetDevice(masterDeviceId);
    cudaStreamSynchronize(masterStream1); CUERR
    dp_cells = avg_length_2 * avg_length;
    TIMERSTOP_CUDA(FULLSCAN_CUDA)

    CUERR;

    cudaEventDestroy(masterevent1);
    cudaStreamDestroy(masterStream1);

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
	    std::cout << totalOverFlowNumber << " total overflow positions \n";
		cout << "Query Length:" << lengths[i] << " Header: ";
		cout << batch.headers[i] << '\n';

		for(int j = 0; j < 10; ++j) {
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
		}
    }

    CUERR;

    for(int i = 0; i < numGpus; i++){
        //std::cout << "destroy " << i << "\n";
        cudaSetDevice(deviceIds[i]); CUERR;
        destroyGpuWorkingSet(workingSets[i]);
    }

}
