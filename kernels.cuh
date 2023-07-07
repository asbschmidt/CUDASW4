#ifndef KERNELS_CUH
#define KERNELS_CUH


#include "convert.cuh"

#include <cuda_fp16.h>

#define MAX_ACC_HALF2 2048.0 // 1024 ??


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


//__constant__ char cQuery[8*1024];
__constant__ char4 constantQuery4[2048];
__constant__ char cBLOSUM62_dev[21*21];
//__constant__ char2 cQuery2[4*1024];

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
    const size_t base_S0 = devOffsets[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)]];

    int length_S1 = devLengths[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)+1]];
    size_t base_S1 = devOffsets[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)+1]];

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
template <int group_size, int numRegs, class PositionsIterator> 
__launch_bounds__(256,2)
__global__
void NW_local_affine_Protein_many_pass_half2(
    const char * devChars,
    float * devAlignmentScores,
    __half2 * devTempHcol2,
    __half2 * devTempEcol2,
    const size_t* devOffsets,
    const size_t* devLengths,
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
    const size_t base_S0 = devOffsets[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)]]-devOffsets[0];

	int length_S1 = devLengths[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)+1]];
	size_t base_S1 = devOffsets[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)+1]]-devOffsets[0];

    // printf("tid %d, %lu, %lu\n", threadIdx.x, base_S0, base_S1);
    //__syncthreads();


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

    auto checkHEindex = [&](auto x, int line){
        // if(x < 0){printf("line %d\n", line);}
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
        //#pragma unroll //UNROLLHERE
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
       //#pragma unroll //UNROLLHERE
       for (int i=0; i<numRegs; i++) {

           if (offset_isc+numRegs*(thid%group_size)+i >= length_S0) subject[i] = 1; // 20;
           else{
                //printf("tid %d, i %d, offset_isc %d, base_S0 %lu, total %d\n", threadIdx.x, i, offset_isc, base_S0, offset_isc+base_S0+numRegs*(thid%group_size)+i);
            subject[i] = devChars[offset_isc+base_S0+numRegs*(thid%group_size)+i];
           }

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
        //#pragma unroll //UNROLLHERE
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
            checkHEindex(offset_out, __LINE__);
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

    //printf("tid %d, offset_out %d, from_thread_id %d\n", threadIdx.x, offset_out, from_thread_id);
    if (thid>=from_thread_id) {
        checkHEindex(offset_out-from_thread_id, __LINE__);
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
        checkHEindex(offset_in, __LINE__);
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
                checkHEindex(offset_out, __LINE__);
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
                checkHEindex(offset_in, __LINE__);
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
            checkHEindex(offset_out-from_thread_id, __LINE__);
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
    checkHEindex(offset_in, __LINE__);
    H_temp_in = devTempHcol[offset_in];
    E_temp_in = devTempEcol[offset_in];
    offset_in += group_size;

    init_penalties_local(0);
    //init_local_score_profile_BLOSUM62((passes-1)*(32*numRegs));
    init_local_score_profile_BLOSUM62(passes > 1 ? (passes-1)*(32*numRegs) : 0);
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
                checkHEindex(offset_in, __LINE__);
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
__launch_bounds__(256,1)
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
    const size_t base_S0 = devOffsets[alignmentId_checklast_0]-devOffsets[0];

    //if ((blid == 2) && (!thid)) printf("Values in Block: %d, in Thread: %d, length_S0: %d, base_S0: %d\n", blid, thid, length_S0, base_S0);

	int length_S1 = length_S0;
	size_t base_S1 = base_S0;
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
    const size_t base_S0 = devOffsets[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)]];

    int length_S1 = devLengths[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)+1]];
    size_t base_S1 = devOffsets[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)+1]];

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
    const size_t base = devOffsets[d_positions_of_selected_lengths[(32/group_size)*blid+(thid%check_last)/group_size]]-devOffsets[0]; //[(32/group_size)*blid+thid/group_size];
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
    const size_t base = devOffsets[d_positions_of_selected_lengths[(32/group_size)*blid+(thid%check_last)/group_size]]; //[(32/group_size)*blid+thid/group_size];
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
    const size_t base = devOffsets[d_positions_of_selected_lengths[blid]]-devOffsets[0]; // devOffsets[(32/group_size)*blid+thid/group_size];

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
    const size_t base = devOffsets[d_positions_of_selected_lengths[blid]]; // devOffsets[(32/group_size)*blid+thid/group_size];

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
    const size_t base = devOffsets[blid];
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



#endif

