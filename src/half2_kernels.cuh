#ifndef HALF2_KERNELS_CUH
#define HALF2_KERNELS_CUH


#include <cuda_fp16.h>
#include "blosum.hpp"
#include "config.hpp"

namespace cudasw4{

template <int group_size, int numRegs, int blosumDim, class PositionsIterator> 
struct Half2Aligner{
    static_assert(2 <= numRegs && numRegs % 2 == 0, "Half2Aligner does not support odd number of numRegs");
    static_assert(1 <= group_size && group_size <= 32 && ((group_size & (group_size - 1)) == 0), "Half2Aligner requires power-of-two sub-warp size");

    static constexpr float negInftyFloat = -1000.0f;

    static constexpr int deviceBlosumDimCexpr = blosumDim;
    static constexpr int deviceBlosumDimCexprSquared = deviceBlosumDimCexpr * deviceBlosumDimCexpr;

    __half2* shared_blosum;

    int numSelected;
    float gap_open;
    float gap_extend;
    PositionsIterator d_positions_of_selected_lengths;
    const char* devChars;
    __half2* devTempHcol2;
    __half2* devTempEcol2;
    const size_t* devOffsets;
    const SequenceLengthT* devLengths;

    __device__
    Half2Aligner(
        __half2* shared_blosum_,
        const char* devChars_,
        __half2* devTempHcol2_,
        __half2* devTempEcol2_,
        const size_t* devOffsets_,
        const SequenceLengthT* devLengths_,
        PositionsIterator d_positions_of_selected_lengths_,
        int numSelected_,
        float gap_open_,
        float gap_extend_
    ) : shared_blosum(shared_blosum_),
        devChars(devChars_),
        devTempHcol2(devTempHcol2_),
        devTempEcol2(devTempEcol2_),
        devOffsets(devOffsets_),
        devLengths(devLengths_),
        d_positions_of_selected_lengths(d_positions_of_selected_lengths_),
        numSelected(numSelected_),
        gap_open(gap_open_),
        gap_extend(gap_extend_)
    {
        for (int i=threadIdx.x; i<deviceBlosumDimCexprSquared; i+=blockDim.x) {
            __half2 temp0;
            temp0.x = deviceBlosum[deviceBlosumDimCexpr*(i/deviceBlosumDimCexpr)+(i%deviceBlosumDimCexpr)];
            for (int j=0; j<deviceBlosumDimCexpr; j++) {
                temp0.y = deviceBlosum[deviceBlosumDimCexpr*(i/deviceBlosumDimCexpr)+j];
                shared_blosum[(i/deviceBlosumDimCexpr) * deviceBlosumDimCexprSquared + deviceBlosumDimCexpr*(i%deviceBlosumDimCexpr)+j]=temp0;
            }
        }
        __syncthreads();
    }

    __device__ 
    void computeCheckLast(int& check_last, int& check_last2) const{
        check_last = blockDim.x/group_size;
        check_last2 = 0;
        if (blockIdx.x == gridDim.x-1) {
            if (numSelected % (2*blockDim.x/group_size)) {
                check_last = (numSelected/2) % (blockDim.x/group_size);
                check_last2 = numSelected%2;
                check_last = check_last + check_last2;
            }
        }
        check_last = check_last * group_size;
    }

    __device__
    SequenceLengthT getPaddedQueryLength(SequenceLengthT queryLength) const{
        //pad query length to char4, add warpsize char4 border.
        return SDIV(queryLength, 4) * 4 + 32 * sizeof(char4);
    }

    __device__
    void checkHEindex(int x, SequenceLengthT queryLength, int line) const{
        // const SequenceLengthT currentQueryLengthWithPadding = getPaddedQueryLength(queryLength);
        // assert(x >= 0);
        // assert(x < currentQueryLengthWithPadding);
    };

    __device__
    void initial_calc32_local_affine_float(const int value, char query_letter, __half2& E, __half2& penalty_here31, __half2 penalty_diag, __half2 penalty_left, __half2& maximum, 
        int (&subject)[numRegs], 
        __half2 (&penalty_here_array)[numRegs],
        __half2 (&F_here_array)[numRegs]
    ) const{
        const __half2* const sbt_row = &shared_blosum[int(query_letter) * deviceBlosumDimCexprSquared];

        const __half2 score2_0 = sbt_row[subject[0]];
        //score2.y = sbt_row[subject1[0].x];
        __half2 penalty_temp0 = penalty_here_array[0];
        if (!value || (threadIdx.x%group_size)) E = __hmax2(__hadd2(E,__float2half2_rn(gap_extend)), __hadd2(penalty_left,__float2half2_rn(gap_open)));
        F_here_array[0] = __hmax2(__hadd2(F_here_array[0],__float2half2_rn(gap_extend)), __hadd2(penalty_here_array[0],__float2half2_rn(gap_open)));
        const __half2 temp0_0 = __hmax2(__hmax2(__hadd2(penalty_diag,score2_0), __hmax2(E, F_here_array[0])), __float2half2_rn(0.0));
        penalty_here_array[0] = temp0_0;
        maximum = __hmax2(temp0_0, maximum);

        const __half2 score2_1 = sbt_row[subject[1]];
        //score2.y = sbt_row[subject1[0].y];
        __half2 penalty_temp1 = penalty_here_array[1];
        E = __hmax2(__hadd2(E,__float2half2_rn(gap_extend)), __hadd2(penalty_here_array[0],__float2half2_rn(gap_open)));
        F_here_array[1] = __hmax2(__hadd2(F_here_array[1],__float2half2_rn(gap_extend)), __hadd2(penalty_here_array[1],__float2half2_rn(gap_open)));
        penalty_here_array[1] = __hmax2(__hmax2(__hadd2(penalty_temp0,score2_1), __hmax2(E, F_here_array[1])), __float2half2_rn(0.0));
        const __half2 temp0_1 = penalty_here_array[1];
        maximum = __hmax2(temp0_1,maximum);

        #pragma unroll
        for (int i=1; i<numRegs/2; i++) {
            const __half2 score2_2i = sbt_row[subject[2*i]];
            //score2.y = sbt_row[subject1[i].x];
            penalty_temp0 = penalty_here_array[2*i];
            E = __hmax2(__hadd2(E,__float2half2_rn(gap_extend)), __hadd2(penalty_here_array[2*i-1],__float2half2_rn(gap_open)));
            F_here_array[2*i] = __hmax2(__hadd2(F_here_array[2*i],__float2half2_rn(gap_extend)), __hadd2(penalty_here_array[2*i],__float2half2_rn(gap_open)));
            penalty_here_array[2*i] = __hmax2(__hmax2(__hadd2(penalty_temp1,score2_2i), __hmax2(E, F_here_array[2*i])), __float2half2_rn(0.0));
            const __half2 temp0_2 = penalty_here_array[2*i]; 
            maximum = __hmax2(temp0_2,maximum);

            const __half2 score2_2i1 = sbt_row[subject[2*i+1]];
            //score2.y = sbt_row[subject1[i].y];
            penalty_temp1 = penalty_here_array[2*i+1];
            E = __hmax2(__hadd2(E,__float2half2_rn(gap_extend)), __hadd2(penalty_here_array[2*i],__float2half2_rn(gap_open)));
            F_here_array[2*i+1] = __hmax2(__hadd2(F_here_array[2*i+1],__float2half2_rn(gap_extend)), __hadd2(penalty_here_array[2*i+1],__float2half2_rn(gap_open)));
            penalty_here_array[2*i+1] = __hmax2(__hmax2(__hadd2(penalty_temp0,score2_2i1), __hmax2(E, F_here_array[2*i+1])), __float2half2_rn(0.0));
            const __half2 temp0_3 = penalty_here_array[2*i+1];
            maximum = __hmax2(temp0_3,maximum);
        }

        penalty_here31 = penalty_here_array[numRegs-1];
        E = __hmax2(E+__float2half2_rn(gap_extend), penalty_here31+__float2half2_rn(gap_open));
        #pragma unroll //UNROLLHERE
        for (int i=0; i<numRegs; i++) F_here_array[i] = __hmax2(__hadd2(F_here_array[i],__float2half2_rn(gap_extend)), __hadd2(penalty_here_array[i],__float2half2_rn(gap_open)));
    };

    __device__
    void calc32_local_affine_float(char query_letter, __half2& E, __half2& penalty_here31, __half2 penalty_diag, __half2& maximum, 
        int (&subject)[numRegs],
        __half2 (&penalty_here_array)[numRegs],
        __half2 (&F_here_array)[numRegs]
    ) const{
        const __half2* const sbt_row = &shared_blosum[int(query_letter) * deviceBlosumDimCexprSquared];

        // if(threadIdx.x < 1){
        //     for(int t = 0; t < group_size; t++){
        //         if(t == threadIdx.x){
        //             printf("tid %d penalty_here_array\n", threadIdx.x);
        //             for(int i = 0; i < numRegs; i++){
        //                 printf("(%f %f)", float(penalty_here_array[i].x), float(penalty_here_array[i].y));
        //             }
        //             printf("\n");
        //             printf("tid %d F_here_array\n", threadIdx.x);
        //             for(int i = 0; i < numRegs; i++){
        //                 printf("(%f %f)", float(F_here_array[i].x), float(F_here_array[i].y));
        //             }
        //             printf("\n");
        //         }
        //     }
        // }

        const __half2 score2_0 = sbt_row[subject[0]];
        //score2.y = sbt_row[subject1[0].x];
        __half2 penalty_temp0 = penalty_here_array[0];
        penalty_here_array[0] = __hmax2(__hmax2(__hadd2(penalty_diag,score2_0), __hmax2(E, F_here_array[0])),__float2half2_rn(0.0));
        //maximum = __hmax2(penalty_here_array[0],maximum);
        __half2 penalty_temp1 = __hadd2(penalty_here_array[0],__float2half2_rn(gap_open));
        E = __hmax2(__hadd2(E,__float2half2_rn(gap_extend)), penalty_temp1);
        F_here_array[0] = __hmax2(__hadd2(F_here_array[0],__float2half2_rn(gap_extend)), penalty_temp1);

        const __half2 score2_1 = sbt_row[subject[1]];
        //score2.y = sbt_row[subject1[0].y];
        penalty_temp1 = penalty_here_array[1];
        penalty_here_array[1] = __hmax2(__hmax2(__hadd2(penalty_temp0,score2_1), __hmax2(E, F_here_array[1])),__float2half2_rn(0.0));
        //maximum = __hmax2(penalty_here_array[1],maximum);
        penalty_temp0 = __hadd2(penalty_here_array[1],__float2half2_rn(gap_open));
        E = __hmax2(__hadd2(E,__float2half2_rn(gap_extend)), penalty_temp0);
        F_here_array[1] = __hmax2(__hadd2(F_here_array[1],__float2half2_rn(gap_extend)), penalty_temp0);
		maximum = __hmax2(maximum, __hmax2(penalty_here_array[1],penalty_here_array[0]));
        ////if(threadIdx.x < 1) printf("calc32. maximum %f %f, score2_0 %f %f, score2_1 %f %f\n", float(maximum.x), float(maximum.y), float(score2_0.x), float(score2_0.y), float(score2_1.x), float(score2_1.y));

        #pragma unroll
        for (int i=1; i<numRegs/2; i++) {
            const __half2 score2_2i = sbt_row[subject[2*i]];
            penalty_temp0 = penalty_here_array[2*i];
            penalty_here_array[2*i] = __hmax2(__hmax2(__hadd2(penalty_temp1,score2_2i), __hmax2(E, F_here_array[2*i])),__float2half2_rn(0.0));
            //maximum = __hmax2(penalty_here_array[2*i],maximum);
            penalty_temp1 = __hadd2(penalty_here_array[2*i],__float2half2_rn(gap_open));
            E = __hmax2(__hadd2(E,__float2half2_rn(gap_extend)), penalty_temp1);
            F_here_array[2*i] = __hmax2(__hadd2(F_here_array[2*i],__float2half2_rn(gap_extend)), penalty_temp1);

            const __half2 score2_2i1 = sbt_row[subject[2*i+1]];
            penalty_temp1 = penalty_here_array[2*i+1];
            penalty_here_array[2*i+1] = __hmax2(__hmax2(__hadd2(penalty_temp0,score2_2i1), __hmax2(E, F_here_array[2*i+1])),__float2half2_rn(0.0));
            //maximum = __hmax2(penalty_here_array[2*i+1],maximum);
            penalty_temp0 = __hadd2(penalty_here_array[2*i+1],__float2half2_rn(gap_open));
            E = __hmax2(__hadd2(E,__float2half2_rn(gap_extend)), penalty_temp0);
            F_here_array[2*i+1] = __hmax2(__hadd2(F_here_array[2*i+1],__float2half2_rn(gap_extend)), penalty_temp0);
			maximum = __hmax2(maximum,__hmax2(penalty_here_array[2*i+1],penalty_here_array[2*i]));
            ////if(threadIdx.x < 1) printf("calc32. maximum %f %f, score2_2i %f %f, score2_2i1 %f %f\n", float(maximum.x), float(maximum.y), float(score2_2i.x), float(score2_2i.y), float(score2_2i1.x), float(score2_2i1.y));
        }

        //for (int i=0; i<numRegs/4; i++)
		 //   maximum = __hmax2(maximum,__hmax2(__hmax2(penalty_here_array[4*i],penalty_here_array[4*i+1]),__hmax2(penalty_here_array[4*i+2],penalty_here_array[4*i+3])));
        penalty_here31 = penalty_here_array[numRegs-1];
    }

    __device__    
    void init_penalties_local(int value, __half2& penalty_diag, __half2& penalty_left, 
        __half2 (&penalty_here_array)[numRegs], 
        __half2 (&F_here_array)[numRegs]
    ) const{
        //ZERO = __float2half2_rn(negInftyFloat);
        penalty_left = __float2half2_rn(negInftyFloat);
        penalty_diag = __float2half2_rn(negInftyFloat);
        //E = __float2half2_rn(negInftyFloat);
        #pragma unroll
        for (int i=0; i<numRegs; i++) penalty_here_array[i] = __float2half2_rn(negInftyFloat);
        #pragma unroll //UNROLLHERE
        for (int i=0; i<numRegs; i++) F_here_array[i] = __float2half2_rn(negInftyFloat);
        if (threadIdx.x % group_size == 0) {
            penalty_left = __floats2half2_rn(0,0);
            penalty_diag = __floats2half2_rn(0,0);
            #pragma unroll
            for (int i=0; i<numRegs; i++) penalty_here_array[i] = __floats2half2_rn(0,0);
        }
        if (threadIdx.x % group_size == 1) {
            penalty_left = __floats2half2_rn(0,0);
        }
    }

    __device__
    void load_subject_regs(SequenceLengthT offset_isc, int (&subject)[numRegs], 
        const char* const devS0, const SequenceLengthT length_S0, 
        const char* const devS1, const SequenceLengthT length_S1
    ) const{
        #pragma unroll //UNROLLHERE
        for (int i=0; i<numRegs; i++) {

            if (offset_isc+numRegs*(threadIdx.x%group_size)+i >= length_S0) subject[i] = (deviceBlosumDimCexpr-1); // 20;
            else{
                
                subject[i] = devS0[offset_isc+numRegs*(threadIdx.x%group_size)+i];
            }

            if (offset_isc+numRegs*(threadIdx.x%group_size)+i >= length_S1) subject[i] += (deviceBlosumDimCexpr-1)*deviceBlosumDimCexpr; // 20*deviceBlosumDimCexpr;
            else subject[i] += deviceBlosumDimCexpr* devS1[offset_isc+numRegs*(threadIdx.x%group_size)+i];
        }

    }

    __device__
    void shuffle_query(char new_letter, char& query_letter) const{
        query_letter = __shfl_up_sync(0xFFFFFFFF, query_letter, 1, 32);
        const int group_id = threadIdx.x % group_size;
        if (!group_id) query_letter = new_letter;
    }

    __device__
    void shuffle_new_query(char4& new_query_letter4) const{
        const int temp = __shfl_down_sync(0xFFFFFFFF, *((int*)(&new_query_letter4)), 1, 32);
        new_query_letter4 = *((char4*)(&temp));
    }

    __device__
    void shuffle_affine_penalty(
        __half2 new_penalty_left, __half2 new_E_left, __half2& E, 
        __half2 penalty_here31, __half2& penalty_diag, __half2& penalty_left
    ) const{
        penalty_diag = penalty_left;
        penalty_left = __shfl_up_sync(0xFFFFFFFF, penalty_here31, 1, 32);
        E = __shfl_up_sync(0xFFFFFFFF, E, 1, 32);
        const int group_id = threadIdx.x % group_size;
        if (!group_id) {
            penalty_left = new_penalty_left;
            E = new_E_left;
        }
    }

    __device__
    void shuffle_H_E_temp_out(__half2& H_temp_out, __half2& E_temp_out) const{
        const uint32_t temp = __shfl_down_sync(0xFFFFFFFF, *((int*)(&H_temp_out)), 1, 32);
        H_temp_out = *((half2*)(&temp));
        const uint32_t temp2 = __shfl_down_sync(0xFFFFFFFF, *((int*)(&E_temp_out)), 1, 32);
        E_temp_out = *((half2*)(&temp2));
    }

    __device__
    void shuffle_H_E_temp_in(__half2& H_temp_in, __half2& E_temp_in) const{
        const uint32_t temp = __shfl_down_sync(0xFFFFFFFF, *((int*)(&H_temp_in)), 1, 32);
        H_temp_in = *((half2*)(&temp));
        const uint32_t temp2 = __shfl_down_sync(0xFFFFFFFF, *((int*)(&E_temp_in)), 1, 32);
        E_temp_in = *((half2*)(&temp2));
    }

    __device__
    void set_H_E_temp_out(__half2 E, __half2 penalty_here31, __half2& H_temp_out, __half2& E_temp_out) const{
        if (threadIdx.x % group_size == 31) {
            H_temp_out = penalty_here31;
            E_temp_out = E;
        }
    }

    __device__
    void computeFirstPass(__half2& maximum, const char* const devS0, const SequenceLengthT length_S0, 
        const char* const devS1, const SequenceLengthT length_S1,
        const char4* query4,
        SequenceLengthT queryLength
    ) const{
        // FIRST PASS (of many passes)
        // Note first pass has always full seqeunce length

        int counter = 1;
        char query_letter = 20;
        char4 new_query_letter4 = query4[threadIdx.x%group_size];
        if (threadIdx.x % group_size== 0) query_letter = new_query_letter4.x;

        int check_last;
        int check_last2;
        computeCheckLast(check_last, check_last2);
        const size_t numGroupsPerBlock = blockDim.x / group_size;
        const size_t groupIdInBlock = threadIdx.x / group_size;
        const size_t groupIdInGrid = numGroupsPerBlock * size_t(blockIdx.x) + groupIdInBlock;
        const size_t base_3 = groupIdInGrid * size_t(getPaddedQueryLength(queryLength)); //temp of both subjects is packed into half2
        
        __half2* const devTempHcol = (&devTempHcol2[base_3]);
        __half2* const devTempEcol = (&devTempEcol2[base_3]);

        const int group_id = threadIdx.x % group_size;
        int offset = group_id + group_size;
        int offset_out = group_id;


        __half2 E = __float2half2_rn(negInftyFloat);
        __half2 penalty_here31;
        __half2 penalty_diag;
        __half2 penalty_left;
        __half2 H_temp_out;
        __half2 E_temp_out;
        int subject[numRegs];
        __half2 penalty_here_array[numRegs];
        __half2 F_here_array[numRegs];

        
        init_penalties_local(0, penalty_diag, penalty_left, penalty_here_array, F_here_array);
        load_subject_regs(0, subject, devS0, length_S0, devS1, length_S1);
        initial_calc32_local_affine_float(0, query_letter, E, penalty_here31, penalty_diag, penalty_left, maximum, subject, penalty_here_array, F_here_array);
        shuffle_query(new_query_letter4.y, query_letter);
        shuffle_affine_penalty(__float2half2_rn(0.0), __float2half2_rn(negInftyFloat), E, penalty_here31, penalty_diag, penalty_left);

        //shuffle_max();
        calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
        shuffle_query(new_query_letter4.z, query_letter);
        shuffle_affine_penalty(__float2half2_rn(0.0), __float2half2_rn(negInftyFloat), E, penalty_here31, penalty_diag, penalty_left);

        //shuffle_max();
        calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
        shuffle_query(new_query_letter4.w, query_letter);
        shuffle_affine_penalty(__float2half2_rn(0.0), __float2half2_rn(negInftyFloat), E, penalty_here31, penalty_diag, penalty_left);
        shuffle_new_query(new_query_letter4);
        counter++;

        for (SequenceLengthT k = 4; k <= queryLength+28; k+=4) {
            //shuffle_max();
            calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out(E, penalty_here31, H_temp_out, E_temp_out);
            shuffle_H_E_temp_out(H_temp_out, E_temp_out);

            shuffle_query(new_query_letter4.x, query_letter);
            shuffle_affine_penalty(__float2half2_rn(0.0), __float2half2_rn(negInftyFloat), E, penalty_here31, penalty_diag, penalty_left);

            //shuffle_max();
            calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out(E, penalty_here31, H_temp_out, E_temp_out);
            shuffle_H_E_temp_out(H_temp_out, E_temp_out);

            shuffle_query(new_query_letter4.y, query_letter);
            shuffle_affine_penalty(__float2half2_rn(0.0), __float2half2_rn(negInftyFloat), E, penalty_here31, penalty_diag, penalty_left);

            //shuffle_max();
            calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out(E, penalty_here31, H_temp_out, E_temp_out);
            shuffle_H_E_temp_out(H_temp_out, E_temp_out);

            shuffle_query(new_query_letter4.z, query_letter);
            shuffle_affine_penalty(__float2half2_rn(0.0), __float2half2_rn(negInftyFloat), E, penalty_here31, penalty_diag, penalty_left);

            //shuffle_max();
            calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);

            set_H_E_temp_out(E, penalty_here31, H_temp_out, E_temp_out);
            if (counter%8 == 0 && counter > 8) {
                checkHEindex(offset_out, queryLength, __LINE__);
                devTempHcol[offset_out]=H_temp_out;
                devTempEcol[offset_out]=E_temp_out;
                offset_out += group_size;
            }
            if (k != queryLength+28) shuffle_H_E_temp_out(H_temp_out, E_temp_out);
            shuffle_query(new_query_letter4.w, query_letter);
            shuffle_affine_penalty(__float2half2_rn(0.0), __float2half2_rn(negInftyFloat), E, penalty_here31, penalty_diag, penalty_left);
            shuffle_new_query(new_query_letter4);
            if (counter%group_size == 0) {
                new_query_letter4 = query4[offset];
                offset += group_size;
            }
            counter++;
        }
        //if (queryLength % 4 == 0) {
            //temp = __shfl_up_sync(0xFFFFFFFF, *((int*)(&H_temp_out)), 1, 32);
            //H_temp_out = *((half2*)(&temp));
            //temp = __shfl_up_sync(0xFFFFFFFF, *((int*)(&E_temp_out)), 1, 32);
            //E_temp_out = *((half2*)(&temp));
        //}
        if (queryLength%4 == 1) {
            //shuffle_max();
            calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out(E, penalty_here31, H_temp_out, E_temp_out);
        }

        if (queryLength%4 == 2) {
            //shuffle_max();
            calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out(E, penalty_here31, H_temp_out, E_temp_out);
            shuffle_H_E_temp_out(H_temp_out, E_temp_out);
            shuffle_query(new_query_letter4.x, query_letter);
            shuffle_affine_penalty(__float2half2_rn(0.0), __float2half2_rn(negInftyFloat), E, penalty_here31, penalty_diag, penalty_left);
            //shuffle_max();
            calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out(E, penalty_here31, H_temp_out, E_temp_out);
        }
        if (queryLength%4 == 3) {
            //shuffle_max();
            calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out(E, penalty_here31, H_temp_out, E_temp_out);
            shuffle_H_E_temp_out(H_temp_out, E_temp_out);
            shuffle_query(new_query_letter4.x, query_letter);
            shuffle_affine_penalty(__float2half2_rn(0.0), __float2half2_rn(negInftyFloat), E, penalty_here31, penalty_diag, penalty_left);
            //shuffle_max();
            calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out(E, penalty_here31, H_temp_out, E_temp_out);
            shuffle_H_E_temp_out(H_temp_out, E_temp_out);
            shuffle_query(new_query_letter4.y, query_letter);
            shuffle_affine_penalty(__float2half2_rn(0.0), __float2half2_rn(negInftyFloat), E, penalty_here31, penalty_diag, penalty_left);
            //shuffle_max();
            calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out(E, penalty_here31, H_temp_out, E_temp_out);
        }
        const int final_out = queryLength % 32;
        const int from_thread_id = 32 - final_out;

        //printf("tid %d, offset_out %d, from_thread_id %d\n", threadIdx.x, offset_out, from_thread_id);
        if (threadIdx.x>=from_thread_id) {
            checkHEindex(offset_out-from_thread_id, queryLength, __LINE__);
            devTempHcol[offset_out-from_thread_id]=H_temp_out;
            devTempEcol[offset_out-from_thread_id]=E_temp_out;
        }
    }

    __device__
    void computeMiddlePass(int pass, __half2& maximum, const char* const devS0, const SequenceLengthT length_S0, 
        const char* const devS1, const SequenceLengthT length_S1,
        const char4* query4,
        SequenceLengthT queryLength
    ) const{
        int counter = 1;
        char query_letter = 20;
        char4 new_query_letter4 = query4[threadIdx.x%group_size];
        if (threadIdx.x % group_size== 0) query_letter = new_query_letter4.x;

        int check_last;
        int check_last2;
        computeCheckLast(check_last, check_last2);
        const size_t numGroupsPerBlock = blockDim.x / group_size;
        const size_t groupIdInBlock = threadIdx.x / group_size;
        const size_t groupIdInGrid = numGroupsPerBlock * size_t(blockIdx.x) + groupIdInBlock;
        const size_t base_3 = groupIdInGrid * size_t(getPaddedQueryLength(queryLength)); //temp of both subjects is packed into half2
        
        __half2* const devTempHcol = (&devTempHcol2[base_3]);
        __half2* const devTempEcol = (&devTempEcol2[base_3]);

        const int group_id = threadIdx.x % group_size;
        int offset = group_id + group_size;
        int offset_out = group_id;
        int offset_in = group_id;
        checkHEindex(offset_in, queryLength, __LINE__);
        __half2 H_temp_in = devTempHcol[offset_in];
        __half2 E_temp_in = devTempEcol[offset_in];
        offset_in += group_size;

        __half2 E = __float2half2_rn(negInftyFloat);
        __half2 penalty_here31;
        __half2 penalty_diag;
        __half2 penalty_left;
        __half2 H_temp_out;
        __half2 E_temp_out;
        int subject[numRegs];
        __half2 penalty_here_array[numRegs];
        __half2 F_here_array[numRegs];

        init_penalties_local(0, penalty_diag, penalty_left, penalty_here_array, F_here_array);
        load_subject_regs(pass*(32*numRegs), subject, devS0, length_S0, devS1, length_S1);

        if (!group_id) {
            penalty_left = H_temp_in;
            E = E_temp_in;
        }
        shuffle_H_E_temp_in(H_temp_in, E_temp_in);

        initial_calc32_local_affine_float(1, query_letter, E, penalty_here31, penalty_diag, penalty_left, maximum, subject, penalty_here_array, F_here_array);
        shuffle_query(new_query_letter4.y, query_letter);
        shuffle_affine_penalty(H_temp_in, E_temp_in, E, penalty_here31, penalty_diag, penalty_left);
        shuffle_H_E_temp_in(H_temp_in, E_temp_in);

        //shuffle_max();
        calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
        shuffle_query(new_query_letter4.z, query_letter);
        shuffle_affine_penalty(H_temp_in, E_temp_in, E, penalty_here31, penalty_diag, penalty_left);
        shuffle_H_E_temp_in(H_temp_in, E_temp_in);

        //shuffle_max();
        calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
        shuffle_query(new_query_letter4.w, query_letter);
        shuffle_affine_penalty(H_temp_in, E_temp_in, E, penalty_here31, penalty_diag, penalty_left);
        shuffle_H_E_temp_in(H_temp_in, E_temp_in);
        shuffle_new_query(new_query_letter4);
        counter++;

        for (SequenceLengthT k = 4; k <= queryLength+28; k+=4) {
            //shuffle_max();
            calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out(E, penalty_here31, H_temp_out, E_temp_out);
            shuffle_H_E_temp_out(H_temp_out, E_temp_out);

            shuffle_query(new_query_letter4.x, query_letter);
            shuffle_affine_penalty(H_temp_in, E_temp_in, E, penalty_here31, penalty_diag, penalty_left);
            shuffle_H_E_temp_in(H_temp_in, E_temp_in);

            //shuffle_max();
            calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out(E, penalty_here31, H_temp_out, E_temp_out);
            shuffle_H_E_temp_out(H_temp_out, E_temp_out);

            shuffle_query(new_query_letter4.y, query_letter);
            shuffle_affine_penalty(H_temp_in, E_temp_in, E, penalty_here31, penalty_diag, penalty_left);
            shuffle_H_E_temp_in(H_temp_in, E_temp_in);

            //shuffle_max();
            calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out(E, penalty_here31, H_temp_out, E_temp_out);
            shuffle_H_E_temp_out(H_temp_out, E_temp_out);

            shuffle_query(new_query_letter4.z, query_letter);
            shuffle_affine_penalty(H_temp_in, E_temp_in, E, penalty_here31, penalty_diag, penalty_left);
            shuffle_H_E_temp_in(H_temp_in, E_temp_in);

            //shuffle_max();
            calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out(E, penalty_here31, H_temp_out, E_temp_out);
            if (counter%8 == 0 && counter > 8) {
                checkHEindex(offset_out, queryLength, __LINE__);
                devTempHcol[offset_out]=H_temp_out;
                devTempEcol[offset_out]=E_temp_out;
                offset_out += group_size;
            }
            if (k != queryLength+28) shuffle_H_E_temp_out(H_temp_out, E_temp_out);
            shuffle_query(new_query_letter4.w, query_letter);
            shuffle_affine_penalty(H_temp_in, E_temp_in, E, penalty_here31, penalty_diag, penalty_left);
            shuffle_new_query(new_query_letter4);
            if (counter%group_size == 0) {
                new_query_letter4 = query4[offset];
                offset += group_size;
            }
            shuffle_H_E_temp_in(H_temp_in, E_temp_in);
            if (counter%8 == 0) {
                checkHEindex(offset_in, queryLength, __LINE__);
                H_temp_in = devTempHcol[offset_in];
                E_temp_in = devTempEcol[offset_in];
                offset_in += group_size;
            }
            counter++;
        }
        //if (queryLength % 4 == 0) {
            //temp = __shfl_up_sync(0xFFFFFFFF, *((int*)(&H_temp_out)), 1, 32);
            //H_temp_out = *((half2*)(&temp));
            //temp = __shfl_up_sync(0xFFFFFFFF, *((int*)(&E_temp_out)), 1, 32);
            //E_temp_out = *((half2*)(&temp));
        //}
        if (queryLength%4 == 1) {
            //shuffle_max();
            calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out(E, penalty_here31, H_temp_out, E_temp_out);
        }

        if (queryLength%4 == 2) {
            //shuffle_max();
            calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out(E, penalty_here31, H_temp_out, E_temp_out);
            shuffle_H_E_temp_out(H_temp_out, E_temp_out);
            shuffle_query(new_query_letter4.x, query_letter);
            shuffle_affine_penalty(H_temp_in, E_temp_in, E, penalty_here31, penalty_diag, penalty_left);
            shuffle_H_E_temp_in(H_temp_in, E_temp_in);
            //shuffle_max();
            calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out(E, penalty_here31, H_temp_out, E_temp_out);
        }
        if (queryLength%4 == 3) {
            //shuffle_max();
            calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out(E, penalty_here31, H_temp_out, E_temp_out);
            shuffle_H_E_temp_out(H_temp_out, E_temp_out);
            shuffle_query(new_query_letter4.x, query_letter);
            shuffle_affine_penalty(H_temp_in, E_temp_in, E, penalty_here31, penalty_diag, penalty_left);
            shuffle_H_E_temp_in(H_temp_in, E_temp_in);
            //shuffle_max();
            calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out(E, penalty_here31, H_temp_out, E_temp_out);
            shuffle_H_E_temp_out(H_temp_out, E_temp_out);
            shuffle_query(new_query_letter4.y, query_letter);
            shuffle_affine_penalty(H_temp_in, E_temp_in, E, penalty_here31, penalty_diag, penalty_left);
            shuffle_H_E_temp_in(H_temp_in, E_temp_in);
            //shuffle_max();
            calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out(E, penalty_here31, H_temp_out, E_temp_out);
        }
        const int final_out = queryLength % 32;
        const int from_thread_id = 32 - final_out;

        if (threadIdx.x>=from_thread_id) {
            checkHEindex(offset_out-from_thread_id, queryLength, __LINE__);
            devTempHcol[offset_out-from_thread_id]=H_temp_out;
            devTempEcol[offset_out-from_thread_id]=E_temp_out;
        }
    }

    __device__ 
    void computeFinalPass(int passes, __half2& maximum, const char* const devS0, const SequenceLengthT length_S0, 
        const char* const devS1, const SequenceLengthT length_S1,
        const char4* query4,
        SequenceLengthT queryLength
    ) const{
        int counter = 1;
        char query_letter = 20;
        char4 new_query_letter4 = query4[threadIdx.x%group_size];
        if (threadIdx.x % group_size== 0) query_letter = new_query_letter4.x;

        int check_last;
        int check_last2;
        computeCheckLast(check_last, check_last2);
        const size_t numGroupsPerBlock = blockDim.x / group_size;
        const size_t groupIdInBlock = threadIdx.x / group_size;
        const size_t groupIdInGrid = numGroupsPerBlock * size_t(blockIdx.x) + groupIdInBlock;
        const size_t base_3 = groupIdInGrid * size_t(getPaddedQueryLength(queryLength)); //temp of both subjects is packed into half2
        
        __half2* const devTempHcol = (&devTempHcol2[base_3]);
        __half2* const devTempEcol = (&devTempEcol2[base_3]);

        const int group_id = threadIdx.x % group_size;
        int offset = group_id + group_size;
        int offset_in = group_id;
        checkHEindex(offset_in, queryLength, __LINE__);
        __half2 H_temp_in = devTempHcol[offset_in];
        __half2 E_temp_in = devTempEcol[offset_in];
        offset_in += group_size;

        const int length = max(length_S0, length_S1);
        const int thread_result = ((length-1)%(group_size*numRegs))/numRegs; 

        __half2 E = __float2half2_rn(negInftyFloat);
        __half2 penalty_here31;
        __half2 penalty_diag;
        __half2 penalty_left;
        int subject[numRegs];
        __half2 penalty_here_array[numRegs];
        __half2 F_here_array[numRegs];

        init_penalties_local(0, penalty_diag, penalty_left, penalty_here_array, F_here_array);
        load_subject_regs((passes-1)*(32*numRegs), subject, devS0, length_S0, devS1, length_S1);
        //copy_H_E_temp_in();
        if (!group_id) {
            penalty_left = H_temp_in;
            E = E_temp_in;
        }
        shuffle_H_E_temp_in(H_temp_in, E_temp_in);

        initial_calc32_local_affine_float(1, query_letter, E, penalty_here31, penalty_diag, penalty_left, maximum, subject, penalty_here_array, F_here_array);
        shuffle_query(new_query_letter4.y, query_letter);
        shuffle_affine_penalty(H_temp_in, E_temp_in, E, penalty_here31, penalty_diag, penalty_left);
        shuffle_H_E_temp_in(H_temp_in, E_temp_in);
        //if (queryLength+thread_result >=2) {
        if(1 < queryLength+thread_result){
            //shuffle_max();
            calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            shuffle_query(new_query_letter4.z, query_letter);
            //copy_H_E_temp_in();
            shuffle_affine_penalty(H_temp_in, E_temp_in, E, penalty_here31, penalty_diag, penalty_left);
            shuffle_H_E_temp_in(H_temp_in, E_temp_in);
        }

        //if (queryLength+thread_result >=3) {
        if(2 < queryLength+thread_result){
            //shuffle_max();
            calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            shuffle_query(new_query_letter4.w, query_letter);
            //copy_H_E_temp_in();
            shuffle_affine_penalty(H_temp_in, E_temp_in, E, penalty_here31, penalty_diag, penalty_left);
            shuffle_H_E_temp_in(H_temp_in, E_temp_in);
            shuffle_new_query(new_query_letter4);
            counter++;
        }
        //if (queryLength+thread_result >=4) {
        if(3 < queryLength+thread_result){
            SequenceLengthT k;
            //for (k = 5; k < lane_2+thread_result-2; k+=4) {
            //for (k = 4; k <= queryLength+(thread_result-3); k+=4) {
            for (k = 3; k < queryLength+thread_result-3; k+=4) {
                //shuffle_max();
                calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);

                shuffle_query(new_query_letter4.x, query_letter);
                //copy_H_E_temp_in();
                shuffle_affine_penalty(H_temp_in, E_temp_in, E, penalty_here31, penalty_diag, penalty_left);
                shuffle_H_E_temp_in(H_temp_in, E_temp_in);

                //shuffle_max();
                calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
                shuffle_query(new_query_letter4.y, query_letter);
                //copy_H_E_temp_in();
                shuffle_affine_penalty(H_temp_in, E_temp_in, E, penalty_here31, penalty_diag, penalty_left);
                shuffle_H_E_temp_in(H_temp_in, E_temp_in);

                //shuffle_max();
                calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
                shuffle_query(new_query_letter4.z, query_letter);
                //copy_H_E_temp_in();
                shuffle_affine_penalty(H_temp_in, E_temp_in, E, penalty_here31, penalty_diag, penalty_left);
                shuffle_H_E_temp_in(H_temp_in, E_temp_in);

                //shuffle_max();
                calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
                shuffle_query(new_query_letter4.w, query_letter);
                //copy_H_E_temp_in();
                shuffle_affine_penalty(H_temp_in, E_temp_in, E, penalty_here31, penalty_diag, penalty_left);
                shuffle_new_query(new_query_letter4);
                if (counter%group_size == 0) {
                    new_query_letter4 = query4[offset];
                    offset += group_size;
                }
                shuffle_H_E_temp_in(H_temp_in, E_temp_in);
                if (counter%8 == 0) {
                    checkHEindex(offset_in, queryLength, __LINE__);
                    H_temp_in = devTempHcol[offset_in];
                    E_temp_in = devTempEcol[offset_in];
                    offset_in += group_size;
                }
                counter++;
            }

            //if ((k-1)-(queryLength+thread_result) > 0) {
            if(k < queryLength+thread_result){
                //shuffle_max();
                calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
                shuffle_query(new_query_letter4.x, query_letter);
                shuffle_affine_penalty(H_temp_in, E_temp_in, E, penalty_here31, penalty_diag, penalty_left);
                shuffle_H_E_temp_in(H_temp_in, E_temp_in);
                k++;
            }


            //if ((k-1)-(queryLength+thread_result) > 0) {
            if(k < queryLength+thread_result){
                //shuffle_max();
                calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
                shuffle_query(new_query_letter4.y, query_letter);
                shuffle_affine_penalty(H_temp_in, E_temp_in, E, penalty_here31, penalty_diag, penalty_left);
                shuffle_H_E_temp_in(H_temp_in, E_temp_in);
                k++;
            }

            //if ((k-1)-(queryLength+thread_result) > 0) {
            if(k < queryLength+thread_result){
                //shuffle_max();
                calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            }
        }
    }

    __device__ 
    void computeSinglePass(__half2& maximum, const char* const devS0, const SequenceLengthT length_S0, 
        const char* const devS1, const SequenceLengthT length_S1, const SequenceLengthT warpMaxLength,
        const char4* query4,
        SequenceLengthT queryLength
    ) const{
        int counter = 1;
        char query_letter = 20;
        char4 new_query_letter4 = query4[threadIdx.x%group_size];
        if (threadIdx.x % group_size== 0) query_letter = new_query_letter4.x;


        const int group_id = threadIdx.x % group_size;
        int offset = group_id + group_size;

        const int thread_result = ((warpMaxLength-1)%(group_size*numRegs))/numRegs; 

        __half2 E = __float2half2_rn(negInftyFloat);
        __half2 penalty_here31;
        __half2 penalty_diag;
        __half2 penalty_left;
        int subject[numRegs];
        __half2 penalty_here_array[numRegs];
        __half2 F_here_array[numRegs];

        init_penalties_local(0, penalty_diag, penalty_left, penalty_here_array, F_here_array);
        load_subject_regs(0, subject, devS0, length_S0, devS1, length_S1);

        // if(threadIdx.x < group_size){
        //     for(int t = 0; t < group_size; t++){
        //         if(t == threadIdx.x){
        //             printf("tid %d subject\n", threadIdx.x);
        //             for(int i = 0; i < numRegs; i++){
        //                 printf("%d ", int(subject[i]));
        //             }
        //             printf("\n");
        //         }
        //     }
        // }

        initial_calc32_local_affine_float(0, query_letter, E, penalty_here31, penalty_diag, penalty_left, maximum, subject, penalty_here_array, F_here_array);
        shuffle_query(new_query_letter4.y, query_letter);
        shuffle_affine_penalty(__float2half2_rn(0.0), __float2half2_rn(negInftyFloat), E, penalty_here31, penalty_diag, penalty_left);
//if(threadIdx.x < 1) printf("A tid %d maximum %f %f, %d %d %d %d, %d\n", threadIdx.x, float(maximum.x), float(maximum.y), int(new_query_letter4.x), int(new_query_letter4.y), int(new_query_letter4.z), int(new_query_letter4.w), int(query_letter));
        //if (queryLength+thread_result >=2) {
        if(1 < queryLength+thread_result){
            //shuffle_max();
            calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            shuffle_query(new_query_letter4.z, query_letter);
            shuffle_affine_penalty(__float2half2_rn(0.0), __float2half2_rn(negInftyFloat), E, penalty_here31, penalty_diag, penalty_left);
//if(threadIdx.x < 1) printf("B tid %d maximum %f %f, %d %d %d %d, %d\n", threadIdx.x, float(maximum.x), float(maximum.y), int(new_query_letter4.x), int(new_query_letter4.y), int(new_query_letter4.z), int(new_query_letter4.w), int(query_letter));        
        }

        //if (queryLength+thread_result >=3) {
        if(2 < queryLength+thread_result){
            //shuffle_max();
            calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            shuffle_query(new_query_letter4.w, query_letter);
            shuffle_affine_penalty(__float2half2_rn(0.0), __float2half2_rn(negInftyFloat), E, penalty_here31, penalty_diag, penalty_left);
            shuffle_new_query(new_query_letter4);
            counter++;
//if(threadIdx.x < 1) printf("C tid %d maximum %f %f, %d %d %d %d, %d\n", threadIdx.x, float(maximum.x), float(maximum.y), int(new_query_letter4.x), int(new_query_letter4.y), int(new_query_letter4.z), int(new_query_letter4.w), int(query_letter));
        }
        //if (queryLength+thread_result >=4) {
        if(3 < queryLength+thread_result){
            SequenceLengthT k;
            //for (k = 5; k < lane_2+thread_result-2; k+=4) {
            //for (k = 4; k <= queryLength+(thread_result-3); k+=4) {
            for (k = 3; k < queryLength+thread_result-3; k+=4) {
                //shuffle_max();
                calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
//if(threadIdx.x < 1) printf("D tid %d maximum %f %f, %d %d %d %d, %d\n", threadIdx.x, float(maximum.x), float(maximum.y), int(new_query_letter4.x), int(new_query_letter4.y), int(new_query_letter4.z), int(new_query_letter4.w), int(query_letter));
                shuffle_query(new_query_letter4.x, query_letter);
                shuffle_affine_penalty(__float2half2_rn(0.0), __float2half2_rn(negInftyFloat), E, penalty_here31, penalty_diag, penalty_left);

                //shuffle_max();
                calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
                shuffle_query(new_query_letter4.y, query_letter);
                shuffle_affine_penalty(__float2half2_rn(0.0), __float2half2_rn(negInftyFloat), E, penalty_here31, penalty_diag, penalty_left);
//if(threadIdx.x < 1) printf("E tid %d maximum %f %f, %d %d %d %d, %d\n", threadIdx.x, float(maximum.x), float(maximum.y), int(new_query_letter4.x), int(new_query_letter4.y), int(new_query_letter4.z), int(new_query_letter4.w), int(query_letter));
                //shuffle_max();
                calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
                shuffle_query(new_query_letter4.z, query_letter);
                shuffle_affine_penalty(__float2half2_rn(0.0), __float2half2_rn(negInftyFloat), E, penalty_here31, penalty_diag, penalty_left);
//if(threadIdx.x < 1) printf("F tid %d maximum %f %f, %d %d %d %d, %d\n", threadIdx.x, float(maximum.x), float(maximum.y), int(new_query_letter4.x), int(new_query_letter4.y), int(new_query_letter4.z), int(new_query_letter4.w), int(query_letter));
                //shuffle_max();
                calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
                shuffle_query(new_query_letter4.w, query_letter);
                shuffle_affine_penalty(__float2half2_rn(0.0), __float2half2_rn(negInftyFloat), E, penalty_here31, penalty_diag, penalty_left);
                shuffle_new_query(new_query_letter4);
                if (counter%group_size == 0) {
                    new_query_letter4 = query4[offset];
                    offset += group_size;
                }
                counter++;
//if(threadIdx.x < 1) printf("G tid %d maximum %f %f, %d %d %d %d, %d\n", threadIdx.x, float(maximum.x), float(maximum.y), int(new_query_letter4.x), int(new_query_letter4.y), int(new_query_letter4.z), int(new_query_letter4.w), int(query_letter));                
            }

            //if(threadIdx.x < 1) printf("tid %d, k %d queryLength %d thread_result %d\n", threadIdx.x, k, queryLength, thread_result);


// (int(196) - 1) - (int(189) + unsigned(7))
            //if ((k-1)-(queryLength+thread_result) > 0) {
            if(k < queryLength+thread_result){
                //shuffle_max();
                calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
                shuffle_query(new_query_letter4.x, query_letter);
                shuffle_affine_penalty(__float2half2_rn(0.0), __float2half2_rn(negInftyFloat), E, penalty_here31, penalty_diag, penalty_left);
                k++;
//if(threadIdx.x < 1) printf("H tid %d maximum %f %f, %d %d %d %d, %d\n", threadIdx.x, float(maximum.x), float(maximum.y), int(new_query_letter4.x), int(new_query_letter4.y), int(new_query_letter4.z), int(new_query_letter4.w), int(query_letter));
            }


            //if ((k-1)-(queryLength+thread_result) > 0) {
            if(k < queryLength+thread_result){
                //shuffle_max();
                calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
                shuffle_query(new_query_letter4.y, query_letter);
                shuffle_affine_penalty(__float2half2_rn(0.0), __float2half2_rn(negInftyFloat), E, penalty_here31, penalty_diag, penalty_left);
                k++;
//if(threadIdx.x < 1) printf("I tid %d maximum %f %f, %d %d %d %d, %d\n", threadIdx.x, float(maximum.x), float(maximum.y), int(new_query_letter4.x), int(new_query_letter4.y), int(new_query_letter4.z), int(new_query_letter4.w), int(query_letter));                
            }

            

            //if ((k-1)-(queryLength+thread_result) > 0) {
            if(k < queryLength+thread_result){
                //shuffle_max();
                calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
//if(threadIdx.x < 1) printf("J tid %d maximum %f %f, %d %d %d %d, %d\n", threadIdx.x, float(maximum.x), float(maximum.y), int(new_query_letter4.x), int(new_query_letter4.y), int(new_query_letter4.z), int(new_query_letter4.w), int(query_letter));                
            }
        }
    }

    template<class ScoreOutputIterator>
    __device__
    void computeMultiPass(
        ScoreOutputIterator const devAlignmentScores,
        const bool overflow_check, 
        int* const d_overflow_number, 
        ReferenceIdT* const d_overflow_positions,
        const char4* query4,
        SequenceLengthT queryLength
    ) const{
        int check_last;
        int check_last2;
        computeCheckLast(check_last, check_last2);

        const SequenceLengthT length_S0 = devLengths[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blockIdx.x+2*((threadIdx.x%check_last)/group_size)]];
        const size_t base_S0 = devOffsets[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blockIdx.x+2*((threadIdx.x%check_last)/group_size)]]-devOffsets[0];
        SequenceLengthT length_S1 = devLengths[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blockIdx.x+2*((threadIdx.x%check_last)/group_size)+1]];
        size_t base_S1 = devOffsets[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blockIdx.x+2*((threadIdx.x%check_last)/group_size)+1]]-devOffsets[0];
        if (blockIdx.x == gridDim.x-1){
            if (check_last2){
                if (((threadIdx.x%check_last) >= check_last-group_size) && ((threadIdx.x%check_last) < check_last)) {
                    length_S1 = length_S0;
                    base_S1 = base_S0;
                }
            }
        }

        const char* const devS0 = &devChars[base_S0];
        const char* const devS1 = &devChars[base_S1];

        const SequenceLengthT length = max(length_S0, length_S1);
        const int passes = (length + (group_size*numRegs) - 1) / (group_size*numRegs);
        // constexpr int length = 4096;
        // constexpr int passes = (length + (group_size*numRegs) - 1) / (group_size*numRegs);

        __half2 maximum = __float2half2_rn(0.0);

        computeFirstPass(maximum, devS0, length_S0, devS1, length_S1, query4, queryLength);

        for (int pass = 1; pass < passes-1; pass++) {
            computeMiddlePass(pass, maximum, devS0, length_S0, devS1, length_S1, query4, queryLength);
        }

        computeFinalPass(passes, maximum, devS0, length_S0, devS1, length_S1, query4, queryLength);

        for (int offset=group_size/2; offset>0; offset/=2){
            maximum = __hmax2(maximum,__shfl_down_sync(0xFFFFFFFF,maximum,offset,group_size));
        }

        const int group_id = threadIdx.x % group_size;
        if (!group_id) {

            // check for overflow
            if (overflow_check){
                half max_half2 = __float2half_rn(MAX_ACC_HALF2);
                const ReferenceIdT alignmentNumber0 = 2*(blockDim.x/group_size)*blockIdx.x+2*(threadIdx.x/group_size);
                if(alignmentNumber0 < numSelected){
                    if (maximum.y >= max_half2) {
                        //overflow happened
                        const int pos_overflow = atomicAdd(d_overflow_number,1);
                        d_overflow_positions[pos_overflow] = d_positions_of_selected_lengths[alignmentNumber0];
                    }else{
                        //no overflow happened, update score
                        devAlignmentScores[d_positions_of_selected_lengths[alignmentNumber0]] =  maximum.y;
                    }
                }
                const ReferenceIdT alignmentNumber1 = 2*(blockDim.x/group_size)*blockIdx.x+2*(threadIdx.x/group_size)+1;
                if(alignmentNumber1 < numSelected){
                    if (maximum.x >= max_half2) {
                        const int pos_overflow = atomicAdd(d_overflow_number,1);
                        d_overflow_positions[pos_overflow] = d_positions_of_selected_lengths[alignmentNumber1];
                    }else{
                        //no overflow happened, update score
                        devAlignmentScores[d_positions_of_selected_lengths[alignmentNumber1]] =  maximum.x;
                    }
                }
            }else{
                //update all computed scores without checking for overflow
                int check_last;
                int check_last2;
                computeCheckLast(check_last, check_last2);
                if (blockIdx.x < gridDim.x-1) {
                    devAlignmentScores[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blockIdx.x+2*(threadIdx.x/group_size)]] =  maximum.y; // lane_2+thread_result+1-queryLength%4; penalty_here_array[(length-1)%numRegs];
                    devAlignmentScores[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blockIdx.x+2*(threadIdx.x/group_size)+1]] =  maximum.x; // lane_2+thread_result+1-queryLength%4; penalty_here_array[(length-1)%numRegs];
                } else {
                    devAlignmentScores[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blockIdx.x+2*((threadIdx.x%check_last)/group_size)]] =  maximum.y; // lane_2+thread_result+1-queryLength%4; penalty_here_array[(length-1)%numRegs];
                    if (!check_last2 || (threadIdx.x%check_last) < check_last-group_size) devAlignmentScores[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blockIdx.x+2*((threadIdx.x%check_last)/group_size)+1]] =  maximum.x; // lane_2+thread_result+1-queryLength%4; penalty_here_array[(length-1)%numRegs];
                }
            }
        }
    }

    template<class ScoreOutputIterator>
    __device__
    void computeSinglePass(
        ScoreOutputIterator const devAlignmentScores,
        const bool overflow_check, 
        int* const d_overflow_number, 
        ReferenceIdT* const d_overflow_positions,
        const char4* query4,
        SequenceLengthT queryLength
    ) const{
        int check_last;
        int check_last2;
        computeCheckLast(check_last, check_last2);

        const ReferenceIdT alignmentId_checklast_0 = d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blockIdx.x+2*((threadIdx.x%check_last)/group_size)];
        const ReferenceIdT alignmentId_checklast_1 = d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blockIdx.x+2*((threadIdx.x%check_last)/group_size)+1];
        const ReferenceIdT alignmentId_0 = d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blockIdx.x+2*(threadIdx.x/group_size)];
        const ReferenceIdT alignmentId_1 = d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blockIdx.x+2*(threadIdx.x/group_size)+1];


        const SequenceLengthT length_S0 = devLengths[alignmentId_checklast_0];
        const size_t base_S0 = devOffsets[alignmentId_checklast_0]-devOffsets[0];
        SequenceLengthT length_S1 = length_S0;
        size_t base_S1 = base_S0;

        if ((blockIdx.x < gridDim.x-1) || (!check_last2) || ((threadIdx.x%check_last) < check_last-group_size) || ((threadIdx.x%check_last) >= check_last)) {
            length_S1 = devLengths[alignmentId_checklast_1];
            base_S1 = devOffsets[alignmentId_checklast_1]-devOffsets[0];
        }


        const char* const devS0 = &devChars[base_S0];
        const char* const devS1 = &devChars[base_S1];

        const SequenceLengthT temp_length = max(length_S0, length_S1);
        const SequenceLengthT warpMaxLength = warp_max_reduce_broadcast(0xFFFFFFFF, temp_length);
        const int passes = (warpMaxLength + (group_size*numRegs) - 1) / (group_size*numRegs);
        if(passes == 1){
            __half2 maximum = __float2half2_rn(0.0);

            computeSinglePass(maximum, devS0, length_S0, devS1, length_S1, warpMaxLength, query4, queryLength);

            // if(length_S0 == 251){
            //     printf("251 y, %f\n", float(maximum.y));
            // }
            // if(length_S1 == 251){
            //     printf("251 x, %f\n", float(maximum.x));
            // }

            for (int offset=group_size/2; offset>0; offset/=2){
                maximum = __hmax2(maximum,__shfl_down_sync(0xFFFFFFFF,maximum,offset,group_size));
            }

            {
                if(threadIdx.x == 0){
                    // printf("final maximum %f %f\n", float(maximum.x), float(maximum.y));
                }
            }

            const int group_id = threadIdx.x % group_size;
            if (!group_id) {

                // check for overflow
                if (overflow_check){
                    half max_half2 = __float2half_rn(MAX_ACC_HALF2);
                    const ReferenceIdT alignmentNumber0 = 2*(blockDim.x/group_size)*blockIdx.x+2*(threadIdx.x/group_size);
                    if(alignmentNumber0 < numSelected){
                        if (maximum.y >= max_half2) {
                            //overflow happened
                            const int pos_overflow = atomicAdd(d_overflow_number,1);
                            d_overflow_positions[pos_overflow] = d_positions_of_selected_lengths[alignmentNumber0];
                        }else{
                            //no overflow happened, update score
                            devAlignmentScores[d_positions_of_selected_lengths[alignmentNumber0]] =  maximum.y;
                        }
                    }
                    const ReferenceIdT alignmentNumber1 = 2*(blockDim.x/group_size)*blockIdx.x+2*(threadIdx.x/group_size)+1;
                    if(alignmentNumber1 < numSelected){
                        if (maximum.x >= max_half2) {
                            const int pos_overflow = atomicAdd(d_overflow_number,1);
                            d_overflow_positions[pos_overflow] = d_positions_of_selected_lengths[alignmentNumber1];
                        }else{
                            //no overflow happened, update score
                            devAlignmentScores[d_positions_of_selected_lengths[alignmentNumber1]] =  maximum.x;
                        }
                    }
                }else{
                    //update all computed scores without checking for overflow
                    int check_last;
                    int check_last2;
                    computeCheckLast(check_last, check_last2);
                    if (blockIdx.x < gridDim.x-1) {
                        devAlignmentScores[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blockIdx.x+2*(threadIdx.x/group_size)]] =  maximum.y; // lane_2+thread_result+1-queryLength%4; penalty_here_array[(length-1)%numRegs];
                        devAlignmentScores[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blockIdx.x+2*(threadIdx.x/group_size)+1]] =  maximum.x; // lane_2+thread_result+1-queryLength%4; penalty_here_array[(length-1)%numRegs];
                    } else {
                        devAlignmentScores[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blockIdx.x+2*((threadIdx.x%check_last)/group_size)]] =  maximum.y; // lane_2+thread_result+1-queryLength%4; penalty_here_array[(length-1)%numRegs];
                        if (!check_last2 || (threadIdx.x%check_last) < check_last-group_size) devAlignmentScores[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blockIdx.x+2*((threadIdx.x%check_last)/group_size)+1]] =  maximum.x; // lane_2+thread_result+1-queryLength%4; penalty_here_array[(length-1)%numRegs];
                    }
                }
            }
        }
    }
};



// Needleman-Wunsch (NW): global alignment with linear gap penalty
// numRegs values per thread
// uses a single warp per CUDA thread block;
// every groupsize threads computes an alignmen score
// devTempHcol2 and devTempEcol2 each must have length numBlocks * (blocksize / group_size) * (SDIV(queryLength, 4) * 4 + 32 * sizeof(char4));

template <int blocksize, int group_size, int numRegs, int blosumDim, class ScoreOutputIterator, class PositionsIterator> 
#if __CUDA_ARCH__ >= 800
__launch_bounds__(blocksize,2)
#else
__launch_bounds__(blocksize)
#endif
__global__
void NW_local_affine_multi_pass_half2(
    __grid_constant__ const char * const devChars,
    __grid_constant__ ScoreOutputIterator const devAlignmentScores,
    __grid_constant__ __half2 * const devTempHcol2,
    __grid_constant__ __half2 * const devTempEcol2,
    __grid_constant__ const size_t* const devOffsets,
    __grid_constant__ const SequenceLengthT* const devLengths,
    __grid_constant__ PositionsIterator const d_positions_of_selected_lengths,
    __grid_constant__ const int numSelected,
	__grid_constant__ ReferenceIdT* const d_overflow_positions,
	__grid_constant__ int* const d_overflow_number,
	__grid_constant__ const bool overflow_check,
    __grid_constant__ const char4* const query4,
    __grid_constant__ const SequenceLengthT queryLength,
    __grid_constant__ const float gap_open,
    __grid_constant__ const float gap_extend
) {
    static_assert(blocksize % group_size == 0);

    __builtin_assume(blockDim.x == blocksize);
    __builtin_assume(blockDim.x % group_size == 0);
    
    using Processor = Half2Aligner<group_size, numRegs, blosumDim, PositionsIterator>;
    extern __shared__ __half2 shared_blosum[];
    //__shared__ typename Processor::BLOSUM62_SMEM shared_blosum;

    Processor processor(
        shared_blosum,
        devChars,
        devTempHcol2,
        devTempEcol2,
        devOffsets,
        devLengths,
        d_positions_of_selected_lengths,
        numSelected,
        gap_open,
        gap_extend
    );

    processor.computeMultiPass(devAlignmentScores, overflow_check, d_overflow_number, d_overflow_positions, query4, queryLength);
}



template <int blocksize, int group_size, int numRegs, class ScoreOutputIterator, class PositionsIterator> 
void call_NW_local_affine_multi_pass_half2(
    BlosumType /*blosumType*/,
    const char * const devChars,
    ScoreOutputIterator const devAlignmentScores,
    __half2 * const devTempHcol2,
    __half2 * const devTempEcol2,
    const size_t* const devOffsets,
    const SequenceLengthT* const devLengths,
    PositionsIterator const d_positions_of_selected_lengths,
    const int numSelected,
	ReferenceIdT* const d_overflow_positions,
	int* const d_overflow_number,
	const bool overflow_check,
    const char4* query4,
    const SequenceLengthT queryLength,
    const float gap_open,
    const float gap_extend,
    cudaStream_t stream
) {
    constexpr int groupsPerBlock = blocksize / group_size;
    constexpr int alignmentsPerGroup = 2;
    constexpr int alignmentsPerBlock = groupsPerBlock * alignmentsPerGroup;

    int smem = sizeof(__half2) * hostBlosumDim * hostBlosumDim * hostBlosumDim;

    if(hostBlosumDim == 21){
        auto kernel = NW_local_affine_multi_pass_half2<blocksize, group_size, numRegs, 21, ScoreOutputIterator, PositionsIterator>;
        assert(smem <= 48 * 1024);

        dim3 block = blocksize;
        dim3 grid = SDIV(numSelected, alignmentsPerBlock);

        kernel<<<grid, block, smem, stream>>>(
            devChars,
            devAlignmentScores,
            devTempHcol2,
            devTempEcol2,
            devOffsets,
            devLengths,
            d_positions_of_selected_lengths,
            numSelected,
            d_overflow_positions,
            d_overflow_number,
            overflow_check,
            query4,
            queryLength,
            gap_open,
            gap_extend
        ); CUERR;
    #ifdef CAN_USE_FULL_BLOSUM
    }else if(hostBlosumDim == 25){
        auto kernel = NW_local_affine_multi_pass_half2<blocksize, group_size, numRegs, 25, ScoreOutputIterator, PositionsIterator>;
        auto setSmemKernelAttribute = [&](){
            static bool isSet = false;
            if(!isSet){
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
                isSet = true;
            }
        };

        setSmemKernelAttribute();

        dim3 block = blocksize;
        dim3 grid = SDIV(numSelected, alignmentsPerBlock);

        kernel<<<grid, block, smem, stream>>>(
            devChars,
            devAlignmentScores,
            devTempHcol2,
            devTempEcol2,
            devOffsets,
            devLengths,
            d_positions_of_selected_lengths,
            numSelected,
            d_overflow_positions,
            d_overflow_number,
            overflow_check,
            query4,
            queryLength,
            gap_open,
            gap_extend
        ); CUERR;
    #endif
    }else{
        assert(false);
    }
}


// Needleman-Wunsch (NW): global alignment with linear gap penalty
// numRegs values per thread
// uses a single warp per CUDA thread block;
// every groupsize threads computes an alignmen score
template <int blocksize, int group_size, int numRegs, int blosumDim, class ScoreOutputIterator, class PositionsIterator> 
#if __CUDA_ARCH__ >= 800
__launch_bounds__(blocksize,2)
//__launch_bounds__(512,1)
#else
__launch_bounds__(blocksize)
#endif
__global__
void NW_local_affine_single_pass_half2(
    __grid_constant__ const char * const devChars,
    __grid_constant__ ScoreOutputIterator const devAlignmentScores,
    __grid_constant__ const size_t* const devOffsets,
    __grid_constant__ const SequenceLengthT* const devLengths,
    __grid_constant__ PositionsIterator const d_positions_of_selected_lengths,
    __grid_constant__ const int numSelected,
	__grid_constant__ ReferenceIdT* const d_overflow_positions,
	__grid_constant__ int* const d_overflow_number,
	__grid_constant__ const bool overflow_check,
    __grid_constant__ const char4* const query4,
    __grid_constant__ const SequenceLengthT queryLength,
    __grid_constant__ const float gap_open,
    __grid_constant__ const float gap_extend
) {
    static_assert(blocksize % group_size == 0);

    __builtin_assume(blockDim.x == blocksize);
    __builtin_assume(blockDim.x % group_size == 0);

    using Processor = Half2Aligner<group_size, numRegs, blosumDim, PositionsIterator>;
    extern __shared__ __half2 shared_blosum[];
    //__shared__ typename Processor::BLOSUM62_SMEM shared_blosum;

    Processor processor(
        shared_blosum,
        devChars,
        nullptr,
        nullptr,
        devOffsets,
        devLengths,
        d_positions_of_selected_lengths,
        numSelected,
        gap_open,
        gap_extend
    );

    processor.computeSinglePass(devAlignmentScores, overflow_check, d_overflow_number, d_overflow_positions, query4, queryLength);
}


template <int blocksize, int group_size, int numRegs, class ScoreOutputIterator, class PositionsIterator> 
void call_NW_local_affine_single_pass_half2(
    BlosumType /*blosumType*/,
    const char * const devChars,
    ScoreOutputIterator const devAlignmentScores,
    const size_t* const devOffsets,
    const SequenceLengthT* const devLengths,
    PositionsIterator const d_positions_of_selected_lengths,
    const int numSelected,
	ReferenceIdT* const d_overflow_positions,
	int* const d_overflow_number,
	const bool overflow_check,
    const char4* query4,
    const SequenceLengthT queryLength,
    const float gap_open,
    const float gap_extend,
    cudaStream_t stream
){
    
    constexpr int groupsPerBlock = blocksize / group_size;
    constexpr int alignmentsPerGroup = 2;
    constexpr int alignmentsPerBlock = groupsPerBlock * alignmentsPerGroup;

    int smem = sizeof(__half2) * hostBlosumDim * hostBlosumDim * hostBlosumDim;

    if(hostBlosumDim == 21){
        auto kernel = NW_local_affine_single_pass_half2<blocksize, group_size, numRegs, 21, ScoreOutputIterator, PositionsIterator>;
        assert(smem <= 48 * 1024);

        dim3 grid = (numSelected + alignmentsPerBlock - 1) / alignmentsPerBlock;
        kernel<<<grid, blocksize, smem, stream>>>(
            devChars,
            devAlignmentScores,
            devOffsets,
            devLengths,
            d_positions_of_selected_lengths,
            numSelected,
            d_overflow_positions,
            d_overflow_number,
            overflow_check,
            query4,
            queryLength,
            gap_open,
            gap_extend
        );
    #ifdef CAN_USE_FULL_BLOSUM
    }else if(hostBlosumDim == 25){
        auto kernel = NW_local_affine_single_pass_half2<blocksize, group_size, numRegs, 25, ScoreOutputIterator, PositionsIterator>;
        auto setSmemKernelAttribute = [&](){
            static bool isSet = false;
            if(!isSet){
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
                isSet = true;
            }
        };

        setSmemKernelAttribute();

        dim3 grid = (numSelected + alignmentsPerBlock - 1) / alignmentsPerBlock;
        kernel<<<grid, blocksize, smem, stream>>>(
            devChars,
            devAlignmentScores,
            devOffsets,
            devLengths,
            d_positions_of_selected_lengths,
            numSelected,
            d_overflow_positions,
            d_overflow_number,
            overflow_check,
            query4,
            queryLength,
            gap_open,
            gap_extend
        );
    #endif
    } else{
        assert(false);
    }

}

} //namespace cudasw4

#endif