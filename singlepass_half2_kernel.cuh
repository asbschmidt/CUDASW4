#ifndef SINGLEPASS_HALF2_KERNEL_CUH
#define SINGLEPASS_HALF2_KERNEL_CUH

#include <cuda_fp16.h>
#include "blosum.hpp"

template <int group_size, int numRegs, class PositionsIterator> 
struct SinglePassHalf2{

    static constexpr float negInftyFloat = -1000.0f;

    __half2* shared_blosum;

    //int& deviceBlosumDim;
    int numSelected;
    int length_2;
    float gap_open;
    float gap_extend;
    PositionsIterator d_positions_of_selected_lengths;
    const char* devChars;
    const size_t* devOffsets;
    const size_t* devLengths;

    __device__
    SinglePassHalf2(
        __half2* shared_blosum_,
        const char* devChars_,
        const size_t* devOffsets_,
        const size_t* devLengths_,
        PositionsIterator d_positions_of_selected_lengths_,
        int numSelected_,
        int length_2_,
        float gap_open_,
        float gap_extend_
    ) : shared_blosum(shared_blosum_),
        //deviceBlosumDim(deviceBlosumDim),
        devChars(devChars_),
        devOffsets(devOffsets_),
        devLengths(devLengths_),
        d_positions_of_selected_lengths(d_positions_of_selected_lengths_),
        numSelected(numSelected_),
        length_2(length_2_),
        gap_open(gap_open_),
        gap_extend(gap_extend_)
    {

        for (int i=threadIdx.x; i<deviceBlosumDimSquared; i+=blockDim.x) {
            __half2 temp0;
            const int A = (i/deviceBlosumDim);
            const int B = (i%deviceBlosumDim);
            temp0.x = deviceBlosum[deviceBlosumDim*A+B];
            for (int j=0; j<deviceBlosumDim; j++) {
                temp0.y = deviceBlosum[deviceBlosumDim*A+j];
                shared_blosum[A * deviceBlosumDimSquared + deviceBlosumDim*(B)+j]=temp0;
                //shared_blosum[(A * deviceBlosumDim + B) * deviceBlosumDim + j]=temp0;
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
    void checkHEindex(int x, int line) const{
        // if(x < 0){printf("line %d\n", line);}
        // assert(x >= 0); //positive index
        // assert(2*(blockDim.x/group_size)*blockIdx.x * length_2 <= base_3 + x);
        // assert(base_3+x < 2*(blockDim.x/group_size)*(blockIdx.x+1) * length_2);
    };

    __device__
    void initial_calc32_local_affine_float(const int value, char query_letter, __half2& E, __half2& penalty_here31, __half2 penalty_diag, __half2 penalty_left, __half2& maximum, 
        int (&subject)[numRegs], 
        __half2 (&penalty_here_array)[numRegs],
        __half2 (&F_here_array)[numRegs]
    ) const{
        const __half2* const sbt_row = &shared_blosum[int(query_letter) * deviceBlosumDimSquared];

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
        const __half2* const sbt_row = &shared_blosum[int(query_letter) * deviceBlosumDimSquared];

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
    void init_local_score_profile_BLOSUM62(int offset_isc, int (&subject)[numRegs], 
        const char* const devS0, const int length_S0, 
        const char* const devS1, const int length_S1
    ) const{
    //     if (!offset_isc) {
    //         for (int i=threadIdx.x; i<deviceBlosumDim*deviceBlosumDim; i+=blockDim.x) {
    //             const int A = 
    //             __half2 temp0;
    //             temp0.x = deviceBlosum[deviceBlosumDim*(i/deviceBlosumDim)+(i%deviceBlosumDim)];
    //             for (int j=0; j<deviceBlosumDim; j++) {
    //                 temp0.y = deviceBlosum[deviceBlosumDim*(i/deviceBlosumDim)+j];
    //                 //shared_blosum[(i/deviceBlosumDim) * deviceBlosumDim * deviceBlosumDim + deviceBlosumDim*(i%deviceBlosumDim)+j]=temp0;
    //                 shared_blosum[((i/deviceBlosumDim) * deviceBlosumDim + (i%deviceBlosumDim)) * deviceBlosumDim + j]=temp0;
    //             }
    //         }
    //         __syncthreads();
	//    }
       #pragma unroll //UNROLLHERE
       for (int i=0; i<numRegs; i++) {

           if (offset_isc+numRegs*(threadIdx.x%group_size)+i >= length_S0) subject[i] = 1; // 20;
           else{
                //printf("tid %d, i %d, offset_isc %d, base_S0 %lu, total %d\n", threadIdx.x, i, offset_isc, base_S0, offset_isc+base_S0+numRegs*(threadIdx.x%group_size)+i);
            subject[i] = devS0[offset_isc+numRegs*(threadIdx.x%group_size)+i]; //devChars[offset_isc+base_S0+numRegs*(threadIdx.x%group_size)+i];
           }

           if (offset_isc+numRegs*(threadIdx.x%group_size)+i >= length_S1) subject[i] += 1*deviceBlosumDim; // 20*deviceBlosumDim;
           else subject[i] += deviceBlosumDim* devS1[offset_isc+numRegs*(threadIdx.x%group_size)+i]; //devChars[offset_isc+base_S1+numRegs*(threadIdx.x%group_size)+i];
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
    void computeSinglePass(__half2& maximum, const char* const devS0, const int length_S0, 
        const char* const devS1, const int length_S1, const int warpMaxLength
    ) const{
        int counter = 1;
        char query_letter = 20;
        char4 new_query_letter4 = constantQuery4[threadIdx.x%group_size];
        if (threadIdx.x % group_size== 0) query_letter = new_query_letter4.x;


        const int group_id = threadIdx.x % group_size;
        int offset = group_id + group_size;

        const uint32_t thread_result = ((warpMaxLength-1)%(group_size*numRegs))/numRegs; 

        __half2 E = __float2half2_rn(negInftyFloat);
        __half2 penalty_here31;
        __half2 penalty_diag;
        __half2 penalty_left;
        int subject[numRegs];
        __half2 penalty_here_array[numRegs];
        __half2 F_here_array[numRegs];

        init_penalties_local(0, penalty_diag, penalty_left, penalty_here_array, F_here_array);
        init_local_score_profile_BLOSUM62(0, subject, devS0, length_S0, devS1, length_S1);

        initial_calc32_local_affine_float(0, query_letter, E, penalty_here31, penalty_diag, penalty_left, maximum, subject, penalty_here_array, F_here_array);
        shuffle_query(new_query_letter4.y, query_letter);
        shuffle_affine_penalty(__float2half2_rn(0.0), __float2half2_rn(negInftyFloat), E, penalty_here31, penalty_diag, penalty_left);

        if (length_2+thread_result >=2) {
            //shuffle_max();
            calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            shuffle_query(new_query_letter4.z, query_letter);
            shuffle_affine_penalty(__float2half2_rn(0.0), __float2half2_rn(negInftyFloat), E, penalty_here31, penalty_diag, penalty_left);
        }

        if (length_2+thread_result >=3) {
            //shuffle_max();
            calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            shuffle_query(new_query_letter4.w, query_letter);
            shuffle_affine_penalty(__float2half2_rn(0.0), __float2half2_rn(negInftyFloat), E, penalty_here31, penalty_diag, penalty_left);
            shuffle_new_query(new_query_letter4);
            counter++;
        }
        if (length_2+thread_result >=4) {
            int k;
            //for (k = 5; k < lane_2+thread_result-2; k+=4) {
            for (k = 4; k <= length_2+(thread_result-3); k+=4) {
                //shuffle_max();
                calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);

                shuffle_query(new_query_letter4.x, query_letter);
                shuffle_affine_penalty(__float2half2_rn(0.0), __float2half2_rn(negInftyFloat), E, penalty_here31, penalty_diag, penalty_left);

                //shuffle_max();
                calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
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
                if (counter%group_size == 0) {
                    new_query_letter4 = constantQuery4[offset];
                    offset += group_size;
                }
                counter++;
            }

            if ((k-1)-(length_2+thread_result) > 0) {
                //shuffle_max();
                calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
                shuffle_query(new_query_letter4.x, query_letter);
                shuffle_affine_penalty(__float2half2_rn(0.0), __float2half2_rn(negInftyFloat), E, penalty_here31, penalty_diag, penalty_left);
                k++;
            }


            if ((k-1)-(length_2+thread_result) > 0) {
                //shuffle_max();
                calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
                shuffle_query(new_query_letter4.y, query_letter);
                shuffle_affine_penalty(__float2half2_rn(0.0), __float2half2_rn(negInftyFloat), E, penalty_here31, penalty_diag, penalty_left);
                k++;
            }

            if ((k-1)-(length_2+thread_result) > 0) {
                //shuffle_max();
                calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            }
        }
    }

    template<class ScoreOutputIterator>
    __device__
    void compute(
        ScoreOutputIterator const devAlignmentScores,
        const bool overflow_check, 
        int* const d_overflow_number, 
        size_t* const d_overflow_positions
    ) const{
        int check_last;
        int check_last2;
        computeCheckLast(check_last, check_last2);

        const size_t alignmentId_checklast_0 = d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blockIdx.x+2*((threadIdx.x%check_last)/group_size)];
        const size_t alignmentId_checklast_1 = d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blockIdx.x+2*((threadIdx.x%check_last)/group_size)+1];
        const size_t alignmentId_0 = d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blockIdx.x+2*(threadIdx.x/group_size)];
        const size_t alignmentId_1 = d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blockIdx.x+2*(threadIdx.x/group_size)+1];


        const int length_S0 = devLengths[alignmentId_checklast_0];
        const size_t base_S0 = devOffsets[alignmentId_checklast_0]-devOffsets[0];
        int length_S1 = length_S0;
        size_t base_S1 = base_S0;

        if ((blockIdx.x < gridDim.x-1) || (!check_last2) || ((threadIdx.x%check_last) < check_last-group_size) || ((threadIdx.x%check_last) >= check_last)) {
            length_S1 = devLengths[alignmentId_checklast_1];
            base_S1 = devOffsets[alignmentId_checklast_1]-devOffsets[0];
        }

        const char* const devS0 = &devChars[base_S0];
        const char* const devS1 = &devChars[base_S1];

        const int temp_length = max(length_S0, length_S1);
        const int warpMaxLength = warp_max_reduce_broadcast(0xFFFFFFFF, temp_length);
        const int passes = (warpMaxLength + (group_size*numRegs) - 1) / (group_size*numRegs);
        if(passes == 1){
            __half2 maximum = __float2half2_rn(0.0);

            computeSinglePass(maximum, devS0, length_S0, devS1, length_S1, warpMaxLength);

            // if(length_S0 == 251){
            //     printf("251 y, %f\n", float(maximum.y));
            // }
            // if(length_S1 == 251){
            //     printf("251 x, %f\n", float(maximum.x));
            // }

            for (int offset=group_size/2; offset>0; offset/=2){
                maximum = __hmax2(maximum,__shfl_down_sync(0xFFFFFFFF,maximum,offset,group_size));
            }

            const int group_id = threadIdx.x % group_size;
            if (!group_id) {

                // check for overflow
                if (overflow_check){
                    half max_half2 = __float2half_rn(MAX_ACC_HALF2);
                    const int alignmentNumber0 = 2*(blockDim.x/group_size)*blockIdx.x+2*(threadIdx.x/group_size);
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
                    const int alignmentNumber1 = 2*(blockDim.x/group_size)*blockIdx.x+2*(threadIdx.x/group_size)+1;
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
                        devAlignmentScores[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blockIdx.x+2*(threadIdx.x/group_size)]] =  maximum.y; // lane_2+thread_result+1-length_2%4; penalty_here_array[(length-1)%numRegs];
                        devAlignmentScores[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blockIdx.x+2*(threadIdx.x/group_size)+1]] =  maximum.x; // lane_2+thread_result+1-length_2%4; penalty_here_array[(length-1)%numRegs];
                    } else {
                        devAlignmentScores[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blockIdx.x+2*((threadIdx.x%check_last)/group_size)]] =  maximum.y; // lane_2+thread_result+1-length_2%4; penalty_here_array[(length-1)%numRegs];
                        if (!check_last2 || (threadIdx.x%check_last) < check_last-group_size) devAlignmentScores[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blockIdx.x+2*((threadIdx.x%check_last)/group_size)+1]] =  maximum.x; // lane_2+thread_result+1-length_2%4; penalty_here_array[(length-1)%numRegs];
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
template <int group_size, int numRegs, class ScoreOutputIterator, class PositionsIterator> 
#if __CUDA_ARCH__ >= 800
__launch_bounds__(256,2)
//__launch_bounds__(512,1)
#else
__launch_bounds__(256)
#endif
__global__
void NW_local_affine_Protein_single_pass_half2_new(
    __grid_constant__ const char * const devChars,
    __grid_constant__ ScoreOutputIterator const devAlignmentScores,
    __grid_constant__ const size_t* const devOffsets,
    __grid_constant__ const size_t* const devLengths,
    __grid_constant__ PositionsIterator const d_positions_of_selected_lengths,
    __grid_constant__ const int numSelected,
	__grid_constant__ size_t* const d_overflow_positions,
	__grid_constant__ int* const d_overflow_number,
	__grid_constant__ const bool overflow_check,
    __grid_constant__ const int length_2,
    __grid_constant__ const float gap_open,
    __grid_constant__ const float gap_extend
) {
    // if(threadIdx.x == 0 && blockIdx.x == 0){
    //     printf("constantquery\n");
    //     for(int i = 0; i < 2048; i++){
    //         printf("%d %d %d %d", int(constantQuery4[i].x), int(constantQuery4[i].y), int(constantQuery4[i].z), int(constantQuery4[i].w));
    //     }
    //     printf("\n");
    //     printf("constantblosum\n");
    //     for(int i = 0; i < deviceBlosumDimSquared; i++){
    //         printf("%d %d %d %d", int(deviceBlosum[i]));
    //     }
    //     printf("\n");

    //     printf("offset %lu\n", devOffsets[0]);
    //     printf("length %lu\n", devLengths[0]);
    //     printf("d_positions_of_selected_lengths %lu\n", d_positions_of_selected_lengths[0]);
    //     printf("length_2 %d\n", length_2);
    //     printf("gap_open %f\n", gap_open);
    //     printf("gap_extend %f\n", gap_extend);
    // }
    using Processor = SinglePassHalf2<group_size, numRegs, PositionsIterator>;
    extern __shared__ __half2 shared_blosum[];
    //__shared__ typename Processor::BLOSUM62_SMEM shared_blosum;

    Processor processor(
        shared_blosum,
        devChars,
        devOffsets,
        devLengths,
        d_positions_of_selected_lengths,
        numSelected,
        length_2,
        gap_open,
        gap_extend
    );

    processor.compute(devAlignmentScores, overflow_check, d_overflow_number, d_overflow_positions);
}


template <int blocksize, int group_size, int numRegs, class ScoreOutputIterator, class PositionsIterator> 
void call_NW_local_affine_Protein_single_pass_half2_new(
    BlosumType blosumType,
    const char * const devChars,
    ScoreOutputIterator const devAlignmentScores,
    const size_t* const devOffsets,
    const size_t* const devLengths,
    PositionsIterator const d_positions_of_selected_lengths,
    const int numSelected,
	size_t* const d_overflow_positions,
	int* const d_overflow_number,
	const bool overflow_check,
    const int length_2,
    const float gap_open,
    const float gap_extend,
    cudaStream_t stream
){
    
    constexpr int groupsPerBlock = blocksize / group_size;
    constexpr int alignmentsPerGroup = 2;
    constexpr int alignmentsPerBlock = groupsPerBlock * alignmentsPerGroup;

    int smem = sizeof(__half2) * hostBlosumDim * hostBlosumDim * hostBlosumDim;

    auto kernel = NW_local_affine_Protein_single_pass_half2_new<group_size, numRegs, ScoreOutputIterator, PositionsIterator>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

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
        length_2,
        gap_open,
        gap_extend
    );

}

#endif