#ifndef MANYPASS_HALF2_KERNEL_CUH
#define MANYPASS_HALF2_KERNEL_CUH

#include <cuda_fp16.h>

template <int group_size, int numRegs, class PositionsIterator> 
struct ManyPassHalf2{
    using BLOSUM62_SMEM = __half2[21][21*21];

    static constexpr float negInftyFloat = -1000.0f;

    BLOSUM62_SMEM& shared_BLOSUM62;

    //input params
    const char* devChars;
    float* devAlignmentScores;
    __half2* devTempHcol2;
    __half2* devTempEcol2;
    const size_t* devOffsets;
    const size_t* devLengths;
    PositionsIterator d_positions_of_selected_lengths;
    int numSelected;
    size_t* d_overflow_positions;
    int* d_overflow_number;
    bool overflow_check;
    int length_2;
    float gap_open;
    float gap_extend;

    int thid;

    //variables
    int subject[numRegs];
    __half2 penalty_here_array[numRegs];
    __half2 F_here_array[numRegs];

    __half2 penalty_diag;
    __half2 penalty_left;
    __half2 penalty_here31;
    __half2 E;
    __half2 maximum;
    __half2 H_temp_out;
    __half2 H_temp_in;
    __half2 E_temp_out;
    __half2 E_temp_in;

    int group_id;
    int length_S0;
    size_t base_S0;
	int length_S1;
	size_t base_S1;
    __half2* devTempHcol;
    __half2* devTempEcol;

    __device__
    ManyPassHalf2(
        BLOSUM62_SMEM& shared_BLOSUM62_,
        const char* devChars_,
        float* devAlignmentScores_,
        __half2* devTempHcol2_,
        __half2* devTempEcol2_,
        const size_t* devOffsets_,
        const size_t* devLengths_,
        PositionsIterator d_positions_of_selected_lengths_,
        int numSelected_,
        size_t* d_overflow_positions_,
        int* d_overflow_number_,
        bool overflow_check_,
        int length_2_,
        float gap_open_,
        float gap_extend_
    ) : shared_BLOSUM62(shared_BLOSUM62_),
        devChars(devChars_),
        devAlignmentScores(devAlignmentScores_),
        devTempHcol2(devTempHcol2_),
        devTempEcol2(devTempEcol2_),
        devOffsets(devOffsets_),
        devLengths(devLengths_),
        d_positions_of_selected_lengths(d_positions_of_selected_lengths_),
        numSelected(numSelected_),
        d_overflow_positions(d_overflow_positions_),
        d_overflow_number(d_overflow_number_),
        overflow_check(overflow_check_),
        length_2(length_2_),
        gap_open(gap_open_),
        gap_extend(gap_extend_),
        thid(threadIdx.x)
    {
        const unsigned int blid = blockIdx.x;
        group_id = thid%group_size;

        int check_last;
        int check_last2;
        computeCheckLast(check_last, check_last2);

        check_last = blockDim.x/group_size;
        check_last2 = 0;
        if (blid == gridDim.x-1) {
            if (numSelected % (2*blockDim.x/group_size)) {
                check_last = (numSelected/2) % (blockDim.x/group_size);
                check_last2 = numSelected%2;
                check_last = check_last + check_last2;
            }
        }
        check_last = check_last * group_size;

        length_S0 = devLengths[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)]];
        base_S0 = devOffsets[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)]]-devOffsets[0];
        length_S1 = devLengths[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)+1]];
        base_S1 = devOffsets[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size)+1]]-devOffsets[0];
        if (blid == gridDim.x-1){
            if (check_last2){
                if (((thid%check_last) >= check_last-group_size) && ((thid%check_last) < check_last)) {
                    length_S1 = length_S0;
                    base_S1 = base_S0;
                }
            }
        }

        // query_letter = 20;
        // new_query_letter4 = constantQuery4[thid%group_size];
        // if (thid % group_size== 0) query_letter = new_query_letter4.x;
        

        const size_t base_3 = (2*(size_t(blockDim.x)/group_size)*size_t(blid)+2*((thid%check_last)/group_size))*size_t(length_2);
        maximum = __float2half2_rn(0.0);
        devTempHcol = (half2*)(&devTempHcol2[base_3]);
        devTempEcol = (half2*)(&devTempEcol2[base_3]);

    }

    __device__ 
    void computeCheckLast(int& check_last, int& check_last2) const{
        const unsigned int blid = blockIdx.x;
        check_last = blockDim.x/group_size;
        check_last2 = 0;
        if (blid == gridDim.x-1) {
            if (numSelected % (2*blockDim.x/group_size)) {
                check_last = (numSelected/2) % (blockDim.x/group_size);
                check_last2 = numSelected%2;
                check_last = check_last + check_last2;
            }
        }
        check_last = check_last * group_size;
    }

    __device__
    void checkHEindex(int x, int line){
        // if(x < 0){printf("line %d\n", line);}
        // assert(x >= 0); //positive index
        // assert(2*(blockDim.x/group_size)*blid * length_2 <= base_3 + x);
        // assert(base_3+x < 2*(blockDim.x/group_size)*(blid+1) * length_2);
    };

    __device__
    void initial_calc32_local_affine_float(const int value, char query_letter){
        const __half2* const sbt_row = shared_BLOSUM62[query_letter];

        const __half2 score2_0 = sbt_row[subject[0]];
        //score2.y = sbt_row[subject1[0].x];
        __half2 penalty_temp0 = penalty_here_array[0];
        if (!value || (thid%group_size)) E = __hmax2(__hadd2(E,__float2half2_rn(gap_extend)), __hadd2(penalty_left,__float2half2_rn(gap_open)));
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
    void calc32_local_affine_float(char query_letter){
        const __half2* const sbt_row = shared_BLOSUM62[query_letter];

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
    };

    __device__    
    void init_penalties_local(int value) {
        //ZERO = __float2half2_rn(negInftyFloat);
        penalty_left = __float2half2_rn(negInftyFloat);
        penalty_diag = __float2half2_rn(negInftyFloat);
        E = __float2half2_rn(negInftyFloat);
        #pragma unroll
        for (int i=0; i<numRegs; i++) penalty_here_array[i] = __float2half2_rn(negInftyFloat);
        #pragma unroll //UNROLLHERE
        for (int i=0; i<numRegs; i++) F_here_array[i] = __float2half2_rn(negInftyFloat);
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

    __device__
    void init_local_score_profile_BLOSUM62(int offset_isc) {
        if (!offset_isc) {
            for (int i=thid; i<21*21; i+=blockDim.x) {
                __half2 temp0;
                temp0.x = cBLOSUM62_dev[21*(i/21)+(i%21)];
                for (int j=0; j<21; j++) {
                    temp0.y = cBLOSUM62_dev[21*(i/21)+j];
                    shared_BLOSUM62[i/21][21*(i%21)+j]=temp0;
                }
            }
            __syncthreads();
	   }
       #pragma unroll //UNROLLHERE
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

    __device__
    char shuffle_query(char new_letter, char query_letter) {
        query_letter = __shfl_up_sync(0xFFFFFFFF, query_letter, 1, 32);
        if (!group_id) query_letter = new_letter;
        return query_letter;
    };

    __device__
    char4 shuffle_new_query(char4 new_query_letter4) {
        const int temp = __shfl_down_sync(0xFFFFFFFF, *((int*)(&new_query_letter4)), 1, 32);
        new_query_letter4 = *((char4*)(&temp));
        return new_query_letter4;
    };

    __device__
    void shuffle_affine_penalty(__half2 new_penalty_left, __half2 new_E_left) {
        penalty_diag = penalty_left;
        penalty_left = __shfl_up_sync(0xFFFFFFFF, penalty_here31, 1, 32);
        E = __shfl_up_sync(0xFFFFFFFF, E, 1, 32);
        if (!group_id) {
            penalty_left = new_penalty_left;
            E = new_E_left;
        }
    };

    __device__
    void shuffle_H_E_temp_out() {
        const uint32_t temp = __shfl_down_sync(0xFFFFFFFF, *((int*)(&H_temp_out)), 1, 32);
        H_temp_out = *((half2*)(&temp));
        const uint32_t temp2 = __shfl_down_sync(0xFFFFFFFF, *((int*)(&E_temp_out)), 1, 32);
        E_temp_out = *((half2*)(&temp2));
    };

    __device__
    void shuffle_H_E_temp_in() {
        const uint32_t temp = __shfl_down_sync(0xFFFFFFFF, *((int*)(&H_temp_in)), 1, 32);
        H_temp_in = *((half2*)(&temp));
        const uint32_t temp2 = __shfl_down_sync(0xFFFFFFFF, *((int*)(&E_temp_in)), 1, 32);
        E_temp_in = *((half2*)(&temp2));

    };

    __device__
    void set_H_E_temp_out() {
        if (thid % group_size == 31) {
            H_temp_out = penalty_here31;
            E_temp_out = E;
        }
    };

    __device__
    void computeFirstPass(){
        // FIRST PASS (of many passes)
        // Note first pass has always full seqeunce length

        int offset = group_id + group_size;
        int offset_out = group_id;
        int counter = 1;
        char query_letter = 20;
        char4 new_query_letter4 = constantQuery4[thid%group_size];
        if (thid % group_size== 0) query_letter = new_query_letter4.x;
        
        init_penalties_local(0);
        init_local_score_profile_BLOSUM62(0);
        initial_calc32_local_affine_float(0, query_letter);
        query_letter = shuffle_query(new_query_letter4.y, query_letter);
        shuffle_affine_penalty(__float2half2_rn(0.0), __float2half2_rn(negInftyFloat));

        //shuffle_max();
        calc32_local_affine_float(query_letter);
        query_letter = shuffle_query(new_query_letter4.z, query_letter);
        shuffle_affine_penalty(__float2half2_rn(0.0), __float2half2_rn(negInftyFloat));

        //shuffle_max();
        calc32_local_affine_float(query_letter);
        query_letter = shuffle_query(new_query_letter4.w, query_letter);
        shuffle_affine_penalty(__float2half2_rn(0.0), __float2half2_rn(negInftyFloat));
        new_query_letter4 = shuffle_new_query(new_query_letter4);
        counter++;

        for (int k = 4; k <= length_2+28; k+=4) {
            //shuffle_max();
            calc32_local_affine_float(query_letter);
            set_H_E_temp_out();
            shuffle_H_E_temp_out();

            query_letter = shuffle_query(new_query_letter4.x, query_letter);
            shuffle_affine_penalty(__float2half2_rn(0.0),__float2half2_rn(negInftyFloat));

            //shuffle_max();
            calc32_local_affine_float(query_letter);
            set_H_E_temp_out();
            shuffle_H_E_temp_out();

            query_letter = shuffle_query(new_query_letter4.y, query_letter);
            shuffle_affine_penalty(__float2half2_rn(0.0),__float2half2_rn(negInftyFloat));

            //shuffle_max();
            calc32_local_affine_float(query_letter);
            set_H_E_temp_out();
            shuffle_H_E_temp_out();

            query_letter = shuffle_query(new_query_letter4.z, query_letter);
            shuffle_affine_penalty(__float2half2_rn(0.0),__float2half2_rn(negInftyFloat));

            //shuffle_max();
            calc32_local_affine_float(query_letter);

            set_H_E_temp_out();
            if (counter%8 == 0 && counter > 8) {
                checkHEindex(offset_out, __LINE__);
                devTempHcol[offset_out]=H_temp_out;
                devTempEcol[offset_out]=E_temp_out;
                offset_out += group_size;
            }
            if (k != length_2+28) shuffle_H_E_temp_out();
            query_letter = shuffle_query(new_query_letter4.w, query_letter);
            shuffle_affine_penalty(__float2half2_rn(0.0),__float2half2_rn(negInftyFloat));
            new_query_letter4 = shuffle_new_query(new_query_letter4);
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
            calc32_local_affine_float(query_letter);
            set_H_E_temp_out();
        }

        if (length_2%4 == 2) {
            //shuffle_max();
            calc32_local_affine_float(query_letter);
            set_H_E_temp_out();
            shuffle_H_E_temp_out();
            query_letter = shuffle_query(new_query_letter4.x, query_letter);
            shuffle_affine_penalty(__float2half2_rn(0.0),__float2half2_rn(negInftyFloat));
            //shuffle_max();
            calc32_local_affine_float(query_letter);
            set_H_E_temp_out();
        }
        if (length_2%4 == 3) {
            //shuffle_max();
            calc32_local_affine_float(query_letter);
            set_H_E_temp_out();
            shuffle_H_E_temp_out();
            query_letter = shuffle_query(new_query_letter4.x, query_letter);
            shuffle_affine_penalty(__float2half2_rn(0.0),__float2half2_rn(negInftyFloat));
            //shuffle_max();
            calc32_local_affine_float(query_letter);
            set_H_E_temp_out();
            shuffle_H_E_temp_out();
            query_letter = shuffle_query(new_query_letter4.y, query_letter);
            shuffle_affine_penalty(__float2half2_rn(0.0),__float2half2_rn(negInftyFloat));
            //shuffle_max();
            calc32_local_affine_float(query_letter);
            set_H_E_temp_out();
        }
        const int final_out = length_2 % 32;
        const int from_thread_id = 32 - final_out;

        //printf("tid %d, offset_out %d, from_thread_id %d\n", threadIdx.x, offset_out, from_thread_id);
        if (thid>=from_thread_id) {
            checkHEindex(offset_out-from_thread_id, __LINE__);
            devTempHcol[offset_out-from_thread_id]=H_temp_out;
            devTempEcol[offset_out-from_thread_id]=E_temp_out;
        }
    }

    __device__
    void computeMiddlePass(int pass){
        int counter = 1;
        char query_letter = 20;
        char4 new_query_letter4 = constantQuery4[thid%group_size];
        if (thid % group_size== 0) query_letter = new_query_letter4.x;

        int offset = group_id + group_size;
        int offset_out = group_id;
        int offset_in = group_id;
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

        initial_calc32_local_affine_float(1, query_letter);
        query_letter = shuffle_query(new_query_letter4.y, query_letter);
        shuffle_affine_penalty(H_temp_in,E_temp_in);
        shuffle_H_E_temp_in();

        //shuffle_max();
        calc32_local_affine_float(query_letter);
        query_letter = shuffle_query(new_query_letter4.z, query_letter);
        shuffle_affine_penalty(H_temp_in,E_temp_in);
        shuffle_H_E_temp_in();

        //shuffle_max();
        calc32_local_affine_float(query_letter);
        query_letter = shuffle_query(new_query_letter4.w, query_letter);
        shuffle_affine_penalty(H_temp_in,E_temp_in);
        shuffle_H_E_temp_in();
        new_query_letter4 = shuffle_new_query(new_query_letter4);
        counter++;

        for (int k = 4; k <= length_2+28; k+=4) {
            //shuffle_max();
            calc32_local_affine_float(query_letter);
            set_H_E_temp_out();
            shuffle_H_E_temp_out();

            query_letter = shuffle_query(new_query_letter4.x, query_letter);
            shuffle_affine_penalty(H_temp_in,E_temp_in);
            shuffle_H_E_temp_in();

            //shuffle_max();
            calc32_local_affine_float(query_letter);
            set_H_E_temp_out();
            shuffle_H_E_temp_out();

            query_letter = shuffle_query(new_query_letter4.y, query_letter);
            shuffle_affine_penalty(H_temp_in,E_temp_in);
            shuffle_H_E_temp_in();

            //shuffle_max();
            calc32_local_affine_float(query_letter);
            set_H_E_temp_out();
            shuffle_H_E_temp_out();

            query_letter = shuffle_query(new_query_letter4.z, query_letter);
            shuffle_affine_penalty(H_temp_in,E_temp_in);
            shuffle_H_E_temp_in();

            //shuffle_max();
            calc32_local_affine_float(query_letter);
            set_H_E_temp_out();
            if (counter%8 == 0 && counter > 8) {
                checkHEindex(offset_out, __LINE__);
                devTempHcol[offset_out]=H_temp_out;
                devTempEcol[offset_out]=E_temp_out;
                offset_out += group_size;
            }
            if (k != length_2+28) shuffle_H_E_temp_out();
            query_letter = shuffle_query(new_query_letter4.w, query_letter);
            shuffle_affine_penalty(H_temp_in,E_temp_in);
            new_query_letter4 = shuffle_new_query(new_query_letter4);
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
            calc32_local_affine_float(query_letter);
            set_H_E_temp_out();
        }

        if (length_2%4 == 2) {
            //shuffle_max();
            calc32_local_affine_float(query_letter);
            set_H_E_temp_out();
            shuffle_H_E_temp_out();
            query_letter = shuffle_query(new_query_letter4.x, query_letter);
            shuffle_affine_penalty(H_temp_in,E_temp_in);
            shuffle_H_E_temp_in();
            //shuffle_max();
            calc32_local_affine_float(query_letter);
            set_H_E_temp_out();
        }
        if (length_2%4 == 3) {
            //shuffle_max();
            calc32_local_affine_float(query_letter);
            set_H_E_temp_out();
            shuffle_H_E_temp_out();
            query_letter = shuffle_query(new_query_letter4.x, query_letter);
            shuffle_affine_penalty(H_temp_in,E_temp_in);
            shuffle_H_E_temp_in();
            //shuffle_max();
            calc32_local_affine_float(query_letter);
            set_H_E_temp_out();
            shuffle_H_E_temp_out();
            query_letter = shuffle_query(new_query_letter4.y, query_letter);
            shuffle_affine_penalty(H_temp_in,E_temp_in);
            shuffle_H_E_temp_in();
            //shuffle_max();
            calc32_local_affine_float(query_letter);
            set_H_E_temp_out();
        }
        const int final_out = length_2 % 32;
        const int from_thread_id = 32 - final_out;

        if (thid>=from_thread_id) {
            checkHEindex(offset_out-from_thread_id, __LINE__);
            devTempHcol[offset_out-from_thread_id]=H_temp_out;
            devTempEcol[offset_out-from_thread_id]=E_temp_out;
        }
    }

    __device__ 
    void computeFinalPass(int passes){
        int counter = 1;
        char query_letter = 20;
        char4 new_query_letter4 = constantQuery4[thid%group_size];
        if (thid % group_size== 0) query_letter = new_query_letter4.x;

        int offset = group_id + group_size;
        int offset_in = group_id;
        checkHEindex(offset_in, __LINE__);
        H_temp_in = devTempHcol[offset_in];
        E_temp_in = devTempEcol[offset_in];
        offset_in += group_size;

        const int length = max(length_S0, length_S1);
        const uint32_t thread_result = ((length-1)%(group_size*numRegs))/numRegs; 

        init_penalties_local(0);
        init_local_score_profile_BLOSUM62((passes-1)*(32*numRegs));
        //copy_H_E_temp_in();
        if (!group_id) {
            penalty_left = H_temp_in;
            E = E_temp_in;
        }
        shuffle_H_E_temp_in();

        initial_calc32_local_affine_float(1, query_letter);
        query_letter = shuffle_query(new_query_letter4.y, query_letter);
        shuffle_affine_penalty(H_temp_in,E_temp_in);
        shuffle_H_E_temp_in();
        if (length_2+thread_result >=2) {
            //shuffle_max();
            calc32_local_affine_float(query_letter);
            query_letter = shuffle_query(new_query_letter4.z, query_letter);
            //copy_H_E_temp_in();
            shuffle_affine_penalty(H_temp_in,E_temp_in);
            shuffle_H_E_temp_in();
        }

        if (length_2+thread_result >=3) {
            //shuffle_max();
            calc32_local_affine_float(query_letter);
            query_letter = shuffle_query(new_query_letter4.w, query_letter);
            //copy_H_E_temp_in();
            shuffle_affine_penalty(H_temp_in,E_temp_in);
            shuffle_H_E_temp_in();
            new_query_letter4 = shuffle_new_query(new_query_letter4);
            counter++;
        }
        if (length_2+thread_result >=4) {
            int k;
            //for (k = 5; k < lane_2+thread_result-2; k+=4) {
            for (k = 4; k <= length_2+(thread_result-3); k+=4) {
                //shuffle_max();
                calc32_local_affine_float(query_letter);

                query_letter = shuffle_query(new_query_letter4.x, query_letter);
                //copy_H_E_temp_in();
                shuffle_affine_penalty(H_temp_in,E_temp_in);
                shuffle_H_E_temp_in();

                //shuffle_max();
                calc32_local_affine_float(query_letter);
                query_letter = shuffle_query(new_query_letter4.y, query_letter);
                //copy_H_E_temp_in();
                shuffle_affine_penalty(H_temp_in,E_temp_in);
                shuffle_H_E_temp_in();

                //shuffle_max();
                calc32_local_affine_float(query_letter);
                query_letter = shuffle_query(new_query_letter4.z, query_letter);
                //copy_H_E_temp_in();
                shuffle_affine_penalty(H_temp_in,E_temp_in);
                shuffle_H_E_temp_in();

                //shuffle_max();
                calc32_local_affine_float(query_letter);
                query_letter = shuffle_query(new_query_letter4.w, query_letter);
                //copy_H_E_temp_in();
                shuffle_affine_penalty(H_temp_in,E_temp_in);
                new_query_letter4 = shuffle_new_query(new_query_letter4);
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

            if ((k-1)-(length_2+thread_result) > 0) {
                //shuffle_max();
                calc32_local_affine_float(query_letter);
                query_letter = shuffle_query(new_query_letter4.x, query_letter);
                shuffle_affine_penalty(H_temp_in,E_temp_in);
                shuffle_H_E_temp_in();
                k++;
            }


            if ((k-1)-(length_2+thread_result) > 0) {
                //shuffle_max();
                calc32_local_affine_float(query_letter);
                query_letter = shuffle_query(new_query_letter4.y, query_letter);
                shuffle_affine_penalty(H_temp_in,E_temp_in);
                shuffle_H_E_temp_in();
                k++;
            }

            if ((k-1)-(length_2+thread_result) > 0) {
                //shuffle_max();
                calc32_local_affine_float(query_letter);
            }
        }
    }

    __device__
    void compute(){
        const int length = max(length_S0, length_S1);
        const int passes = (length + (group_size*numRegs) - 1) / (group_size*numRegs);
        // constexpr int length = 4096;
        // constexpr int passes = (length + (group_size*numRegs) - 1) / (group_size*numRegs);

        computeFirstPass();

        for (int pass = 1; pass < passes-1; pass++) {
            computeMiddlePass(pass);
        }

        computeFinalPass(passes);

        for (int offset=group_size/2; offset>0; offset/=2){
            maximum = __hmax2(maximum,__shfl_down_sync(0xFFFFFFFF,maximum,offset,group_size));
        }

        if (!group_id) {
            const unsigned int blid = blockIdx.x;
            int check_last;
            int check_last2;
            computeCheckLast(check_last, check_last2);

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
};

// Needleman-Wunsch (NW): global alignment with linear gap penalty
// numRegs values per thread
// uses a single warp per CUDA thread block;
// every groupsize threads computes an alignmen score
template <int group_size, int numRegs, class PositionsIterator> 
__launch_bounds__(256,1)
//__launch_bounds__(128,1)
__global__
void NW_local_affine_Protein_many_pass_half2_new(
    // const char * devChars,
    // float * devAlignmentScores,
    // __half2 * devTempHcol2,
    // __half2 * devTempEcol2,
    // const size_t* devOffsets,
    // const size_t* devLengths,
    // PositionsIterator d_positions_of_selected_lengths,
    // const int numSelected,
	// size_t* d_overflow_positions,
	// int* d_overflow_number,
	// const bool overflow_check,
    // const int length_2,
    // const float gap_open,
    // const float gap_extend
    __grid_constant__ const char * const devChars,
    __grid_constant__ float * const devAlignmentScores,
    __grid_constant__ __half2 * const devTempHcol2,
    __grid_constant__ __half2 * const devTempEcol2,
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

    __shared__ __half2 shared_BLOSUM62[21][21*21];
    int subject[numRegs];

    //__shared__ __half2 devTempHcol[1024];
    //__shared__ __half2 devTempEcol[1024];

    // auto getNegInfinity = [](){
    //     constexpr unsigned int asUint = 3822117840u;
    //     return *((const __half2*)&asUint);
    // };

    constexpr float negInftyFloat = -1000.0f;

    //const __half2 NEGINFINITY2 = __float2half2_rn(negInftyFloat);
    //const __half2 NEGINFINITY2 = getNegInfinity();
    // if(threadIdx.x == 0 && blockIdx.x == 0){
    //     printf("%u\n", *((unsigned int*)&NEGINFINITY2));
    // }
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

    __half2 penalty_left, penalty_diag;
    __half2 penalty_here31;
    __half2 penalty_here_array[numRegs];
    __half2 F_here_array[numRegs];
    __half2 E = __float2half2_rn(negInftyFloat);

	const int base_3 = (2*(blockDim.x/group_size)*blid+2*((thid%check_last)/group_size))*length_2;
    // if(blockIdx.x == 0){
    //     printf("tid %d, base_3 %d, ql %d\n", threadIdx.x, base_3, length_2);
    // }
	//const int base_3 = (2*(blockDim.x/group_size)*blid+2*(thid/group_size))*(SDIV(length_2, group_size) * group_size);
    __half2 maximum = __float2half2_rn(0.0);
    __half2 * const devTempHcol = (half2*)(&devTempHcol2[base_3]);
    __half2 * const devTempEcol = (half2*)(&devTempEcol2[base_3]);

    auto checkHEindex = [&](auto x, int line){
        // if(x < 0){printf("line %d\n", line);}
        // assert(x >= 0); //positive index
        // assert(2*(blockDim.x/group_size)*blid * length_2 <= base_3 + x);
        // assert(base_3+x < 2*(blockDim.x/group_size)*(blid+1) * length_2);
    };


    auto init_penalties_local = [&](const auto& value) {
        //ZERO = __float2half2_rn(negInftyFloat);
        penalty_left = __float2half2_rn(negInftyFloat);
        penalty_diag = __float2half2_rn(negInftyFloat);
        E = __float2half2_rn(negInftyFloat);
        #pragma unroll
        for (int i=0; i<numRegs; i++) penalty_here_array[i] = __float2half2_rn(negInftyFloat);
        #pragma unroll //UNROLLHERE
        for (int i=0; i<numRegs; i++) F_here_array[i] = __float2half2_rn(negInftyFloat);
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


    auto init_local_score_profile_BLOSUM62 = [&](const auto& offset_isc) {
        if (!offset_isc) {
            for (int i=thid; i<21*21; i+=blockDim.x) {
                __half2 temp0;
                temp0.x = cBLOSUM62_dev[21*(i/21)+(i%21)];
                for (int j=0; j<21; j++) {
                    temp0.y = cBLOSUM62_dev[21*(i/21)+j];
                    shared_BLOSUM62[i/21][21*(i%21)+j]=temp0;
                }
            }
            __syncthreads();
	   }
       #pragma unroll //UNROLLHERE
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

    auto initial_calc32_local_affine_float = [&](const auto& value){
        const __half2* const sbt_row = shared_BLOSUM62[query_letter];

        const __half2 score2_0 = sbt_row[subject[0]];
        //score2.y = sbt_row[subject1[0].x];
        __half2 penalty_temp0 = penalty_here_array[0];
        if (!value || (thid%group_size)) E = __hmax2(__hadd2(E,gap_extend2), __hadd2(penalty_left,gap_open2));
        F_here_array[0] = __hmax2(__hadd2(F_here_array[0],gap_extend2), __hadd2(penalty_here_array[0],gap_open2));
        const __half2 temp0_0 = __hmax2(__hmax2(__hadd2(penalty_diag,score2_0), __hmax2(E, F_here_array[0])), __float2half2_rn(0.0));
        penalty_here_array[0] = temp0_0;
        maximum = __hmax2(temp0_0, maximum);

        const __half2 score2_1 = sbt_row[subject[1]];
        //score2.y = sbt_row[subject1[0].y];
        __half2 penalty_temp1 = penalty_here_array[1];
        E = __hmax2(__hadd2(E,gap_extend2), __hadd2(penalty_here_array[0],gap_open2));
        F_here_array[1] = __hmax2(__hadd2(F_here_array[1],gap_extend2), __hadd2(penalty_here_array[1],gap_open2));
        penalty_here_array[1] = __hmax2(__hmax2(__hadd2(penalty_temp0,score2_1), __hmax2(E, F_here_array[1])), __float2half2_rn(0.0));
        const __half2 temp0_1 = penalty_here_array[1];
        maximum = __hmax2(temp0_1,maximum);

        #pragma unroll
        for (int i=1; i<numRegs/2; i++) {
            const __half2 score2_2i = sbt_row[subject[2*i]];
            //score2.y = sbt_row[subject1[i].x];
            penalty_temp0 = penalty_here_array[2*i];
            E = __hmax2(__hadd2(E,gap_extend2), __hadd2(penalty_here_array[2*i-1],gap_open2));
            F_here_array[2*i] = __hmax2(__hadd2(F_here_array[2*i],gap_extend2), __hadd2(penalty_here_array[2*i],gap_open2));
            penalty_here_array[2*i] = __hmax2(__hmax2(__hadd2(penalty_temp1,score2_2i), __hmax2(E, F_here_array[2*i])), __float2half2_rn(0.0));
            const __half2 temp0_2 = penalty_here_array[2*i]; 
            maximum = __hmax2(temp0_2,maximum);

            const __half2 score2_2i1 = sbt_row[subject[2*i+1]];
            //score2.y = sbt_row[subject1[i].y];
            penalty_temp1 = penalty_here_array[2*i+1];
            E = __hmax2(__hadd2(E,gap_extend2), __hadd2(penalty_here_array[2*i],gap_open2));
            F_here_array[2*i+1] = __hmax2(__hadd2(F_here_array[2*i+1],gap_extend2), __hadd2(penalty_here_array[2*i+1],gap_open2));
            penalty_here_array[2*i+1] = __hmax2(__hmax2(__hadd2(penalty_temp0,score2_2i1), __hmax2(E, F_here_array[2*i+1])), __float2half2_rn(0.0));
            const __half2 temp0_3 = penalty_here_array[2*i+1];
            maximum = __hmax2(temp0_3,maximum);
        }

        penalty_here31 = penalty_here_array[numRegs-1];
        E = __hmax2(E+gap_extend2, penalty_here31+gap_open2);
        #pragma unroll //UNROLLHERE
        for (int i=0; i<numRegs; i++) F_here_array[i] = __hmax2(__hadd2(F_here_array[i],gap_extend2), __hadd2(penalty_here_array[i],gap_open2));
    };


    auto calc32_local_affine_float = [&](){
        const __half2* const sbt_row = shared_BLOSUM62[query_letter];

        const __half2 score2_0 = sbt_row[subject[0]];
        //score2.y = sbt_row[subject1[0].x];
        __half2 penalty_temp0 = penalty_here_array[0];
        penalty_here_array[0] = __hmax2(__hmax2(__hadd2(penalty_diag,score2_0), __hmax2(E, F_here_array[0])),__float2half2_rn(0.0));
        //maximum = __hmax2(penalty_here_array[0],maximum);
        __half2 penalty_temp1 = __hadd2(penalty_here_array[0],gap_open2);
        E = __hmax2(__hadd2(E,gap_extend2), penalty_temp1);
        F_here_array[0] = __hmax2(__hadd2(F_here_array[0],gap_extend2), penalty_temp1);

        const __half2 score2_1 = sbt_row[subject[1]];
        //score2.y = sbt_row[subject1[0].y];
        penalty_temp1 = penalty_here_array[1];
        penalty_here_array[1] = __hmax2(__hmax2(__hadd2(penalty_temp0,score2_1), __hmax2(E, F_here_array[1])),__float2half2_rn(0.0));
        //maximum = __hmax2(penalty_here_array[1],maximum);
        penalty_temp0 = __hadd2(penalty_here_array[1],gap_open2);
        E = __hmax2(__hadd2(E,gap_extend2), penalty_temp0);
        F_here_array[1] = __hmax2(__hadd2(F_here_array[1],gap_extend2), penalty_temp0);
		maximum = __hmax2(maximum, __hmax2(penalty_here_array[1],penalty_here_array[0]));

        #pragma unroll
        for (int i=1; i<numRegs/2; i++) {
            const __half2 score2_2i = sbt_row[subject[2*i]];
            penalty_temp0 = penalty_here_array[2*i];
            penalty_here_array[2*i] = __hmax2(__hmax2(__hadd2(penalty_temp1,score2_2i), __hmax2(E, F_here_array[2*i])),__float2half2_rn(0.0));
            //maximum = __hmax2(penalty_here_array[2*i],maximum);
            penalty_temp1 = __hadd2(penalty_here_array[2*i],gap_open2);
            E = __hmax2(__hadd2(E,gap_extend2), penalty_temp1);
            F_here_array[2*i] = __hmax2(__hadd2(F_here_array[2*i],gap_extend2), penalty_temp1);

            const __half2 score2_2i1 = sbt_row[subject[2*i+1]];
            penalty_temp1 = penalty_here_array[2*i+1];
            penalty_here_array[2*i+1] = __hmax2(__hmax2(__hadd2(penalty_temp0,score2_2i1), __hmax2(E, F_here_array[2*i+1])),__float2half2_rn(0.0));
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

    assert(passes > 1);

    //constexpr int passes = 4;

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

    auto shuffle_H_E_temp_out = [&]() {
        const uint32_t temp = __shfl_down_sync(0xFFFFFFFF, *((int*)(&H_temp_out)), 1, 32);
        H_temp_out = *((half2*)(&temp));
        const uint32_t temp2 = __shfl_down_sync(0xFFFFFFFF, *((int*)(&E_temp_out)), 1, 32);
        E_temp_out = *((half2*)(&temp2));
    };

    auto shuffle_H_E_temp_in = [&]() {
        const uint32_t temp = __shfl_down_sync(0xFFFFFFFF, *((int*)(&H_temp_in)), 1, 32);
        H_temp_in = *((half2*)(&temp));
        const uint32_t temp2 = __shfl_down_sync(0xFFFFFFFF, *((int*)(&E_temp_in)), 1, 32);
        E_temp_in = *((half2*)(&temp2));

    };

    auto shuffle_new_query = [&]() {
        const uint32_t temp = __shfl_down_sync(0xFFFFFFFF, *((int*)(&new_query_letter4)), 1, 32);
        new_query_letter4 = *((char4*)(&temp));
    };

    auto set_H_E_temp_out = [&]() {
        if (thid % group_size == 31) {
            H_temp_out = penalty_here31;
            E_temp_out = E;
        }
    };


    const uint32_t thread_result = ((length-1)%(group_size*numRegs))/numRegs;              // 0, 1, ..., or 31

    // FIRST PASS (of many passes)
    // Note first pass has always full seqeunce length
    init_penalties_local(0);
    init_local_score_profile_BLOSUM62(0);
    initial_calc32_local_affine_float(0);
    shuffle_query(new_query_letter4.y);
    shuffle_affine_penalty(__float2half2_rn(0.0), __float2half2_rn(negInftyFloat));

    //shuffle_max();
    calc32_local_affine_float();
    shuffle_query(new_query_letter4.z);
    shuffle_affine_penalty(__float2half2_rn(0.0), __float2half2_rn(negInftyFloat));

    //shuffle_max();
    calc32_local_affine_float();
    shuffle_query(new_query_letter4.w);
    shuffle_affine_penalty(__float2half2_rn(0.0), __float2half2_rn(negInftyFloat));
    shuffle_new_query();
    counter++;

    for (int k = 4; k <= length_2+28; k+=4) {
        //shuffle_max();
        calc32_local_affine_float();
        set_H_E_temp_out();
        shuffle_H_E_temp_out();

        shuffle_query(new_query_letter4.x);
        shuffle_affine_penalty(__float2half2_rn(0.0),__float2half2_rn(negInftyFloat));

        //shuffle_max();
        calc32_local_affine_float();
        set_H_E_temp_out();
        shuffle_H_E_temp_out();

        shuffle_query(new_query_letter4.y);
        shuffle_affine_penalty(__float2half2_rn(0.0),__float2half2_rn(negInftyFloat));

        //shuffle_max();
        calc32_local_affine_float();
        set_H_E_temp_out();
        shuffle_H_E_temp_out();

        shuffle_query(new_query_letter4.z);
        shuffle_affine_penalty(__float2half2_rn(0.0),__float2half2_rn(negInftyFloat));

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
        shuffle_affine_penalty(__float2half2_rn(0.0),__float2half2_rn(negInftyFloat));
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
        shuffle_affine_penalty(__float2half2_rn(0.0),__float2half2_rn(negInftyFloat));
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
        shuffle_affine_penalty(__float2half2_rn(0.0),__float2half2_rn(negInftyFloat));
        //shuffle_max();
        calc32_local_affine_float();
        set_H_E_temp_out();
        shuffle_H_E_temp_out();
        shuffle_query(new_query_letter4.y);
        shuffle_affine_penalty(__float2half2_rn(0.0),__float2half2_rn(negInftyFloat));
        //shuffle_max();
        calc32_local_affine_float();
        set_H_E_temp_out();
    }
    const int final_out = length_2 % 32;
    const int from_thread_id = 32 - final_out;

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

        for (int k = 4; k <= length_2+28; k+=4) {
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
        const int final_out = length_2 % 32;
        const int from_thread_id = 32 - final_out;

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
        int k;
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
__launch_bounds__(256,2)
//__launch_bounds__(128,1)
__global__
void NW_local_affine_Protein_many_pass_half2_new2(
    // const char * devChars,
    // float * devAlignmentScores,
    // __half2 * devTempHcol2,
    // __half2 * devTempEcol2,
    // const size_t* devOffsets,
    // const size_t* devLengths,
    // PositionsIterator d_positions_of_selected_lengths,
    // const int numSelected,
	// size_t* d_overflow_positions,
	// int* d_overflow_number,
	// const bool overflow_check,
    // const int length_2,
    // const float gap_open,
    // const float gap_extend
    __grid_constant__ const char * const devChars,
    __grid_constant__ float * const devAlignmentScores,
    __grid_constant__ __half2 * const devTempHcol2,
    __grid_constant__ __half2 * const devTempEcol2,
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

    __shared__ __half2 shared_BLOSUM62[21][21*21];

    ManyPassHalf2<group_size, numRegs, PositionsIterator> processor(
        shared_BLOSUM62,
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
        length_2,
        gap_open,
        gap_extend
    );

    processor.compute();
}

#endif