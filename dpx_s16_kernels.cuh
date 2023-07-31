#ifndef DPX_S16_KERNELS_CUH
#define DPX_S16_KERNELS_CUH


#include <cuda_fp16.h>
#include "blosum.hpp"

template <int group_size, int numRegs, int blosumDim, class PositionsIterator> 
struct DPXAligner_s16{
    static constexpr short negInfty = -1000;

    static constexpr int deviceBlosumDimCexpr = blosumDim;
    static constexpr int deviceBlosumDimCexprSquared = deviceBlosumDimCexpr * deviceBlosumDimCexpr;

    short2* shared_blosum;

    int numSelected;
    short gap_open;
    short gap_extend;
    PositionsIterator d_positions_of_selected_lengths;
    const char* devChars;
    short2* devTempHcol2;
    short2* devTempEcol2;
    const size_t* devOffsets;
    const size_t* devLengths;

    __device__
    DPXAligner_s16(
        short2* shared_blosum_,
        const char* devChars_,
        short2* devTempHcol2_,
        short2* devTempEcol2_,
        const size_t* devOffsets_,
        const size_t* devLengths_,
        PositionsIterator d_positions_of_selected_lengths_,
        int numSelected_,
        short gap_open_,
        short gap_extend_
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
            // short2 temp0;
            // temp0.x = deviceBlosum[deviceBlosumDimCexpr*(i/deviceBlosumDimCexpr)+(i%deviceBlosumDimCexpr)];
            // for (int j=0; j<deviceBlosumDimCexpr; j++) {
            //     temp0.y = deviceBlosum[deviceBlosumDimCexpr*(i/deviceBlosumDimCexpr)+j];
            //     shared_blosum[(i/deviceBlosumDimCexpr) * deviceBlosumDimCexprSquared + deviceBlosumDimCexpr*(i%deviceBlosumDimCexpr)+j]=temp0;
            // }
            const short first = deviceBlosum[deviceBlosumDimCexpr*(i/deviceBlosumDimCexpr)+(i%deviceBlosumDimCexpr)];
            for (int j=0; j<deviceBlosumDimCexpr; j++) {
                const short second = deviceBlosum[deviceBlosumDimCexpr*(i/deviceBlosumDimCexpr)+j];
                shared_blosum[(i/deviceBlosumDimCexpr) * deviceBlosumDimCexprSquared + deviceBlosumDimCexpr*(i%deviceBlosumDimCexpr)+j]
                    = make_short2(first, second);
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
        // assert(2*(blockDim.x/group_size)*blockIdx.x * queryLength <= base_3 + x);
        // assert(base_3+x < 2*(blockDim.x/group_size)*(blockIdx.x+1) * queryLength);
    };

    __device__
    void initial_calc32_local_affine_float(const int value, char query_letter, short2& E, short2& penalty_here31, short2 penalty_diag, short2 penalty_left, short2& maximum, 
        int (&subject)[numRegs], 
        short2 (&penalty_here_array)[numRegs],
        short2 (&F_here_array)[numRegs]
    ) const{
        const short2* const sbt_row = &shared_blosum[int(query_letter) * deviceBlosumDimCexprSquared];

        const short2 score2_0 = sbt_row[subject[0]];
        short2 penalty_temp0 = penalty_here_array[0];
        if (!value || (threadIdx.x%group_size)) E = viaddmax(E, make_short2(gap_extend, gap_extend),viadd(penalty_left,make_short2(gap_open, gap_open)));
        F_here_array[0] = viaddmax(F_here_array[0],make_short2(gap_extend, gap_extend),viadd(penalty_here_array[0],make_short2(gap_open, gap_open)));
        penalty_here_array[0] = vimax3(viadd(penalty_diag,score2_0),E, F_here_array[0]);
        //maximum = vimax(temp0,maximum);

        const short2 score2_1 = sbt_row[subject[1]];
        short2 penalty_temp1 = penalty_here_array[1];
        E = viaddmax(E,make_short2(gap_extend, gap_extend),viadd(penalty_here_array[0],make_short2(gap_open, gap_open)));
        F_here_array[1] = viaddmax(F_here_array[1],make_short2(gap_extend, gap_extend),viadd(penalty_here_array[1],make_short2(gap_open, gap_open)));
        penalty_here_array[1] = vimax3(viadd(penalty_temp0,score2_1),E, F_here_array[1]);
        const short2 temp0_1 = penalty_here_array[1];
        //maximum = vimax(temp0_1,maximum);
        maximum = vimax3(maximum,penalty_here_array[0],penalty_here_array[1]);

        #pragma unroll
        for (int i=1; i<numRegs/2; i++) {
            const short2 score2_2i = sbt_row[subject[2*i]];
            penalty_temp0 = penalty_here_array[2*i];
            E = vimax(viadd(E,make_short2(gap_extend, gap_extend)), viadd(penalty_here_array[2*i-1],make_short2(gap_open, gap_open)));
            F_here_array[2*i] = vimax(viadd(F_here_array[2*i],make_short2(gap_extend, gap_extend)), viadd(penalty_here_array[2*i],make_short2(gap_open, gap_open)));
            penalty_here_array[2*i] = vimax3(viadd(penalty_temp1,score2_2i), E, F_here_array[2*i]);
            //maximum = vimax(temp0,maximum);

            const short2 score2_2i1 = sbt_row[subject[2*i+1]];
            penalty_temp1 = penalty_here_array[2*i+1];
            E = vimax(viadd(E,make_short2(gap_extend, gap_extend)), viadd(penalty_here_array[2*i],make_short2(gap_open, gap_open)));
            F_here_array[2*i+1] = vimax(viadd(F_here_array[2*i+1],make_short2(gap_extend, gap_extend)), viadd(penalty_here_array[2*i+1],make_short2(gap_open, gap_open)));
            penalty_here_array[2*i+1] = vimax3(viadd(penalty_temp0,score2_2i1), E, F_here_array[2*i+1]);
            //maximum = vimax(temp0,maximum);
			maximum = vimax3(maximum,penalty_here_array[2*i],penalty_here_array[2*i+1]);
        }

        penalty_here31 = penalty_here_array[numRegs-1];
        E = vimax(viadd(E, make_short2(gap_extend, gap_extend)), viadd(penalty_here31, make_short2(gap_open, gap_open)));
        #pragma unroll //UNROLLHERE
        for (int i=0; i<numRegs; i++) F_here_array[i] = vimax(viadd(F_here_array[i],make_short2(gap_extend, gap_extend)), viadd(penalty_here_array[i],make_short2(gap_open, gap_open)));
    };

    __device__
    void calc32_local_affine_float(char query_letter, short2& E, short2& penalty_here31, short2 penalty_diag, short2& maximum, 
        int (&subject)[numRegs],
        short2 (&penalty_here_array)[numRegs],
        short2 (&F_here_array)[numRegs]
    ) const{
        const short2* const sbt_row = &shared_blosum[int(query_letter) * deviceBlosumDimCexprSquared];

        const short2 score2_0 = sbt_row[subject[0]];
        short2 penalty_temp0 = penalty_here_array[0];
        penalty_here_array[0] = vimax3(viadd(penalty_diag,score2_0), E, F_here_array[0]);
        short2 penalty_temp1 = viadd(penalty_here_array[0],make_short2(gap_open, gap_open));
        E = viaddmax(E,make_short2(gap_extend, gap_extend), penalty_temp1);
        F_here_array[0] = viaddmax(F_here_array[0],make_short2(gap_extend, gap_extend), penalty_temp1);

        const short2 score2_1 = sbt_row[subject[1]];
        penalty_temp1 = penalty_here_array[1];
        penalty_here_array[1] = vimax3(viadd(penalty_temp0,score2_1), E, F_here_array[1]);
        penalty_temp0 = viadd(penalty_here_array[1],make_short2(gap_open, gap_open));
        E = viaddmax(E,make_short2(gap_extend, gap_extend), penalty_temp0);
        F_here_array[1] = viaddmax(F_here_array[1],make_short2(gap_extend, gap_extend),penalty_temp0);
		maximum = vimax3(maximum,penalty_here_array[0],penalty_here_array[1]);

        #pragma unroll
        for (int i=1; i<numRegs/2; i++) {
            const short2 score2_2i = sbt_row[subject[2*i]];
            penalty_temp0 = penalty_here_array[2*i];
            penalty_here_array[2*i] = vimax3(viadd(penalty_temp1,score2_2i), E, F_here_array[2*i]);
            penalty_temp1 = viadd(penalty_here_array[2*i],make_short2(gap_open, gap_open));
            E = viaddmax(E,make_short2(gap_extend, gap_extend), penalty_temp1);
            F_here_array[2*i] = viaddmax(F_here_array[2*i],make_short2(gap_extend, gap_extend), penalty_temp1);

            const short2 score2_2i1 = sbt_row[subject[2*i+1]];
            penalty_temp1 = penalty_here_array[2*i+1];
            penalty_here_array[2*i+1] = vimax3(viadd(penalty_temp0,score2_2i1), E, F_here_array[2*i+1]);
            penalty_temp0 = viadd(penalty_here_array[2*i+1],make_short2(gap_open, gap_open));
            E = viaddmax(E,make_short2(gap_extend, gap_extend), penalty_temp0);
            F_here_array[2*i+1] = viaddmax(F_here_array[2*i+1],make_short2(gap_extend, gap_extend),penalty_temp0);
            maximum = vimax3(maximum,penalty_here_array[2*i],penalty_here_array[2*i+1]);
        }

        //for (int i=0; i<numRegs/4; i++)
		 //   maximum = __hmax2(maximum,__hmax2(__hmax2(penalty_here_array[4*i],penalty_here_array[4*i+1]),__hmax2(penalty_here_array[4*i+2],penalty_here_array[4*i+3])));
        penalty_here31 = penalty_here_array[numRegs-1];
    }

    __device__    
    void init_penalties_local(int value, short2& penalty_diag, short2& penalty_left, 
        short2 (&penalty_here_array)[numRegs], 
        short2 (&F_here_array)[numRegs]
    ) const{

        penalty_left = make_short2(negInfty, negInfty);
        penalty_diag = make_short2(negInfty, negInfty);

        #pragma unroll
        for (int i=0; i<numRegs; i++) penalty_here_array[i] = make_short2(negInfty, negInfty);
        #pragma unroll //UNROLLHERE
        for (int i=0; i<numRegs; i++) F_here_array[i] = make_short2(negInfty, negInfty);
        if (threadIdx.x % group_size == 0) {
            penalty_left = make_short2(0,0);
            penalty_diag = make_short2(0,0);
            #pragma unroll
            for (int i=0; i<numRegs; i++) penalty_here_array[i] = make_short2(0,0);
        }
        if (threadIdx.x % group_size == 1) {
            penalty_left = make_short2(0,0);
        }
    }

    __device__
    void init_local_score_profile(int offset_isc, int (&subject)[numRegs], 
        const char* const devS0, const int length_S0, 
        const char* const devS1, const int length_S1
    ) const{
        #pragma unroll //UNROLLHERE
        for (int i=0; i<numRegs; i++) {

            if (offset_isc+numRegs*(threadIdx.x%group_size)+i >= length_S0) subject[i] = 1; // 20;
            else{
                
                subject[i] = devS0[offset_isc+numRegs*(threadIdx.x%group_size)+i];
            }

            if (offset_isc+numRegs*(threadIdx.x%group_size)+i >= length_S1) subject[i] += 1*deviceBlosumDimCexpr; // 20*deviceBlosumDimCexpr;
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
        short2 new_penalty_left, short2 new_E_left, short2& E, 
        short2 penalty_here31, short2& penalty_diag, short2& penalty_left
    ) const{
        penalty_diag = penalty_left;
        //TODO normal shuffle
        penalty_left = shfl_up_2xint16(0xFFFFFFFF, penalty_here31, 1, 32);
        E = shfl_up_2xint16(0xFFFFFFFF, E, 1, 32);
        const int group_id = threadIdx.x % group_size;
        if (!group_id) {
            penalty_left = new_penalty_left;
            E = new_E_left;
        }
    }

    __device__
    void shuffle_H_E_temp_out(short2& H_temp_out, short2& E_temp_out) const{
        const uint32_t temp = __shfl_down_sync(0xFFFFFFFF, *((int*)(&H_temp_out)), 1, 32);
        H_temp_out = *((short2*)(&temp));
        const uint32_t temp2 = __shfl_down_sync(0xFFFFFFFF, *((int*)(&E_temp_out)), 1, 32);
        E_temp_out = *((short2*)(&temp2));
    }

    __device__
    void shuffle_H_E_temp_in(short2& H_temp_in, short2& E_temp_in) const{
        const uint32_t temp = __shfl_down_sync(0xFFFFFFFF, *((int*)(&H_temp_in)), 1, 32);
        H_temp_in = *((short2*)(&temp));
        const uint32_t temp2 = __shfl_down_sync(0xFFFFFFFF, *((int*)(&E_temp_in)), 1, 32);
        E_temp_in = *((short2*)(&temp2));
    }

    __device__
    void set_H_E_temp_out(short2 E, short2 penalty_here31, short2& H_temp_out, short2& E_temp_out) const{
        if (threadIdx.x % group_size == 31) {
            H_temp_out = penalty_here31;
            E_temp_out = E;
        }
    }

    __device__
    void computeFirstPass(short2& maximum, const char* const devS0, const int length_S0, 
        const char* const devS1, const int length_S1,
        const char4* query4,
        int queryLength
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
        const size_t base_3 = (2*(size_t(blockDim.x)/group_size)*size_t(blockIdx.x)+2*((threadIdx.x%check_last)/group_size))*size_t(queryLength);
        short2* const devTempHcol = (&devTempHcol2[base_3]);
        short2* const devTempEcol = (&devTempEcol2[base_3]);

        const int group_id = threadIdx.x % group_size;
        int offset = group_id + group_size;
        int offset_out = group_id;


        short2 E = make_short2(negInfty, negInfty);
        short2 penalty_here31;
        short2 penalty_diag;
        short2 penalty_left;
        short2 H_temp_out;
        short2 E_temp_out;
        int subject[numRegs];
        short2 penalty_here_array[numRegs];
        short2 F_here_array[numRegs];

        
        init_penalties_local(0, penalty_diag, penalty_left, penalty_here_array, F_here_array);
        init_local_score_profile(0, subject, devS0, length_S0, devS1, length_S1);
        initial_calc32_local_affine_float(0, query_letter, E, penalty_here31, penalty_diag, penalty_left, maximum, subject, penalty_here_array, F_here_array);
        shuffle_query(new_query_letter4.y, query_letter);
        shuffle_affine_penalty(make_short2(0,0), make_short2(negInfty, negInfty), E, penalty_here31, penalty_diag, penalty_left);

        //shuffle_max();
        calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
        shuffle_query(new_query_letter4.z, query_letter);
        shuffle_affine_penalty(make_short2(0,0), make_short2(negInfty, negInfty), E, penalty_here31, penalty_diag, penalty_left);

        //shuffle_max();
        calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
        shuffle_query(new_query_letter4.w, query_letter);
        shuffle_affine_penalty(make_short2(0,0), make_short2(negInfty, negInfty), E, penalty_here31, penalty_diag, penalty_left);
        shuffle_new_query(new_query_letter4);
        counter++;

        for (int k = 4; k <= queryLength+28; k+=4) {
            //shuffle_max();
            calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out(E, penalty_here31, H_temp_out, E_temp_out);
            shuffle_H_E_temp_out(H_temp_out, E_temp_out);

            shuffle_query(new_query_letter4.x, query_letter);
            shuffle_affine_penalty(make_short2(0,0), make_short2(negInfty, negInfty), E, penalty_here31, penalty_diag, penalty_left);

            //shuffle_max();
            calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out(E, penalty_here31, H_temp_out, E_temp_out);
            shuffle_H_E_temp_out(H_temp_out, E_temp_out);

            shuffle_query(new_query_letter4.y, query_letter);
            shuffle_affine_penalty(make_short2(0,0), make_short2(negInfty, negInfty), E, penalty_here31, penalty_diag, penalty_left);

            //shuffle_max();
            calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out(E, penalty_here31, H_temp_out, E_temp_out);
            shuffle_H_E_temp_out(H_temp_out, E_temp_out);

            shuffle_query(new_query_letter4.z, query_letter);
            shuffle_affine_penalty(make_short2(0,0), make_short2(negInfty, negInfty), E, penalty_here31, penalty_diag, penalty_left);

            //shuffle_max();
            calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);

            set_H_E_temp_out(E, penalty_here31, H_temp_out, E_temp_out);
            if (counter%8 == 0 && counter > 8) {
                checkHEindex(offset_out, __LINE__);
                devTempHcol[offset_out]=H_temp_out;
                devTempEcol[offset_out]=E_temp_out;
                offset_out += group_size;
            }

            if (k != queryLength+28) shuffle_H_E_temp_out(H_temp_out, E_temp_out);
            shuffle_query(new_query_letter4.w, query_letter);
            shuffle_affine_penalty(make_short2(0,0), make_short2(negInfty, negInfty), E, penalty_here31, penalty_diag, penalty_left);
            shuffle_new_query(new_query_letter4);
            if (counter%group_size == 0) {
                new_query_letter4 = query4[offset];
                offset += group_size;
            }
            counter++;
        }
        // if (queryLength % 4 == 0) {
        //     const int temp0 = __shfl_up_sync(0xFFFFFFFF, *((int*)(&H_temp_out)), 1, 32);
        //     H_temp_out = *((short2*)(&temp0));
        //     const int temp1 = __shfl_up_sync(0xFFFFFFFF, *((int*)(&E_temp_out)), 1, 32);
        //     E_temp_out = *((short2*)(&temp1));
        // }
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
            shuffle_affine_penalty(make_short2(0,0), make_short2(negInfty, negInfty), E, penalty_here31, penalty_diag, penalty_left);
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
            shuffle_affine_penalty(make_short2(0,0), make_short2(negInfty, negInfty), E, penalty_here31, penalty_diag, penalty_left);
            //shuffle_max();
            calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out(E, penalty_here31, H_temp_out, E_temp_out);
            shuffle_H_E_temp_out(H_temp_out, E_temp_out);
            shuffle_query(new_query_letter4.y, query_letter);
            shuffle_affine_penalty(make_short2(0,0), make_short2(negInfty, negInfty), E, penalty_here31, penalty_diag, penalty_left);
            //shuffle_max();
            calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out(E, penalty_here31, H_temp_out, E_temp_out);
        }
        const int final_out = queryLength % 32;
        const int from_thread_id = 32 - final_out;

        //printf("tid %d, offset_out %d, from_thread_id %d\n", threadIdx.x, offset_out, from_thread_id);
        if (threadIdx.x>=from_thread_id) {
            checkHEindex(offset_out-from_thread_id, __LINE__);
            devTempHcol[offset_out-from_thread_id]=H_temp_out;
            devTempEcol[offset_out-from_thread_id]=E_temp_out;
        }
    }

    __device__
    void computeMiddlePass(int pass, short2& maximum, const char* const devS0, const int length_S0, 
        const char* const devS1, const int length_S1,
        const char4* query4,
        int queryLength
    ) const{
        int counter = 1;
        char query_letter = 20;
        char4 new_query_letter4 = query4[threadIdx.x%group_size];
        if (threadIdx.x % group_size== 0) query_letter = new_query_letter4.x;

        int check_last;
        int check_last2;
        computeCheckLast(check_last, check_last2);
        const size_t base_3 = (2*(size_t(blockDim.x)/group_size)*size_t(blockIdx.x)+2*((threadIdx.x%check_last)/group_size))*size_t(queryLength);
        short2* const devTempHcol = (&devTempHcol2[base_3]);
        short2* const devTempEcol = (&devTempEcol2[base_3]);

        const int group_id = threadIdx.x % group_size;
        int offset = group_id + group_size;
        int offset_out = group_id;
        int offset_in = group_id;
        checkHEindex(offset_in, __LINE__);
        short2 H_temp_in = devTempHcol[offset_in];
        short2 E_temp_in = devTempEcol[offset_in];
        offset_in += group_size;

        short2 E = make_short2(negInfty, negInfty);
        short2 penalty_here31;
        short2 penalty_diag;
        short2 penalty_left;
        short2 H_temp_out;
        short2 E_temp_out;
        int subject[numRegs];
        short2 penalty_here_array[numRegs];
        short2 F_here_array[numRegs];

        init_penalties_local(0, penalty_diag, penalty_left, penalty_here_array, F_here_array);
        init_local_score_profile(pass*(32*numRegs), subject, devS0, length_S0, devS1, length_S1);

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

        for (int k = 4; k <= queryLength+28; k+=4) {
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
                checkHEindex(offset_out, __LINE__);
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
                checkHEindex(offset_in, __LINE__);
                H_temp_in = devTempHcol[offset_in];
                E_temp_in = devTempEcol[offset_in];
                offset_in += group_size;
            }
            counter++;
        }
        // if (queryLength % 4 == 0) {
        //     const int temp0 = __shfl_up_sync(0xFFFFFFFF, *((int*)(&H_temp_out)), 1, 32);
        //     H_temp_out = *((short2*)(&temp0));
        //     const int temp1 = __shfl_up_sync(0xFFFFFFFF, *((int*)(&E_temp_out)), 1, 32);
        //     E_temp_out = *((short2*)(&temp1));
        // }
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
            checkHEindex(offset_out-from_thread_id, __LINE__);
            devTempHcol[offset_out-from_thread_id]=H_temp_out;
            devTempEcol[offset_out-from_thread_id]=E_temp_out;
        }
    }

    __device__ 
    void computeFinalPass(int passes, short2& maximum, const char* const devS0, const int length_S0, 
        const char* const devS1, const int length_S1,
        const char4* query4,
        int queryLength
    ) const{
        int counter = 1;
        char query_letter = 20;
        char4 new_query_letter4 = query4[threadIdx.x%group_size];
        if (threadIdx.x % group_size== 0) query_letter = new_query_letter4.x;

        int check_last;
        int check_last2;
        computeCheckLast(check_last, check_last2);
        const size_t base_3 = (2*(size_t(blockDim.x)/group_size)*size_t(blockIdx.x)+2*((threadIdx.x%check_last)/group_size))*size_t(queryLength);
        short2* const devTempHcol = (&devTempHcol2[base_3]);
        short2* const devTempEcol = (&devTempEcol2[base_3]);

        const int group_id = threadIdx.x % group_size;
        int offset = group_id + group_size;
        int offset_in = group_id;
        checkHEindex(offset_in, __LINE__);
        short2 H_temp_in = devTempHcol[offset_in];
        short2 E_temp_in = devTempEcol[offset_in];
        offset_in += group_size;

        const int length = max(length_S0, length_S1);
        const uint32_t thread_result = ((length-1)%(group_size*numRegs))/numRegs; 

        short2 E = make_short2(negInfty, negInfty);
        short2 penalty_here31;
        short2 penalty_diag;
        short2 penalty_left;
        int subject[numRegs];
        short2 penalty_here_array[numRegs];
        short2 F_here_array[numRegs];

        init_penalties_local(0, penalty_diag, penalty_left, penalty_here_array, F_here_array);
        init_local_score_profile((passes-1)*(32*numRegs), subject, devS0, length_S0, devS1, length_S1);
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
        if (queryLength+thread_result >=2) {
            //shuffle_max();
            calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            shuffle_query(new_query_letter4.z, query_letter);
            //copy_H_E_temp_in();
            shuffle_affine_penalty(H_temp_in, E_temp_in, E, penalty_here31, penalty_diag, penalty_left);
            shuffle_H_E_temp_in(H_temp_in, E_temp_in);
        }

        if (queryLength+thread_result >=3) {
            //shuffle_max();
            calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            shuffle_query(new_query_letter4.w, query_letter);
            //copy_H_E_temp_in();
            shuffle_affine_penalty(H_temp_in, E_temp_in, E, penalty_here31, penalty_diag, penalty_left);
            shuffle_H_E_temp_in(H_temp_in, E_temp_in);
            shuffle_new_query(new_query_letter4);
            counter++;
        }
        if (queryLength+thread_result >=4) {
            int k;
            //for (k = 5; k < lane_2+thread_result-2; k+=4) {
            for (k = 4; k <= queryLength+(thread_result-3); k+=4) {
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
                    checkHEindex(offset_in, __LINE__);
                    H_temp_in = devTempHcol[offset_in];
                    E_temp_in = devTempEcol[offset_in];
                    offset_in += group_size;
                }
                counter++;
            }

            if ((k-1)-(queryLength+thread_result) > 0) {
                //shuffle_max();
                calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
                shuffle_query(new_query_letter4.x, query_letter);
                shuffle_affine_penalty(H_temp_in, E_temp_in, E, penalty_here31, penalty_diag, penalty_left);
                shuffle_H_E_temp_in(H_temp_in, E_temp_in);
                k++;
            }


            if ((k-1)-(queryLength+thread_result) > 0) {
                //shuffle_max();
                calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
                shuffle_query(new_query_letter4.y, query_letter);
                shuffle_affine_penalty(H_temp_in, E_temp_in, E, penalty_here31, penalty_diag, penalty_left);
                shuffle_H_E_temp_in(H_temp_in, E_temp_in);
                k++;
            }

            if ((k-1)-(queryLength+thread_result) > 0) {
                //shuffle_max();
                calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            }
        }
    }

    __device__ 
    void computeSinglePass(short2& maximum, const char* const devS0, const int length_S0, 
        const char* const devS1, const int length_S1, const int warpMaxLength,
        const char4* query4,
        int queryLength
    ) const{
        int counter = 1;
        char query_letter = 20;
        char4 new_query_letter4 = query4[threadIdx.x%group_size];
        if (threadIdx.x % group_size== 0) query_letter = new_query_letter4.x;


        const int group_id = threadIdx.x % group_size;
        int offset = group_id + group_size;

        const uint32_t thread_result = ((warpMaxLength-1)%(group_size*numRegs))/numRegs; 

        short2 E = make_short2(negInfty, negInfty);
        short2 penalty_here31;
        short2 penalty_diag;
        short2 penalty_left;
        int subject[numRegs];
        short2 penalty_here_array[numRegs];
        short2 F_here_array[numRegs];

        init_penalties_local(0, penalty_diag, penalty_left, penalty_here_array, F_here_array);
        init_local_score_profile(0, subject, devS0, length_S0, devS1, length_S1);

        initial_calc32_local_affine_float(0, query_letter, E, penalty_here31, penalty_diag, penalty_left, maximum, subject, penalty_here_array, F_here_array);
        shuffle_query(new_query_letter4.y, query_letter);
        shuffle_affine_penalty(make_short2(0,0), make_short2(negInfty, negInfty), E, penalty_here31, penalty_diag, penalty_left);

        if (queryLength+thread_result >=2) {
            //shuffle_max();
            calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            shuffle_query(new_query_letter4.z, query_letter);
            shuffle_affine_penalty(make_short2(0,0), make_short2(negInfty, negInfty), E, penalty_here31, penalty_diag, penalty_left);
        }

        if (queryLength+thread_result >=3) {
            //shuffle_max();
            calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            shuffle_query(new_query_letter4.w, query_letter);
            shuffle_affine_penalty(make_short2(0,0), make_short2(negInfty, negInfty), E, penalty_here31, penalty_diag, penalty_left);
            shuffle_new_query(new_query_letter4);
            counter++;
        }
        if (queryLength+thread_result >=4) {
            int k;
            //for (k = 5; k < lane_2+thread_result-2; k+=4) {
            for (k = 4; k <= queryLength+(thread_result-3); k+=4) {
                //shuffle_max();
                calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);

                shuffle_query(new_query_letter4.x, query_letter);
                shuffle_affine_penalty(make_short2(0,0), make_short2(negInfty, negInfty), E, penalty_here31, penalty_diag, penalty_left);

                //shuffle_max();
                calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
                shuffle_query(new_query_letter4.y, query_letter);
                shuffle_affine_penalty(make_short2(0,0), make_short2(negInfty, negInfty), E, penalty_here31, penalty_diag, penalty_left);

                //shuffle_max();
                calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
                shuffle_query(new_query_letter4.z, query_letter);
                shuffle_affine_penalty(make_short2(0,0), make_short2(negInfty, negInfty), E, penalty_here31, penalty_diag, penalty_left);

                //shuffle_max();
                calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
                shuffle_query(new_query_letter4.w, query_letter);
                shuffle_affine_penalty(make_short2(0,0), make_short2(negInfty, negInfty), E, penalty_here31, penalty_diag, penalty_left);
                shuffle_new_query(new_query_letter4);
                if (counter%group_size == 0) {
                    new_query_letter4 = query4[offset];
                    offset += group_size;
                }
                counter++;
            }

            if ((k-1)-(queryLength+thread_result) > 0) {
                //shuffle_max();
                calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
                shuffle_query(new_query_letter4.x, query_letter);
                shuffle_affine_penalty(make_short2(0,0), make_short2(negInfty, negInfty), E, penalty_here31, penalty_diag, penalty_left);
                k++;
            }


            if ((k-1)-(queryLength+thread_result) > 0) {
                //shuffle_max();
                calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
                shuffle_query(new_query_letter4.y, query_letter);
                shuffle_affine_penalty(make_short2(0,0), make_short2(negInfty, negInfty), E, penalty_here31, penalty_diag, penalty_left);
                k++;
            }

            if ((k-1)-(queryLength+thread_result) > 0) {
                //shuffle_max();
                calc32_local_affine_float(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            }
        }
    }

    template<class ScoreOutputIterator>
    __device__
    void computeMultiPass(
        ScoreOutputIterator const devAlignmentScores,
        const bool overflow_check, 
        int* const d_overflow_number, 
        size_t* const d_overflow_positions,
        const char4* query4,
        int queryLength
    ) const{
        int check_last;
        int check_last2;
        computeCheckLast(check_last, check_last2);

        const int length_S0 = devLengths[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blockIdx.x+2*((threadIdx.x%check_last)/group_size)]];
        const size_t base_S0 = devOffsets[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blockIdx.x+2*((threadIdx.x%check_last)/group_size)]]-devOffsets[0];
        int length_S1 = devLengths[d_positions_of_selected_lengths[2*(blockDim.x/group_size)*blockIdx.x+2*((threadIdx.x%check_last)/group_size)+1]];
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

        const int length = max(length_S0, length_S1);
        const int passes = (length + (group_size*numRegs) - 1) / (group_size*numRegs);
        // constexpr int length = 4096;
        // constexpr int passes = (length + (group_size*numRegs) - 1) / (group_size*numRegs);

        short2 maximum = make_short2(0,0);

        computeFirstPass(maximum, devS0, length_S0, devS1, length_S1, query4, queryLength);

        for (int pass = 1; pass < passes-1; pass++) {
            computeMiddlePass(pass, maximum, devS0, length_S0, devS1, length_S1, query4, queryLength);
        }

        computeFinalPass(passes, maximum, devS0, length_S0, devS1, length_S1, query4, queryLength);

        for (int offset=group_size/2; offset>0; offset/=2){
            maximum = vimax(maximum,shfl_down_2xint16(0xFFFFFFFF,maximum,offset,group_size));
        }

        const int group_id = threadIdx.x % group_size;
        if (!group_id) {

            // check for overflow
            if (overflow_check){
                //TODO which value for max_short ???
                short max_short = MAX_ACC_SHORT;
                const int alignmentNumber0 = 2*(blockDim.x/group_size)*blockIdx.x+2*(threadIdx.x/group_size);
                if(alignmentNumber0 < numSelected){
                    if (maximum.y >= max_short) {
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
                    if (maximum.x >= max_short) {
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
        size_t* const d_overflow_positions,
        const char4* query4,
        int queryLength
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
            short2 maximum = make_short2(0,0);

            computeSinglePass(maximum, devS0, length_S0, devS1, length_S1, warpMaxLength, query4, queryLength);

            for (int offset=group_size/2; offset>0; offset/=2){
                maximum = vimax(maximum,shfl_down_2xint16(0xFFFFFFFF,maximum,offset,group_size));
            }

            const int group_id = threadIdx.x % group_size;
            if (!group_id) {

                // check for overflow
                if (overflow_check){
                    //TODO which value for max_short ???
                    short max_short = MAX_ACC_SHORT;
                    const int alignmentNumber0 = 2*(blockDim.x/group_size)*blockIdx.x+2*(threadIdx.x/group_size);
                    if(alignmentNumber0 < numSelected){
                        if (maximum.y >= max_short) {
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
                        if (maximum.x >= max_short) {
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
template <int group_size, int numRegs, int blosumDim, class ScoreOutputIterator, class PositionsIterator> 
#if __CUDA_ARCH__ >= 800
__launch_bounds__(256,2)
//__launch_bounds__(512,1)
#else
__launch_bounds__(256)
#endif
__global__
void NW_local_affine_many_pass_s16_DPX_new(
    __grid_constant__ const char * const devChars,
    __grid_constant__ ScoreOutputIterator const devAlignmentScores,
    __grid_constant__ short2 * const devTempHcol2,
    __grid_constant__ short2 * const devTempEcol2,
    __grid_constant__ const size_t* const devOffsets,
    __grid_constant__ const size_t* const devLengths,
    __grid_constant__ PositionsIterator const d_positions_of_selected_lengths,
    __grid_constant__ const int numSelected,
	__grid_constant__ size_t* const d_overflow_positions,
	__grid_constant__ int* const d_overflow_number,
	__grid_constant__ const bool overflow_check,
    __grid_constant__ const char4* const query4,
    __grid_constant__ const int queryLength,
    __grid_constant__ const short gap_open,
    __grid_constant__ const short gap_extend
) {
    using Processor = DPXAligner_s16<group_size, numRegs, blosumDim, PositionsIterator>;
    extern __shared__ short2 shared_blosum_dpx_s16[];
    //__shared__ typename Processor::BLOSUM62_SMEM shared_blosum;

    Processor processor(
        shared_blosum_dpx_s16,
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

    processor.computeMultiPass(
        devAlignmentScores, 
        overflow_check, d_overflow_number, d_overflow_positions,
        query4,
        queryLength
    );
}



template <int blocksize, int group_size, int numRegs, class ScoreOutputIterator, class PositionsIterator> 
void call_NW_local_affine_many_pass_s16_DPX_new(
    BlosumType /*blosumType*/,
    const char * const devChars,
    ScoreOutputIterator const devAlignmentScores,
    short2 * const devTempHcol2,
    short2 * const devTempEcol2,
    const size_t* const devOffsets,
    const size_t* const devLengths,
    PositionsIterator const d_positions_of_selected_lengths,
    const int numSelected,
	size_t* const d_overflow_positions,
	int* const d_overflow_number,
	const bool overflow_check,
    const char4* query4,
    const int queryLength,
    const short gap_open,
    const short gap_extend,
    cudaStream_t stream
) {
    constexpr int groupsPerBlock = blocksize / group_size;
    constexpr int alignmentsPerGroup = 2;
    constexpr int alignmentsPerBlock = groupsPerBlock * alignmentsPerGroup;

    int smem = sizeof(short2) * hostBlosumDim * hostBlosumDim * hostBlosumDim;

    if(hostBlosumDim == 21){
        auto kernel = NW_local_affine_many_pass_s16_DPX_new<group_size, numRegs, 21, ScoreOutputIterator, PositionsIterator>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

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
        auto kernel = NW_local_affine_many_pass_s16_DPX_new<group_size, numRegs, 25, ScoreOutputIterator, PositionsIterator>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

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
__launch_bounds__(256,2)
//__launch_bounds__(512,1)
#else
__launch_bounds__(256)
#endif
__global__
void NW_local_affine_single_pass_s16_DPX_new(
    __grid_constant__ const char * const devChars,
    __grid_constant__ ScoreOutputIterator const devAlignmentScores,
    __grid_constant__ const size_t* const devOffsets,
    __grid_constant__ const size_t* const devLengths,
    __grid_constant__ PositionsIterator const d_positions_of_selected_lengths,
    __grid_constant__ const int numSelected,
	__grid_constant__ size_t* const d_overflow_positions,
	__grid_constant__ int* const d_overflow_number,
	__grid_constant__ const bool overflow_check,
    __grid_constant__ const char4* const query4,
    __grid_constant__ const int queryLength,
    __grid_constant__ const short gap_open,
    __grid_constant__ const short gap_extend
) {
    __builtin_assume(blockDim.x == blocksize);
    __builtin_assume(blockDim.x % group_size == 0);


    using Processor = DPXAligner_s16<group_size, numRegs, blosumDim, PositionsIterator>;
    extern __shared__ short2 shared_blosum_dpx_s16[];
    //__shared__ typename Processor::BLOSUM62_SMEM shared_blosum;

    Processor processor(
        shared_blosum_dpx_s16,
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

    processor.computeSinglePass(
        devAlignmentScores, 
        overflow_check, d_overflow_number, d_overflow_positions,
        query4,
        queryLength
    );
}


template <int blocksize, int group_size, int numRegs, class ScoreOutputIterator, class PositionsIterator> 
void call_NW_local_affine_single_pass_s16_DPX_new(
    BlosumType /*blosumType*/,
    const char * const devChars,
    ScoreOutputIterator const devAlignmentScores,
    const size_t* const devOffsets,
    const size_t* const devLengths,
    PositionsIterator const d_positions_of_selected_lengths,
    const int numSelected,
	size_t* const d_overflow_positions,
	int* const d_overflow_number,
	const bool overflow_check,
    const char4* query4,
    const int queryLength,
    const short gap_open,
    const short gap_extend,
    cudaStream_t stream
){
    
    constexpr int groupsPerBlock = blocksize / group_size;
    constexpr int alignmentsPerGroup = 2;
    constexpr int alignmentsPerBlock = groupsPerBlock * alignmentsPerGroup;

    int smem = sizeof(short2) * hostBlosumDim * hostBlosumDim * hostBlosumDim;

    if(hostBlosumDim == 21){
        auto kernel = NW_local_affine_single_pass_s16_DPX_new<blocksize, group_size, numRegs, 21, ScoreOutputIterator, PositionsIterator>;
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
            query4,
            queryLength,
            gap_open,
            gap_extend
        );
    #ifdef CAN_USE_FULL_BLOSUM
    }else if(hostBlosumDim == 25){
        auto kernel = NW_local_affine_single_pass_s16_DPX_new<blocksize, group_size, numRegs, 25, ScoreOutputIterator, PositionsIterator>;
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



#endif