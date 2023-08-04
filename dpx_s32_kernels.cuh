#ifndef DPX_S32_KERNELS_CUH
#define DPX_S32_KERNELS_CUH

#include "blosum.hpp"
#include "config.hpp"

namespace cudasw4{

template <int numRegs, int blosumDim, class PositionsIterator> 
struct DPXAligner_s32{
    static constexpr int group_size = 32;

    static constexpr int negInfty = -10000;

    static constexpr int deviceBlosumDimCexpr = blosumDim;
    static constexpr int deviceBlosumDimCexprSquared = deviceBlosumDimCexpr * deviceBlosumDimCexpr;

    int* shared_blosum;

    int numSelected;
    int gap_open;
    int gap_extend;
    PositionsIterator d_positions_of_selected_lengths;
    const char* devChars;
    int2* devTempHcol2;
    int2* devTempEcol2;
    const size_t* devOffsets;
    const SequenceLengthT* devLengths;

    __device__
    DPXAligner_s32(
        int* shared_blosum_,
        const char* devChars_,
        int2* devTempHcol2_,
        int2* devTempEcol2_,
        const size_t* devOffsets_,
        const SequenceLengthT* devLengths_,
        PositionsIterator d_positions_of_selected_lengths_,
        int gap_open_,
        int gap_extend_
    ) : shared_blosum(shared_blosum_),
        devChars(devChars_),
        devTempHcol2(devTempHcol2_),
        devTempEcol2(devTempEcol2_),
        devOffsets(devOffsets_),
        devLengths(devLengths_),
        d_positions_of_selected_lengths(d_positions_of_selected_lengths_),
        gap_open(gap_open_),
        gap_extend(gap_extend_)
    {
        for (int i=threadIdx.x; i<deviceBlosumDimSquared; i+=32){
            shared_blosum[(i/deviceBlosumDim) * deviceBlosumDim + (i%deviceBlosumDim)]=deviceBlosum[i];
        }
        __syncwarp();


    }


    __device__
    void checkHEindex(int x, int line) const{
        // if(x < 0){printf("line %d\n", line);}
        // assert(x >= 0); //positive index
        // assert(2*(blockDim.x/group_size)*blockIdx.x * queryLength <= base_3 + x);
        // assert(base_3+x < 2*(blockDim.x/group_size)*(blockIdx.x+1) * queryLength);
    };

    __device__
    void initial_calc32_local_affine_int(const int value, char query_letter, int& E, int& penalty_here31, int penalty_diag, int penalty_left, int& maximum, 
        int (&subject)[numRegs], 
        int (&penalty_here_array)[numRegs],
        int (&F_here_array)[numRegs]
    ) const{
        const int* const sbt_row = &shared_blosum[int(query_letter) * deviceBlosumDim];

        const int score2_0 = sbt_row[subject[0]];
        int penalty_temp0 = penalty_here_array[0];
        if (!value || (threadIdx.x%group_size)) E = max(E+gap_extend, penalty_left+gap_open);
        F_here_array[0] = max(F_here_array[0]+gap_extend, penalty_here_array[0]+gap_open);
        penalty_here_array[0] = __vimax3_s32_relu(penalty_diag + score2_0, E, F_here_array[0]);


        const int score2_1 = sbt_row[subject[1]];
        int penalty_temp1 = penalty_here_array[1];
        E = max(E+gap_extend, penalty_here_array[0]+gap_open);
        F_here_array[1] = max(F_here_array[1]+gap_extend, penalty_here_array[1]+gap_open);
        penalty_here_array[1] = __vimax3_s32_relu(penalty_temp0 + score2_1, E, F_here_array[1]);
        maximum = __vimax3_s32(maximum,penalty_here_array[0],penalty_here_array[1]);

        #pragma unroll
        for (int i=1; i<numRegs/2; i++) {
            const int score2_2i = sbt_row[subject[2*i]];
            //score2.y = sbt_row[subject1[i].x];
            penalty_temp0 = penalty_here_array[2*i];
            E = max(E+gap_extend, penalty_here_array[2*i-1]+gap_open);
            F_here_array[2*i] = max(F_here_array[2*i]+gap_extend, penalty_here_array[2*i]+gap_open);
            penalty_here_array[2*i] = __vimax3_s32_relu(penalty_temp1+score2_2i,E,F_here_array[2*i]);

            const int score2_2i1 = sbt_row[subject[2*i+1]];
            //score2.y = sbt_row[subject1[i].y];
            penalty_temp1 = penalty_here_array[2*i+1];
            E = max(E+gap_extend, penalty_here_array[2*i]+gap_open);
            F_here_array[2*i+1] = max(F_here_array[2*i+1]+gap_extend, penalty_here_array[2*i+1]+gap_open);
            penalty_here_array[2*i+1] = __vimax3_s32_relu(penalty_temp0+score2_2i1,E,F_here_array[2*i+1]);
			maximum = __vimax3_s32(maximum,penalty_here_array[2*i],penalty_here_array[2*i+1]);
        }

        penalty_here31 = penalty_here_array[numRegs-1];
        E = max(E+gap_extend, penalty_here31+gap_open);
        #pragma unroll //UNROLLHERE
        for (int i=0; i<numRegs; i++) F_here_array[i] = max(F_here_array[i]+gap_extend, penalty_here_array[i]+gap_open);
    };

    __device__
    void calc32_local_affine_int(char query_letter, int& E, int& penalty_here31, int penalty_diag, int& maximum, 
        int (&subject)[numRegs],
        int (&penalty_here_array)[numRegs],
        int (&F_here_array)[numRegs]
    ) const{
        const int* const sbt_row = &shared_blosum[int(query_letter) * deviceBlosumDim];

        const int score2_0 = sbt_row[subject[0]];
        int penalty_temp0 = penalty_here_array[0];
        penalty_here_array[0] = __vimax3_s32_relu(penalty_diag + score2_0,E,F_here_array[0]);
        int penalty_temp1 = penalty_here_array[0]+gap_open;
        E = __viaddmax_s32(E,gap_extend,penalty_temp1);
        F_here_array[0] = __viaddmax_s32(F_here_array[0],gap_extend,penalty_temp1);

        const int score2_1 = sbt_row[subject[1]];
        penalty_temp1 = penalty_here_array[1];
        penalty_here_array[1] = __vimax3_s32_relu(penalty_temp0+score2_1,E,F_here_array[1]);
        penalty_temp0 = penalty_here_array[1]+gap_open;
        E = __viaddmax_s32(E,gap_extend,penalty_temp0);
        F_here_array[1] = __viaddmax_s32(F_here_array[1],gap_extend,penalty_temp0);
        maximum = __vimax3_s32(maximum,penalty_here_array[0],penalty_here_array[1]);


        #pragma unroll
        for (int i=1; i<numRegs/2; i++) {
            const int score2_2i = sbt_row[subject[2*i]];
            penalty_temp0 = penalty_here_array[2*i];
            penalty_here_array[2*i] = __vimax3_s32_relu(penalty_temp1+score2_2i,E,F_here_array[2*i]);
            //if (penalty_here_array[2*i] > maximum) maximum = penalty_here_array[2*i];
            penalty_temp1 = penalty_here_array[2*i]+gap_open;
            E = __viaddmax_s32(E,gap_extend,penalty_temp1);
            F_here_array[2*i] = __viaddmax_s32(F_here_array[2*i],gap_extend,penalty_temp1);

            const int score2_2i1 = sbt_row[subject[2*i+1]];
            penalty_temp1 = penalty_here_array[2*i+1];
            penalty_here_array[2*i+1] = __vimax3_s32_relu(penalty_temp0+score2_2i1,E,F_here_array[2*i+1]);
            //if (penalty_here_array[2*i+1] > maximum) maximum = penalty_here_array[2*i+1];
            penalty_temp0 = penalty_here_array[2*i+1]+gap_open;
            E = __viaddmax_s32(E,gap_extend,penalty_temp0);
            F_here_array[2*i+1] = __viaddmax_s32(F_here_array[2*i+1],gap_extend,penalty_temp0);
            maximum = __vimax3_s32(maximum,penalty_here_array[2*i],penalty_here_array[2*i+1]);
        }

		//for (int i=0; i<numRegs/4; i++)
		//	maximum = max(maximum,max(max(penalty_here_array[4*i],penalty_here_array[4*i+1]), max(penalty_here_array[4*i+2],penalty_here_array[4*i+3])));
        penalty_here31 = penalty_here_array[numRegs-1];
    }

    __device__    
    void init_penalties_local(int value, int& penalty_diag, int& penalty_left, 
        int (&penalty_here_array)[numRegs], 
        int (&F_here_array)[numRegs]
    ) const{
        penalty_left = negInfty;
        penalty_diag = negInfty;
        #pragma unroll
        for (int i=0; i<numRegs; i++) penalty_here_array[i] = negInfty;
        #pragma unroll //UNROLLHERE
        for (int i=0; i<numRegs; i++) F_here_array[i] = negInfty;
        if (threadIdx.x % group_size == 0) {
            penalty_left = 0;
            penalty_diag = 0;
            #pragma unroll
            for (int i=0; i<numRegs; i++) penalty_here_array[i] = 0;
        }
        if (threadIdx.x % group_size == 1) {
            penalty_left = 0;
        }
    }

    __device__
    void init_local_score_profile_BLOSUM62(SequenceLengthT offset_isc, int (&subject)[numRegs], 
        const char* const devS0, const SequenceLengthT length_S0
    ) const{
        // if (!offset_isc) {
        //     for (int i=threadIdx.x; i<deviceBlosumDimSquared; i+=32) shared_blosum[(i/deviceBlosumDim) * deviceBlosumDim + (i%deviceBlosumDim)]=deviceBlosum[i];
        //     __syncwarp();
        // }

        #pragma unroll //UNROLLHERE
        for (int i=0; i<numRegs; i++) {

            if (offset_isc+numRegs*(threadIdx.x%group_size)+i >= length_S0) subject[i] = 1; // 20;
            else{                
                subject[i] = devS0[offset_isc+numRegs*(threadIdx.x%group_size)+i];
            }
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
        int new_penalty_left, int new_E_left, int& E, 
        int penalty_here31, int& penalty_diag, int& penalty_left
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
    void shuffle_H_E_temp_out(int2& H_temp_out, int2& E_temp_out) const{
        const double temp = __shfl_down_sync(0xFFFFFFFF, *((double*)(&H_temp_out)), 1, 32);
        H_temp_out = *((int2*)(&temp));
        const double temp2 = __shfl_down_sync(0xFFFFFFFF, *((double*)(&E_temp_out)), 1, 32);
        E_temp_out = *((int2*)(&temp2));
    }

    __device__
    void shuffle_H_E_temp_in(int2& H_temp_in, int2& E_temp_in) const{
        const double temp = __shfl_down_sync(0xFFFFFFFF, *((double*)(&H_temp_in)), 1, 32);
        H_temp_in = *((int2*)(&temp));
        const double temp2 = __shfl_down_sync(0xFFFFFFFF, *((double*)(&E_temp_in)), 1, 32);
        E_temp_in = *((int2*)(&temp2));
    }

    __device__
    void set_H_E_temp_out_x(int penalty_here31, int E, int2& H_temp_out, int2& E_temp_out) const{
        if (threadIdx.x == 31) {
            H_temp_out.x = penalty_here31;
            E_temp_out.x = E;
        }
    };

    __device__
    void set_H_E_temp_out_y(int penalty_here31, int E, int2& H_temp_out, int2& E_temp_out) const{
        if (threadIdx.x == 31) {
            H_temp_out.y = penalty_here31;
            E_temp_out.y = E;
        }
    };



    __device__
    void computeFirstPass(
        int& maximum, 
        const char* const devS0, 
        const SequenceLengthT length_S0,
        const char4* query4,
        SequenceLengthT queryLength
    ) const{
        // FIRST PASS (of many passes)
        // Note first pass has always full seqeunce length

        int counter = 1;
        char query_letter = 20;
        char4 new_query_letter4 = query4[threadIdx.x%group_size];
        if (threadIdx.x % group_size== 0) query_letter = new_query_letter4.x;


        const size_t base_3 = size_t(blockIdx.x)*size_t(queryLength);
        int2* const devTempHcol = (&devTempHcol2[base_3]);
        int2* const devTempEcol = (&devTempEcol2[base_3]);

        const int group_id = threadIdx.x % group_size;
        int offset = group_id + group_size;
        int offset_out = group_id;


        int E = negInfty;
        int penalty_here31;
        int penalty_diag;
        int penalty_left;
        int2 H_temp_out;
        int2 E_temp_out;
        int subject[numRegs];
        int penalty_here_array[numRegs];
        int F_here_array[numRegs];
       
        init_penalties_local(0, penalty_diag, penalty_left, penalty_here_array, F_here_array);
        init_local_score_profile_BLOSUM62(0, subject, devS0, length_S0);
        initial_calc32_local_affine_int(0, query_letter, E, penalty_here31, penalty_diag, penalty_left, maximum, subject, penalty_here_array, F_here_array);
        shuffle_query(new_query_letter4.y, query_letter);
        shuffle_affine_penalty(0.f, negInfty, E, penalty_here31, penalty_diag, penalty_left);

        //shuffle_max();
        calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
        shuffle_query(new_query_letter4.z, query_letter);
        shuffle_affine_penalty(0.f, negInfty, E, penalty_here31, penalty_diag, penalty_left);

        //shuffle_max();
        calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
        shuffle_query(new_query_letter4.w, query_letter);
        shuffle_affine_penalty(0.f, negInfty, E, penalty_here31, penalty_diag, penalty_left);
        shuffle_new_query(new_query_letter4);
        counter++;

        for (int k = 4; k <= 28; k+=4) {
            //shuffle_max();
            calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            shuffle_query(new_query_letter4.x, query_letter);
            shuffle_affine_penalty(0.f, negInfty, E, penalty_here31, penalty_diag, penalty_left);

            //shuffle_max();
            calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            shuffle_query(new_query_letter4.y, query_letter);
            shuffle_affine_penalty(0.f, negInfty, E, penalty_here31, penalty_diag, penalty_left);

            //shuffle_max();
            calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            shuffle_query(new_query_letter4.z, query_letter);
            shuffle_affine_penalty(0.f, negInfty, E, penalty_here31, penalty_diag, penalty_left);

            //shuffle_max();
            calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            shuffle_query(new_query_letter4.w, query_letter);
            shuffle_affine_penalty(0.f, negInfty, E, penalty_here31, penalty_diag, penalty_left);
            shuffle_new_query(new_query_letter4);
            counter++;
        }

        for (SequenceLengthT k = 32; k <= queryLength+28; k+=4) {
            //shuffle_max();
            calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out_x(penalty_here31, E, H_temp_out, E_temp_out);
            shuffle_query(new_query_letter4.x, query_letter);
            shuffle_affine_penalty(0.f, negInfty, E, penalty_here31, penalty_diag, penalty_left);

            //shuffle_max();
            calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);            
            set_H_E_temp_out_y(penalty_here31, E, H_temp_out, E_temp_out);
            shuffle_H_E_temp_out(H_temp_out, E_temp_out);
            shuffle_query(new_query_letter4.y, query_letter);
            shuffle_affine_penalty(0.f, negInfty, E, penalty_here31, penalty_diag, penalty_left);

            //shuffle_max();
            calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out_x(penalty_here31, E, H_temp_out, E_temp_out);
            shuffle_query(new_query_letter4.z, query_letter);
            shuffle_affine_penalty(0.f, negInfty, E, penalty_here31, penalty_diag, penalty_left);

            //shuffle_max();
            calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out_y(penalty_here31, E, H_temp_out, E_temp_out);

            if ((counter+8)%16 == 0 && counter > 8) {
                checkHEindex(offset_out, __LINE__);
                devTempHcol[offset_out]=H_temp_out;
                devTempEcol[offset_out]=E_temp_out;
                offset_out += group_size;
            }

            shuffle_H_E_temp_out(H_temp_out, E_temp_out);
            shuffle_query(new_query_letter4.w, query_letter);
            shuffle_affine_penalty(0.f, negInfty, E, penalty_here31, penalty_diag, penalty_left);
            shuffle_new_query(new_query_letter4);
            if (counter%group_size == 0) {
                new_query_letter4 = query4[offset];
                offset += group_size;
            }
            counter++;
        }
        if (queryLength % 4 == 0) {
            const double temp1 = __shfl_up_sync(0xFFFFFFFF, *((double*)(&H_temp_out)), 1, 32);
            H_temp_out = *((int2*)(&temp1));
            const double temp2 = __shfl_up_sync(0xFFFFFFFF, *((double*)(&E_temp_out)), 1, 32);
            E_temp_out = *((int2*)(&temp2));
        }

        if (queryLength%4 == 1) {
            //shuffle_max();
            calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out_x(penalty_here31, E, H_temp_out, E_temp_out);
            set_H_E_temp_out_y(penalty_here31, E, H_temp_out, E_temp_out);
        }

        if (queryLength%4 == 2) {
            //shuffle_max();
            calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out_x(penalty_here31, E, H_temp_out, E_temp_out);
            shuffle_query(new_query_letter4.x, query_letter);
            shuffle_affine_penalty(0.f, negInfty, E, penalty_here31, penalty_diag, penalty_left);
            //shuffle_max();
            calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out_y(penalty_here31, E, H_temp_out, E_temp_out);
        }
        if (queryLength%4 == 3) {
            //shuffle_max();
            calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out_x(penalty_here31, E, H_temp_out, E_temp_out);
            shuffle_query(new_query_letter4.x, query_letter);
            shuffle_affine_penalty(0.f, negInfty, E, penalty_here31, penalty_diag, penalty_left);
            //shuffle_max();
            calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out_y(penalty_here31, E, H_temp_out, E_temp_out);
            shuffle_H_E_temp_out(H_temp_out, E_temp_out);
            shuffle_query(new_query_letter4.y, query_letter);
            shuffle_affine_penalty(0.f, negInfty, E, penalty_here31, penalty_diag, penalty_left);
            //shuffle_max();
            calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out_x(penalty_here31, E, H_temp_out, E_temp_out);
            set_H_E_temp_out_y(penalty_here31, E, H_temp_out, E_temp_out);
        }
        const int final_out = queryLength % 64;
        const int from_thread_id = 32 - ((final_out+1)/2);

        //printf("tid %d, offset_out %d, from_thread_id %d\n", threadIdx.x, offset_out, from_thread_id);
        if (threadIdx.x>=from_thread_id) {
            checkHEindex(offset_out-from_thread_id, __LINE__);
            devTempHcol[offset_out-from_thread_id]=H_temp_out;
            devTempEcol[offset_out-from_thread_id]=E_temp_out;
        }
    }

    __device__
    void computeMiddlePass(
        int pass, 
        int& maximum, 
        const char* const devS0, 
        const SequenceLengthT length_S0,
        const char4* query4,
        SequenceLengthT queryLength
    ) const{
        int counter = 1;
        char query_letter = 20;
        char4 new_query_letter4 = query4[threadIdx.x%group_size];
        if (threadIdx.x % group_size== 0) query_letter = new_query_letter4.x;


        const size_t base_3 = size_t(blockIdx.x)*size_t(queryLength);
        int2* const devTempHcol = (&devTempHcol2[base_3]);
        int2* const devTempEcol = (&devTempEcol2[base_3]);

        const int group_id = threadIdx.x % group_size;
        int offset = group_id + group_size;
        int offset_out = group_id;
        int offset_in = group_id;
        checkHEindex(offset_in, __LINE__);
        int2 H_temp_in = devTempHcol[offset_in];
        int2 E_temp_in = devTempEcol[offset_in];
        offset_in += group_size;

        int E = negInfty;
        int penalty_here31;
        int penalty_diag;
        int penalty_left;
        int2 H_temp_out;
        int2 E_temp_out;
        int subject[numRegs];
        int penalty_here_array[numRegs];
        int F_here_array[numRegs];

        init_penalties_local(gap_open+(pass*32*numRegs-1)*gap_extend, penalty_diag, penalty_left, penalty_here_array, F_here_array);
        init_local_score_profile_BLOSUM62(pass*(32*numRegs), subject, devS0, length_S0);

        if (!group_id) {
            penalty_left = H_temp_in.x;
            E = E_temp_in.x;
        }


        initial_calc32_local_affine_int(1, query_letter, E, penalty_here31, penalty_diag, penalty_left, maximum, subject, penalty_here_array, F_here_array);
        shuffle_query(new_query_letter4.y, query_letter);
        shuffle_affine_penalty(H_temp_in.y, E_temp_in.y, E, penalty_here31, penalty_diag, penalty_left);
        shuffle_H_E_temp_in(H_temp_in, E_temp_in);

        //shuffle_max();
        calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
        shuffle_query(new_query_letter4.z, query_letter);
        shuffle_affine_penalty(H_temp_in.x, E_temp_in.x, E, penalty_here31, penalty_diag, penalty_left);

        //shuffle_max();
        calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
        shuffle_query(new_query_letter4.w, query_letter);
        shuffle_affine_penalty(H_temp_in.y, E_temp_in.y, E, penalty_here31, penalty_diag, penalty_left);
        shuffle_H_E_temp_in(H_temp_in, E_temp_in);
        shuffle_new_query(new_query_letter4);
        counter++;

        for (int k = 4; k <= 28; k+=4) {
            //shuffle_max();
            calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            shuffle_query(new_query_letter4.x, query_letter);
            shuffle_affine_penalty(H_temp_in.x, E_temp_in.x, E, penalty_here31, penalty_diag, penalty_left);

            //shuffle_max();
            calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            shuffle_query(new_query_letter4.y, query_letter);
            shuffle_affine_penalty(H_temp_in.y, E_temp_in.y, E, penalty_here31, penalty_diag, penalty_left);
            shuffle_H_E_temp_in(H_temp_in, E_temp_in);

            //shuffle_max();
            calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            shuffle_query(new_query_letter4.z, query_letter);
            shuffle_affine_penalty(H_temp_in.x, E_temp_in.x, E, penalty_here31, penalty_diag, penalty_left);

            //shuffle_max();
            calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            shuffle_query(new_query_letter4.w, query_letter);
            shuffle_affine_penalty(H_temp_in.y, E_temp_in.y, E, penalty_here31, penalty_diag, penalty_left);
            shuffle_new_query(new_query_letter4);
            shuffle_H_E_temp_in(H_temp_in, E_temp_in);

            counter++;
        }
        for (SequenceLengthT k = 32; k <= queryLength+28; k+=4) {
            //shuffle_max();
            calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out_x(penalty_here31, E, H_temp_out, E_temp_out);
            shuffle_query(new_query_letter4.x, query_letter);
            shuffle_affine_penalty(H_temp_in.x, E_temp_in.x, E, penalty_here31, penalty_diag, penalty_left);

            //shuffle_max();
            calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out_y(penalty_here31, E, H_temp_out, E_temp_out);
            shuffle_H_E_temp_out(H_temp_out, E_temp_out);
            shuffle_query(new_query_letter4.y, query_letter);
            shuffle_affine_penalty(H_temp_in.y, E_temp_in.y, E, penalty_here31, penalty_diag, penalty_left);
            shuffle_H_E_temp_in(H_temp_in, E_temp_in);

            //shuffle_max();
            calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out_x(penalty_here31, E, H_temp_out, E_temp_out);
            shuffle_query(new_query_letter4.z, query_letter);
            shuffle_affine_penalty(H_temp_in.x, E_temp_in.x, E, penalty_here31, penalty_diag, penalty_left);

            //shuffle_max();
            calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out_y(penalty_here31, E, H_temp_out, E_temp_out);

            if ((counter+8)%16 == 0 && counter > 8) {
                checkHEindex(offset_out, __LINE__);
                devTempHcol[offset_out]=H_temp_out;
                devTempEcol[offset_out]=E_temp_out;
                offset_out += group_size;
            }
            shuffle_H_E_temp_out(H_temp_out, E_temp_out);
            shuffle_query(new_query_letter4.w, query_letter);
            shuffle_affine_penalty(H_temp_in.y, E_temp_in.y, E, penalty_here31, penalty_diag, penalty_left);
            shuffle_new_query(new_query_letter4);
            if (counter%group_size == 0) {
                new_query_letter4 = query4[offset];
                offset += group_size;
            }
            shuffle_H_E_temp_in(H_temp_in, E_temp_in);
            if (counter%16 == 0) {
                checkHEindex(offset_in, __LINE__);
                H_temp_in = devTempHcol[offset_in];
                E_temp_in = devTempEcol[offset_in];
                offset_in += group_size;
            }
            counter++;
        }

        if (queryLength % 4 == 0) {
            const double temp1 = __shfl_up_sync(0xFFFFFFFF, *((double*)(&H_temp_out)), 1, 32);
            H_temp_out = *((int2*)(&temp1));
            const double temp2 = __shfl_up_sync(0xFFFFFFFF, *((double*)(&E_temp_out)), 1, 32);
            E_temp_out = *((int2*)(&temp2));
        }
        if (queryLength % 4 == 1) {
            //shuffle_max();
            calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out_x(penalty_here31, E, H_temp_out, E_temp_out);
            set_H_E_temp_out_y(penalty_here31, E, H_temp_out, E_temp_out);
        }        

        if (queryLength%4 == 2) {
            //shuffle_max();
            calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out_x(penalty_here31, E, H_temp_out, E_temp_out);
            shuffle_query(new_query_letter4.x, query_letter);
            shuffle_affine_penalty(H_temp_in.x, E_temp_in.x, E, penalty_here31, penalty_diag, penalty_left);
            calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out_y(E, penalty_here31, H_temp_out, E_temp_out);
        }
        if (queryLength%4 == 3) {
            //shuffle_max();
            calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out_x(E, penalty_here31, H_temp_out, E_temp_out);
            shuffle_query(new_query_letter4.x, query_letter);
            shuffle_affine_penalty(H_temp_in.x, E_temp_in.x, E, penalty_here31, penalty_diag, penalty_left);
            //shuffle_max();
            calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out_y(E, penalty_here31, H_temp_out, E_temp_out);
            shuffle_H_E_temp_out(H_temp_out, E_temp_out);
            shuffle_query(new_query_letter4.y, query_letter);
            shuffle_affine_penalty(H_temp_in.y, E_temp_in.y, E, penalty_here31, penalty_diag, penalty_left);

            calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            set_H_E_temp_out_x(E, penalty_here31, H_temp_out, E_temp_out);
            set_H_E_temp_out_y(E, penalty_here31, H_temp_out, E_temp_out);
        }
        const int final_out = queryLength % 64;
        const int from_thread_id = 32 - ((final_out+1)/2);

        if (threadIdx.x>=from_thread_id) {
            checkHEindex(offset_out-from_thread_id, __LINE__);
            devTempHcol[offset_out-from_thread_id]=H_temp_out;
            devTempEcol[offset_out-from_thread_id]=E_temp_out;
        }
    }

    __device__ 
    void computeFinalPass(
        int passes, 
        int& maximum, 
        const char* const devS0, 
        const SequenceLengthT length_S0,
        const char4* query4,
        SequenceLengthT queryLength
    ) const{
        int counter = 1;
        char query_letter = 20;
        char4 new_query_letter4 = query4[threadIdx.x%group_size];
        if (threadIdx.x % group_size== 0) query_letter = new_query_letter4.x;


        const size_t base_3 = size_t(blockIdx.x)*size_t(queryLength);
        int2* const devTempHcol = (&devTempHcol2[base_3]);
        int2* const devTempEcol = (&devTempEcol2[base_3]);

        const int group_id = threadIdx.x % group_size;
        int offset = group_id + group_size;
        int offset_in = group_id;
        checkHEindex(offset_in, __LINE__);
        int2 H_temp_in = devTempHcol[offset_in];
        int2 E_temp_in = devTempEcol[offset_in];
        offset_in += group_size;

        const uint32_t thread_result = ((length_S0-1)%(32*numRegs))/numRegs;

        int E = negInfty;
        int penalty_here31;
        int penalty_diag;
        int penalty_left;
        int subject[numRegs];
        int penalty_here_array[numRegs];
        int F_here_array[numRegs];

        init_penalties_local(gap_open+((passes-1)*32*numRegs-1)*gap_extend, penalty_diag, penalty_left, penalty_here_array, F_here_array);
        init_local_score_profile_BLOSUM62((passes-1)*(32*numRegs), subject, devS0, length_S0);
        //copy_H_E_temp_in();
        if (!group_id) {
            penalty_left = H_temp_in.x;
            E = E_temp_in.x;
        }

        initial_calc32_local_affine_int(1, query_letter, E, penalty_here31, penalty_diag, penalty_left, maximum, subject, penalty_here_array, F_here_array);
        shuffle_query(new_query_letter4.y, query_letter);
        shuffle_affine_penalty(H_temp_in.y, E_temp_in.y, E, penalty_here31, penalty_diag, penalty_left);
        shuffle_H_E_temp_in(H_temp_in, E_temp_in);
        if (queryLength+thread_result >=2) {
            //shuffle_max();
            calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            shuffle_query(new_query_letter4.z, query_letter);
            //copy_H_E_temp_in();
            shuffle_affine_penalty(H_temp_in.x, E_temp_in.x, E, penalty_here31, penalty_diag, penalty_left);
        }

        if (queryLength+thread_result >=3) {
            //shuffle_max();
            calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            shuffle_query(new_query_letter4.w, query_letter);
            //copy_H_E_temp_in();
            shuffle_affine_penalty(H_temp_in.y, E_temp_in.y, E, penalty_here31, penalty_diag, penalty_left);
            shuffle_H_E_temp_in(H_temp_in, E_temp_in);
            shuffle_new_query(new_query_letter4);
            counter++;
        }
        if (queryLength+thread_result >=4) {
            SequenceLengthT k;
            //for (k = 5; k < lane_2+thread_result-2; k+=4) {
            for (k = 4; k <= queryLength+(thread_result-3); k+=4) {
                //shuffle_max();
                calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);

                shuffle_query(new_query_letter4.x, query_letter);
                //copy_H_E_temp_in();
                shuffle_affine_penalty(H_temp_in.x, E_temp_in.x, E, penalty_here31, penalty_diag, penalty_left);

                //shuffle_max();
                calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
                shuffle_query(new_query_letter4.y, query_letter);
                //copy_H_E_temp_in();
                shuffle_affine_penalty(H_temp_in.y, E_temp_in.y, E, penalty_here31, penalty_diag, penalty_left);
                shuffle_H_E_temp_in(H_temp_in, E_temp_in);

                //shuffle_max();
                calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
                shuffle_query(new_query_letter4.z, query_letter);
                //copy_H_E_temp_in();
                shuffle_affine_penalty(H_temp_in.x, E_temp_in.x, E, penalty_here31, penalty_diag, penalty_left);

                //shuffle_max();
                calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
                shuffle_query(new_query_letter4.w, query_letter);
                //copy_H_E_temp_in();
                shuffle_affine_penalty(H_temp_in.y, E_temp_in.y, E, penalty_here31, penalty_diag, penalty_left);
                shuffle_new_query(new_query_letter4);
                if (counter%group_size == 0) {
                    new_query_letter4 = query4[offset];
                    offset += group_size;
                }
                shuffle_H_E_temp_in(H_temp_in, E_temp_in);
                if (counter%16 == 0) {
                    checkHEindex(offset_in, __LINE__);
                    H_temp_in = devTempHcol[offset_in];
                    E_temp_in = devTempEcol[offset_in];
                    offset_in += group_size;
                }
                counter++;
            }

            if ((k-1)-(queryLength+thread_result) > 0) {
                //shuffle_max();
                calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
                shuffle_query(new_query_letter4.x, query_letter);
                shuffle_affine_penalty(H_temp_in.x, E_temp_in.x, E, penalty_here31, penalty_diag, penalty_left);
                k++;
            }


            if ((k-1)-(queryLength+thread_result) > 0) {
                //shuffle_max();
                calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
                shuffle_query(new_query_letter4.y, query_letter);
                shuffle_affine_penalty(H_temp_in.y, E_temp_in.y, E, penalty_here31, penalty_diag, penalty_left);
                shuffle_H_E_temp_in(H_temp_in, E_temp_in);
                k++;
            }

            if ((k-1)-(queryLength+thread_result) > 0) {
                //shuffle_max();
                calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            }
        }
    }

    __device__ 
    void computeSinglePass(
        int& maximum, 
        const char* const devS0, 
        const SequenceLengthT length_S0,
        const char4* query4,
        SequenceLengthT queryLength
    ) const{
        int counter = 1;
        char query_letter = 20;
        char4 new_query_letter4 = query4[threadIdx.x%group_size];
        if (threadIdx.x % group_size== 0) query_letter = new_query_letter4.x;

        const int group_id = threadIdx.x % group_size;
        int offset = group_id + group_size;
        int offset_in = group_id;
        checkHEindex(offset_in, __LINE__);
        offset_in += group_size;

        const uint32_t thread_result = ((length_S0-1)%(32*numRegs))/numRegs;

        int E = negInfty;
        int penalty_here31;
        int penalty_diag;
        int penalty_left;
        int subject[numRegs];
        int penalty_here_array[numRegs];
        int F_here_array[numRegs];

        init_penalties_local(0, penalty_diag, penalty_left, penalty_here_array, F_here_array);
        init_local_score_profile_BLOSUM62(0, subject, devS0, length_S0);

        initial_calc32_local_affine_int(0, query_letter, E, penalty_here31, penalty_diag, penalty_left, maximum, subject, penalty_here_array, F_here_array);
        shuffle_query(new_query_letter4.y, query_letter);
        shuffle_affine_penalty(0.f, negInfty, E, penalty_here31, penalty_diag, penalty_left);

        if (queryLength+thread_result >=2) {
            //shuffle_max();
            calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            shuffle_query(new_query_letter4.z, query_letter);
            shuffle_affine_penalty(0.f, negInfty, E, penalty_here31, penalty_diag, penalty_left);
        }

        if (queryLength+thread_result >=3) {
            //shuffle_max();
            calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            shuffle_query(new_query_letter4.w, query_letter);
            //copy_H_E_temp_in();
            shuffle_affine_penalty(0.f, negInfty, E, penalty_here31, penalty_diag, penalty_left);
            shuffle_new_query(new_query_letter4);
            counter++;
        }
        if (queryLength+thread_result >=4) {
            SequenceLengthT k;

            for (k = 4; k <= queryLength+(thread_result-3); k+=4) {
                //shuffle_max();
                calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);

                shuffle_query(new_query_letter4.x, query_letter);
                //copy_H_E_temp_in();
                shuffle_affine_penalty(0.f, negInfty, E, penalty_here31, penalty_diag, penalty_left);

                //shuffle_max();
                calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
                shuffle_query(new_query_letter4.y, query_letter);
                //copy_H_E_temp_in();
                shuffle_affine_penalty(0.f, negInfty, E, penalty_here31, penalty_diag, penalty_left);

                //shuffle_max();
                calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
                shuffle_query(new_query_letter4.z, query_letter);
                //copy_H_E_temp_in();
                shuffle_affine_penalty(0.f, negInfty, E, penalty_here31, penalty_diag, penalty_left);

                //shuffle_max();
                calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
                shuffle_query(new_query_letter4.w, query_letter);
                //copy_H_E_temp_in();
                shuffle_affine_penalty(0.f, negInfty, E, penalty_here31, penalty_diag, penalty_left);
                shuffle_new_query(new_query_letter4);
                if (counter%group_size == 0) {
                    new_query_letter4 = query4[offset];
                    offset += group_size;
                }
                counter++;
            }

            if ((k-1)-(queryLength+thread_result) > 0) {
                //shuffle_max();
                calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
                shuffle_query(new_query_letter4.x, query_letter);
                shuffle_affine_penalty(0.f, negInfty, E, penalty_here31, penalty_diag, penalty_left);
                k++;
            }


            if ((k-1)-(queryLength+thread_result) > 0) {
                //shuffle_max();
                calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
                shuffle_query(new_query_letter4.y, query_letter);
                shuffle_affine_penalty(0.f, negInfty, E, penalty_here31, penalty_diag, penalty_left);
                k++;
            }

            if ((k-1)-(queryLength+thread_result) > 0) {
                //shuffle_max();
                calc32_local_affine_int(query_letter, E, penalty_here31, penalty_diag, maximum, subject, penalty_here_array, F_here_array);
            }
        }
    }

    template<class ScoreOutputIterator>
    __device__
    void compute(
        ScoreOutputIterator const devAlignmentScores,
        const char4* query4,
        SequenceLengthT queryLength
    ) const{


        const SequenceLengthT length_S0 = devLengths[d_positions_of_selected_lengths[blockIdx.x]];
        const size_t base_S0 = devOffsets[d_positions_of_selected_lengths[blockIdx.x]]-devOffsets[0];

        const char* const devS0 = &devChars[base_S0];

        const int passes = (length_S0 + (group_size*numRegs) - 1) / (group_size*numRegs);

        int maximum = 0;

        if(passes == 1){
            computeSinglePass(maximum, devS0, length_S0, query4, queryLength);
        }else{

            computeFirstPass(maximum, devS0, length_S0, query4, queryLength);

            for (int pass = 1; pass < passes-1; pass++) {
                computeMiddlePass(pass, maximum, devS0, length_S0, query4, queryLength);
            }

            computeFinalPass(passes, maximum, devS0, length_S0, query4, queryLength);
        }

        for (int offset=group_size/2; offset>0; offset/=2){
            maximum = max(maximum,__shfl_down_sync(0xFFFFFFFF,maximum,offset,group_size));
        }

        const int group_id = threadIdx.x % group_size;
        if (!group_id) {
            devAlignmentScores[d_positions_of_selected_lengths[blockIdx.x]] = maximum;
        }
    }

};



// Needleman-Wunsch (NW): global alignment with linear gap penalty
// numRegs values per thread
// uses a single warp per CUDA thread block;
// every groupsize threads computes an alignmen score
template <int numRegs, int blosumDim, class ScoreOutputIterator, class PositionsIterator> 
__launch_bounds__(32,16)
//__launch_bounds__(32)
__global__
void NW_local_affine_s32_DPX_new(
    __grid_constant__ const char * const devChars,
    __grid_constant__ ScoreOutputIterator const devAlignmentScores,
    __grid_constant__ int2 * const devTempHcol2,
    __grid_constant__ int2 * const devTempEcol2,
    __grid_constant__ const size_t* const devOffsets,
    __grid_constant__ const SequenceLengthT* const devLengths,
    __grid_constant__ PositionsIterator const d_positions_of_selected_lengths,
    __grid_constant__ const char4* const query4,
    __grid_constant__ const SequenceLengthT queryLength,
    __grid_constant__ const int gap_open,
    __grid_constant__ const int gap_extend
) {
    using Processor = DPXAligner_s32<numRegs, blosumDim, PositionsIterator>;

    //__shared__ typename Processor::BLOSUM62_SMEM shared_blosum;

    //25 is max blosum dimension
    __shared__ int shared_blosum[25 * 25];

    Processor processor(
        shared_blosum,
        devChars,
        devTempHcol2,
        devTempEcol2,
        devOffsets,
        devLengths,
        d_positions_of_selected_lengths,
        gap_open,
        gap_extend
    );

    processor.compute(devAlignmentScores, query4, queryLength);
}

template <int numRegs, class ScoreOutputIterator, class PositionsIterator> 
void call_NW_local_affine_s32_DPX_new(
    BlosumType /*blosumType*/,
    const char * const devChars,
    ScoreOutputIterator const devAlignmentScores,
    int2 * const devTempHcol2,
    int2 * const devTempEcol2,
    const size_t* const devOffsets,
    const SequenceLengthT* const devLengths,
    PositionsIterator const d_positions_of_selected_lengths,
    const int numSelected,
    const char4* query4,
    const SequenceLengthT queryLength,
    const int gap_open,
    const int gap_extend,
    cudaStream_t stream
) {

    if(hostBlosumDim == 21){
        auto kernel = NW_local_affine_s32_DPX_new<numRegs, 21, ScoreOutputIterator, PositionsIterator>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 0);

        dim3 block = 32;
        dim3 grid = numSelected;

        kernel<<<grid, block, 0, stream>>>(
            devChars,
            devAlignmentScores,
            devTempHcol2,
            devTempEcol2,
            devOffsets,
            devLengths,
            d_positions_of_selected_lengths,
            query4,
            queryLength,
            gap_open,
            gap_extend
        ); CUERR;
    #ifdef CAN_USE_FULL_BLOSUM
    }else if(hostBlosumDim == 25){
        auto kernel = NW_local_affine_s32_DPX_new<numRegs, 25, ScoreOutputIterator, PositionsIterator>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 0);

        dim3 block = 32;
        dim3 grid = numSelected;

        kernel<<<grid, block, 0, stream>>>(
            devChars,
            devAlignmentScores,
            devTempHcol2,
            devTempEcol2,
            devOffsets,
            devLengths,
            d_positions_of_selected_lengths,
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


template <int numRegs, int blosumDim, class ScoreOutputIterator, class PositionsIterator>
__launch_bounds__(1,1)
__global__
void launch_process_overflow_alignments_kernel_NW_local_affine_s32_DPX_new(
    __grid_constant__ const int* const d_overflow_number,
    __grid_constant__ int2* const d_temp,
    __grid_constant__ const size_t maxTempBytes,
    __grid_constant__ const char * const devChars,
    __grid_constant__ ScoreOutputIterator const devAlignmentScores,
    __grid_constant__ const size_t* const devOffsets,
    __grid_constant__ const SequenceLengthT* const devLengths,
    __grid_constant__ PositionsIterator const d_positions_of_selected_lengths,
    __grid_constant__ const char4* const query4,
    __grid_constant__ const SequenceLengthT queryLength,
    __grid_constant__ const int gap_open,
    __grid_constant__ const int gap_extend
){
    const int numOverflow = *d_overflow_number;
    if(numOverflow > 0){
        const SequenceLengthT currentQueryLengthWithPadding = SDIV(queryLength, 4) * 4 + sizeof(char4) * 32;
        const size_t tempBytesPerSubjectPerBuffer = sizeof(int2) * currentQueryLengthWithPadding;
        const size_t maxSubjectsPerIteration = std::min(size_t(numOverflow), maxTempBytes / (tempBytesPerSubjectPerBuffer * 2));

        int2* d_tempHcol2 = d_temp;
        int2* d_tempEcol2 = (int2*)(((char*)d_tempHcol2) + maxSubjectsPerIteration * tempBytesPerSubjectPerBuffer);

        const int numIters =  SDIV(numOverflow, maxSubjectsPerIteration);
        for(int iter = 0; iter < numIters; iter++){
            const size_t begin = iter * maxSubjectsPerIteration;
            const size_t end = iter < numIters-1 ? (iter+1) * maxSubjectsPerIteration : numOverflow;
            const size_t num = end - begin;

            cudaMemsetAsync(d_temp, 0, tempBytesPerSubjectPerBuffer * 2 * num, 0);

            // cudaMemsetAsync(d_tempHcol2, 0, tempBytesPerSubjectPerBuffer * num, 0);
            // cudaMemsetAsync(d_tempEcol2, 0, tempBytesPerSubjectPerBuffer * num, 0);

            NW_local_affine_s32_DPX_new<numRegs, blosumDim><<<num, 32>>>(
                devChars, 
                devAlignmentScores,
                d_tempHcol2, 
                d_tempEcol2, 
                devOffsets, 
                devLengths, 
                d_positions_of_selected_lengths + begin, 
                query4,
                queryLength, 
                gap_open, 
                gap_extend
            );
        }
    }
}


template <int numRegs, class ScoreOutputIterator, class PositionsIterator> 
void call_launch_process_overflow_alignments_kernel_NW_local_affine_s32_DPX_new(
    const int* const d_overflow_number,
    int2* const d_temp,
    const size_t maxTempBytes,
    const char * const devChars,
    ScoreOutputIterator const devAlignmentScores,
    const size_t* const devOffsets,
    const SequenceLengthT* const devLengths,
    PositionsIterator const d_positions_of_selected_lengths,
    const char4* const query4,
    const SequenceLengthT queryLength,
    const int gap_open,
    const int gap_extend,
    cudaStream_t stream
){
    if(hostBlosumDim == 21){
        auto kernel = launch_process_overflow_alignments_kernel_NW_local_affine_s32_DPX_new<numRegs, 21, ScoreOutputIterator, PositionsIterator>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 0);

        kernel<<<1, 1, 0, stream>>>(
            d_overflow_number,
            d_temp,
            maxTempBytes,
            devChars,
            devAlignmentScores,
            devOffsets,
            devLengths,
            d_positions_of_selected_lengths,
            query4,
            queryLength,
            gap_open,
            gap_extend
        ); CUERR;
    #ifdef CAN_USE_FULL_BLOSUM
    }else if(hostBlosumDim == 25){
        auto kernel = launch_process_overflow_alignments_kernel_NW_local_affine_s32_DPX_new<numRegs, 25, ScoreOutputIterator, PositionsIterator>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 0);

        kernel<<<1, 1, 0, stream>>>(
            d_overflow_number,
            d_temp,
            maxTempBytes,
            devChars,
            devAlignmentScores,
            devOffsets,
            devLengths,
            d_positions_of_selected_lengths,
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

}

#endif