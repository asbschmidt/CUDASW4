#ifndef CONVERT_CUH
#define CONVERT_CUH

#ifdef __CUDACC__
__host__ __device__ __forceinline__
#endif
char convert_AA_alphabetical(char AA_norm) {
    AA_norm = AA_norm-65;
    if ((AA_norm >= 0) && (AA_norm <=8)) return AA_norm;
    if ((AA_norm >= 10) && (AA_norm <=13)) return AA_norm-1;
    if ((AA_norm >= 15) && (AA_norm <=19)) return AA_norm-2;
    if ((AA_norm >= 21) && (AA_norm <=22)) return AA_norm-3;
    if (AA_norm == 24) return AA_norm-4;
    return 1; // else
};

#ifdef __CUDACC__
__host__ __device__ __forceinline__
#endif
char convert_AA(const char& AA) {
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
    return 20; //  else
};

#endif