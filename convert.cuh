#ifndef CONVERT_CUH
#define CONVERT_CUH

namespace cudasw4{

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
    return 20; //  else
};

#ifdef __CUDACC__
__host__ __device__ __forceinline__
#endif
char inverse_convert_AA(const char& AA) {
    if (AA == 0) return 'A';
    if (AA == 1) return 'R';
    if (AA == 2) return 'N';
    if (AA == 3) return 'D';
    if (AA == 4) return 'C';
    if (AA == 5) return 'Q';
    if (AA == 6) return 'E';
    if (AA == 7) return 'G';
    if (AA == 8) return 'H';
    if (AA == 9) return 'I';
    if (AA == 10) return 'L';
    if (AA == 11) return 'K';
    if (AA == 12) return 'M';
    if (AA == 13) return 'F';
    if (AA == 14) return 'P';
    if (AA == 15) return 'S';
    if (AA == 16) return 'T';
    if (AA == 17) return 'W';
    if (AA == 18) return 'Y';
    if (AA == 19) return 'V';
    return '-'; //  else
};

} //namespace cudasw4

#endif