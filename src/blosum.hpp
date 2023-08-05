#ifndef BLOSUM_HPP
#define BLOSUM_HPP

#include "types.hpp"
#include "util.cuh"
#include <array>
#include <string>
#include <vector>

namespace cudasw4{

#ifdef __CUDACC__

extern __constant__ char deviceBlosum[25*25];
extern __constant__ int deviceBlosumDim;
extern __constant__ int deviceBlosumDimSquared;

#endif

extern char hostBlosum[25*25];
extern int hostBlosumDim;
extern int hostBlosumDimSquared;

//set host and device global blosum variables
void setProgramWideBlosum(BlosumType blosumType, const std::vector<int>& deviceIds);

} //namespace cudasw4

#endif