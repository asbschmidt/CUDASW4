#ifndef BLOSUM_HPP
#define BLOSUM_HPP

#include "blosumTypes.hpp"

#include <array>
#include <string>

#ifdef __CUDACC__
__constant__ char deviceBlosum[25*25];
__constant__ int deviceBlosumDim;
__constant__ int deviceBlosumDimSquared;
#endif

char hostBlosum[25*25];
int hostBlosumDim;
int hostBlosumDimSquared;

//set host and device global variables
void setProgramWideBlosum(BlosumType blosumType);

void setProgramWideBlosum(BlosumType blosumType){
    switch(blosumType){
        case BlosumType::BLOSUM45:
            {
                const auto blosum = BLOSUM45::get1D();
                const int dim = BLOSUM45::dim;
                hostBlosumDim = dim;
                hostBlosumDimSquared = dim * dim;
                auto it = std::copy(blosum.begin(), blosum.end(), hostBlosum);
                assert(std::distance(hostBlosum, it) <= 25 * 25);                
            }
            break;
        case BlosumType::BLOSUM50:
            {
                const auto blosum = BLOSUM50::get1D();
                const int dim = BLOSUM50::dim;
                hostBlosumDim = dim;
                hostBlosumDimSquared = dim * dim;
                auto it = std::copy(blosum.begin(), blosum.end(), hostBlosum);
                assert(std::distance(hostBlosum, it) <= 25 * 25);                
            }
            break;
        case BlosumType::BLOSUM62:
            {
                const auto blosum = BLOSUM62::get1D();
                const int dim = BLOSUM62::dim;
                hostBlosumDim = dim;
                hostBlosumDimSquared = dim * dim;
                auto it = std::copy(blosum.begin(), blosum.end(), hostBlosum);
                assert(std::distance(hostBlosum, it) <= 25 * 25);                
            }
            break;
        case BlosumType::BLOSUM80:
            {
                const auto blosum = BLOSUM80::get1D();
                const int dim = BLOSUM80::dim;
                hostBlosumDim = dim;
                hostBlosumDimSquared = dim * dim;
                auto it = std::copy(blosum.begin(), blosum.end(), hostBlosum);
                assert(std::distance(hostBlosum, it) <= 25 * 25);                
            }
            break;
        case BlosumType::BLOSUM45_20:
            {
                const auto blosum = BLOSUM45_20::get1D();
                const int dim = BLOSUM45_20::dim;
                hostBlosumDim = dim;
                hostBlosumDimSquared = dim * dim;
                auto it = std::copy(blosum.begin(), blosum.end(), hostBlosum);
                assert(std::distance(hostBlosum, it) <= 25 * 25);                
            }
            break;
        case BlosumType::BLOSUM50_20:
            {
                const auto blosum = BLOSUM50_20::get1D();
                const int dim = BLOSUM50_20::dim;
                hostBlosumDim = dim;
                hostBlosumDimSquared = dim * dim;
                auto it = std::copy(blosum.begin(), blosum.end(), hostBlosum);
                assert(std::distance(hostBlosum, it) <= 25 * 25);                
            }
            break;
        case BlosumType::BLOSUM62_20:
            {
                const auto blosum = BLOSUM62_20::get1D();
                const int dim = BLOSUM62_20::dim;
                hostBlosumDim = dim;
                hostBlosumDimSquared = dim * dim;
                auto it = std::copy(blosum.begin(), blosum.end(), hostBlosum);
                assert(std::distance(hostBlosum, it) <= 25 * 25);                
            }
            break;
        case BlosumType::BLOSUM80_20:
            {
                const auto blosum = BLOSUM80_20::get1D();
                const int dim = BLOSUM80_20::dim;
                hostBlosumDim = dim;
                hostBlosumDimSquared = dim * dim;
                auto it = std::copy(blosum.begin(), blosum.end(), hostBlosum);
                assert(std::distance(hostBlosum, it) <= 25 * 25);                
            }
            break;
        default:
            assert(false && "unimplemented blosum copy");
            break;
    }
#ifdef __CUDACC__
    int numGpus = 0;
    cudaGetDeviceCount(&numGpus); CUERR;

    int old = 0;
    cudaGetDevice(&old); CUERR;
    for(int gpu = 0; gpu < numGpus; gpu++){
        cudaSetDevice(gpu); CUERR;
        cudaMemcpyToSymbol(deviceBlosum, &(hostBlosum[0]), sizeof(char) * hostBlosumDim * hostBlosumDim); CUERR;
        cudaMemcpyToSymbol(deviceBlosumDim, &hostBlosumDim, sizeof(int)); CUERR;
        cudaMemcpyToSymbol(deviceBlosumDimSquared, &hostBlosumDimSquared, sizeof(int)); CUERR;
    }
    cudaSetDevice(old); CUERR;
#endif    
}

#endif