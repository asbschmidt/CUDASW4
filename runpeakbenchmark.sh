#!/bin/bash


# 0 = half2, 1 = dpxs16, 2 = dpxs32, 3 = float
kerneltype=0
nsysprofile=0
if [ "$#" -ge 1 ]; then
    kerneltype=$1
fi
if [ "$#" -ge 2 ]; then
    nsysprofile=$2
fi

echo "kerneltype $kerneltype, nsys $nsysprofile"

if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
    export CUDA_VISIBLE_DEVICES=0
fi


ALLQUERIESFILE=allqueries.fasta


make -j release

pseudosize=1000000
commonargs="--query $ALLQUERIESFILE --top 0 --verbose --uploadFull --prefetchDBFile --mat blosum62"

if [ $kerneltype -eq 0 ]; then
    kernelargs="--singlePassType Half2 --manyPassType_small Half2 --manyPassType_large Float"
    for pseudolength in 128 256 512 768 1024 2048; do
        echo "run pseudo half2 $pseudolength"

        if [ $nsysprofile -eq 0 ]; then
            /usr/bin/time -v ./align $commonargs $kernelargs --pseudodb $pseudosize $pseudolength \
                > "results_pseudo_"$pseudosize"_"$pseudolength"_half2.txt" 2>&1;
        else
            nsys profile -f true -o "pseudo_"$pseudosize"_"$pseudolength_"half2_nsys" ./align $commonargs $kernelargs --pseudodb $pseudosize $pseudolength
        fi
    done
    
elif [ $kerneltype -eq 1 ]; then

    kernelargs="--singlePassType DPXs16 --manyPassType_small DPXs16 --manyPassType_large DPXs32"
    for pseudolength in 128 256 512 768 1024 2048; do
        echo "run pseudo dpx_s16 $pseudolength"

        if [ $nsysprofile -eq 0 ]; then
            /usr/bin/time -v ./align $commonargs $kernelargs --pseudodb $pseudosize $pseudolength \
                > "results_pseudo_"$pseudosize"_"$pseudolength"_dpxs16.txt" 2>&1;
        else
            nsys profile -f true -o "pseudo_"$pseudosize"_"$pseudolength_"dpxs16_nsys" ./align $commonargs $kernelargs --pseudodb $pseudosize $pseudolength
        fi
    done

    
elif [ $kerneltype -eq 2 ]; then

    kernelargs="--singlePassType DPXs32"
    for pseudolength in 128 256 512 768 1024; do
        echo "run pseudo dpx_s32 $pseudolength"

        if [ $nsysprofile -eq 0 ]; then
            /usr/bin/time -v ./align $commonargs $kernelargs --pseudodb $pseudosize $pseudolength \
                > "results_pseudo_"$pseudosize"_"$pseudolength"_dpxs32.txt" 2>&1;
        else
            nsys profile -f true -o "pseudo_"$pseudosize"_"$pseudolength_"dpxs32_nsys" ./align $commonargs $kernelargs --pseudodb $pseudosize $pseudolength
        fi
    done

elif [ $kerneltype -eq 3 ]; then

    kernelargs="--singlePassType Float"
    for pseudolength in 128 256 512 768 1024; do
        echo "run pseudo float $pseudolength"

        if [ $nsysprofile -eq 0 ]; then
            /usr/bin/time -v ./align $commonargs $kernelargs --pseudodb $pseudosize $pseudolength \
                > "results_pseudo_"$pseudosize"_"$pseudolength"_float.txt" 2>&1;
        else
            nsys profile -f true -o "pseudo_"$pseudosize"_"$pseudolength_"float_nsys" ./align $commonargs $kernelargs --pseudodb $pseudosize $pseudolength
        fi
    done

fi






