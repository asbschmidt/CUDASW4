#!/bin/bash

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

NAME=sprot

DBFOLDER=benchmarkdbs
DBSRCURL=https://ftp.expasy.org/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz # 88 megabyte file
DBSRCFILENAME=uniprot_sprot.fasta.gz
DBSRCFULLPATH=$DBFOLDER/$DBSRCFILENAME
DBPREFIX=$DBFOLDER"/"$NAME"_db"

ALLQUERIESFILE=allqueries.fasta

./benchmarksetup.sh $DBFOLDER $DBSRCURL $DBSRCFILENAME $DBSRCFULLPATH $DBPREFIX


commonargs="--query $ALLQUERIESFILE --db $DBPREFIX --top 0 --verbose --uploadFull --prefetchDBFiles --mat blosum62"

if [ $kerneltype -eq 0 ]; then
    echo "run half2"
    kernelargs="--singlePassType Half2 --manyPassType_small Half2 --manyPassType_large Float"

    if [ $nsysprofile -eq 0 ]; then
        /usr/bin/time -v ./align $commonargs $kernelargs > "results_"$NAME"_half2.txt" 2>&1;
    else
        nsys profile -f true -o $NAME"_half2_nsys" ./align $commonargs $kernelargs
    fi
elif [ $kerneltype -eq 1 ]; then
    echo "run dpx"
    kernelargs="--singlePassType DPXs16 --manyPassType_small DPXs16 --manyPassType_large DPXs32"
    
    if [ $nsysprofile -eq 0 ]; then
        /usr/bin/time -v ./align $commonargs $kernelargs > "results_"$NAME"_dpx.txt" 2>&1;
    else
        nsys profile -f true -o $NAME"_dpx_nsys" ./align $commonargs $kernelargs
    fi
fi






