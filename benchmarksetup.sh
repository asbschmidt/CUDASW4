#!/bin/bash

if [ "$#" -ne 5 ]; then
    echo "$0 wrong number of arguments"
    exit
fi


DBFOLDER=$1
DBSRCURL=$2
DBSRCFILENAME=$3
DBSRCFULLPATH=$4
DBPREFIX=$5

make -j release

mkdir -p $DBFOLDER

if [ ! -e  $DBPREFIX"0chars" ]; then

    if [ ! -e $DBSRCFULLPATH ]; then
        wget -O $DBSRCFULLPATH $DBSRCURL
    fi

    ./makedb $DBSRCFULLPATH $DBPREFIX
fi