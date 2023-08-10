




# CUDASW4
CUDASW++4.0: Ultra-fast GPU-based Smith-Waterman Protein Sequence Database Search

## Software requirements
* zlib
* make
* C++17 compiler
* CUDA Toolkit 12 or newer

## Hardware requirements
*   A modern CUDA-capable GPU of generation Ampère or newer. We have tested CUDASW4 on Ampère (sm_80), Ada Lovelace (sm_89), and Hopper (sm_90). Older generations lack hardware-support for specific instructions and may run at reduced speeds or may not run at all.


## Download
`git clone https://github.com/asbschmidt/CUDASW4.git`

## Build
Our software has two components, **makedb** and **align** . **makedb** is used to construct a database which can be queried by **align**.

The build step compiles the GPU code for all GPU archictectures of GPUs detected in the system. The CUDA environment variable `CUDA_VISIBLE_DEVICES` can be used to control the detected GPUs. If `CUDA_VISIBLE_DEVICES` is not set, it will default to all GPUs in the system.

* Build makedb: `make makedb`

* Build align: `make align`

* Build align for the GPU architecture of GPUs 0 and 1: `CUDA_VISIBLE_DEVICES=0,1 make align`

## Database construction
Use makedb to create a database from a fasta file. The file can be gzip'ed.
At the moment, it is required that the complete uncompressed fasta file fits into RAM. This may change in the future.

```
mkdir -p dbfolder
./makedb input.fa(.gz) dbfolder/dbname
```

## Querying the database
Use **align** to query the database. **align** has two mandatory arguments. 1. The query file which contains all queries
2. The path to the reference database. Run `./align --help` to get a complete list of options.

Similar to the build process, **align** will use all GPUs that are set with `CUDA_VISIBLE_DEVICES`, or all GPUs if `CUDA_VISIBLE_DEVICES` is not set. 

```
./align --query queries.fa(.gz) --db dbfolder/dbname

CUDA_VISIBLE_DEVICES=0,1 ./align --query queries.fa(.gz) --db dbfolder/dbname
```

## Scoring options

```
    --top val : Output the val best scores. Default val = 10.
    --mat val : Set substitution matrix. Supported values: blosum45, blosum50, blosum62, blosum80. Default val = blosum62.
    --gop val : Gap open score. Overwrites blosum-dependent default score.
    --gex val : Gap extend score. Overwrites blosum-dependent default score.
```

The default gap scores are listed in the following table.
|            | blosum45 | blosum50 | blosum62 | blosum80 |
|------------|----------|----------|----------|----------|
| gap_open   | -13      | -13      | -11      | -10      |
| gap_extend | -2       | -2       | -1       | -1       |

## Memory options

```
    --maxGpuMem val : Try not to use more than val bytes of gpu memory per gpu. This is not a hard limit. Can use suffix K,M,G. All available gpu memory by default.
    --maxTempBytes val : Size of temp storage in GPU memory. Can use suffix K,M,G. Default val = 4G
    --maxBatchBytes val : Process DB in batches of at most val bytes. Can use suffix K,M,G. Default val = 128M
    --maxBatchSequences val : Process DB in batches of at most val sequences. Default val = 10000000
```

Depending on the database size and available total GPU memory, the database is transferred to the GPU once for all queries, or it is processed in batches which requires a transfer for each query. Above options give some control over memory usage. For best performance, the complete database must fit into `maxGpuMem` times the number of used GPUs.

## Other options
```
    --verbose : More console output. Shows timings.
    --printLengthPartitions : Print number of sequences per length partition in db.
    --interactive : Loads DB, then waits for sequence input by user
    --help : Print all options
```


# Benchmark instructions

## Peak performance benchmark
`./runpeakbenchmark.sh kerneltype [nsys]`

kerneltype 0: half2, kerneltype 1: dpx_s16, kerneltype 2: dpx_s32, kerneltype 3: float

if nsys = 1, create nsys profile

## sprot benchmark
`./runsprotbenchmark.sh kerneltype [nsys]`

kerneltype 0: half2, kerneltype 1: dpx

if nsys = 1, create nsys profile

## refseq50 benchmark
`./runrefseq50benchmark.sh kerneltype [nsys]`

kerneltype 0: half2, kerneltype 1: dpx

if nsys = 1, create nsys profile