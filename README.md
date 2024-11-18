# gpuPairHMM
gpuPairHMM: Ultra-fast GPU-based PairHMM for DNA Variant Calling

## Software requirements
* Linux operating system with compatible CUDA Toolkit 12 or newer
* C++17 compiler

## Hardware requirements
*   We have tested gpuPairHMM on the following GPU architectures: Volta (sm_70), Ampere (sm_80, sm_86), Ada (sm_89), and Hopper (sm_90). 


## Download
`git clone https://github.com/asbschmidt/gpuPairHMM.git`


## Build

Build the executable `gpuPairHMM` with `make`

The build step compiles the GPU code for all GPU archictectures of GPUs detected in the system. The CUDA environment variable `CUDA_VISIBLE_DEVICES` can be used to control the detected GPUs. If `CUDA_VISIBLE_DEVICES` is not set, it will default to all GPUs in the system.

## Usage
```
./gpuPairHMM 

    --inputfile filename : Specify input file
    --outputfile filename : Specify output file
    --verbose : More console output (optional)
```

gpuPairHMM uses GPU 0. Use the CUDA environment variable `CUDA_VISIBLE_DEVICES` to select the GPU in multi-GPU systems

### File format
Input file is a line-based format of records. 
First line of a record specifies number of reads $R$ and number haplotypes $H$ to align against each read. Next $R$ lines specify reads, followed by $H$ lines for haplotype sequences. The lines for reads must contain five strings of equal length, separated by a single space. The strings are: read sequence, read quality, insertion quality, deletion quality, and gcp quality.

Example record:
```
2 2
CTGTGTCCATGTCAGAGCAATGGCCCAAGTCTGGGCCTGGG 888888888888'88888888'8888888888888'8'888 IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIN IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIN +++++++++++++++++++++++++++++++++++++++++
ATGTCAGAGCAATGGCCCAAGTCTGGGTCTGGG <AFFFKKKKKKKKKKKKKKKKKKKKKKKKKKKK IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIN IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIN +++++++++++++++++++++++++++++++++
CTGTGTCCATGTCAGAGCAACGGCCCAAGTCTGGGTCTGGG
CTGTGTCCATGTCAGAGCAATGGCCCAAGTCTGGGTCTGGG
```

The output file repeats the first line of an input record, then outputs $R$ lines with $H$ scores each.

Example output:
```
2 2
-5.97153 -3.1966
-6.34043 -1.66333
```

## Benchmark commands
`./gpuPairHMM --peakBenchFloat` will measure the peak performance of different kernel configurations (i.e. read lengths)

`./gpuPairHMM --inputfile filename --filebenchmark [--checkResults]` will benchmark different kernel approaches on the given input file. --checkResults will additionally recompute all alignments on the cpu and compare them to the gpu results.
Our benchmark datasets are publicly available at: [https://zenodo.org/records/13928573](https://zenodo.org/records/13928573)

