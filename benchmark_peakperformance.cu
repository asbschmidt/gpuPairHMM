//nvcc -O3 -g -std=c++17 -arch=sm_80 -lineinfo --expt-relaxed-constexpr -rdc=true --extended-lambda -Xcompiler="-fopenmp" playground.cu -o playground

#include "utility_kernels.cuh"
#include "pairhmm_kernels.cuh"

#include <sstream>
#include <limits>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <optional>
#include <numeric>
#include <random>
#include <chrono>
#include <thread>

using namespace std::chrono_literals;


#include <algorithm>
#include <iterator>
#include <cuda_fp16.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>
#include <thrust/functional.h>
#include <thrust/device_malloc_allocator.h>

#include <cub/cub.cuh>

#include <cuda/functional>

#include "cuda_helpers.cuh"
#include "timers.cuh"

#include <omp.h>


#ifndef SDIV
#define SDIV(x,y)(((x)+(y)-1)/(y))
#endif




#define TIMERSTART_CUDA(label)                                                  \
    cudaEvent_t start##label, stop##label;                                 \
    float time##label;                                                     \
    cudaEventCreate(&start##label);                                        \
    cudaEventCreate(&stop##label);                                         \
    cudaEventRecord(start##label, 0);

#define TIMERSTOP_CUDA(label)                                                   \
            cudaEventRecord(stop##label, 0);                                   \
            cudaEventSynchronize(stop##label);                                 \
            cudaEventElapsedTime(&time##label, start##label, stop##label);     \
            std::cout << "TIMING: " << time##label << " ms " << (dp_cells)/(time##label*1e6) << " GCUPS (" << #label << ")" << std::endl;

#define TIMERSTART_CUDA_STREAM(label, stream)                                                  \
    cudaEvent_t start##label, stop##label;                                 \
    float time##label;                                                     \
    cudaEventCreate(&start##label);                                        \
    cudaEventCreate(&stop##label);                                         \
    cudaEventRecord(start##label, stream);

#define TIMERSTOP_CUDA_STREAM(label, stream)                                                   \
            cudaEventRecord(stop##label, stream);                                   \
            cudaEventSynchronize(stop##label);                                 \
            cudaEventElapsedTime(&time##label, start##label, stop##label);     \
            std::cout << "TIMING: " << time##label << " ms " << (dp_cells)/(time##label*1e6) << " GCUPS (" << #label << ")" << std::endl;


std::vector<float> generate_ph2pr ( int max_ph2pr=128){

    std::vector<float> ph2pr;
    float tmp = 0;
    ph2pr.reserve(max_ph2pr);

    for (int i=0; i < max_ph2pr; ++i){
        tmp = pow( 10.0, (-i) / 10.0);
        ph2pr.push_back(tmp);
        //std::cout << "ph2pr " << tmp << "\n";
    }

    return ph2pr;


}




template<int group_size, int numRegs>
void runPeakBenchFloatImpl_singletile(int sequencelength, int timingiterations){
    const int readLength = sequencelength;
    const int hapLength = sequencelength;
    const int paddedReadLength = SDIV(readLength, 4) * 4;
    const int paddedHapLength = SDIV(hapLength, 4) * 4;
    const int numReadsInSequenceGroup = 32;
    const int numHapsInSequenceGroup = 32;

    const size_t readBytes = 64 * 1024 * 1024;
    const size_t hapBytes = 64 * 1024 * 1024;

    if(readLength > group_size * numRegs){
        std::cout << "read length " << readLength << " > " << group_size << " * " << numRegs << ". Skipping\n";
        return;
    }

    
    std::vector<char> readData(readBytes);
    std::vector<char> hapData(hapBytes);
    const char* letters = "ACGT";
    
    #pragma omp parallel
    {
        std::mt19937 gen(42 + omp_get_thread_num());
        std::uniform_int_distribution<> dist(0,3);
        #pragma omp for
        for(size_t i = 0; i < readBytes; i++){
            readData[i] = letters[dist(gen)];
        }
    }
    hapData = readData;
    
    const int numReadsInReadData = readBytes / paddedReadLength;
    const int numHapsInHapData = hapBytes / paddedHapLength;
    const int maxNumSequenceGroupsInGeneratedData = std::min(numReadsInReadData / numReadsInSequenceGroup, numHapsInHapData / numHapsInSequenceGroup);
    const int numSequenceGroups = maxNumSequenceGroupsInGeneratedData;
    const int totalNumReads = maxNumSequenceGroupsInGeneratedData * numReadsInSequenceGroup;
    const int totalNumHaps = maxNumSequenceGroupsInGeneratedData * numHapsInSequenceGroup;
    size_t totalNumAlignments_sz = size_t(numReadsInSequenceGroup) * size_t(numHapsInSequenceGroup) * size_t(numSequenceGroups);
    if(totalNumAlignments_sz > size_t(std::numeric_limits<int>::max())){
        std::cout << "Warning: number of alignments (" << totalNumAlignments_sz << ") does not fit in int. gcups may be wrong.\n";

        totalNumAlignments_sz = std::min(size_t(100'000'000), totalNumAlignments_sz);
    }
    const int totalNumAlignments = totalNumAlignments_sz;
    const size_t totalDPCells = size_t(readLength) * size_t(hapLength) * size_t(numReadsInSequenceGroup) * size_t(numHapsInSequenceGroup) * size_t(numSequenceGroups);

    // std::cout << "numReadsInReadData " << numReadsInReadData
    //     << ", numHapsInHapData " << numHapsInHapData
    //     << ", numSequenceGroups " << numSequenceGroups
    //     << ", totalNumReads " << totalNumReads
    //     << ", totalNumHaps " << totalNumHaps
    //     << ", totalNumAlignments " << totalNumAlignments
    //     << "\n";
    
    std::vector<int> readLengths(totalNumReads, readLength);
    std::vector<int> hapLengths(totalNumHaps, hapLength);
    std::vector<char> baseQuals(readBytes, 'I');
    std::vector<char> insQuals(readBytes, 'I');
    std::vector<char> delQuals(readBytes, 'I');

    std::vector<int> readBeginOffsets(totalNumReads);
    for(int i = 0; i < totalNumReads; i++){
        readBeginOffsets[i] = i * paddedReadLength;
    }
    std::vector<int> hapBeginOffsets(totalNumHaps);
    for(int i = 0; i < totalNumHaps; i++){
        hapBeginOffsets[i] = i * paddedHapLength;
    }


    std::vector<int> numReadsPerSequenceGroup(numSequenceGroups, numReadsInSequenceGroup);
    std::vector<int> numHapsPerSequenceGroup(numSequenceGroups, numHapsInSequenceGroup);
    std::vector<int> numAlignmentsPerSequenceGroup(numSequenceGroups, numReadsInSequenceGroup * numHapsInSequenceGroup);
    std::vector<int> numReadsPerSequenceGroupPrefixSum(numSequenceGroups);
    std::vector<int> numHapsPerSequenceGroupPrefixSum(numSequenceGroups);
    std::vector<int> numAlignmentsPerSequenceGroupInclusivePrefixSum(numSequenceGroups);
    std::exclusive_scan(numReadsPerSequenceGroup.begin(), numReadsPerSequenceGroup.end(), numReadsPerSequenceGroupPrefixSum.begin(), int(0));
    std::exclusive_scan(numHapsPerSequenceGroup.begin(), numHapsPerSequenceGroup.end(), numHapsPerSequenceGroupPrefixSum.begin(), int(0));
    std::inclusive_scan(numAlignmentsPerSequenceGroup.begin(), numAlignmentsPerSequenceGroup.end(), numAlignmentsPerSequenceGroupInclusivePrefixSum.begin());


    thrust::device_vector<uint8_t> d_readData = readData;
    thrust::device_vector<uint8_t> d_hapData = hapData;
    thrust::device_vector<int> d_readLengths = readLengths;
    thrust::device_vector<int> d_hapLengths = hapLengths;
    thrust::device_vector<uint8_t> d_baseQuals = baseQuals;
    thrust::device_vector<uint8_t> d_insQuals = insQuals;
    thrust::device_vector<uint8_t> d_delQuals = delQuals;
    thrust::device_vector<int> d_readBeginOffsets = readBeginOffsets;
    thrust::device_vector<int> d_hapBeginOffsets = hapBeginOffsets;
    thrust::device_vector<int> d_numReadsPerSequenceGroup = numReadsPerSequenceGroup;
    thrust::device_vector<int> d_numHapsPerSequenceGroup = numHapsPerSequenceGroup;
    thrust::device_vector<int> d_numAlignmentsPerSequenceGroup = numAlignmentsPerSequenceGroup;
    thrust::device_vector<int> d_numReadsPerSequenceGroupPrefixSum = numReadsPerSequenceGroupPrefixSum;
    thrust::device_vector<int> d_numHapsPerSequenceGroupPrefixSum = numHapsPerSequenceGroupPrefixSum;
    thrust::device_vector<int> d_numAlignmentsPerSequenceGroupInclusivePrefixSum = numAlignmentsPerSequenceGroupInclusivePrefixSum;

    thrust::device_vector<int> d_numIndicesPerSequenceGroup = d_numReadsPerSequenceGroup;
    thrust::device_vector<int> d_indicesPerSequenceGroup(totalNumReads);
    thrust::transform(
        thrust::make_counting_iterator(0), 
        thrust::make_counting_iterator(totalNumReads),
        d_indicesPerSequenceGroup.begin(),
        cuda::proclaim_return_type<int>([numReadsInSequenceGroup] __device__ (int i){ return i % numReadsInSequenceGroup; })
    );

    thrust::device_vector<int> d_resultOffsetsPerSequenceGroup(totalNumAlignments);
    thrust::transform(
        d_numReadsPerSequenceGroup.begin(),
        d_numReadsPerSequenceGroup.begin() + numSequenceGroups,
        d_numHapsPerSequenceGroup.begin(),
        d_resultOffsetsPerSequenceGroup.begin(),
        thrust::multiplies<int>{}
    );
    thrust::exclusive_scan(
        d_resultOffsetsPerSequenceGroup.begin(),
        d_resultOffsetsPerSequenceGroup.begin() + numSequenceGroups,
        d_resultOffsetsPerSequenceGroup.begin()
    );

    thrust::device_vector<float> d_results(totalNumAlignments,-42);

    convert_DNA<<<SDIV(d_readData.size(), 512), 512>>>(d_readData.data().get(), d_readData.size());
    convert_DNA<<<SDIV(d_hapData.size(), 512), 512>>>(d_hapData.data().get(), d_hapData.size());

    cudaStream_t stream = cudaStreamLegacy;

    std::vector<double> gcupsVec;
    for(int i = 0; i < timingiterations; i++){
        std::string name = "PairHMM_float_kernel " + std::to_string(group_size) + " " + std::to_string(numRegs);
        constexpr int groupsPerBlock = 32 / group_size;
        const int numBlocks = SDIV(totalNumAlignments, groupsPerBlock);
        helpers::GpuTimer timer(stream, name);
        PairHMM_float_kernel<group_size,numRegs><<<numBlocks, 32, 0, stream>>>(
            d_results.data().get(), 
            d_readData.data().get(), 
            d_hapData.data().get(), 
            d_baseQuals.data().get(), 
            d_insQuals.data().get(),  
            d_delQuals.data().get(), 
            d_readBeginOffsets.data().get(), 
            d_hapBeginOffsets.data().get(), 
            d_readLengths.data().get(), 
            d_hapLengths.data().get(), 
            d_numHapsPerSequenceGroup.data().get(), 
            d_numHapsPerSequenceGroupPrefixSum.data().get(),
            d_indicesPerSequenceGroup.data().get(),
            d_numReadsPerSequenceGroupPrefixSum.data().get(), 
            numSequenceGroups, 
            d_resultOffsetsPerSequenceGroup.data().get(), 
            d_numAlignmentsPerSequenceGroup.data().get(), 
            d_numAlignmentsPerSequenceGroupInclusivePrefixSum.data().get(), 
            totalNumAlignments
        );
        CUERR;
        timer.stop();
        double elapsed = timer.elapsed();
        double gcups = totalDPCells / 1000. / 1000. / 1000.;
        gcups = gcups / (elapsed / 1000);
        gcupsVec.push_back(gcups);
        //timer.printGCUPS(totalDPCells);
    }
    CUERR;
    std::sort(gcupsVec.begin(), gcupsVec.end());
    //erase slowest and fastest run
    if(gcupsVec.size() > 2){
        gcupsVec.erase(gcupsVec.begin());
        gcupsVec.erase(gcupsVec.begin() + gcupsVec.size()-1);
    }
    double avggcups = std::reduce(gcupsVec.begin(), gcupsVec.end()) / gcupsVec.size();
    double mingcups = 0;
    double maxgcups = 0;
    if(gcupsVec.size() > 0){
        mingcups = *std::min_element(gcupsVec.begin(), gcupsVec.end());
        maxgcups = *std::max_element(gcupsVec.begin(), gcupsVec.end());
    }
    std::cout << group_size << "," << numRegs << "," << sequencelength << "," << mingcups << "," << avggcups << "," << maxgcups << "\n";
}




template<int group_size, int numRegs>
void runPeakBenchFloatImpl_singletile_withoutimprovements(int sequencelength, int timingiterations){
    const int readLength = sequencelength;
    const int hapLength = sequencelength;
    const int paddedReadLength = SDIV(readLength, 4) * 4;
    const int paddedHapLength = SDIV(hapLength, 4) * 4;
    const int numReadsInSequenceGroup = 32;
    const int numHapsInSequenceGroup = 32;

    const size_t readBytes = 64 * 1024 * 1024;
    const size_t hapBytes = 64 * 1024 * 1024;

    if(readLength > group_size * numRegs){
        std::cout << "read length " << readLength << " > " << group_size << " * " << numRegs << ". Skipping\n";
        return;
    }

    
    std::vector<char> readData(readBytes);
    std::vector<char> hapData(hapBytes);
    const char* letters = "ACGT";
    
    #pragma omp parallel
    {
        std::mt19937 gen(42 + omp_get_thread_num());
        std::uniform_int_distribution<> dist(0,3);
        #pragma omp for
        for(size_t i = 0; i < readBytes; i++){
            readData[i] = letters[dist(gen)];
        }
    }
    hapData = readData;
    
    const int numReadsInReadData = readBytes / paddedReadLength;
    const int numHapsInHapData = hapBytes / paddedHapLength;
    const int maxNumSequenceGroupsInGeneratedData = std::min(numReadsInReadData / numReadsInSequenceGroup, numHapsInHapData / numHapsInSequenceGroup);
    const int numSequenceGroups = maxNumSequenceGroupsInGeneratedData;
    const int totalNumReads = maxNumSequenceGroupsInGeneratedData * numReadsInSequenceGroup;
    const int totalNumHaps = maxNumSequenceGroupsInGeneratedData * numHapsInSequenceGroup;
    size_t totalNumAlignments_sz = size_t(numReadsInSequenceGroup) * size_t(numHapsInSequenceGroup) * size_t(numSequenceGroups);
    if(totalNumAlignments_sz > size_t(std::numeric_limits<int>::max())){
        std::cout << "Warning: number of alignments (" << totalNumAlignments_sz << ") does not fit in int. gcups may be wrong.\n";

        totalNumAlignments_sz = std::min(size_t(100'000'000), totalNumAlignments_sz);
    }
    const int totalNumAlignments = totalNumAlignments_sz;
    const size_t totalDPCells = size_t(readLength) * size_t(hapLength) * size_t(numReadsInSequenceGroup) * size_t(numHapsInSequenceGroup) * size_t(numSequenceGroups);

    // std::cout << "numReadsInReadData " << numReadsInReadData
    //     << ", numHapsInHapData " << numHapsInHapData
    //     << ", numSequenceGroups " << numSequenceGroups
    //     << ", totalNumReads " << totalNumReads
    //     << ", totalNumHaps " << totalNumHaps
    //     << ", totalNumAlignments " << totalNumAlignments
    //     << "\n";
    
    std::vector<int> readLengths(totalNumReads, readLength);
    std::vector<int> hapLengths(totalNumHaps, hapLength);
    std::vector<char> baseQuals(readBytes, 'I');
    std::vector<char> insQuals(readBytes, 'I');
    std::vector<char> delQuals(readBytes, 'I');

    std::vector<int> readBeginOffsets(totalNumReads);
    for(int i = 0; i < totalNumReads; i++){
        readBeginOffsets[i] = i * paddedReadLength;
    }
    std::vector<int> hapBeginOffsets(totalNumHaps);
    for(int i = 0; i < totalNumHaps; i++){
        hapBeginOffsets[i] = i * paddedHapLength;
    }


    std::vector<int> numReadsPerSequenceGroup(numSequenceGroups, numReadsInSequenceGroup);
    std::vector<int> numHapsPerSequenceGroup(numSequenceGroups, numHapsInSequenceGroup);
    std::vector<int> numAlignmentsPerSequenceGroup(numSequenceGroups, numReadsInSequenceGroup * numHapsInSequenceGroup);
    std::vector<int> numReadsPerSequenceGroupPrefixSum(numSequenceGroups);
    std::vector<int> numHapsPerSequenceGroupPrefixSum(numSequenceGroups);
    std::vector<int> numAlignmentsPerSequenceGroupInclusivePrefixSum(numSequenceGroups);
    std::exclusive_scan(numReadsPerSequenceGroup.begin(), numReadsPerSequenceGroup.end(), numReadsPerSequenceGroupPrefixSum.begin(), int(0));
    std::exclusive_scan(numHapsPerSequenceGroup.begin(), numHapsPerSequenceGroup.end(), numHapsPerSequenceGroupPrefixSum.begin(), int(0));
    std::inclusive_scan(numAlignmentsPerSequenceGroup.begin(), numAlignmentsPerSequenceGroup.end(), numAlignmentsPerSequenceGroupInclusivePrefixSum.begin());


    thrust::device_vector<uint8_t> d_readData = readData;
    thrust::device_vector<uint8_t> d_hapData = hapData;
    thrust::device_vector<int> d_readLengths = readLengths;
    thrust::device_vector<int> d_hapLengths = hapLengths;
    thrust::device_vector<uint8_t> d_baseQuals = baseQuals;
    thrust::device_vector<uint8_t> d_insQuals = insQuals;
    thrust::device_vector<uint8_t> d_delQuals = delQuals;
    thrust::device_vector<int> d_readBeginOffsets = readBeginOffsets;
    thrust::device_vector<int> d_hapBeginOffsets = hapBeginOffsets;
    thrust::device_vector<int> d_numReadsPerSequenceGroup = numReadsPerSequenceGroup;
    thrust::device_vector<int> d_numHapsPerSequenceGroup = numHapsPerSequenceGroup;
    thrust::device_vector<int> d_numAlignmentsPerSequenceGroup = numAlignmentsPerSequenceGroup;
    thrust::device_vector<int> d_numReadsPerSequenceGroupPrefixSum = numReadsPerSequenceGroupPrefixSum;
    thrust::device_vector<int> d_numHapsPerSequenceGroupPrefixSum = numHapsPerSequenceGroupPrefixSum;
    thrust::device_vector<int> d_numAlignmentsPerSequenceGroupInclusivePrefixSum = numAlignmentsPerSequenceGroupInclusivePrefixSum;

    thrust::device_vector<int> d_numIndicesPerSequenceGroup = d_numReadsPerSequenceGroup;
    thrust::device_vector<int> d_indicesPerSequenceGroup(totalNumReads);
    thrust::transform(
        thrust::make_counting_iterator(0), 
        thrust::make_counting_iterator(totalNumReads),
        d_indicesPerSequenceGroup.begin(),
        cuda::proclaim_return_type<int>([numReadsInSequenceGroup] __device__ (int i){ return i % numReadsInSequenceGroup; })
    );

    thrust::device_vector<int> d_resultOffsetsPerSequenceGroup(totalNumAlignments);
    thrust::transform(
        d_numReadsPerSequenceGroup.begin(),
        d_numReadsPerSequenceGroup.begin() + numSequenceGroups,
        d_numHapsPerSequenceGroup.begin(),
        d_resultOffsetsPerSequenceGroup.begin(),
        thrust::multiplies<int>{}
    );
    thrust::exclusive_scan(
        d_resultOffsetsPerSequenceGroup.begin(),
        d_resultOffsetsPerSequenceGroup.begin() + numSequenceGroups,
        d_resultOffsetsPerSequenceGroup.begin()
    );

    thrust::device_vector<float> d_results(totalNumAlignments,-42);

    convert_DNA<<<SDIV(d_readData.size(), 512), 512>>>(d_readData.data().get(), d_readData.size());
    convert_DNA<<<SDIV(d_hapData.size(), 512), 512>>>(d_hapData.data().get(), d_hapData.size());

    cudaStream_t stream = cudaStreamLegacy;

    std::vector<double> gcupsVec;
    for(int i = 0; i < timingiterations; i++){
        std::string name = "PairHMM_align_partition_float_allowMultipleSequenceGroupsPerWarp_coalesced_smem_noimprovedResultComputation " + std::to_string(group_size) + " " + std::to_string(numRegs);
        constexpr int groupsPerBlock = 32 / group_size;
        const int numBlocks = SDIV(totalNumAlignments, groupsPerBlock);
        helpers::GpuTimer timer(stream, name);
        PairHMM_align_partition_float_allowMultipleSequenceGroupsPerWarp_coalesced_smem_noimprovedResultComputation<group_size,numRegs><<<numBlocks, 32, 0, stream>>>(
            d_results.data().get(), 
            d_readData.data().get(), 
            d_hapData.data().get(), 
            d_baseQuals.data().get(), 
            d_insQuals.data().get(),  
            d_delQuals.data().get(), 
            d_readBeginOffsets.data().get(), 
            d_hapBeginOffsets.data().get(), 
            d_readLengths.data().get(), 
            d_hapLengths.data().get(), 
            d_numHapsPerSequenceGroup.data().get(), 
            d_numHapsPerSequenceGroupPrefixSum.data().get(),
            d_indicesPerSequenceGroup.data().get(),
            d_numReadsPerSequenceGroupPrefixSum.data().get(), 
            numSequenceGroups, 
            d_resultOffsetsPerSequenceGroup.data().get(), 
            d_numAlignmentsPerSequenceGroup.data().get(), 
            d_numAlignmentsPerSequenceGroupInclusivePrefixSum.data().get(), 
            totalNumAlignments
        );
        CUERR;
        timer.stop();
        double elapsed = timer.elapsed();
        double gcups = totalDPCells / 1000. / 1000. / 1000.;
        gcups = gcups / (elapsed / 1000);
        gcupsVec.push_back(gcups);
        //timer.printGCUPS(totalDPCells);
    }
    CUERR;
    std::sort(gcupsVec.begin(), gcupsVec.end());
    //erase slowest and fastest run
    if(gcupsVec.size() > 2){
        gcupsVec.erase(gcupsVec.begin());
        gcupsVec.erase(gcupsVec.begin() + gcupsVec.size()-1);
    }
    double avggcups = std::reduce(gcupsVec.begin(), gcupsVec.end()) / gcupsVec.size();
    double mingcups = 0;
    double maxgcups = 0;
    if(gcupsVec.size() > 0){
        mingcups = *std::min_element(gcupsVec.begin(), gcupsVec.end());
        maxgcups = *std::max_element(gcupsVec.begin(), gcupsVec.end());
    }
    std::cout << group_size << "," << numRegs << "," << sequencelength << "," << mingcups << "," << avggcups << "," << maxgcups << "\n";
}



template<int group_size, int numRegs>
void runPeakBenchFloatImpl_multitile(int readLength, int hapLength, int timingiterations){
    const int paddedReadLength = SDIV(readLength, 4) * 4;
    const int paddedHapLength = SDIV(hapLength, 4) * 4;
    const int numReadsInSequenceGroup = 32;
    const int numHapsInSequenceGroup = 32;

    const size_t readBytes = 64 * 1024 * 1024;
    const size_t hapBytes = 64 * 1024 * 1024;
   
    std::vector<char> readData(readBytes);
    std::vector<char> hapData(hapBytes);
    const char* letters = "ACGT";
    
    #pragma omp parallel
    {
        std::mt19937 gen(42 + omp_get_thread_num());
        std::uniform_int_distribution<> dist(0,3);
        #pragma omp for
        for(size_t i = 0; i < readBytes; i++){
            readData[i] = letters[dist(gen)];
        }
    }
    
    hapData = readData;
    
    const int numReadsInReadData = readBytes / paddedReadLength;
    const int numHapsInHapData = hapBytes / paddedHapLength;
    const int maxNumSequenceGroupsInGeneratedData = std::min(numReadsInReadData / numReadsInSequenceGroup, numHapsInHapData / numHapsInSequenceGroup);
    const int numSequenceGroups = maxNumSequenceGroupsInGeneratedData;
    const int totalNumReads = maxNumSequenceGroupsInGeneratedData * numReadsInSequenceGroup;
    const int totalNumHaps = maxNumSequenceGroupsInGeneratedData * numHapsInSequenceGroup;
    const int totalNumAlignments = numReadsInSequenceGroup * numHapsInSequenceGroup * numSequenceGroups;
    const size_t totalDPCells = size_t(readLength) * size_t(hapLength) * size_t(numReadsInSequenceGroup) * size_t(numHapsInSequenceGroup) * size_t(numSequenceGroups);

    // std::cout << "numReadsInReadData " << numReadsInReadData
    //     << ", numHapsInHapData " << numHapsInHapData
    //     << ", numSequenceGroups " << numSequenceGroups
    //     << ", totalNumReads " << totalNumReads
    //     << ", totalNumHaps " << totalNumHaps
    //     << ", totalNumAlignments " << totalNumAlignments
    //     << "\n";
    
    std::vector<int> readLengths(totalNumReads, readLength);
    std::vector<int> hapLengths(totalNumHaps, hapLength);
    std::vector<char> baseQuals(readBytes, 'I');
    std::vector<char> insQuals(readBytes, 'I');
    std::vector<char> delQuals(readBytes, 'I');

    std::vector<int> readBeginOffsets(totalNumReads);
    for(int i = 0; i < totalNumReads; i++){
        readBeginOffsets[i] = i * paddedReadLength;
    }
    std::vector<int> hapBeginOffsets(totalNumHaps);
    for(int i = 0; i < totalNumHaps; i++){
        hapBeginOffsets[i] = i * paddedHapLength;
    }


    std::vector<int> numReadsPerSequenceGroup(numSequenceGroups, numReadsInSequenceGroup);
    std::vector<int> numHapsPerSequenceGroup(numSequenceGroups, numHapsInSequenceGroup);
    std::vector<int> numAlignmentsPerSequenceGroup(numSequenceGroups, numReadsInSequenceGroup * numHapsInSequenceGroup);
    std::vector<int> numReadsPerSequenceGroupPrefixSum(numSequenceGroups);
    std::vector<int> numHapsPerSequenceGroupPrefixSum(numSequenceGroups);
    std::vector<int> numAlignmentsPerSequenceGroupInclusivePrefixSum(numSequenceGroups);
    std::exclusive_scan(numReadsPerSequenceGroup.begin(), numReadsPerSequenceGroup.end(), numReadsPerSequenceGroupPrefixSum.begin(), int(0));
    std::exclusive_scan(numHapsPerSequenceGroup.begin(), numHapsPerSequenceGroup.end(), numHapsPerSequenceGroupPrefixSum.begin(), int(0));
    std::inclusive_scan(numAlignmentsPerSequenceGroup.begin(), numAlignmentsPerSequenceGroup.end(), numAlignmentsPerSequenceGroupInclusivePrefixSum.begin());


    thrust::device_vector<uint8_t> d_readData = readData;
    thrust::device_vector<uint8_t> d_hapData = hapData;
    thrust::device_vector<int> d_readLengths = readLengths;
    thrust::device_vector<int> d_hapLengths = hapLengths;
    thrust::device_vector<uint8_t> d_baseQuals = baseQuals;
    thrust::device_vector<uint8_t> d_insQuals = insQuals;
    thrust::device_vector<uint8_t> d_delQuals = delQuals;
    thrust::device_vector<int> d_readBeginOffsets = readBeginOffsets;
    thrust::device_vector<int> d_hapBeginOffsets = hapBeginOffsets;
    thrust::device_vector<int> d_numReadsPerSequenceGroup = numReadsPerSequenceGroup;
    thrust::device_vector<int> d_numHapsPerSequenceGroup = numHapsPerSequenceGroup;
    thrust::device_vector<int> d_numAlignmentsPerSequenceGroup = numAlignmentsPerSequenceGroup;
    thrust::device_vector<int> d_numReadsPerSequenceGroupPrefixSum = numReadsPerSequenceGroupPrefixSum;
    thrust::device_vector<int> d_numHapsPerSequenceGroupPrefixSum = numHapsPerSequenceGroupPrefixSum;
    thrust::device_vector<int> d_numAlignmentsPerSequenceGroupInclusivePrefixSum = numAlignmentsPerSequenceGroupInclusivePrefixSum;

    thrust::device_vector<int> d_numIndicesPerSequenceGroup = d_numReadsPerSequenceGroup;
    thrust::device_vector<int> d_indicesPerSequenceGroup(totalNumReads);
    thrust::transform(
        thrust::make_counting_iterator(0), 
        thrust::make_counting_iterator(totalNumReads),
        d_indicesPerSequenceGroup.begin(),
        cuda::proclaim_return_type<int>([numReadsInSequenceGroup] __device__ (int i){ return i % numReadsInSequenceGroup; })
    );

    thrust::device_vector<int> d_resultOffsetsPerSequenceGroup(totalNumAlignments);
    thrust::transform(
        d_numReadsPerSequenceGroup.begin(),
        d_numReadsPerSequenceGroup.begin() + numSequenceGroups,
        d_numHapsPerSequenceGroup.begin(),
        d_resultOffsetsPerSequenceGroup.begin(),
        thrust::multiplies<int>{}
    );
    thrust::exclusive_scan(
        d_resultOffsetsPerSequenceGroup.begin(),
        d_resultOffsetsPerSequenceGroup.begin() + numSequenceGroups,
        d_resultOffsetsPerSequenceGroup.begin()
    );

    thrust::device_vector<float> d_results(totalNumAlignments,-42);

    convert_DNA<<<SDIV(d_readData.size(), 512), 512>>>(d_readData.data().get(), d_readData.size());
    convert_DNA<<<SDIV(d_hapData.size(), 512), 512>>>(d_hapData.data().get(), d_hapData.size());

    constexpr size_t GB1 = 1u << 30;
    const size_t tempStorageBytes = 2 * GB1;
    thrust::device_vector<char> d_temp(tempStorageBytes);

    cudaStream_t stream = cudaStreamLegacy;

    std::vector<double> gcupsVec;
    for(int i = 0; i < timingiterations; i++){
        std::string name = "PairHMM_float_multitile_kernel " + std::to_string(group_size) + " " + std::to_string(numRegs);
        helpers::GpuTimer timer(stream, name);
        call_PairHMM_float_multitile_kernel<group_size,numRegs>(
            d_temp.data().get(),
            d_temp.size(),
            hapLength,
            d_results.data().get(), 
            d_readData.data().get(), 
            d_hapData.data().get(), 
            d_baseQuals.data().get(), 
            d_insQuals.data().get(),  
            d_delQuals.data().get(), 
            d_readBeginOffsets.data().get(), 
            d_hapBeginOffsets.data().get(), 
            d_readLengths.data().get(), 
            d_hapLengths.data().get(), 
            d_numHapsPerSequenceGroup.data().get(), 
            d_numHapsPerSequenceGroupPrefixSum.data().get(),
            d_indicesPerSequenceGroup.data().get(), 
            d_numReadsPerSequenceGroupPrefixSum.data().get(), 
            numSequenceGroups, 
            d_resultOffsetsPerSequenceGroup.data().get(), 
            d_numAlignmentsPerSequenceGroup.data().get(), 
            d_numAlignmentsPerSequenceGroupInclusivePrefixSum.data().get(), 
            totalNumAlignments,
            stream
        );
        CUERR;
        timer.stop();
        double elapsed = timer.elapsed();
        double gcups = totalDPCells / 1000. / 1000. / 1000.;
        gcups = gcups / (elapsed / 1000);
        gcupsVec.push_back(gcups);
        //timer.printGCUPS(totalDPCells);
    }
    CUERR;
    std::sort(gcupsVec.begin(), gcupsVec.end());
    //erase slowest and fastest run
    if(gcupsVec.size() > 2){
        gcupsVec.erase(gcupsVec.begin());
        gcupsVec.erase(gcupsVec.begin() + gcupsVec.size()-1);
    }
    double avggcups = std::reduce(gcupsVec.begin(), gcupsVec.end()) / gcupsVec.size();
    double mingcups = 0;
    double maxgcups = 0;
    if(gcupsVec.size() > 0){
        mingcups = *std::min_element(gcupsVec.begin(), gcupsVec.end());
        maxgcups = *std::max_element(gcupsVec.begin(), gcupsVec.end());
    }
    std::cout << group_size << "," << numRegs << "," << hapLength << "," << readLength << "," << mingcups << "," << avggcups << "," << maxgcups << "\n";
}




void runPeakBenchFloat(int timingiterations){
    // std::cout << "runPeakBenchFloat\n";

    #define RUN_SINGLE_TILE(group_size, numRegs){ \
        const int sequencelength = group_size * numRegs; \
        runPeakBenchFloatImpl_singletile<group_size, numRegs>(sequencelength, timingiterations); \
        std::this_thread::sleep_for(2s); \
    }

    #define RUN_SINGLE_TILE_withoutimprovements(group_size, numRegs){ \
        const int sequencelength = group_size * numRegs; \
        runPeakBenchFloatImpl_singletile_withoutimprovements<group_size, numRegs>(sequencelength, timingiterations); \
        std::this_thread::sleep_for(2s); \
    }

    #define RUN_MULTI_TILE_RECTANGULAR(group_size, numRegs){ \
        const int readlength = 10 * group_size * numRegs; \
        const int haplength = 1 * group_size * numRegs; \
        runPeakBenchFloatImpl_multitile<group_size, numRegs>(readlength, haplength, timingiterations); \
        std::this_thread::sleep_for(2s); \
    }

    #define RUN_MULTI_TILE(group_size, numRegs, sequencelength){ \
        const int readlength = sequencelength; \
        const int haplength = sequencelength; \
        runPeakBenchFloatImpl_multitile<group_size, numRegs>(readlength, haplength, timingiterations); \
        std::this_thread::sleep_for(2s); \
    }

    // std::cout << "RUN_SINGLE_TILE_withoutimprovements\n";
    // RUN_SINGLE_TILE_withoutimprovements(4,4);
    // RUN_SINGLE_TILE_withoutimprovements(4,8);
    // RUN_SINGLE_TILE_withoutimprovements(4,12);
    // RUN_SINGLE_TILE_withoutimprovements(4,16);
    // RUN_SINGLE_TILE_withoutimprovements(4,20);
    // RUN_SINGLE_TILE_withoutimprovements(4,24);
    // RUN_SINGLE_TILE_withoutimprovements(4,28);
    // RUN_SINGLE_TILE_withoutimprovements(4,32);

    // RUN_SINGLE_TILE_withoutimprovements(8,4);
    // RUN_SINGLE_TILE_withoutimprovements(8,8);
    // RUN_SINGLE_TILE_withoutimprovements(8,12);
    // RUN_SINGLE_TILE_withoutimprovements(8,16);
    // RUN_SINGLE_TILE_withoutimprovements(8,20);
    // RUN_SINGLE_TILE_withoutimprovements(8,24);
    // RUN_SINGLE_TILE_withoutimprovements(8,28);
    // RUN_SINGLE_TILE_withoutimprovements(8,32);

    // RUN_SINGLE_TILE_withoutimprovements(16,4);
    // RUN_SINGLE_TILE_withoutimprovements(16,8);
    // RUN_SINGLE_TILE_withoutimprovements(16,12);
    // RUN_SINGLE_TILE_withoutimprovements(16,16);
    // RUN_SINGLE_TILE_withoutimprovements(16,20);
    // RUN_SINGLE_TILE_withoutimprovements(16,24);
    // RUN_SINGLE_TILE_withoutimprovements(16,28);
    // RUN_SINGLE_TILE_withoutimprovements(16,32);

    // RUN_SINGLE_TILE_withoutimprovements(32,4);
    // RUN_SINGLE_TILE_withoutimprovements(32,8);
    // RUN_SINGLE_TILE_withoutimprovements(32,12);
    // RUN_SINGLE_TILE_withoutimprovements(32,16);
    // RUN_SINGLE_TILE_withoutimprovements(32,20);
    // RUN_SINGLE_TILE_withoutimprovements(32,24);
    // RUN_SINGLE_TILE_withoutimprovements(32,28);
    // RUN_SINGLE_TILE_withoutimprovements(32,32);

    std::cout << "RUN_SINGLE_TILE\n";
    RUN_SINGLE_TILE(4,4);
    RUN_SINGLE_TILE(4,8);
    RUN_SINGLE_TILE(4,12);
    RUN_SINGLE_TILE(4,16);
    RUN_SINGLE_TILE(4,20);
    RUN_SINGLE_TILE(4,24);
    RUN_SINGLE_TILE(4,28);
    RUN_SINGLE_TILE(4,32);

    RUN_SINGLE_TILE(8,4);
    RUN_SINGLE_TILE(8,8);
    RUN_SINGLE_TILE(8,12);
    RUN_SINGLE_TILE(8,16);
    RUN_SINGLE_TILE(8,20);
    RUN_SINGLE_TILE(8,24);
    RUN_SINGLE_TILE(8,28);
    RUN_SINGLE_TILE(8,32);

    RUN_SINGLE_TILE(16,4);
    RUN_SINGLE_TILE(16,8);
    RUN_SINGLE_TILE(16,12);
    RUN_SINGLE_TILE(16,16);
    RUN_SINGLE_TILE(16,20);
    RUN_SINGLE_TILE(16,24);
    RUN_SINGLE_TILE(16,28);
    RUN_SINGLE_TILE(16,32);

    RUN_SINGLE_TILE(32,4);
    RUN_SINGLE_TILE(32,8);
    RUN_SINGLE_TILE(32,12);
    RUN_SINGLE_TILE(32,16);
    RUN_SINGLE_TILE(32,20);
    RUN_SINGLE_TILE(32,24);
    RUN_SINGLE_TILE(32,28);
    RUN_SINGLE_TILE(32,32);


    std::cout << "RUN_MULTI_TILE_RECTANGULAR\n";
    RUN_MULTI_TILE_RECTANGULAR(4,4);
    RUN_MULTI_TILE_RECTANGULAR(4,8);
    RUN_MULTI_TILE_RECTANGULAR(4,12);
    RUN_MULTI_TILE_RECTANGULAR(4,16);
    RUN_MULTI_TILE_RECTANGULAR(4,20);
    RUN_MULTI_TILE_RECTANGULAR(4,24);
    RUN_MULTI_TILE_RECTANGULAR(4,28);
    RUN_MULTI_TILE_RECTANGULAR(4,32);

    RUN_MULTI_TILE_RECTANGULAR(8,4);
    RUN_MULTI_TILE_RECTANGULAR(8,8);
    RUN_MULTI_TILE_RECTANGULAR(8,12);
    RUN_MULTI_TILE_RECTANGULAR(8,16);
    RUN_MULTI_TILE_RECTANGULAR(8,20);
    RUN_MULTI_TILE_RECTANGULAR(8,24);
    RUN_MULTI_TILE_RECTANGULAR(8,28);
    RUN_MULTI_TILE_RECTANGULAR(8,32);

    RUN_MULTI_TILE_RECTANGULAR(16,4);
    RUN_MULTI_TILE_RECTANGULAR(16,8);
    RUN_MULTI_TILE_RECTANGULAR(16,12);
    RUN_MULTI_TILE_RECTANGULAR(16,16);
    RUN_MULTI_TILE_RECTANGULAR(16,20);
    RUN_MULTI_TILE_RECTANGULAR(16,24);
    RUN_MULTI_TILE_RECTANGULAR(16,28);
    RUN_MULTI_TILE_RECTANGULAR(16,32);

    RUN_MULTI_TILE_RECTANGULAR(32,4);
    RUN_MULTI_TILE_RECTANGULAR(32,8);
    RUN_MULTI_TILE_RECTANGULAR(32,12);
    RUN_MULTI_TILE_RECTANGULAR(32,16);
    RUN_MULTI_TILE_RECTANGULAR(32,20);
    RUN_MULTI_TILE_RECTANGULAR(32,24);
    RUN_MULTI_TILE_RECTANGULAR(32,28);
    RUN_MULTI_TILE_RECTANGULAR(32,32);


    std::cout << "RUN_MULTI_TILE squared\n";
    RUN_MULTI_TILE(16,16, 1024)
    RUN_MULTI_TILE(16,16, 3072)
    RUN_MULTI_TILE(16,16, 4096)
    RUN_MULTI_TILE(16,16, 8192)
    RUN_MULTI_TILE(16,16, 16384)

    RUN_MULTI_TILE(16,32, 1024)
    RUN_MULTI_TILE(16,32, 2048)
    RUN_MULTI_TILE(16,32, 3072)
    RUN_MULTI_TILE(16,32, 4096)
    RUN_MULTI_TILE(16,32, 8192)
    RUN_MULTI_TILE(16,32, 16384)

    RUN_MULTI_TILE(32,16, 1024)
    RUN_MULTI_TILE(32,16, 2048)
    RUN_MULTI_TILE(32,16, 3072)
    RUN_MULTI_TILE(32,16, 4096)
    RUN_MULTI_TILE(32,16, 8192)
    RUN_MULTI_TILE(32,16, 16384)

    RUN_MULTI_TILE(32,32, 1024)
    RUN_MULTI_TILE(32,32, 2048)
    RUN_MULTI_TILE(32,32, 3072)
    RUN_MULTI_TILE(32,32, 4096)
    RUN_MULTI_TILE(32,32, 8192)
    RUN_MULTI_TILE(32,32, 16384)
}




int main(const int /*argc*/, char const * const /*argv*/[])
{

    constexpr int MAX_PH2PR_INDEX = 128;

    int timingiterations = 10;

    std::vector<float> ph2pr = generate_ph2pr(MAX_PH2PR_INDEX);
    const int deviceId = 0;
    cudaSetDevice(deviceId); CUERR;
    cudaMemcpyToSymbol(cPH2PR,ph2pr.data(),MAX_PH2PR_INDEX*sizeof(float));

    size_t releaseThreshold = UINT64_MAX;
    cudaMemPool_t memPool;
    cudaDeviceGetDefaultMemPool(&memPool, deviceId);
    cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &releaseThreshold); CUERR;



    runPeakBenchFloat(timingiterations);

    return 0;
}
