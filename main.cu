#include <sstream>
#include <limits>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>

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

#include <cub/cub.cuh>

#include "cuda_helpers.cuh"
#include "Context.h"

#include <omp.h>

using std::cout;
using std::copy;




#define SDIV(x,y)(((x)+(y)-1)/(y))


//#define MAX_PH2PR_INDEX 128

__constant__ float cPH2PR[128];

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



struct PartitionLimits{
    static constexpr int numPartitions(){
        return 10;
    }
    PartitionLimits() = default;
    int boundaries[10]{
        64,  // 48
        //80,
        96,
        128,
        160,
        192,
        256,
        320,
        384,
        512,
        576
    };
};
constexpr int numPartitions = PartitionLimits::numPartitions();





#include <thrust/device_malloc_allocator.h>

#include <iostream>

template<class T>
struct ThrustCudaMallocAsyncAllocator : thrust::device_malloc_allocator<T> {
    using value_type = T;
    using super_t = thrust::device_malloc_allocator<T>;

    using pointer = typename super_t::pointer;
    using size_type = typename super_t::size_type;
    using reference = typename super_t::reference;
    using const_reference = typename super_t::const_reference;

    cudaStream_t stream{};

    ThrustCudaMallocAsyncAllocator(cudaStream_t stream_)
        : stream(stream_){
        
    }

    pointer allocate(size_type n){
        T* ptr = nullptr;
        cudaError_t status = cudaMallocAsync(&ptr, n * sizeof(T), stream);
        if(status != cudaSuccess){
            cudaGetLastError(); //reset error state
            std::cerr << "ThrustCudaMallocAsyncAllocator cuda error when allocating " << (n * sizeof(T)) << " bytes: " << cudaGetErrorString(status) << "\n";
            throw std::bad_alloc();
        }
        return thrust::device_pointer_cast(ptr);
    }

    void deallocate(pointer ptr, size_type /*n*/){
        cudaError_t status = cudaFreeAsync(ptr.get(), stream);
        if(status != cudaSuccess){
            cudaGetLastError(); //reset error state
            throw std::bad_alloc();
        }
    }
};



__global__
void partitionIndicesKernel(
    int* numIndicesPerPartitionPerBatch,
    int* indicesPerPartitionPerBatch,
    const int* read_lengths,
    const int* numReadsPerBatchPrefixSum,
    int numBatches,
    int numReads
){
    const PartitionLimits partitionLimits;

    for(int batchId = blockIdx.x; batchId < numBatches; batchId += gridDim.x){
        const int offset = numReadsPerBatchPrefixSum[batchId];
        const int nextOffset = batchId < numBatches - 1 ? numReadsPerBatchPrefixSum[batchId+1] : numReads;
        const int numReadsInBatch = nextOffset - offset;
        const int* readLengthsOfBatch = read_lengths + offset;

        for(int r = threadIdx.x; r < numReadsInBatch; r += blockDim.x){
            const int length = readLengthsOfBatch[r];
            for(int p = 0; p < numPartitions; p++){
                if(length <= partitionLimits.boundaries[p]){
                    const int pos = atomicAdd(&numIndicesPerPartitionPerBatch[p * numBatches + batchId], 1);
                    indicesPerPartitionPerBatch[p * numReads + offset + pos] = r;
                    break;
                }
            }
        }
    }
}

__global__
void computeNumWarpsPerBatchKernel(
    int* numWarpsPerBatch,
    const int* numIndicesPerBatch,
    int numBatches,
    int alignmentsPerWarp
){
    for(int batchId = threadIdx.x + blockIdx.x * blockDim.x; batchId < numBatches; batchId += gridDim.x * blockDim.x){
        numWarpsPerBatch[batchId] = (numIndicesPerBatch[batchId] + alignmentsPerWarp - 1) / alignmentsPerWarp;
    }
}

__global__
void gatherNumWarpsPerPartitionFromInclPrefixSumKernel(
    int* numWarpsPerPartition,
    const int* numWarpsPerBatchInclusivePrefixSumPerPartition,
    int numBatches,
    int numPartitions
){
    for(int tid = threadIdx.x + blockIdx.x * blockDim.x; tid < numPartitions; tid += gridDim.x * blockDim.x){
        numWarpsPerPartition[tid] = numWarpsPerBatchInclusivePrefixSumPerPartition[tid * numBatches + (numBatches - 1)];
    }
}


__global__
void computeAlignmentsPerPartitionPerBatch(
    int* d_numAlignmentsPerPartitionPerBatch,
    const int* d_numIndicesPerPartitionPerBatch,
    const int* d_numHaplotypesPerBatch,
    int numPartitions, 
    int numBatches
){
    for(int partition = blockIdx.y; partition < numPartitions; partition += gridDim.y){
        for(int batch = threadIdx.x + blockIdx.x * blockDim.x; batch < numBatches; batch += gridDim.x * blockDim.x){
            d_numAlignmentsPerPartitionPerBatch[partition * numBatches + batch]
                = d_numIndicesPerPartitionPerBatch[partition * numBatches + batch] * d_numHaplotypesPerBatch[batch];
        }
    }
}

__global__
void processKernel(
    const int* numIndicesPerBatch,
    const int* indicesPerBatch,
    const int* numWarpsPerBatch,
    const int* numWarpsPerBatchInclusivePrefixSum,
    const int* numReadsPerBatchPrefixSum,
    int numBatches,
    int numGroupsPerWarp,
    int groupSize
){
    //constexpr int numGroupsPerWarp = 4;
    //constexpr int groupSize = 8;

    const int warpIdInGrid = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
    const int laneInWarp = threadIdx.x % 32;
    const int groupIdInWarp = laneInWarp / groupSize;
    const int laneInGroup = threadIdx.x % groupSize;

    const int batchId = thrust::distance(
        numWarpsPerBatchInclusivePrefixSum,
        thrust::upper_bound(thrust::seq,
            numWarpsPerBatchInclusivePrefixSum,
            numWarpsPerBatchInclusivePrefixSum + numBatches,
            warpIdInGrid
        )
    );
    if(laneInWarp == 0){
        printf("warp %d, batchId %d\n", warpIdInGrid, batchId);
    }
    if(batchId < numBatches){
        const int offset_read_batches = numReadsPerBatchPrefixSum[batchId];
        if(laneInWarp == 0){
            printf("warp %d, batchId %d, offset_read_batches %d\n", warpIdInGrid, batchId, offset_read_batches);
        }

        const int numWarpsForBatch = numWarpsPerBatch[batchId];
        const int warpIdInBatch = warpIdInGrid - (numWarpsPerBatchInclusivePrefixSum[batchId]-numWarpsForBatch);
        const int groupIdInBatch = numGroupsPerWarp * warpIdInBatch + groupIdInWarp;
        const int numGroupsForBatch = numIndicesPerBatch[batchId];
        if(groupIdInBatch < numGroupsForBatch){
            int readToProcessInBatch = indicesPerBatch[offset_read_batches + groupIdInBatch];

            if(laneInGroup == 0){
                printf("warp: %d. warpIdInBatch %d, groupId in warp: %d. groupId in batch: %d. readToProcessInBatch %d\n", warpIdInGrid, warpIdInBatch, groupIdInWarp, groupIdInBatch, readToProcessInBatch);
            }
        }else{
            if(laneInGroup == 0){
                printf("warp: %d. warpIdInBatch %d, groupId in warp: %d. groupId in batch: %d. unused\n", warpIdInGrid, warpIdInBatch, groupIdInWarp, groupIdInBatch);
            }
        }

    }else{
        //left-over warp
        if(laneInWarp == 0){
            printf("warp %d unused\n", warpIdInGrid);
        }
    }
}



template <int group_size, int numRegs> 
__global__
void PairHMM_align_partition_half_allowMultipleBatchesPerWarp(
    const uint8_t * read_chars,
    const uint8_t * hap_chars,
    const uint8_t * base_quals,
    const uint8_t * ins_quals,
    const uint8_t * del_quals,
    float * devAlignmentScores,
    const int* read_offsets,
    const int* hap_offsets,
    const int* read_length,
    const int* hap_length,
    const int *reads_in_batch,
    const int *haps_in_batch,
    const int *offset_hap_batches,

    const int* numIndicesPerBatch,
    const int* indicesPerBatch,
    const int* numReadsPerBatchPrefixSum,
    const int numBatches,
    const int* resultOffsetsPerBatch,

    const int* numAlignmentsPerBatch,
    const int* numAlignmentsPerBatchInclusivePrefixSum,
    const int numAlignments
) {

    alignas(8) __shared__ __half2 lambda_array[5][16*numRegs];

    float M[numRegs], I, D[numRegs];
    half2 alpha[numRegs/2], delta[numRegs/2], sigma[numRegs/2];
    float Results[numRegs];

    const int threadIdInGroup = threadIdx.x % group_size;
    const int groupIdInBlock = threadIdx.x / group_size;
    const int groupIdInGrid = (threadIdx.x + blockIdx.x * blockDim.x) / group_size;
    const unsigned int myGroupMask = __match_any_sync(0xFFFFFFFF, groupIdInGrid); //compute mask for all threads with same groupIdInGrid

    if(groupIdInGrid < numAlignments){

        const int batchIdByGroupId = thrust::distance(
            numAlignmentsPerBatchInclusivePrefixSum,
            thrust::upper_bound(thrust::seq,
                numAlignmentsPerBatchInclusivePrefixSum,
                numAlignmentsPerBatchInclusivePrefixSum + numBatches,
                groupIdInGrid
            )
        );
        const int batchId = min(batchIdByGroupId, numBatches-1);
        const int groupIdInBatch = groupIdInGrid - (batchId == 0 ? 0 : numAlignmentsPerBatchInclusivePrefixSum[batchId-1]);
        const int hapToProcessInBatch = groupIdInBatch % haps_in_batch[batchId];
        const int readIndexToProcessInBatch = groupIdInBatch / haps_in_batch[batchId];

        const int offset_read_batches = numReadsPerBatchPrefixSum[batchId];
        const int readToProcessInBatch = indicesPerBatch[offset_read_batches + readIndexToProcessInBatch];

        const int read_nr = readToProcessInBatch;
        const int global_read_id = read_nr + offset_read_batches;


        const int byteOffsetForRead = read_offsets[global_read_id];
        const int readLength = read_length[global_read_id];

        const int b_h_off = offset_hap_batches[batchId];
        const int bytesOffsetForHap = hap_offsets[hapToProcessInBatch+b_h_off];
        const char4* const HapsAsChar4 = reinterpret_cast<const char4*>(&hap_chars[bytesOffsetForHap]);
        const int haploLength = hap_length[hapToProcessInBatch+b_h_off];

        // if(groupIdInGrid < 10 && group_size == 8 && numRegs == 8){
        //     if(threadIdInGroup == 0){
        //         printf("group %d, myGroupMask %u, batchId %d, groupIdInBatch %d, hapToProcessInBatch %d, readIndexToProcessInBatch %d, readToProcessInBatch %d, readLength %d, haploLength %d, numAlignments %d\n",
        //             groupIdInGrid, myGroupMask, batchId, groupIdInBatch, hapToProcessInBatch, readIndexToProcessInBatch, readToProcessInBatch, readLength, haploLength, numAlignments);
        //     }
        // }

        const float eps = 0.1;
        const float beta = 0.9;
        float M_l, D_l, M_ul, D_ul, I_ul;
        float penalty_temp0, penalty_temp1, penalty_temp2, penalty_temp3;
        float init_D;

        const float constant = ::cuda::std::numeric_limits<float>::max() / 16;

        auto load_PSSM = [&]() {

            char4 temp0, temp1;
            const half one_half = 1.0;
            const half three = 3.0;
            const char4* QualsAsChar4 = reinterpret_cast<const char4*>(&base_quals[byteOffsetForRead]);
            const char4* ReadsAsChar4 = reinterpret_cast<const char4*>(&read_chars[byteOffsetForRead]);
            for (int i=threadIdInGroup; i<(readLength+3)/4; i+=group_size) {
                half2 temp_h2, temp_h3;
                temp0 = QualsAsChar4[i];
                temp1 = ReadsAsChar4[i];
                temp_h2.x = cPH2PR[uint8_t(temp0.x)];
                temp_h2.y = cPH2PR[uint8_t(temp0.y)];
                temp_h3.x = temp_h2.x/three;
                temp_h3.y = temp_h2.y/three;
                for (int j=0; j<5; j++) lambda_array[j][2*i+groupIdInBlock*(group_size*numRegs/2)] = temp_h3;
                if (temp1.x <= 4) lambda_array[temp1.x][2*i+groupIdInBlock*(group_size*numRegs/2)].x = one_half - temp_h2.x;
                if (temp1.y <= 4) lambda_array[temp1.y][2*i+groupIdInBlock*(group_size*numRegs/2)].y = one_half - temp_h2.y;
                temp_h2.x = cPH2PR[uint8_t(temp0.z)];
                temp_h2.y = cPH2PR[uint8_t(temp0.w)];
                temp_h3.x = temp_h2.x/three;
                temp_h3.y = temp_h2.y/three;
                for (int j=0; j<5; j++) lambda_array[j][2*i+1+groupIdInBlock*(group_size*numRegs/2)] = temp_h3;
                if (temp1.z <= 4) lambda_array[temp1.z][2*i+1+groupIdInBlock*(group_size*numRegs/2)].x = one_half - temp_h2.x;
                if (temp1.w <= 4) lambda_array[temp1.w][2*i+1+groupIdInBlock*(group_size*numRegs/2)].y = one_half - temp_h2.y;
            }

            __syncwarp(myGroupMask);

        };

        auto load_probabilities = [&]() {
            char4 temp0, temp1;
            const char4* InsQualsAsChar4 = reinterpret_cast<const char4*>(&ins_quals[byteOffsetForRead]);
            const char4* DelQualsAsChar4 = reinterpret_cast<const char4*>(&del_quals[byteOffsetForRead]);
            for (int i=0; i<numRegs/4; i++) {
                if (threadIdInGroup*numRegs/4+i < (readLength+3)/4) {

                    temp0 = InsQualsAsChar4[threadIdInGroup*numRegs/4+i];
                    temp1 = DelQualsAsChar4[threadIdInGroup*numRegs/4+i];

                    //delta[4*i] = cPH2PR[uint8_t(temp0.x)];
                    //delta[4*i+1] = cPH2PR[uint8_t(temp0.y)];
                    //delta[4*i+2] = cPH2PR[uint8_t(temp0.z)];
                    //delta[4*i+3] = cPH2PR[uint8_t(temp0.w)];
                    delta[2*i] = __floats2half2_rn(cPH2PR[uint8_t(temp0.x)],cPH2PR[uint8_t(temp0.y)]);
                    delta[2*i+1] = __floats2half2_rn(cPH2PR[uint8_t(temp0.z)],cPH2PR[uint8_t(temp0.w)]);

                    //sigma[4*i] = cPH2PR[uint8_t(temp1.x)];
                    //sigma[4*i+1] = cPH2PR[uint8_t(temp1.y)];
                    //sigma[4*i+2] = cPH2PR[uint8_t(temp1.z)];
                    //sigma[4*i+3] = cPH2PR[uint8_t(temp1.w)];
                    sigma[2*i] = __floats2half2_rn(cPH2PR[uint8_t(temp1.x)],cPH2PR[uint8_t(temp1.y)]);
                    sigma[2*i+1] = __floats2half2_rn(cPH2PR[uint8_t(temp1.z)],cPH2PR[uint8_t(temp1.w)]);

                    //alpha[4*i] = 1.0 - (delta[4*i] + sigma[4*i]);
                    //alpha[4*i+1] = 1.0 - (delta[4*i+1] + sigma[4*i+1]);
                    //alpha[4*i+2] = 1.0 - (delta[4*i+2] + sigma[4*i+2]);
                    //alpha[4*i+3] = 1.0 - (delta[4*i+3] + sigma[4*i+3]);
                    alpha[2*i] = __float2half2_rn(1.0) - __hadd2(delta[2*i], sigma[2*i]);
                    alpha[2*i+1] = __float2half2_rn(1.0) - __hadd2(delta[2*i+1], sigma[2*i+1]);
                }
            }

        };

        auto compute_probabilities = [&]() {
            char4 temp0, temp1;
            const char4* InsQualsAsChar4 = reinterpret_cast<const char4*>(&ins_quals[byteOffsetForRead]);
            const char4* DelQualsAsChar4 = reinterpret_cast<const char4*>(&del_quals[byteOffsetForRead]);

            auto computePH2PR = [](int i) -> float{
                // return pow(10.0,  (-i) / 10.0);
                // return __powf(10.0f,  (-i) / 10.0f);
                // return exp10((-i) / 10.0);
                return __exp10f((-i) / 10.0f);
            };

            for (int i=0; i<numRegs/4; i++) {
                if (threadIdInGroup*numRegs/4+i < (readLength+3)/4) {

                    // temp0 = InsQualsAsChar4[threadIdInGroup*numRegs/4+i];
                    // temp1 = DelQualsAsChar4[threadIdInGroup*numRegs/4+i];
                    temp0 = InsQualsAsChar4[0];
                    temp1 = DelQualsAsChar4[0];

                    delta[2*i] = __floats2half2_rn(computePH2PR(temp0.x),computePH2PR(temp0.y));
                    delta[2*i+1] = __floats2half2_rn(computePH2PR(temp0.z),computePH2PR(temp0.w));

                    sigma[2*i] = __floats2half2_rn(computePH2PR(temp1.x),computePH2PR(temp1.y));
                    sigma[2*i+1] = __floats2half2_rn(computePH2PR(temp1.z),computePH2PR(temp1.w));

                    alpha[2*i] = __float2half2_rn(1.0) - __hadd2(delta[2*i], sigma[2*i]);
                    alpha[2*i+1] = __float2half2_rn(1.0) - __hadd2(delta[2*i+1], sigma[2*i+1]);
                }
            }

        };

        auto init_penalties = [&]() {
            #pragma unroll
            for (int i=0; i<numRegs; i++) M[i] = D[i] = Results[i] = 0.0;
            M_l = M_ul = D_ul = I_ul = D_l = I = 0.0;
            if (!threadIdInGroup) D_l = D_ul = init_D;
        };


        char hap_letter;

        __half2 score2;
        __half2 *sbt_row;

        auto calc_DP_float = [&](){

            sbt_row = lambda_array[hap_letter];
            float2 foo = *((float2*)&sbt_row[threadIdx.x*numRegs/2]);
            memcpy(&score2, &foo.x, sizeof(__half2));
            penalty_temp0 = M[0];
            penalty_temp1 = D[0];
            M[0] = float(score2.x) * fmaf(alpha[0].x,M_ul,beta*(I_ul+D_ul));
            D[0] = fmaf(sigma[0].x,penalty_temp0,eps*D[0]);
            I = fmaf(delta[0].x,M_ul,eps*I_ul);
            Results[0] += M[0] + I;
            penalty_temp2 = M[1];
            penalty_temp3 = D[1];
            M[1] = float(score2.y) * fmaf(alpha[0].y,penalty_temp0,beta*(I+penalty_temp1));
            D[1] = fmaf(sigma[0].y,penalty_temp2,eps*D[1]);
            I = fmaf(delta[0].y,penalty_temp0,eps*I);
            Results[1] += M[1] + I;

            memcpy(&score2, &foo.y, sizeof(__half2));
            penalty_temp0 = M[2];
            penalty_temp1 = D[2];
            M[2] = float(score2.x) * fmaf(alpha[1].x,penalty_temp2,beta*(I+penalty_temp3));
            D[2] = fmaf(sigma[1].x,penalty_temp0,eps*D[2]);
            I = fmaf(delta[1].x,penalty_temp2,eps*I);
            Results[2] += M[2] + I;
            penalty_temp2 = M[3];
            penalty_temp3 = D[3];
            M[3] = float(score2.y) * fmaf(alpha[1].y,penalty_temp0,beta*(I+penalty_temp1));
            D[3] = fmaf(sigma[1].y,penalty_temp2,eps*D[3]);
            I = fmaf(delta[1].y,penalty_temp0,eps*I);
            Results[3] += M[3] + I;

            #pragma unroll
            for (int i=1; i<numRegs/4; i++) {
                float2 foo = *((float2*)&sbt_row[threadIdx.x*numRegs/2+2*i]);
                memcpy(&score2, &foo.x, sizeof(__half2));
                penalty_temp0 = M[4*i];
                penalty_temp1 = D[4*i];
                M[4*i] = float(score2.x) * fmaf(alpha[2*i].x,penalty_temp2,beta*(I+penalty_temp3));
                D[4*i] = fmaf(sigma[2*i].x,penalty_temp0,eps*D[4*i]);
                I = fmaf(delta[2*i].x,penalty_temp2,eps*I);
                Results[4*i] += M[4*i] + I;
                penalty_temp2 = M[4*i+1];
                penalty_temp3 = D[4*i+1];
                M[4*i+1] = float(score2.y) * fmaf(alpha[2*i].y,penalty_temp0,beta*(I+penalty_temp1));
                D[4*i+1] = fmaf(sigma[2*i].y,penalty_temp2,eps*D[4*i+1]);
                I = fmaf(delta[2*i].y,penalty_temp0,eps*I);
                Results[4*i+1] += M[4*i+1] + I;

                memcpy(&score2, &foo.y, sizeof(__half2));
                penalty_temp0 = M[4*i+2];
                penalty_temp1 = D[4*i+2];
                M[4*i+2] = float(score2.x) * fmaf(alpha[2*i+1].x,penalty_temp2,beta*(I+penalty_temp3));
                D[4*i+2] = fmaf(sigma[2*i+1].x,penalty_temp0,eps*D[4*i+2]);
                I = fmaf(delta[2*i+1].x,penalty_temp2,eps*I);
                Results[4*i+2] += M[4*i+2] + I;

                penalty_temp2 = M[4*i+3];
                penalty_temp3 = D[4*i+3];
                M[4*i+3] = float(score2.y) * fmaf(alpha[2*i+1].y,penalty_temp0,beta*(I+penalty_temp1));
                D[4*i+3] = fmaf(sigma[2*i+1].y,penalty_temp2,eps*D[4*i+3]);
                I = fmaf(delta[2*i+1].y,penalty_temp0,eps*I);
                Results[4*i+3] += M[4*i+3] + I;
            }
        };

        auto shuffle_penalty = [&]() {
            M_ul = M_l;
            D_ul = D_l;

            M_l = __shfl_up_sync(myGroupMask, M[numRegs-1], 1, group_size);
            I_ul = __shfl_up_sync(myGroupMask, I, 1, group_size);
            D_l = __shfl_up_sync(myGroupMask, D[numRegs-1], 1, group_size);

            if (!threadIdInGroup) {
                M_l = I_ul = 0.0;
                D_l = init_D;
            }
        };

        int result_thread = (readLength-1)/numRegs;
        int result_reg = (readLength-1)%numRegs;

        load_PSSM();
        load_probabilities();
        // compute_probabilities();

        init_D = constant/haploLength;
        init_penalties();

        char4 new_hap_letter4;
        hap_letter = 4;
        int k;
        for (k=0; k<haploLength-3; k+=4) {
            new_hap_letter4 = HapsAsChar4[k/4];
            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.x;
            calc_DP_float();
            shuffle_penalty();

            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.y;
            calc_DP_float();
            shuffle_penalty();

            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.z;
            calc_DP_float();
            shuffle_penalty();

            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.w;
            calc_DP_float();
            shuffle_penalty();
        }
        if (haploLength%4 >= 1) {
            new_hap_letter4 = HapsAsChar4[k/4];
            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.x;
            calc_DP_float();
            shuffle_penalty();
        }
        if (haploLength%4 >= 2) {
            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.y;
            calc_DP_float();
            shuffle_penalty();
        }
        if (haploLength%4 >= 3) {
            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.w;
            calc_DP_float();
            shuffle_penalty();
        }
        for (k=0; k<result_thread; k++) {
            // hap_letter = __shfl_up_sync(__activemask(), hap_letter, 1, 32);
            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            calc_DP_float();
            shuffle_penalty(); // shuffle_penalty_active();
        }
        // adjust I values
        I = fmaf(delta[0].x,M_ul,eps*I_ul);
        Results[0] += I;
        I = fmaf(delta[0].y,M[0],eps*I);
        Results[1] += I;
        for (int p=1; p<numRegs/2; p++) {
            I = fmaf(delta[p].x,M[2*p-1],eps*I);
            Results[2*p] += I;
            I = fmaf(delta[p].y,M[2*p],eps*I);
            Results[2*p+1] += I;
        }
        // adjust I values
        //I = fmaf(delta[0],M_ul,eps*I_ul);
        //Results[0] += I;
        //for (int p=1; p<numRegs; p++) {
        //    I = fmaf(delta[p],M[p-1],eps*I);
        //    Results[p] += I;
        //}


        if (threadIdInGroup == result_thread) {
            float temp_res = Results[result_reg];
            temp_res =  log10f(temp_res) - log10f(constant);
            devAlignmentScores[read_nr*haps_in_batch[batchId]+hapToProcessInBatch+resultOffsetsPerBatch[batchId]] = temp_res;
        }
    }


}




template <int group_size, int numRegs> __global__
void PairHMM_align_partition_half(
    const uint8_t * read_chars,
    const uint8_t * hap_chars,
    const uint8_t * base_quals,
    const uint8_t * ins_quals,
    const uint8_t * del_quals,
    float * devAlignmentScores,
    const int* read_offsets,
    const int* hap_offsets,
    const int* read_length,
    const int* hap_length,
    const int *reads_in_batch,
    const int *haps_in_batch,
    const int *offset_hap_batches,

    const int* numIndicesPerBatch,
    const int* indicesPerBatch,
    const int* numWarpsPerBatch,
    const int* numWarpsPerBatchInclusivePrefixSum,
    const int* numReadsPerBatchPrefixSum,
    const int numBatches,
    const int* resultOffsetsPerBatch
) {
    alignas(8) __shared__ __half2 lambda_array[5][16*numRegs];

    float M[numRegs], I, D[numRegs];
    half2 alpha[numRegs/2], delta[numRegs/2], sigma[numRegs/2];
    float Results[numRegs];

    const int blid = blockIdx.x;
    const int thid = threadIdx.x;
    const int group_id = thid%group_size;
    const int group_nr = thid/group_size;
    const int numGroupsPerWarp = 32/group_size;

    const unsigned int myGroupMask = __match_any_sync(0xFFFFFFFF, group_nr); //compute mask for all threads with same group_nr

    const int batchId = thrust::distance(
        numWarpsPerBatchInclusivePrefixSum,
        thrust::upper_bound(thrust::seq,
            numWarpsPerBatchInclusivePrefixSum,
            numWarpsPerBatchInclusivePrefixSum + numBatches,
            blid
        )
    );

    //if(thid == 0) printf("warp %d, batchId %d\n", blid, batchId);

    const int offset_read_batches = numReadsPerBatchPrefixSum[batchId];
    //if(thid == 0)  printf("warp %d, batchId %d, offset_read_batches %d\n", blid, batchId, offset_read_batches);

    const int numWarpsForBatch = numWarpsPerBatch[batchId];
    const int warpIdInBatch = blid - (numWarpsPerBatchInclusivePrefixSum[batchId]-numWarpsForBatch);
    const int groupIdInBatch = numGroupsPerWarp * warpIdInBatch + group_nr;
    const int numGroupsForBatch = numIndicesPerBatch[batchId];
    int readToProcessInBatch;
    if(groupIdInBatch < numGroupsForBatch){
        readToProcessInBatch = indicesPerBatch[offset_read_batches + groupIdInBatch];
        //if(group_id == 0) printf("warp: %d. warpIdInBatch %d, groupId in warp: %d. groupId in batch: %d. readToProcessInBatch %d\n", blid, warpIdInBatch, group_nr, groupIdInBatch, readToProcessInBatch);
    }else{
        readToProcessInBatch = indicesPerBatch[offset_read_batches + 0];
        //if(group_id == 0) printf("warp: %d. warpIdInBatch %d, groupId in warp: %d. groupId in batch: %d. unused\n", blid, warpIdInBatch, group_nr, groupIdInBatch);
    }



    int read_nr = readToProcessInBatch; // (blockDim.x/group_size)*blid + group_nr;
    int global_read_id = read_nr + offset_read_batches;

    const int base = read_offsets[global_read_id];
    const int length = read_length[global_read_id];

    const float eps = 0.1;
    const float beta = 0.9;
    float M_l, D_l, M_ul, D_ul, I_ul;
    float penalty_temp0, penalty_temp1, penalty_temp2, penalty_temp3;
    float init_D;

    const float constant = ::cuda::std::numeric_limits<float>::max() / 16;

    auto load_PSSM = [&]() {

        char4 temp0, temp1;
        const half one_half = 1.0;
        const half three = 3.0;
        const char4* QualsAsChar4 = reinterpret_cast<const char4*>(&base_quals[base]);
        const char4* ReadsAsChar4 = reinterpret_cast<const char4*>(&read_chars[base]);
        for (int i=group_id; i<(length+3)/4; i+=group_size) {
            half2 temp_h2, temp_h3;
            temp0 = QualsAsChar4[i];
            temp1 = ReadsAsChar4[i];
            temp_h2.x = cPH2PR[uint8_t(temp0.x)];
            temp_h2.y = cPH2PR[uint8_t(temp0.y)];
            temp_h3.x = temp_h2.x/three;
            temp_h3.y = temp_h2.y/three;
            for (int j=0; j<5; j++) lambda_array[j][2*i+group_nr*(group_size*numRegs/2)] = temp_h3;
            if (temp1.x <= 4) lambda_array[temp1.x][2*i+group_nr*(group_size*numRegs/2)].x = one_half - temp_h2.x;
            if (temp1.y <= 4) lambda_array[temp1.y][2*i+group_nr*(group_size*numRegs/2)].y = one_half - temp_h2.y;
            temp_h2.x = cPH2PR[uint8_t(temp0.z)];
            temp_h2.y = cPH2PR[uint8_t(temp0.w)];
            temp_h3.x = temp_h2.x/three;
            temp_h3.y = temp_h2.y/three;
            for (int j=0; j<5; j++) lambda_array[j][2*i+1+group_nr*(group_size*numRegs/2)] = temp_h3;
            if (temp1.z <= 4) lambda_array[temp1.z][2*i+1+group_nr*(group_size*numRegs/2)].x = one_half - temp_h2.x;
            if (temp1.w <= 4) lambda_array[temp1.w][2*i+1+group_nr*(group_size*numRegs/2)].y = one_half - temp_h2.y;
        }

        __syncwarp(myGroupMask);

    };

    auto load_probabilities = [&]() {
        char4 temp0, temp1;
        const char4* InsQualsAsChar4 = reinterpret_cast<const char4*>(&ins_quals[base]);
        const char4* DelQualsAsChar4 = reinterpret_cast<const char4*>(&del_quals[base]);
        for (int i=0; i<numRegs/4; i++) {
            if (group_id*numRegs/4+i < (length+3)/4) {

                temp0 = InsQualsAsChar4[group_id*numRegs/4+i];
                temp1 = DelQualsAsChar4[group_id*numRegs/4+i];

                //delta[4*i] = cPH2PR[uint8_t(temp0.x)];
                //delta[4*i+1] = cPH2PR[uint8_t(temp0.y)];
                //delta[4*i+2] = cPH2PR[uint8_t(temp0.z)];
                //delta[4*i+3] = cPH2PR[uint8_t(temp0.w)];
                delta[2*i] = __floats2half2_rn(cPH2PR[uint8_t(temp0.x)],cPH2PR[uint8_t(temp0.y)]);
                delta[2*i+1] = __floats2half2_rn(cPH2PR[uint8_t(temp0.z)],cPH2PR[uint8_t(temp0.w)]);

                //sigma[4*i] = cPH2PR[uint8_t(temp1.x)];
                //sigma[4*i+1] = cPH2PR[uint8_t(temp1.y)];
                //sigma[4*i+2] = cPH2PR[uint8_t(temp1.z)];
                //sigma[4*i+3] = cPH2PR[uint8_t(temp1.w)];
                sigma[2*i] = __floats2half2_rn(cPH2PR[uint8_t(temp1.x)],cPH2PR[uint8_t(temp1.y)]);
                sigma[2*i+1] = __floats2half2_rn(cPH2PR[uint8_t(temp1.z)],cPH2PR[uint8_t(temp1.w)]);

                //alpha[4*i] = 1.0 - (delta[4*i] + sigma[4*i]);
                //alpha[4*i+1] = 1.0 - (delta[4*i+1] + sigma[4*i+1]);
                //alpha[4*i+2] = 1.0 - (delta[4*i+2] + sigma[4*i+2]);
                //alpha[4*i+3] = 1.0 - (delta[4*i+3] + sigma[4*i+3]);
                alpha[2*i] = __float2half2_rn(1.0) - __hadd2(delta[2*i], sigma[2*i]);
                alpha[2*i+1] = __float2half2_rn(1.0) - __hadd2(delta[2*i+1], sigma[2*i+1]);
            }
        }

    };

    auto init_penalties = [&]() {
        #pragma unroll
        for (int i=0; i<numRegs; i++) M[i] = D[i] = Results[i] = 0.0;
        M_l = M_ul = D_ul = I_ul = D_l = I = 0.0;
        if (!group_id) D_l = D_ul = init_D;
    };


    char hap_letter;

    __half2 score2;
    __half2 *sbt_row;

    auto calc_DP_float = [&](){

        sbt_row = lambda_array[hap_letter];
        float2 foo = *((float2*)&sbt_row[thid*numRegs/2]);
        memcpy(&score2, &foo.x, sizeof(__half2));
        penalty_temp0 = M[0];
        penalty_temp1 = D[0];
        M[0] = float(score2.x) * fmaf(alpha[0].x,M_ul,beta*(I_ul+D_ul));
        D[0] = fmaf(sigma[0].x,penalty_temp0,eps*D[0]);
        I = fmaf(delta[0].x,M_ul,eps*I_ul);
        Results[0] += M[0] + I;
        penalty_temp2 = M[1];
        penalty_temp3 = D[1];
        M[1] = float(score2.y) * fmaf(alpha[0].y,penalty_temp0,beta*(I+penalty_temp1));
        D[1] = fmaf(sigma[0].y,penalty_temp2,eps*D[1]);
        I = fmaf(delta[0].y,penalty_temp0,eps*I);
        Results[1] += M[1] + I;

        memcpy(&score2, &foo.y, sizeof(__half2));
        penalty_temp0 = M[2];
        penalty_temp1 = D[2];
        M[2] = float(score2.x) * fmaf(alpha[1].x,penalty_temp2,beta*(I+penalty_temp3));
        D[2] = fmaf(sigma[1].x,penalty_temp0,eps*D[2]);
        I = fmaf(delta[1].x,penalty_temp2,eps*I);
        Results[2] += M[2] + I;
        penalty_temp2 = M[3];
        penalty_temp3 = D[3];
        M[3] = float(score2.y) * fmaf(alpha[1].y,penalty_temp0,beta*(I+penalty_temp1));
        D[3] = fmaf(sigma[1].y,penalty_temp2,eps*D[3]);
        I = fmaf(delta[1].y,penalty_temp0,eps*I);
        Results[3] += M[3] + I;

        #pragma unroll
        for (int i=1; i<numRegs/4; i++) {
            float2 foo = *((float2*)&sbt_row[thid*numRegs/2+2*i]);
            memcpy(&score2, &foo.x, sizeof(__half2));
            penalty_temp0 = M[4*i];
            penalty_temp1 = D[4*i];
            M[4*i] = float(score2.x) * fmaf(alpha[2*i].x,penalty_temp2,beta*(I+penalty_temp3));
            D[4*i] = fmaf(sigma[2*i].x,penalty_temp0,eps*D[4*i]);
            I = fmaf(delta[2*i].x,penalty_temp2,eps*I);
            Results[4*i] += M[4*i] + I;
            penalty_temp2 = M[4*i+1];
            penalty_temp3 = D[4*i+1];
            M[4*i+1] = float(score2.y) * fmaf(alpha[2*i].y,penalty_temp0,beta*(I+penalty_temp1));
            D[4*i+1] = fmaf(sigma[2*i].y,penalty_temp2,eps*D[4*i+1]);
            I = fmaf(delta[2*i].y,penalty_temp0,eps*I);
            Results[4*i+1] += M[4*i+1] + I;

            memcpy(&score2, &foo.y, sizeof(__half2));
            penalty_temp0 = M[4*i+2];
            penalty_temp1 = D[4*i+2];
            M[4*i+2] = float(score2.x) * fmaf(alpha[2*i+1].x,penalty_temp2,beta*(I+penalty_temp3));
            D[4*i+2] = fmaf(sigma[2*i+1].x,penalty_temp0,eps*D[4*i+2]);
            I = fmaf(delta[2*i+1].x,penalty_temp2,eps*I);
            Results[4*i+2] += M[4*i+2] + I;

            penalty_temp2 = M[4*i+3];
            penalty_temp3 = D[4*i+3];
            M[4*i+3] = float(score2.y) * fmaf(alpha[2*i+1].y,penalty_temp0,beta*(I+penalty_temp1));
            D[4*i+3] = fmaf(sigma[2*i+1].y,penalty_temp2,eps*D[4*i+3]);
            I = fmaf(delta[2*i+1].y,penalty_temp0,eps*I);
            Results[4*i+3] += M[4*i+3] + I;
        }
    };

    auto shuffle_penalty = [&]() {
        M_ul = M_l;
        D_ul = D_l;

        M_l = __shfl_up_sync(myGroupMask, M[numRegs-1], 1, group_size);
        I_ul = __shfl_up_sync(myGroupMask, I, 1, group_size);
        D_l = __shfl_up_sync(myGroupMask, D[numRegs-1], 1, group_size);

        if (!group_id) {
            M_l = I_ul = 0.0;
            D_l = init_D;
        }
    };

    // auto shuffle_penalty_active = [&]() {
    //     M_ul = M_l;
    //     D_ul = D_l;

    //     unsigned mask = __activemask();
    //     M_l = __shfl_up_sync(mask, M[numRegs-1], 1, 32);
    //     I_ul = __shfl_up_sync(mask, I, 1, 32);
    //     D_l = __shfl_up_sync(mask, D[numRegs-1], 1, 32);

    //     if (!group_id) {
    //         M_l = I_ul = 0.0;
    //         D_l = init_D;
    //     }
    // };


    load_PSSM();
    load_probabilities();
    int result_thread = (length-1)/numRegs;
    int result_reg = (length-1)%numRegs;
    int b_h_off = offset_hap_batches[batchId];
    //int h_off = offset_haps[b_h_off];

    for (int i=0; i<haps_in_batch[batchId]; i++) {
        // init_D = constant/hap_length[i]; // why is this i instead of i+b_h_off ?
        init_D = constant/hap_length[i+b_h_off];
        init_penalties();
        const char4* HapsAsChar4 = reinterpret_cast<const char4*>(&hap_chars[hap_offsets[i+b_h_off]]);
        //const char4* HapsAsChar4 = reinterpret_cast<const char4*>(&hap_chars[hap_offsets[i]-hap_offsets[0]]);
        char4 new_hap_letter4;
        hap_letter = 4;
        int k;
        for (k=0; k<hap_length[i+b_h_off]-3; k+=4) {
            new_hap_letter4 = HapsAsChar4[k/4];
            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!group_id) hap_letter = new_hap_letter4.x;
            calc_DP_float();
            shuffle_penalty();

            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!group_id) hap_letter = new_hap_letter4.y;
            calc_DP_float();
            shuffle_penalty();

            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!group_id) hap_letter = new_hap_letter4.z;
            calc_DP_float();
            shuffle_penalty();

            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!group_id) hap_letter = new_hap_letter4.w;
            calc_DP_float();
            shuffle_penalty();


        }

        if (hap_length[i+b_h_off]%4 >= 1) {
            new_hap_letter4 = HapsAsChar4[k/4];
            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!group_id) hap_letter = new_hap_letter4.x;
            calc_DP_float();
            shuffle_penalty();

        }
        if (hap_length[i+b_h_off]%4 >= 2) {
            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!group_id) hap_letter = new_hap_letter4.y;
            calc_DP_float();
            shuffle_penalty();
        }

        if (hap_length[i+b_h_off]%4 >= 3) {
            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!group_id) hap_letter = new_hap_letter4.w;
            calc_DP_float();
            shuffle_penalty();
        }

        for (k=0; k<result_thread; k++) {
            // hap_letter = __shfl_up_sync(__activemask(), hap_letter, 1, 32);
            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            calc_DP_float();
            shuffle_penalty(); // shuffle_penalty_active();
        }
        // adjust I values
        I = fmaf(delta[0].x,M_ul,eps*I_ul);
        Results[0] += I;
        I = fmaf(delta[0].y,M[0],eps*I);
        Results[1] += I;
        for (int p=1; p<numRegs/2; p++) {
            I = fmaf(delta[p].x,M[2*p-1],eps*I);
            Results[2*p] += I;
            I = fmaf(delta[p].y,M[2*p],eps*I);
            Results[2*p+1] += I;
        }
        // adjust I values
        //I = fmaf(delta[0],M_ul,eps*I_ul);
        //Results[0] += I;
        //for (int p=1; p<numRegs; p++) {
        //    I = fmaf(delta[p],M[p-1],eps*I);
        //    Results[p] += I;
        //}


        if (group_id == result_thread) {
            float temp_res = Results[result_reg];
            temp_res =  log10f(temp_res) - log10f(constant);
            if (read_nr < reads_in_batch[batchId]) devAlignmentScores[read_nr*haps_in_batch[batchId]+i+resultOffsetsPerBatch[batchId]] = temp_res;
        //    if ((blockDim.x/group_size)*blid + group_nr < reads_in_batch) devAlignmentScores[read_nr*haps_in_batch+i] = temp_res;
        //if (blid == 0) // if ((blid == 0) && (!group_nr))
        //           printf("Results in Block: %d, in thread: %d, in Register: %d: %f\n", blid, result_thread, result_reg, temp_res);

        }
    }


}



template <int group_size, int numRegs> __global__
void PairHMM_align_partition(
    const uint8_t * read_chars,
    const uint8_t * hap_chars,
    const uint8_t * base_quals,
    const uint8_t * ins_quals,
    const uint8_t * del_quals,
    float * devAlignmentScores,
    const int* read_offsets,
    const int* hap_offsets,
    const int* read_length,
    const int* hap_length,
    const int *reads_in_batch,
    const int *haps_in_batch,
    const int *offset_hap_batches,

    const int* numIndicesPerBatch,
    const int* indicesPerBatch,
    const int* numWarpsPerBatch,
    const int* numWarpsPerBatchInclusivePrefixSum,
    const int* numReadsPerBatchPrefixSum,
    const int numBatches,
    const int* resultOffsetsPerBatch
) {
    alignas(8) __shared__ __half2 lambda_array[5][16*numRegs];

    float M[numRegs], I, D[numRegs];
    float alpha[numRegs], delta[numRegs], sigma[numRegs];
    float Results[numRegs];

    const int blid = blockIdx.x;
    const int thid = threadIdx.x;
    const int group_id = thid%group_size;
    const int group_nr = thid/group_size;
    const int numGroupsPerWarp = 32/group_size;

    const int batchId = thrust::distance(
        numWarpsPerBatchInclusivePrefixSum,
        thrust::upper_bound(thrust::seq,
            numWarpsPerBatchInclusivePrefixSum,
            numWarpsPerBatchInclusivePrefixSum + numBatches,
            blid
        )
    );

    //if(thid == 0) printf("warp %d, batchId %d\n", blid, batchId);

    const int offset_read_batches = numReadsPerBatchPrefixSum[batchId];
    //if(thid == 0)  printf("warp %d, batchId %d, offset_read_batches %d\n", blid, batchId, offset_read_batches);

    const int numWarpsForBatch = numWarpsPerBatch[batchId];
    const int warpIdInBatch = blid - (numWarpsPerBatchInclusivePrefixSum[batchId]-numWarpsForBatch);
    const int groupIdInBatch = numGroupsPerWarp * warpIdInBatch + group_nr;
    const int numGroupsForBatch = numIndicesPerBatch[batchId];
    int readToProcessInBatch;
    if(groupIdInBatch < numGroupsForBatch){
        readToProcessInBatch = indicesPerBatch[offset_read_batches + groupIdInBatch];
        //if(group_id == 0) printf("warp: %d. warpIdInBatch %d, groupId in warp: %d. groupId in batch: %d. readToProcessInBatch %d\n", blid, warpIdInBatch, group_nr, groupIdInBatch, readToProcessInBatch);
    }else{
        readToProcessInBatch = indicesPerBatch[offset_read_batches + 0];
        //if(group_id == 0) printf("warp: %d. warpIdInBatch %d, groupId in warp: %d. groupId in batch: %d. unused\n", blid, warpIdInBatch, group_nr, groupIdInBatch);
    }



    int read_nr = readToProcessInBatch; // (blockDim.x/group_size)*blid + group_nr;
    int global_read_id = read_nr + offset_read_batches;

    const int base = read_offsets[global_read_id];
    const int length = read_length[global_read_id];

    const float eps = 0.1;
    const float beta = 0.9;
    float M_l, D_l, M_ul, D_ul, I_ul;
    float penalty_temp0, penalty_temp1, penalty_temp2, penalty_temp3;
    float init_D;

    const float constant = ::cuda::std::numeric_limits<float>::max() / 16;

    auto load_PSSM = [&]() {

        char4 temp0, temp1;
        const half one_half = 1.0;
        const char4* QualsAsChar4 = reinterpret_cast<const char4*>(&base_quals[base]);
        const char4* ReadsAsChar4 = reinterpret_cast<const char4*>(&read_chars[base]);
        for (int i=group_id; i<(length+3)/4; i+=group_size) {
            half2 temp_h2;
            temp0 = QualsAsChar4[i];
            temp1 = ReadsAsChar4[i];
            temp_h2.x = cPH2PR[uint8_t(temp0.x)];
            temp_h2.y = cPH2PR[uint8_t(temp0.y)];
            for (int j=0; j<5; j++) lambda_array[j][2*i+group_nr*(group_size*numRegs/2)] = temp_h2;
            if (temp1.x <= 4) lambda_array[temp1.x][2*i+group_nr*(group_size*numRegs/2)].x = one_half - temp_h2.x;
            if (temp1.y <= 4) lambda_array[temp1.y][2*i+group_nr*(group_size*numRegs/2)].y = one_half - temp_h2.y;
            temp_h2.x = cPH2PR[uint8_t(temp0.z)];
            temp_h2.y = cPH2PR[uint8_t(temp0.w)];
            for (int j=0; j<5; j++) lambda_array[j][2*i+1+group_nr*(group_size*numRegs/2)] = temp_h2;
            if (temp1.z <= 4) lambda_array[temp1.z][2*i+1+group_nr*(group_size*numRegs/2)].x = one_half - temp_h2.x;
            if (temp1.w <= 4) lambda_array[temp1.w][2*i+1+group_nr*(group_size*numRegs/2)].y = one_half - temp_h2.y;
        }

    };

    auto load_probabilities = [&]() {
        char4 temp0, temp1;
        const char4* InsQualsAsChar4 = reinterpret_cast<const char4*>(&ins_quals[base]);
        const char4* DelQualsAsChar4 = reinterpret_cast<const char4*>(&del_quals[base]);
        for (int i=0; i<numRegs/4; i++) {
            if (group_id*numRegs/4+i < (length+3)/4) {

                temp0 = InsQualsAsChar4[group_id*numRegs/4+i];
                temp1 = DelQualsAsChar4[group_id*numRegs/4+i];

                delta[4*i] = cPH2PR[uint8_t(temp0.x)];
                delta[4*i+1] = cPH2PR[uint8_t(temp0.y)];
                delta[4*i+2] = cPH2PR[uint8_t(temp0.z)];
                delta[4*i+3] = cPH2PR[uint8_t(temp0.w)];

                sigma[4*i] = cPH2PR[uint8_t(temp1.x)];
                sigma[4*i+1] = cPH2PR[uint8_t(temp1.y)];
                sigma[4*i+2] = cPH2PR[uint8_t(temp1.z)];
                sigma[4*i+3] = cPH2PR[uint8_t(temp1.w)];

                alpha[4*i] = 1.0 - (delta[4*i] + sigma[4*i]);
                alpha[4*i+1] = 1.0 - (delta[4*i+1] + sigma[4*i+1]);
                alpha[4*i+2] = 1.0 - (delta[4*i+2] + sigma[4*i+2]);
                alpha[4*i+3] = 1.0 - (delta[4*i+3] + sigma[4*i+3]);
            }
        }

    };

    auto init_penalties = [&]() {
        #pragma unroll
        for (int i=0; i<numRegs; i++) M[i] = D[i] = Results[i] = 0.0;
        M_l = M_ul = D_ul = I_ul = D_l = I = 0.0;
        if (!group_id) D_l = D_ul = init_D;
    };


    char hap_letter;

    __half2 score2;
    __half2 *sbt_row;

    auto calc_DP_float = [&](){

        sbt_row = lambda_array[hap_letter];
        float2 foo = *((float2*)&sbt_row[thid*numRegs/2]);
        memcpy(&score2, &foo.x, sizeof(__half2));
        penalty_temp0 = M[0];
        penalty_temp1 = D[0];
        M[0] = float(score2.x) * fmaf(alpha[0],M_ul,beta*(I_ul+D_ul));
        D[0] = fmaf(sigma[0],penalty_temp0,eps*D[0]);
        I = fmaf(delta[0],M_ul,eps*I_ul);
        Results[0] += M[0] + I;
        penalty_temp2 = M[1];
        penalty_temp3 = D[1];
        M[1] = float(score2.y) * fmaf(alpha[1],penalty_temp0,beta*(I+penalty_temp1));
        D[1] = fmaf(sigma[1],penalty_temp2,eps*D[1]);
        I = fmaf(delta[1],penalty_temp0,eps*I);
        Results[1] += M[1] + I;

        memcpy(&score2, &foo.y, sizeof(__half2));
        penalty_temp0 = M[2];
        penalty_temp1 = D[2];
        M[2] = float(score2.x) * fmaf(alpha[2],penalty_temp2,beta*(I+penalty_temp3));
        D[2] = fmaf(sigma[2],penalty_temp0,eps*D[2]);
        I = fmaf(delta[2],penalty_temp2,eps*I);
        Results[2] += M[2] + I;
        penalty_temp2 = M[3];
        penalty_temp3 = D[3];
        M[3] = float(score2.y) * fmaf(alpha[3],penalty_temp0,beta*(I+penalty_temp1));
        D[3] = fmaf(sigma[3],penalty_temp2,eps*D[3]);
        I = fmaf(delta[3],penalty_temp0,eps*I);
        Results[3] += M[3] + I;

        #pragma unroll
        for (int i=1; i<numRegs/4; i++) {
            float2 foo = *((float2*)&sbt_row[thid*numRegs/2+2*i]);
            memcpy(&score2, &foo.x, sizeof(__half2));
            penalty_temp0 = M[4*i];
            penalty_temp1 = D[4*i];
            M[4*i] = float(score2.x) * fmaf(alpha[4*i],penalty_temp2,beta*(I+penalty_temp3));
            D[4*i] = fmaf(sigma[4*i],penalty_temp0,eps*D[4*i]);
            I = fmaf(delta[4*i],penalty_temp2,eps*I);
            Results[4*i] += M[4*i] + I;
            penalty_temp2 = M[4*i+1];
            penalty_temp3 = D[4*i+1];
            M[4*i+1] = float(score2.y) * fmaf(alpha[4*i+1],penalty_temp0,beta*(I+penalty_temp1));
            D[4*i+1] = fmaf(sigma[4*i+1],penalty_temp2,eps*D[4*i+1]);
            I = fmaf(delta[4*i+1],penalty_temp0,eps*I);
            Results[4*i+1] += M[4*i+1] + I;

            memcpy(&score2, &foo.y, sizeof(__half2));
            penalty_temp0 = M[4*i+2];
            penalty_temp1 = D[4*i+2];
            M[4*i+2] = float(score2.x) * fmaf(alpha[4*i+2],penalty_temp2,beta*(I+penalty_temp3));
            D[4*i+2] = fmaf(sigma[4*i+2],penalty_temp0,eps*D[4*i+2]);
            I = fmaf(delta[4*i+2],penalty_temp2,eps*I);
            Results[4*i+2] += M[4*i+2] + I;

            penalty_temp2 = M[4*i+3];
            penalty_temp3 = D[4*i+3];
            M[4*i+3] = float(score2.y) * fmaf(alpha[4*i+3],penalty_temp0,beta*(I+penalty_temp1));
            D[4*i+3] = fmaf(sigma[4*i+3],penalty_temp2,eps*D[4*i+3]);
            I = fmaf(delta[4*i+3],penalty_temp0,eps*I);
            Results[4*i+3] += M[4*i+3] + I;
        }
    };

    auto shuffle_penalty = [&]() {
        M_ul = M_l;
        D_ul = D_l;

        M_l = __shfl_up_sync(0xFFFFFFFF, M[numRegs-1], 1, 32);
        I_ul = __shfl_up_sync(0xFFFFFFFF, I, 1, 32);
        D_l = __shfl_up_sync(0xFFFFFFFF, D[numRegs-1], 1, 32);

        if (!group_id) {
            M_l = I_ul = 0.0;
            D_l = init_D;
        }
    };

    auto shuffle_penalty_active = [&]() {
        M_ul = M_l;
        D_ul = D_l;

        unsigned mask = __activemask();
        M_l = __shfl_up_sync(mask, M[numRegs-1], 1, 32);
        I_ul = __shfl_up_sync(mask, I, 1, 32);
        D_l = __shfl_up_sync(mask, D[numRegs-1], 1, 32);

        if (!group_id) {
            M_l = I_ul = 0.0;
            D_l = init_D;
        }
    };


    load_PSSM();
    load_probabilities();
    int result_thread = (length-1)/numRegs;
    int result_reg = (length-1)%numRegs;
    int b_h_off = offset_hap_batches[batchId];
    //int h_off = offset_haps[b_h_off];

    for (int i=0; i<haps_in_batch[batchId]; i++) {
        init_D = constant/hap_length[i];
        init_penalties();
        const char4* HapsAsChar4 = reinterpret_cast<const char4*>(&hap_chars[hap_offsets[i+b_h_off]]);
        //const char4* HapsAsChar4 = reinterpret_cast<const char4*>(&hap_chars[hap_offsets[i]-hap_offsets[0]]);
        char4 new_hap_letter4;
        hap_letter = 4;
        int k;
        for (k=0; k<hap_length[i+b_h_off]-3; k+=4) {
            new_hap_letter4 = HapsAsChar4[k/4];
            hap_letter = __shfl_up_sync(0xFFFFFFFF, hap_letter, 1, 32);
            if (!group_id) hap_letter = new_hap_letter4.x;
            calc_DP_float();
            shuffle_penalty();

            hap_letter = __shfl_up_sync(0xFFFFFFFF, hap_letter, 1, 32);
            if (!group_id) hap_letter = new_hap_letter4.y;
            calc_DP_float();
            shuffle_penalty();

            hap_letter = __shfl_up_sync(0xFFFFFFFF, hap_letter, 1, 32);
            if (!group_id) hap_letter = new_hap_letter4.z;
            calc_DP_float();
            shuffle_penalty();

            hap_letter = __shfl_up_sync(0xFFFFFFFF, hap_letter, 1, 32);
            if (!group_id) hap_letter = new_hap_letter4.w;
            calc_DP_float();
            shuffle_penalty();


        }

        if (hap_length[i+b_h_off]%4 >= 1) {
            new_hap_letter4 = HapsAsChar4[k/4];
            hap_letter = __shfl_up_sync(0xFFFFFFFF, hap_letter, 1, 32);
            if (!group_id) hap_letter = new_hap_letter4.x;
            calc_DP_float();
            shuffle_penalty();

        }
        if (hap_length[i+b_h_off]%4 >= 2) {
            hap_letter = __shfl_up_sync(0xFFFFFFFF, hap_letter, 1, 32);
            if (!group_id) hap_letter = new_hap_letter4.y;
            calc_DP_float();
            shuffle_penalty();
        }

        if (hap_length[i+b_h_off]%4 >= 3) {
            hap_letter = __shfl_up_sync(0xFFFFFFFF, hap_letter, 1, 32);
            if (!group_id) hap_letter = new_hap_letter4.w;
            calc_DP_float();
            shuffle_penalty();
        }

        for (k=0; k<result_thread; k++) {
            hap_letter = __shfl_up_sync(__activemask(), hap_letter, 1, 32);
            calc_DP_float();
            shuffle_penalty_active();
        }
        // adjust I values
        I = fmaf(delta[0],M_ul,eps*I_ul);
        Results[0] += I;
        for (int p=1; p<numRegs; p++) {
            I = fmaf(delta[p],M[p-1],eps*I);
            Results[p] += I;
        }


        if (group_id == result_thread) {
            float temp_res = Results[result_reg];
            temp_res =  log10f(temp_res) - log10f(constant);
            if (read_nr < reads_in_batch[batchId]) devAlignmentScores[read_nr*haps_in_batch[batchId]+i+resultOffsetsPerBatch[batchId]] = temp_res;
        //    if ((blockDim.x/group_size)*blid + group_nr < reads_in_batch) devAlignmentScores[read_nr*haps_in_batch+i] = temp_res;
        //if (blid == 0) // if ((blid == 0) && (!group_nr))
        //           printf("Results in Block: %d, in thread: %d, in Register: %d: %f\n", blid, result_thread, result_reg, temp_res);

        }
    }


}



template <int group_size, int numRegs> __global__
void PairHMM_align(
    const uint8_t * read_chars,
    const uint8_t * hap_chars,
    const uint8_t * base_quals,
    const uint8_t * ins_quals,
    const uint8_t * del_quals,
    float * devAlignmentScores,
    const int* read_offsets,
    const int* hap_offsets,
    const int* read_length,
    const int* hap_length,
    const int reads_in_batch,
    const int haps_in_batch
) {
    alignas(8) __shared__ __half2 lambda_array[5][16*numRegs];

    float M[numRegs], I, D[numRegs];
    float alpha[numRegs], delta[numRegs], sigma[numRegs];
    float Results[numRegs];

    const int blid = blockIdx.x;
    const int thid = threadIdx.x;
    const int group_id = thid%group_size;
    const int group_nr = thid/group_size;

    int read_nr = (blockDim.x/group_size)*blid + group_nr;
    if (read_nr >= reads_in_batch) read_nr = 0;

    const int base = read_offsets[read_nr] - read_offsets[0];
    const int length = read_length[read_nr];
    //const int offset = read_offsets[read_nr];




    const float eps = 0.1;
    const float beta = 0.9;
    float M_l, D_l, M_ul, D_ul, I_ul;
    float penalty_temp0, penalty_temp1, penalty_temp2, penalty_temp3;
    float init_D;

    const float constant = ::cuda::std::numeric_limits<float>::max() / 16;

    auto load_PSSM = [&]() {

        char4 temp0, temp1;
        const half one_half = 1.0;
        const half three = 3.0;
        const char4* QualsAsChar4 = reinterpret_cast<const char4*>(&base_quals[base]);
        const char4* ReadsAsChar4 = reinterpret_cast<const char4*>(&read_chars[base]);
        for (int i=group_id; i<(length+3)/4; i+=group_size) {
            half2 temp_h2, temp_h3;
            temp0 = QualsAsChar4[i];
            temp1 = ReadsAsChar4[i];
            temp_h2.x = cPH2PR[uint8_t(temp0.x)];
            temp_h2.y = cPH2PR[uint8_t(temp0.y)];
            temp_h3.x = temp_h2.x/three;
            temp_h3.y = temp_h2.y/three;
            for (int j=0; j<5; j++) lambda_array[j][2*i+group_nr*(group_size*numRegs/2)] = temp_h3;
            if (temp1.x <= 4) lambda_array[temp1.x][2*i+group_nr*(group_size*numRegs/2)].x = one_half - temp_h2.x;
            if (temp1.y <= 4) lambda_array[temp1.y][2*i+group_nr*(group_size*numRegs/2)].y = one_half - temp_h2.y;
            temp_h2.x = cPH2PR[uint8_t(temp0.z)];
            temp_h2.y = cPH2PR[uint8_t(temp0.w)];
            temp_h3.x = temp_h2.x/three;
            temp_h3.y = temp_h2.y/three;
            for (int j=0; j<5; j++) lambda_array[j][2*i+1+group_nr*(group_size*numRegs/2)] = temp_h3;
            if (temp1.z <= 4) lambda_array[temp1.z][2*i+1+group_nr*(group_size*numRegs/2)].x = one_half - temp_h2.x;
            if (temp1.w <= 4) lambda_array[temp1.w][2*i+1+group_nr*(group_size*numRegs/2)].y = one_half - temp_h2.y;
        }

    //    if ((blid == 0) && (!thid)) {
    //        printf("PSSM in Block: %d, in Thread: %d\n", blid, thid);
    //        for (int col=0; col<group_size*numRegs/2; col++) {
    //            for (int letter=0; letter<5; letter++) {
    //                float2 temp_temp = __half22float2(lambda_array[letter][col]);
    //                printf("%f , %f ,",temp_temp.x, temp_temp.y);
    //            }
    //            printf("\n");
    //        }
    //    }

    };

    auto load_probabilities = [&]() {
        char4 temp0, temp1;
        const char4* InsQualsAsChar4 = reinterpret_cast<const char4*>(&ins_quals[base]);
        const char4* DelQualsAsChar4 = reinterpret_cast<const char4*>(&del_quals[base]);
        for (int i=0; i<numRegs/4; i++) {
            if (group_id*numRegs/4+i < (length+3)/4) {
                //if ((blid == 0) && (!thid)) printf("load_probabilities in Block: %d, in Thread: %d, iteration: %i, base: %i, index: %i \n", blid, thid, i, base, group_id*numRegs/4+i);

                temp0 = InsQualsAsChar4[group_id*numRegs/4+i];
                temp1 = DelQualsAsChar4[group_id*numRegs/4+i];

                delta[4*i] = cPH2PR[uint8_t(temp0.x)];
                delta[4*i+1] = cPH2PR[uint8_t(temp0.y)];
                delta[4*i+2] = cPH2PR[uint8_t(temp0.z)];
                delta[4*i+3] = cPH2PR[uint8_t(temp0.w)];

                sigma[4*i] = cPH2PR[uint8_t(temp1.x)];
                sigma[4*i+1] = cPH2PR[uint8_t(temp1.y)];
                sigma[4*i+2] = cPH2PR[uint8_t(temp1.z)];
                sigma[4*i+3] = cPH2PR[uint8_t(temp1.w)];

                alpha[4*i] = 1.0 - (delta[4*i] + sigma[4*i]);
                alpha[4*i+1] = 1.0 - (delta[4*i+1] + sigma[4*i+1]);
                alpha[4*i+2] = 1.0 - (delta[4*i+2] + sigma[4*i+2]);
                alpha[4*i+3] = 1.0 - (delta[4*i+3] + sigma[4*i+3]);
            }
        }

    //    if ((blid == 0) && (!thid)) {
    //        printf("Alpha in Block: %d, in Thread: %d\n", blid, thid);
    //        for (int letter=0; letter<numRegs*group_size; letter++) printf("%f , ",alpha[letter]);
    //        printf("\n");
    //        printf("Delta in Block: %d, in Thread: %d\n", blid, thid);
    //        for (int letter=0; letter<numRegs*group_size; letter++) printf("%f , ",delta[letter]);
    //        printf("\n");
    //        printf("Sigma in Block: %d, in Thread: %d\n", blid, thid);
    //        for (int letter=0; letter<numRegs*group_size; letter++) printf("%f , ",sigma[letter]);
    //        printf("\n");
    //    }

    };

    auto init_penalties = [&]() {

        #pragma unroll
        for (int i=0; i<numRegs; i++) M[i] = D[i] = Results[i] = 0.0;
        M_l = M_ul = D_ul = I_ul = D_l = I = 0.0;
        if (!group_id) D_l = D_ul = init_D;

    //    if ((blid == 0) && (!thid)) printf("Init D_ul: %f \n", D_ul);
    };


    char hap_letter;

    __half2 score2;
    __half2 *sbt_row;

    auto calc_DP_float = [&](){

        sbt_row = lambda_array[hap_letter];
        float2 foo = *((float2*)&sbt_row[thid*numRegs/2]);
        memcpy(&score2, &foo.x, sizeof(__half2));
        penalty_temp0 = M[0];
        penalty_temp1 = D[0];
        M[0] = float(score2.x) * fmaf(alpha[0],M_ul,beta*(I_ul+D_ul));
        D[0] = fmaf(sigma[0],penalty_temp0,eps*D[0]);
        I = fmaf(delta[0],M_ul,eps*I_ul);
        Results[0] += M[0] + I;
        penalty_temp2 = M[1];
        penalty_temp3 = D[1];
        M[1] = float(score2.y) * fmaf(alpha[1],penalty_temp0,beta*(I+penalty_temp1));
        D[1] = fmaf(sigma[1],penalty_temp2,eps*D[1]);
        I = fmaf(delta[1],penalty_temp0,eps*I);
        Results[1] += M[1] + I;

        memcpy(&score2, &foo.y, sizeof(__half2));
        penalty_temp0 = M[2];
        penalty_temp1 = D[2];
        M[2] = float(score2.x) * fmaf(alpha[2],penalty_temp2,beta*(I+penalty_temp3));
        D[2] = fmaf(sigma[2],penalty_temp0,eps*D[2]);
        I = fmaf(delta[2],penalty_temp2,eps*I);
        Results[2] += M[2] + I;
        penalty_temp2 = M[3];
        penalty_temp3 = D[3];
        M[3] = float(score2.y) * fmaf(alpha[3],penalty_temp0,beta*(I+penalty_temp1));
        D[3] = fmaf(sigma[3],penalty_temp2,eps*D[3]);
        I = fmaf(delta[3],penalty_temp0,eps*I);
        Results[3] += M[3] + I;

        #pragma unroll
        for (int i=1; i<numRegs/4; i++) {
            float2 foo = *((float2*)&sbt_row[thid*numRegs/2+2*i]);
            memcpy(&score2, &foo.x, sizeof(__half2));
            penalty_temp0 = M[4*i];
            penalty_temp1 = D[4*i];
            M[4*i] = float(score2.x) * fmaf(alpha[4*i],penalty_temp2,beta*(I+penalty_temp3));
            D[4*i] = fmaf(sigma[4*i],penalty_temp0,eps*D[4*i]);
            I = fmaf(delta[4*i],penalty_temp2,eps*I);
            Results[4*i] += M[4*i] + I;
            penalty_temp2 = M[4*i+1];
            penalty_temp3 = D[4*i+1];
            M[4*i+1] = float(score2.y) * fmaf(alpha[4*i+1],penalty_temp0,beta*(I+penalty_temp1));
            D[4*i+1] = fmaf(sigma[4*i+1],penalty_temp2,eps*D[4*i+1]);
            I = fmaf(delta[4*i+1],penalty_temp0,eps*I);
            Results[4*i+1] += M[4*i+1] + I;

            memcpy(&score2, &foo.y, sizeof(__half2));
            penalty_temp0 = M[4*i+2];
            penalty_temp1 = D[4*i+2];
            M[4*i+2] = float(score2.x) * fmaf(alpha[4*i+2],penalty_temp2,beta*(I+penalty_temp3));
            D[4*i+2] = fmaf(sigma[4*i+2],penalty_temp0,eps*D[4*i+2]);
            I = fmaf(delta[4*i+2],penalty_temp2,eps*I);
            Results[4*i+2] += M[4*i+2] + I;

            penalty_temp2 = M[4*i+3];
            penalty_temp3 = D[4*i+3];
            M[4*i+3] = float(score2.y) * fmaf(alpha[4*i+3],penalty_temp0,beta*(I+penalty_temp1));
            D[4*i+3] = fmaf(sigma[4*i+3],penalty_temp2,eps*D[4*i+3]);
            I = fmaf(delta[4*i+3],penalty_temp0,eps*I);
            Results[4*i+3] += M[4*i+3] + I;
        }
    };

    auto shuffle_penalty = [&]() {
        M_ul = M_l;
        D_ul = D_l;

        M_l = __shfl_up_sync(0xFFFFFFFF, M[numRegs-1], 1, 32);
        I_ul = __shfl_up_sync(0xFFFFFFFF, I, 1, 32);
        D_l = __shfl_up_sync(0xFFFFFFFF, D[numRegs-1], 1, 32);

        if (!group_id) {
            M_l = I_ul = 0.0;
            D_l = init_D;
        }
    };

    auto shuffle_penalty_active = [&]() {
        M_ul = M_l;
        D_ul = D_l;

        unsigned mask = __activemask();
        M_l = __shfl_up_sync(mask, M[numRegs-1], 1, 32);
        I_ul = __shfl_up_sync(mask, I, 1, 32);
        D_l = __shfl_up_sync(mask, D[numRegs-1], 1, 32);

        if (!group_id) {
            M_l = I_ul = 0.0;
            D_l = init_D;
        }
    };


    load_PSSM();
    load_probabilities();
    int result_thread = (length-1)/numRegs;
    int result_reg = (length-1)%numRegs;

    for (int i=0; i<haps_in_batch; i++) {
        init_D = constant/hap_length[i];
        init_penalties();
        const char4* HapsAsChar4 = reinterpret_cast<const char4*>(&hap_chars[hap_offsets[i]-hap_offsets[0]]);
        char4 new_hap_letter4;
        hap_letter = 4;
        int k;
        for (k=0; k<hap_length[i]-3; k+=4) {
            new_hap_letter4 = HapsAsChar4[k/4];
            hap_letter = __shfl_up_sync(0xFFFFFFFF, hap_letter, 1, 32);
            if (!group_id) hap_letter = new_hap_letter4.x;
            calc_DP_float();
            shuffle_penalty();

            hap_letter = __shfl_up_sync(0xFFFFFFFF, hap_letter, 1, 32);
            if (!group_id) hap_letter = new_hap_letter4.y;
            calc_DP_float();
            shuffle_penalty();

            hap_letter = __shfl_up_sync(0xFFFFFFFF, hap_letter, 1, 32);
            if (!group_id) hap_letter = new_hap_letter4.z;
            calc_DP_float();
            shuffle_penalty();

            hap_letter = __shfl_up_sync(0xFFFFFFFF, hap_letter, 1, 32);
            if (!group_id) hap_letter = new_hap_letter4.w;
            calc_DP_float();
            shuffle_penalty();

        //    if ((k==36) && (blid == 0) && (thid==2)) {
        //        printf("Row 39: M in Block: %d, in Thread: %d\n", blid, thid);
        //        for (int letter=0; letter<numRegs; letter++) printf("%f , ",M[letter]);
        //        printf("\n");
        //        printf("D in Block: %d, in Thread: %d\n", blid, thid);
        //        for (int letter=0; letter<numRegs; letter++) printf("%f , ",D[letter]);
        //        printf("\n");
        //        printf("I in Block: %d, in Thread: %d\n", blid, thid);
        //        printf("%f , ",I);
        //        printf("\n");
        //    }
        }

        if (hap_length[i]%4 >= 1) {
            new_hap_letter4 = HapsAsChar4[k/4];
            hap_letter = __shfl_up_sync(0xFFFFFFFF, hap_letter, 1, 32);
            if (!group_id) hap_letter = new_hap_letter4.x;
            calc_DP_float();
            shuffle_penalty();

            //if ((blid == 0) && (thid==4)) {
            //        printf("Row 40: M in Block: %d, in Thread: %d\n", blid, thid);
            //        for (int letter=0; letter<numRegs; letter++) printf("%f , ",M[letter]);
            //        printf("\n");
            //        printf("D in Block: %d, in Thread: %d\n", blid, thid);
            //        for (int letter=0; letter<numRegs; letter++) printf("%f , ",D[letter]);
            //        printf("\n");
            //        printf("I in Block: %d, in Thread: %d\n", blid, thid);
            //       printf("%f , ",I);
            //        printf("\n");
          //}

        }
        if (hap_length[i]%4 >= 2) {
            hap_letter = __shfl_up_sync(0xFFFFFFFF, hap_letter, 1, 32);
            if (!group_id) hap_letter = new_hap_letter4.y;
            calc_DP_float();
            shuffle_penalty();
        }

        if (hap_length[i]%4 >= 3) {
            hap_letter = __shfl_up_sync(0xFFFFFFFF, hap_letter, 1, 32);
            if (!group_id) hap_letter = new_hap_letter4.w;
            calc_DP_float();
            shuffle_penalty();
        }

        for (k=0; k<result_thread; k++) {
            hap_letter = __shfl_up_sync(__activemask(), hap_letter, 1, 32);
            calc_DP_float();
            shuffle_penalty_active();
        }
        // adjust I values
        I = fmaf(delta[0],M_ul,eps*I_ul);
        Results[0] += I;
        for (int p=1; p<numRegs; p++) {
            I = fmaf(delta[p],M[p-1],eps*I);
            Results[p] += I;
        }


        if (group_id == result_thread) {
            float temp_res = Results[result_reg];
            temp_res =  log10f(temp_res) - log10f(constant);
            if ((blockDim.x/group_size)*blid + group_nr < reads_in_batch) devAlignmentScores[read_nr*haps_in_batch+i] = temp_res;
        //if (blid == 0) // if ((blid == 0) && (!group_nr))
        //           printf("Results in Block: %d, in thread: %d, in Register: %d: %f\n", blid, result_thread, result_reg, temp_res);

        }
    }


}


struct batch{

    std::vector<uint8_t> reads;             //all reads padded
    std::vector<uint8_t> haps;              //all haplotypes padded

    std::vector<uint8_t> base_quals;        //base_qual - offset(33)
    std::vector<uint8_t> ins_quals;         //ins_qual - offset(33)
    std::vector<uint8_t> del_quals;         //del_qual - offset(33)
    std::vector<uint8_t> gcp_quals;         //gep_qual - offset(33)

    std::vector<int> batch_haps;            //number of haplotypes per batch
    std::vector<int> batch_reads;           //number of reads/qualities per batch

    std::vector<int> batch_haps_offsets;    //offsets of reads/qualities between batches
    std::vector<int> batch_reads_offsets;   //offsets of haplotypes between batches

    std::vector<int> readlen;               //length of each read/quality
    std::vector<int> haplen;                //length of each haplotype

    std::vector<int> read_offsets;          //offset between reads/qualities
    std::vector<int> hap_offsets;           //offset between hyplotypes

    int lastreadoffset;
    int lasthapoffset;


    /*
    std::vector<float> base_quals;
    std::vector<float> ins_quals;
    std::vector<float> del_quals;
    std::vector<float> gcp_quals;
    */
};

void generate_batch_offsets(batch& batch_){

    int num_batches = batch_.batch_reads.size();

    int batch_readoffset = 0;
    int batch_hapoffset = 0;

    batch_.batch_haps_offsets.reserve(num_batches+1);
    batch_.batch_haps_offsets.push_back(batch_hapoffset);

    batch_.batch_reads_offsets.reserve(num_batches+1);
    batch_.batch_reads_offsets.push_back(batch_hapoffset);

    for (int i=0; i<num_batches; i++){

        batch_readoffset += batch_.batch_reads[i];
        batch_hapoffset += batch_.batch_haps[i];

        batch_.batch_reads_offsets.push_back(batch_readoffset);
        batch_.batch_haps_offsets.push_back(batch_hapoffset);
    }
}

void concat_batch(batch& batch1, batch& batch2) {

    batch1.reads.insert(batch1.reads.end(), batch2.reads.begin(), batch2.reads.end());
    batch1.haps.insert(batch1.haps.end(), batch2.haps.begin(), batch2.haps.end());

    batch1.base_quals.insert(batch1.base_quals.end(), batch2.base_quals.begin(), batch2.base_quals.end());
    batch1.ins_quals.insert(batch1.ins_quals.end(), batch2.ins_quals.begin(), batch2.ins_quals.end());
    batch1.del_quals.insert(batch1.del_quals.end(), batch2.del_quals.begin(), batch2.del_quals.end());
    batch1.gcp_quals.insert(batch1.gcp_quals.end(), batch2.gcp_quals.begin(), batch2.gcp_quals.end());

    batch1.batch_reads.insert(batch1.batch_reads.end(),batch2.batch_reads.begin(),batch2.batch_reads.end());
    batch1.batch_haps.insert(batch1.batch_haps.end(),batch2.batch_haps.begin(),batch2.batch_haps.end());

    batch1.readlen.insert(batch1.readlen.end(),batch2.readlen.begin(),batch2.readlen.end());
    batch1.haplen.insert(batch1.haplen.end(),batch2.haplen.begin(),batch2.haplen.end());

    batch1.read_offsets.insert(batch1.read_offsets.end(),batch2.read_offsets.begin(),batch2.read_offsets.end());
    batch1.hap_offsets.insert(batch1.hap_offsets.end(),batch2.hap_offsets.begin(),batch2.hap_offsets.end());


}

void print_batch(batch& batch_){

    int num_batches = batch_.batch_reads.size();
    int batchreadoffset = 0;
    int batchhapoffset = 0;

    std::cout << num_batches << std::endl;

    for (int i=0; i<num_batches; i++){

        batchreadoffset = batch_.batch_reads_offsets[i];
        batchhapoffset = batch_.batch_haps_offsets[i];

        std::cout << i << std::endl;
        std::cout << batch_.read_offsets[batchreadoffset] << " " << batch_.read_offsets[batchreadoffset+1] << std::endl;
        std::cout << "read: ";
        for (int j=batch_.read_offsets[batchreadoffset]; j < batch_.read_offsets[batchreadoffset+1]; j++ )
            std::cout << batch_.reads[j];

        std::cout << std::endl << "base_qual: ";
        for (int j=batch_.read_offsets[batchreadoffset]; j < batch_.read_offsets[batchreadoffset+1]; j++ )
            std::cout << batch_.base_quals[j]+33;

        std::cout << std::endl << "ins_qual: ";
        for (int j=batch_.read_offsets[batchreadoffset]; j < batch_.read_offsets[batchreadoffset+1]; j++ )
            std::cout << batch_.ins_quals[j]+33;

        std::cout << std::endl << "del_qual: ";
        for (int j=batch_.read_offsets[batchreadoffset]; j < batch_.read_offsets[batchreadoffset+1]; j++ )
            std::cout << batch_.del_quals[j]+33;

        std::cout << std::endl << "gcp_qual: ";
        for (int j=batch_.read_offsets[batchreadoffset]; j < batch_.read_offsets[batchreadoffset+1]; j++ )
            std::cout << batch_.gcp_quals[j]+33;

        std::cout << std::endl << "hap: ";
        for (int j=batch_.hap_offsets[batchhapoffset]; j < batch_.hap_offsets[batchhapoffset+1]; j++ )
            std::cout << batch_.haps[j];
        std::cout << std::endl << std::endl;
    }

}

batch read_batch(std::ifstream& file_, std::vector<float> ph2pr, int nread, int nhapl, bool print=false, int lastreadoffset=0, int lasthapoffset=0){

    batch batch_;

    char bufferchar = 'N';
    std::string linebuffer_;

    std::string read;
    std::string hap;

    std::string base_qual;
    std::string ins_qual;
    std::string del_qual;
    std::string gcp_qual;
    std::string buffer;

    int readl;
    int hapl;
    int buffersize;
    int offset = -33;

    std::size_t split;

    if(print){
        std::cout << "line: " << linebuffer_  << "\n";
        std::cout << "nread: " << nread << "\n";
        std::cout << "nhapl: " << nhapl << "\n";
    }

    int prev_pos = lastreadoffset;
    int curr_pos = lastreadoffset;;

    batch_.batch_haps.push_back(nhapl);
    batch_.batch_reads.push_back(nread);

    for (int i=0; i < nread; i++){
        getline(file_, linebuffer_);
        readl = linebuffer_.find(" ");
        buffersize = readl % 4;
        curr_pos = prev_pos + readl + 4 - buffersize;
        batch_.read_offsets.push_back(curr_pos);
        batch_.readlen.push_back(readl);
        buffer = std::string(4-buffersize, bufferchar);

        read = linebuffer_.substr(0,readl) + buffer;
        base_qual = linebuffer_.substr(readl+1,readl) + buffer;
        ins_qual = linebuffer_.substr(readl*2+2,readl) + buffer;
        del_qual = linebuffer_.substr(readl*3+3, readl) + buffer;
        gcp_qual = linebuffer_.substr(readl*4+4, readl) + buffer;

        if (print){
            std::cout << "read: " << read << "\n";
            std::cout << "base: " << base_qual << "\n";
            std::cout << "ins: " << ins_qual << "\n";
            std::cout << "del: " << del_qual  << "\n";
            std::cout << "gcp: " << gcp_qual << "\n";
            std::cout << "buffersize: " << buffersize << "\n";
            std::cout << "readl: " << batch_.read_offsets[i] << "\n";
        }

        uint8_t t = 0;
        float score = 0;
        for (int j=0; j<readl+4-buffersize; j++){

            batch_.reads.push_back(read[j]);
            batch_.base_quals.push_back(base_qual[j] + offset);
            batch_.ins_quals.push_back(ins_qual[j] + offset);
            batch_.del_quals.push_back(del_qual[j] + offset);
            batch_.gcp_quals.push_back(gcp_qual[j] + offset);

            /*
            batch_.reads.push_back(read[j]);
            batch_.base_quals.push_back(ph2pr[base_qual[j] + offset]);
            batch_.ins_quals.push_back(ph2pr[ins_qual[j] + offset]);
            batch_.del_quals.push_back(ph2pr[del_qual[j] + offset]);
            batch_.gcp_quals.push_back(ph2pr[gcp_qual[j] + offset]);


            t=ins_qual[j] + offset;
            score = ph2pr[t];
            std::cout << ins_qual[j] << " --> t: " << t << " score: " << score << "\n";
            */
        }
        prev_pos = curr_pos;
    }
    batch_.lastreadoffset = curr_pos;

    prev_pos = lasthapoffset;
    curr_pos = lasthapoffset;
    for (int i=0; i < nhapl; i++){
        getline(file_, linebuffer_);
        hapl = linebuffer_.length();
        buffersize = hapl % 4;
        curr_pos = prev_pos + hapl + 4 - buffersize;
        buffer = std::string(4-buffersize, bufferchar);
        batch_.hap_offsets.push_back(curr_pos);
        batch_.haplen.push_back(hapl);

        hap = linebuffer_ + buffer;

        for (int j=0; j<hapl+4-buffersize; j++){

            batch_.haps.push_back(hap[j]);

        }
        if (print)
            std::cout << "hap: " << hap << batch_.haps.size() <<"\n";
        prev_pos = curr_pos;
    }
    batch_.lasthapoffset = curr_pos;

    return batch_;

}

std::ifstream start_stream(std::string const& filename){

    std::ifstream file_;

    if (!filename.empty()) {
        file_.open(filename);

        if (!file_.good()) {
            std::cout << "can't open file " + filename + "\n";
        }
    }
    else {
        std::cout << "no filename was given";
    }

    return file_;


}

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

float transform_quality(std::vector<float>& ph2pr, int8_t char_quality){

    /*

    As seen in: https://github.com/MauricioCarneiro/PairHMM/blob/master/src/pairhmm_impl.h line:116

    */

    return ph2pr[char_quality];
}

double align_host(
    uint8_t* hap_bases,
    int hap_length,
    uint8_t* read_bases,
    int read_length,
    uint8_t* base_quals,
    uint8_t* ins_quals,
    uint8_t* del_quals,
    uint8_t gcp_qual,
    float* ph2pr  // constant
) {
    double result = 0;
    Context<float> ctx;

    const double constant = std::numeric_limits<float>::max() / 16;

    float *M = new float[2*(hap_length+1)];
    float *I = new float[2*(hap_length+1)];
    float *D = new float[2*(hap_length+1)];
    int target_row, source_row;

    for (int col = 0; col<=hap_length; col++) {
        M[col] = I[col] = 0;
        D[col] = constant/hap_length;
    }

    float beta = 1.0 - ph2pr[gcp_qual];
    float eps = ph2pr[gcp_qual];

    for (int row = 1; row<=read_length; row++) {
        target_row = row & 1;
        source_row = !target_row;
        M[target_row*(hap_length+1)] = I[target_row*(hap_length+1)] = 0;
        D[target_row*(hap_length+1)] = 0;
        uint8_t read_base = read_bases[row-1];
        float base_qual = ph2pr[base_quals[row-1]];
        float alpha = 1.0 - (ph2pr[ins_quals[row-1]] + ph2pr[del_quals[row-1]]);
        //float alpha = ctx.set_mm_prob(ins_quals[row-1], del_quals[row-1]);
        float delta = ph2pr[ins_quals[row-1]];
        float sigma = ph2pr[del_quals[row-1]];

        for (int col=1; col<=hap_length; col++) {
            float lambda = ((read_base == hap_bases[col-1]) || (read_base == 'N') || (hap_bases[col-1] == 'N')) ?  1.0 - base_qual : base_qual/3.0;
            M[target_row*(hap_length+1)+col] = lambda*(alpha*M[source_row*(hap_length+1)+col-1] + beta*(I[source_row*(hap_length+1)+col-1]+D[source_row*(hap_length+1)+col-1]));
            I[target_row*(hap_length+1)+col] = delta*M[source_row*(hap_length+1)+col] + eps*I[source_row*(hap_length+1)+col];
            D[target_row*(hap_length+1)+col] = sigma*M[target_row*(hap_length+1)+col-1] + eps*D[target_row*(hap_length+1)+col-1];
            //if ((row == 1) && (col == 1)) {
                //std::cout << "read_pos: " << row << " read_base: " << read_base << " hap_pos: " << col << " hap_base: " << hap_bases[col-1] << "\n";
                //std::cout << "lambda " << lambda << "\n";
                //std::cout << "M-ul: " << M[source_row*(hap_length+1)+col-1] << " alpha: " << lambda << " I-ul: " << I[source_row*(hap_length+1)+col-1] << " beta " << beta << " D-ul: " << D[source_row*(hap_length+1)+col-1] << "\n";
                //std::cout << "M " << M[target_row*(hap_length+1)+col] << "\n";
                //std::cout << "M-u: " << M[source_row*(hap_length+1)+col]  << " delta: " << delta << " I-u: " << I[source_row*(hap_length+1)+col] << " eps " << eps << "\n";
                //std::cout << "I " << I[target_row*(hap_length+1)+col] << "\n";
                //std::cout << "M-l: " << M[target_row*(hap_length+1)+col-1] << " sigma: " << sigma << " D-l: " << D[target_row*(hap_length+1)+col-1] << " eps " << eps << "\n";
                //std::cout << "D " << D[target_row*(hap_length+1)+col] << "\n";
            //}
        }
        //if ((row <= 40) && (row > 32)) {
            //std::cout << "M1: " << M[target_row*(hap_length+1)+1] << " D1: " << D[target_row*(hap_length+1)+1] << " I1: " << I[target_row*(hap_length+1)+1];
            //std::cout << "   M2: " << M[target_row*(hap_length+1)+2] << " D2: " << D[target_row*(hap_length+1)+2] << " I2: " << I[target_row*(hap_length+1)+2] << "\n";
            //std::cout << "   M3: " << M[target_row*(hap_length+1)+3] << " D3: " << D[target_row*(hap_length+1)+3] << " I3: " << I[target_row*(hap_length+1)+3] << "\n";
            //std::cout << "   M4: " << M[target_row*(hap_length+1)+4] << " D4: " << D[target_row*(hap_length+1)+4] << " I4: " << I[target_row*(hap_length+1)+4] << "\n";
            //std::cout << "   M12: " << M[target_row*(hap_length+1)+12] << " D12: " << D[target_row*(hap_length+1)+12] << " I11: " << I[target_row*(hap_length+1)+11] << "\n";
            //std::cout << "M38: " << M[target_row*(hap_length+1)+hap_length-3] << " D38: " << D[target_row*(hap_length+1)+hap_length-3] << " I37: " << I[target_row*(hap_length+1)+hap_length-4];
            //std::cout << "  M37: " << M[target_row*(hap_length+1)+hap_length-4] << " D37: " << D[target_row*(hap_length+1)+hap_length-4] << " I36: " << I[target_row*(hap_length+1)+hap_length-5] << "\n";
        //}
    }
    for (int col=1; col<=hap_length; col++) result += M[target_row*(hap_length+1)+col] + I[target_row*(hap_length+1)+col];
    //std::cout << " score: " << result << "\n";

    result =  log10(result) - log10(constant);

    return result;

}

__global__
void convert_DNA(
    uint8_t * devChars,
    const int size) {

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;

    for(int i = tid; i < size; i += stride){
        uint8_t AA = devChars[i];
        if (AA == 'A') AA = 0;
        else if (AA == 'C') AA = 1;
        else if (AA == 'G') AA = 2;
        else if (AA == 'T') AA = 3;
        else AA = 4;
        devChars[i] = AA;
    }
}


//std::vector<float> align_all_host (batch& batch_){
void align_all_host (batch& batch_,std::vector<float>& ph2pr){
    //std::vector<float> results;

    int counter = 0;
    for (int i=0; i<1; i++) {    //for (int i=0; i<batch_.batch_haps.size(); i++) { // for (int i=0; i<batch_.batch_haps.size(); i++) { //for (int i=0; i<1; i++) {
        std::cout << "Batch: " << i << " #Haplotypes: " << batch_.batch_haps[i] << " #Reads: " << batch_.batch_reads[i] << "\n";
        for (int j=0; j<batch_.batch_haps[i]; j++)
        for (int k=0; k<batch_.batch_reads[i]; k++)  {
        //    for (int j=0; j<batch_.batch_haps[i]; j++) {    //for (int j=0; j<8; j++) { // for (int j=0; j<batch_.batch_haps[i]; j++) {
                counter++;
                int read = batch_.batch_reads_offsets[i]+k;
                int read_len = batch_.readlen[read];
                int hap = batch_.batch_haps_offsets[i]+j;
                int hap_len = batch_.haplen[hap];
                int h_off = batch_.hap_offsets[hap];
                int r_off = batch_.read_offsets[read];
                double score = align_host(&batch_.haps[h_off],hap_len,&batch_.reads[r_off],read_len,&batch_.base_quals[r_off],&batch_.ins_quals[r_off],&batch_.del_quals[r_off],batch_.gcp_quals[r_off],&ph2pr[0]);
                std::cout << counter << " Haplo " << hap << " len: " << hap_len <<  " Read " << read << " len: " << read_len << " score: " << score << "\n";
                //std::cout << counter << " Haplo " << hap << " len: " << hap_len <<  " offset: " << h_off <<  " Read " << read << " len: " << read_len << " offset: " << r_off << " score: " << score << "\n";
            }
    }
}

int main(const int argc, char const * const argv[])
{

    std::string const filename = "mini.in";
    std::ifstream file_;
    std::string linebuffer_;

    int nread;
    int nhapl;

    uint64_t dp_cells = 0;

   const int MAX_PH2PR_INDEX = 128;

    std::size_t split;

    batch result;


    result.read_offsets.push_back(0);
    result.hap_offsets.push_back(0);

    std::vector<float> ph2pr =  generate_ph2pr(MAX_PH2PR_INDEX);

    if (argc > 1){
        file_ = start_stream(argv[1]);
    }
    else {
        file_ = start_stream(filename);
    }


    int counter = 0;
    int lasthap = 0;
    int lastread = 0;

    while (!file_.eof()){

        getline(file_, linebuffer_);
        if (file_.good()){
            split = linebuffer_.find(" ");
            nread = stoi(linebuffer_.substr(0,split));
            nhapl = stoi(linebuffer_.substr(split));

            batch test = read_batch(file_, ph2pr, nread, nhapl, false, lastread, lasthap);
            lasthap = test.lasthapoffset;
            lastread = test.lastreadoffset;
            concat_batch(result, test);

            counter++;
            // std::cout << "batch number: " << counter << "\n";
        }
    }

    generate_batch_offsets(result);
    //print_batch(result); // prints first reads/qualities and haplotype per batch.
    align_all_host(result,ph2pr);

    uint8_t* read_chars       = result.reads.data(); //batch_2.chars.data();
    const uint read_bytes = result.reads.size();
    uint8_t* hap_chars       = result.haps.data(); //batch_2.chars.data();
    const uint hap_bytes = result.haps.size();
    uint8_t* base_qual       = result.base_quals.data(); //batch_2.chars.data();
    uint8_t* ins_qual       = result.ins_quals.data(); //batch_2.chars.data();
    uint8_t* del_qual       = result.del_quals.data(); //batch_2.chars.data();
    int* offset_reads       = result.read_offsets.data(); //batch_2.chars.data();
    int* offset_haps       = result.hap_offsets.data(); //batch_2.chars.data();
    int* read_len       = result.readlen.data(); //batch_2.chars.data();
    int* hap_len       = result.haplen.data(); //batch_2.chars.data();
    int num_reads = result.readlen.size(); //batch_2.chars.data();
    int num_haps = result.haplen.size(); //batch_2.chars.data();
    int num_batches = result.batch_reads.size(); //batch_2.chars.data();
    int* hap_batches       = result.batch_haps.data(); //batch_2.chars.data();
    int* read_batches       = result.batch_reads.data(); //batch_2.chars.data();
    int* offset_hap_batches       = result.batch_haps_offsets.data(); //batch_2.chars.data();
    int* offset_read_batches       = result.batch_reads_offsets.data(); //batch_2.chars.data();

    counter = 0;
    for (int i=0; i<num_batches; i++) counter += read_batches[i] * hap_batches[i];
    const int totalNumberOfAlignments = counter;

    float* alignment_scores_float = nullptr;
    alignment_scores_float = (float *) malloc(sizeof(float)*counter);
    std::cout << "Calculating:  " << counter << " alignments in " << num_batches << " batches \n";
    std::cout << "read_bytes:  " << read_bytes << ", hap_bytes  " << hap_bytes << " \n";
    std::cout << "num_reads:  " << num_reads << ", num_haps  " << num_haps << " \n";


    //determine max lengths in each batch_hatch

    uint64_t *max_read_len = new uint64_t[num_batches];
    uint64_t *max_hap_len = new uint64_t[num_batches];
    uint64_t *avg_read_len = new uint64_t[num_batches];
    uint64_t *avg_hap_len = new uint64_t[num_batches];
    //uint64_t max_read_len[num_batches], max_hap_len[num_batches], avg_read_len[num_batches], avg_hap_len[num_batches];

    //uint64_t dp_cells_avg = 0;
    uint64_t dp_cells_max = 0;
    for (int i=0; i<num_batches; i++) {
        max_read_len[i] = avg_read_len[i] = 0;

        for (int k=0; k<read_batches[i]; k++) {
            int read = offset_read_batches[i]+k;
            int rl= read_len[read];
            if (rl > max_read_len[i]) max_read_len[i] = rl;
            avg_read_len[i] += rl;
        }
        //std::cout << "Batch: " << i << " after first loop \n";
        avg_read_len[i] = avg_read_len[i] / read_batches[i];
        max_hap_len[i] = avg_hap_len[i] = 0;

        for (int k=0; k<hap_batches[i]; k++) {
            int hap = offset_hap_batches[i]+k;
            int hl = hap_len[hap];
            if (hl > max_hap_len[i]) max_hap_len[i] = hl;
            avg_hap_len[i] += hl;
        }
        //std::cout << "Batch: " << i << " after second loop \n";
        avg_hap_len[i] = avg_hap_len[i] / hap_batches[i];
        //for (int k=0; k<read_batches[i]; k++)
        //    for (int j=0; j<hap_batches[i]; j++)
        //       dp_cells += max_read_len[i] * max_hap_len[i]; // [offset_read_batches[i]+k] * hap_len[offset_hap_batches[i]+j];
        uint64_t temp_dp = read_batches[i] * hap_batches[i];
        dp_cells_max += temp_dp * max_read_len[i] * avg_hap_len[i];
        dp_cells += temp_dp * avg_read_len[i] * avg_hap_len[i];
        //std::cout << "Batch: " << i << " #Reads: " << read_batches[i] << " Max_rl: " << max_read_len[i] << " Avg_rl: " << avg_read_len[i] << " #Haps: " << hap_batches[i] << " Max_hl: " << max_hap_len[i] << " avg_hl: " << avg_hap_len[i] << "\n";

    }
    int overall_max_rl = 0;
    int overall_max_hl = 0;
    for (int i=0; i<num_batches; i++) {
        //std::cout << "Batch: " << i << " #Reads: " << read_batches[i] << " Max_rl: " << max_read_len[i] << " Avg_rl: " << avg_read_len[i] << " #Haps: " << hap_batches[i] << " Max_hl: " << max_hap_len[i] << " avg_hl: " << avg_hap_len[i] << "\n";
        if (overall_max_rl < max_read_len[i]) overall_max_rl = max_read_len[i];
        if (overall_max_hl < max_hap_len[i]) overall_max_hl = max_hap_len[i];
    }
    std::cout << "dp_cells: " << dp_cells << " dp_cells_max: " << dp_cells_max << " Max_read_len: " << overall_max_rl << " max_hap_len: " << overall_max_hl << "\n";

    uint8_t* dev_read_chars = nullptr;
    cudaMalloc(&dev_read_chars, read_bytes); CUERR
    uint8_t* dev_hap_chars = nullptr;
    cudaMalloc(&dev_hap_chars, hap_bytes); CUERR
    uint8_t* dev_base_qual = nullptr;
    cudaMalloc(&dev_base_qual, read_bytes); CUERR
    uint8_t* dev_ins_qual = nullptr;
    cudaMalloc(&dev_ins_qual, read_bytes); CUERR
    uint8_t* dev_del_qual = nullptr;
    cudaMalloc(&dev_del_qual, read_bytes); CUERR
    int* dev_offset_reads = nullptr;
    cudaMalloc(&dev_offset_reads, num_reads*sizeof(int)); CUERR
    int* dev_offset_haps = nullptr;
    cudaMalloc(&dev_offset_haps, num_haps*sizeof(int)); CUERR
    int* dev_read_len = nullptr;
    cudaMalloc(&dev_read_len, num_reads*sizeof(int)); CUERR
    int* dev_hap_len = nullptr;
    cudaMalloc(&dev_hap_len, num_haps*sizeof(int)); CUERR
    int* dev_read_batches = nullptr;
    cudaMalloc(&dev_read_batches, num_batches*sizeof(int)); CUERR
    int* dev_hap_batches = nullptr;
    cudaMalloc(&dev_hap_batches, num_batches*sizeof(int)); CUERR
    int* dev_offset_read_batches = nullptr;
    cudaMalloc(&dev_offset_read_batches, num_batches*sizeof(int)); CUERR
    int* dev_offset_hap_batches = nullptr;
    cudaMalloc(&dev_offset_hap_batches, num_batches*sizeof(int)); CUERR
    //int* dev_ph2pr = nullptr;
    //cudaMalloc(&dev_ph2pr, MAX_PH2PR_INDEX*sizeof(float)); CUERR

    float* devAlignmentScoresFloat = nullptr;
    cudaMalloc(&devAlignmentScoresFloat, sizeof(float)*counter); CUERR

    //cudaStream_t streams[num_batches];
    //for (int i=0; i<num_batches; i++) cudaStreamCreate(&streams[i]);

    cudaMemcpyToSymbol(cPH2PR,ph2pr.data(),MAX_PH2PR_INDEX*sizeof(float));
    TIMERSTART_CUDA(DATA_TRANSFER)
    cudaMemcpy(dev_read_chars, read_chars, read_bytes, cudaMemcpyHostToDevice); CUERR
    cudaMemcpy(dev_hap_chars, hap_chars, hap_bytes, cudaMemcpyHostToDevice); CUERR
    cudaMemcpy(dev_base_qual, base_qual, read_bytes, cudaMemcpyHostToDevice); CUERR
    cudaMemcpy(dev_ins_qual, ins_qual, read_bytes, cudaMemcpyHostToDevice); CUERR
    cudaMemcpy(dev_del_qual, del_qual, read_bytes, cudaMemcpyHostToDevice); CUERR
    cudaMemcpy(dev_offset_reads, offset_reads, num_reads*sizeof(int), cudaMemcpyHostToDevice); CUERR
    cudaMemcpy(dev_offset_haps, offset_haps, num_haps*sizeof(int), cudaMemcpyHostToDevice); CUERR
    cudaMemcpy(dev_read_len, read_len, num_reads*sizeof(int), cudaMemcpyHostToDevice); CUERR
    cudaMemcpy(dev_hap_len, hap_len, num_haps*sizeof(int), cudaMemcpyHostToDevice); CUERR
    cudaMemcpy(dev_offset_read_batches, offset_read_batches, num_batches*sizeof(int), cudaMemcpyHostToDevice); CUERR
    cudaMemcpy(dev_offset_hap_batches, offset_hap_batches, num_batches*sizeof(int), cudaMemcpyHostToDevice); CUERR
    cudaMemcpy(dev_read_batches, read_batches, num_batches*sizeof(int), cudaMemcpyHostToDevice); CUERR
    cudaMemcpy(dev_hap_batches, hap_batches, num_batches*sizeof(int), cudaMemcpyHostToDevice); CUERR
    TIMERSTOP_CUDA(DATA_TRANSFER)

    //print_batch(result); // prints first reads/qualities and haplotype per batch.

    convert_DNA<<<num_reads, 128>>>(dev_read_chars,read_bytes);
    convert_DNA<<<num_haps, 128>>>(dev_hap_chars,hap_bytes);

    // cub::DeviceSegmentedSort::SortPairs(
    //     void *d_temp_storage,
    //     std::size_t &temp_storage_bytes,
    //     const KeyT *d_keys_in,
    //     KeyT *d_keys_out,
    //     const ValueT *d_values_in,
    //     ValueT *d_values_out,
    //     num_reads,
    //     num_batches,
    //     BeginOffsetIteratorT d_begin_offsets,
    //     EndOffsetIteratorT d_end_offsets,
    //     cudaStream_t stream = 0
    // );

    thrust::device_vector<int> d_numIndicesPerPartitionPerBatch(num_batches * numPartitions, 0);
    thrust::device_vector<int> d_indicesPerPartitionPerBatch(num_reads * numPartitions, -1);

    partitionIndicesKernel<<<num_batches, 128>>>(
        d_numIndicesPerPartitionPerBatch.data().get(),
        d_indicesPerPartitionPerBatch.data().get(),
        dev_read_len,
        dev_offset_read_batches,
        num_batches,
        num_reads
    );
    CUERR;

    thrust::device_vector<int> d_resultOffsetsPerBatch(num_batches);
    thrust::transform(
        thrust::cuda::par_nosync.on((cudaStream_t)0),
        dev_read_batches,
        dev_read_batches + num_batches,
        dev_hap_batches,
        d_resultOffsetsPerBatch.begin(),
        thrust::multiplies<int>{}
    );
    thrust::exclusive_scan(
        thrust::cuda::par_nosync.on((cudaStream_t)0),
        d_resultOffsetsPerBatch.begin(),
        d_resultOffsetsPerBatch.begin() + num_batches,
        d_resultOffsetsPerBatch.begin()
    );

    for(int i = 0; i < std::min(10, num_batches); i++){
        std::cout << "batch " << i << ", num reads: " << read_batches[i]
            << ", num haps " << hap_batches[i] << ", product: " <<
            read_batches[i] * hap_batches[i]
            << ", offset : " << d_resultOffsetsPerBatch[i] << "\n";
    }


    thrust::device_vector<int> d_numWarpsPerBatchPerPartition(num_batches * numPartitions, 0);
    thrust::device_vector<int> d_numWarpsPerBatchInclusivePrefixSumPerPartition(num_batches * numPartitions, 0);
    thrust::device_vector<int> d_numWarpsPerPartition(numPartitions);

    const int groups_per_warp[numPartitions] = {4,4,2,4,2,2,2,2,2,2};
    for(int p = 0; p < numPartitions; p++){
        cudaStream_t stream = 0;

        const int alignmentsPerWarp = groups_per_warp[p];

        computeNumWarpsPerBatchKernel<<<SDIV(num_batches, 128), 128, 0, stream>>>(
            d_numWarpsPerBatchPerPartition.data().get() + p * num_batches,
            d_numIndicesPerPartitionPerBatch.data().get() + p * num_batches,
            num_batches,
            alignmentsPerWarp
        );

        thrust::inclusive_scan(
            thrust::cuda::par_nosync.on(stream),
            d_numWarpsPerBatchPerPartition.begin() + p * num_batches,
            d_numWarpsPerBatchPerPartition.begin() + (p+1) * num_batches,
            d_numWarpsPerBatchInclusivePrefixSumPerPartition.begin() + p * num_batches
        );
    }
    gatherNumWarpsPerPartitionFromInclPrefixSumKernel<<<SDIV(numPartitions, 128), 128>>>(
        d_numWarpsPerPartition.data().get(),
        d_numWarpsPerBatchInclusivePrefixSumPerPartition.data().get(),
        num_batches,
        numPartitions
    ); CUERR;



    thrust::host_vector<int> h_numIndicesPerPartitionPerBatch = d_numIndicesPerPartitionPerBatch;
    thrust::host_vector<int> h_indicesPerPartitionPerBatch = d_indicesPerPartitionPerBatch;
    thrust::host_vector<int> h_numWarpsPerBatchPerPartition = d_numWarpsPerBatchPerPartition;
    thrust::host_vector<int> h_numWarpsPerBatchInclusivePrefixSumPerPartition = d_numWarpsPerBatchInclusivePrefixSumPerPartition;
    thrust::host_vector<int> h_numWarpsPerPartition = d_numWarpsPerPartition;
#if 0
    std::cout << "warps per partition: ";
    for(int p = 0; p < numPartitions; p++){
        std::cout << h_numWarpsPerPartition[p] << " ";
    }
    std::cout << "\n";



    for(int p = 0; p < numPartitions; p++){
        if(p <= 4){
            std::cout << "Partition p = " << p << "\n";
            std::cout << "numIndicesPerBatch: ";
            for(int b = 0; b < 100; b++){ // or(int b = 0; b < num_batches; b++){
                std::cout << h_numIndicesPerPartitionPerBatch[p * num_batches + b] << ", ";
            }
            std::cout << "\n";

            std::cout << "indicesPerBatch: ";
            for(int b = 0; b < 100; b++){ // for(int b = 0; b < num_batches; b++){
                const int num = h_numIndicesPerPartitionPerBatch[p * num_batches + b];
                for(int i = 0; i < num; i++){
                    std::cout << h_indicesPerPartitionPerBatch[p * num_reads + offset_read_batches[b] + i];
                    if(i != num-1){
                        std::cout << ", ";
                    }
                }
                std::cout << " | ";
            }
            std::cout << "\n";

            std::cout << "numWarpsPerBatch: ";
            for(int b = 0; b < 100; b++){ // for(int b = 0; b < num_batches; b++){
                std::cout << h_numWarpsPerBatchPerPartition[p * num_batches + b] << ", ";
            }
            std::cout << "\n";
            std::cout << "numWarpsPerBatchInclPrefixSum: ";
            for(int b = 0; b < 100; b++){ // for(int b = 0; b < num_batches; b++){
                std::cout << h_numWarpsPerBatchInclusivePrefixSumPerPartition[p * num_batches + b] << ", ";
            }
            std::cout << "\n";
        }
    }
    #endif

/*    for(int p = 0; p < numPartitions; p++){
        if(p == 1){
            cudaStream_t stream = 0;

            const int blocksize = 32;
            const int numWarpsRequired = h_numWarpsPerPartition[p];
            if(numWarpsRequired > 0){
                const int numBlocksRequired = SDIV(numWarpsRequired, (blocksize / 32));

                std::cout << "Kernel for partition p = " << p << ". numWarpsRequired = " << numWarpsRequired << "\n";

                processKernel<<<numBlocksRequired, blocksize, 0, stream>>>(
                    d_numIndicesPerPartitionPerBatch.data().get() + p * num_batches,
                    d_indicesPerPartitionPerBatch.data().get() + p * num_reads,
                    d_numWarpsPerBatchPerPartition.data().get() + p * num_batches,
                    d_numWarpsPerBatchInclusivePrefixSumPerPartition.data().get() + p * num_batches,
                    dev_offset_read_batches,
                    num_batches,
                    groups_per_warp[p],
                    32/groups_per_warp[p]
                ); CUERR;
                cudaDeviceSynchronize(); CUERR;
            }
        }
    }

*/
    int res_off = 0;

/*
    std::cout << "batch 0: " << read_batches[0] << "\n";
    // 64 length , 1 warp with 4 groups

    TIMERSTART_CUDA(PAIR_HMM)
    for (int i=0; i<num_batches; i++) {
        int b_r_off = offset_read_batches[i];
        int b_h_off = offset_hap_batches[i];
        int r_off = offset_reads[b_r_off];
        int h_off = offset_haps[b_h_off];
        //std::cout << i << " batch_read_off: " << b_r_off << " read_off: " << r_off <<  " batch_hap_off: " << b_h_off << " hap_off: " << h_off << " res_off: " << res_off <<"\n";
        //std::cout << i << " read_bytes: " << read_bytes << " hap_bytes: " << hap_bytes <<"\n";
        //if (max_read_len[i] <= 32) PairHMM_align<2, 16><<<(read_batches[i]+15)/16, 32, 0, streams[i]>>>(&(dev_read_chars[r_off]), &(dev_hap_chars[h_off]), &(dev_base_qual[r_off]), &(dev_ins_qual[r_off]), &(dev_del_qual[r_off]), &(devAlignmentScoresFloat[res_off]), &(dev_offset_reads[b_r_off]), &(dev_offset_haps[b_h_off]), &(dev_read_len[b_r_off]), &(dev_hap_len[b_h_off]), read_batches[i], hap_batches[i]);
        //else if (max_read_len[i] <= 40) PairHMM_align<2, 20><<<(read_batches[i]+15)/16, 32, 0, streams[i]>>>(&(dev_read_chars[r_off]), &(dev_hap_chars[h_off]), &(dev_base_qual[r_off]), &(dev_ins_qual[r_off]), &(dev_del_qual[r_off]), &(devAlignmentScoresFloat[res_off]), &(dev_offset_reads[b_r_off]), &(dev_offset_haps[b_h_off]), &(dev_read_len[b_r_off]), &(dev_hap_len[b_h_off]), read_batches[i], hap_batches[i]);
        if (max_read_len[i] <= 48) PairHMM_align<2, 48><<<(read_batches[i]+15)/16, 32, 0, streams[i]>>>(&(dev_read_chars[r_off]), &(dev_hap_chars[h_off]), &(dev_base_qual[r_off]), &(dev_ins_qual[r_off]), &(dev_del_qual[r_off]), &(devAlignmentScoresFloat[res_off]), &(dev_offset_reads[b_r_off]), &(dev_offset_haps[b_h_off]), &(dev_read_len[b_r_off]), &(dev_hap_len[b_h_off]), read_batches[i], hap_batches[i]);
        else if (max_read_len[i] <= 64) PairHMM_align<4, 16><<<(read_batches[i]+7)/8, 32, 0, streams[i]>>>(&(dev_read_chars[r_off]), &(dev_hap_chars[h_off]), &(dev_base_qual[r_off]), &(dev_ins_qual[r_off]), &(dev_del_qual[r_off]), &(devAlignmentScoresFloat[res_off]), &(dev_offset_reads[b_r_off]), &(dev_offset_haps[b_h_off]), &(dev_read_len[b_r_off]), &(dev_hap_len[b_h_off]), read_batches[i], hap_batches[i]);
        else if (max_read_len[i] <= 80) PairHMM_align<4, 20><<<(read_batches[i]+7)/8, 32, 0, streams[i]>>>(&(dev_read_chars[r_off]), &(dev_hap_chars[h_off]), &(dev_base_qual[r_off]), &(dev_ins_qual[r_off]), &(dev_del_qual[r_off]), &(devAlignmentScoresFloat[res_off]), &(dev_offset_reads[b_r_off]), &(dev_offset_haps[b_h_off]), &(dev_read_len[b_r_off]), &(dev_hap_len[b_h_off]), read_batches[i], hap_batches[i]);
        else if (max_read_len[i] <= 96) PairHMM_align<4, 24><<<(read_batches[i]+7)/8, 32, 0, streams[i]>>>(&(dev_read_chars[r_off]), &(dev_hap_chars[h_off]), &(dev_base_qual[r_off]), &(dev_ins_qual[r_off]), &(dev_del_qual[r_off]), &(devAlignmentScoresFloat[res_off]), &(dev_offset_reads[b_r_off]), &(dev_offset_haps[b_h_off]), &(dev_read_len[b_r_off]), &(dev_hap_len[b_h_off]), read_batches[i], hap_batches[i]);
        else if (max_read_len[i] <= 128) PairHMM_align<8, 16><<<(read_batches[i]+3)/4, 32, 0, streams[i]>>>(&(dev_read_chars[r_off]), &(dev_hap_chars[h_off]), &(dev_base_qual[r_off]), &(dev_ins_qual[r_off]), &(dev_del_qual[r_off]), &(devAlignmentScoresFloat[res_off]), &(dev_offset_reads[b_r_off]), &(dev_offset_haps[b_h_off]), &(dev_read_len[b_r_off]), &(dev_hap_len[b_h_off]), read_batches[i], hap_batches[i]);
        else if (max_read_len[i] <= 160) PairHMM_align<8, 20><<<(read_batches[i]+3)/4, 32, 0, streams[i]>>>(&(dev_read_chars[r_off]), &(dev_hap_chars[h_off]), &(dev_base_qual[r_off]), &(dev_ins_qual[r_off]), &(dev_del_qual[r_off]), &(devAlignmentScoresFloat[res_off]), &(dev_offset_reads[b_r_off]), &(dev_offset_haps[b_h_off]), &(dev_read_len[b_r_off]), &(dev_hap_len[b_h_off]), read_batches[i], hap_batches[i]);
        else if (max_read_len[i] <= 192) PairHMM_align<8, 24><<<(read_batches[i]+3)/4, 32, 0, streams[i]>>>(&(dev_read_chars[r_off]), &(dev_hap_chars[h_off]), &(dev_base_qual[r_off]), &(dev_ins_qual[r_off]), &(dev_del_qual[r_off]), &(devAlignmentScoresFloat[res_off]), &(dev_offset_reads[b_r_off]), &(dev_offset_haps[b_h_off]), &(dev_read_len[b_r_off]), &(dev_hap_len[b_h_off]), read_batches[i], hap_batches[i]);
        else if (max_read_len[i] <= 256) PairHMM_align<16, 16><<<(read_batches[i]+1)/2, 32, 0, streams[i]>>>(&(dev_read_chars[r_off]), &(dev_hap_chars[h_off]), &(dev_base_qual[r_off]), &(dev_ins_qual[r_off]), &(dev_del_qual[r_off]), &(devAlignmentScoresFloat[res_off]), &(dev_offset_reads[b_r_off]), &(dev_offset_haps[b_h_off]), &(dev_read_len[b_r_off]), &(dev_hap_len[b_h_off]), read_batches[i], hap_batches[i]);
        else if (max_read_len[i] <= 320) PairHMM_align<16, 20><<<(read_batches[i]+1)/2, 32, 0, streams[i]>>>(&(dev_read_chars[r_off]), &(dev_hap_chars[h_off]), &(dev_base_qual[r_off]), &(dev_ins_qual[r_off]), &(dev_del_qual[r_off]), &(devAlignmentScoresFloat[res_off]), &(dev_offset_reads[b_r_off]), &(dev_offset_haps[b_h_off]), &(dev_read_len[b_r_off]), &(dev_hap_len[b_h_off]), read_batches[i], hap_batches[i]);
        else if (max_read_len[i] <= 384) PairHMM_align<16, 24><<<(read_batches[i]+1)/2, 32, 0, streams[i]>>>(&(dev_read_chars[r_off]), &(dev_hap_chars[h_off]), &(dev_base_qual[r_off]), &(dev_ins_qual[r_off]), &(dev_del_qual[r_off]), &(devAlignmentScoresFloat[res_off]), &(dev_offset_reads[b_r_off]), &(dev_offset_haps[b_h_off]), &(dev_read_len[b_r_off]), &(dev_hap_len[b_h_off]), read_batches[i], hap_batches[i]);
        //else if (max_read_len[i] <= 512) PairHMM_align<32, 16><<<read_batches[i]/1, 32, 0, streams[i]>>>(&(dev_read_chars[r_off]), &(dev_hap_chars[h_off]), &(dev_base_qual[r_off]), &(dev_ins_qual[r_off]), &(dev_del_qual[r_off]), &(devAlignmentScoresFloat[res_off]), &(dev_offset_reads[b_r_off]), &(dev_offset_haps[b_h_off]), &(dev_read_len[b_r_off]), &(dev_hap_len[b_h_off]), read_batches[i], hap_batches[i]); CUERR
        res_off += read_batches[i] * hap_batches[i];
    }
    TIMERSTOP_CUDA(PAIR_HMM)
    cudaMemcpy(alignment_scores_float, devAlignmentScoresFloat, res_off*sizeof(float), cudaMemcpyDeviceToHost);  CUERR
    cudaDeviceSynchronize();
    res_off = 0;
    for (int i=0; i<1; i++) { // for (int i=0; i<num_batches; i++) {
        cout << "Batch:" << i << " Offset: " << res_off << " results: ";
        for(int j = 0; j < 8; j++) // for(int j = 0; j < read_batches[i]; j++)
            for(int k = 0; k < hap_batches[i]; k++)
                cout << " " << alignment_scores_float[res_off+j*hap_batches[i]+k];
        cout << " \n";
        res_off += read_batches[i] * hap_batches[i];
    }
*/
    std::cout << "numPartitions: " << numPartitions << "\n";
    cudaStream_t streams_part[numPartitions];
    for (int i=0; i<numPartitions; i++) cudaStreamCreate(&streams_part[i]);

#if 1    
    cudaMemset(devAlignmentScoresFloat,0,sizeof(float)*counter);
    TIMERSTART_CUDA(PAIR_HMM_PARTITIONED)
    if (h_numWarpsPerPartition[0]) PairHMM_align_partition_half<8,8><<<h_numWarpsPerPartition[0],32,0,streams_part[0]>>>(dev_read_chars, dev_hap_chars, dev_base_qual, dev_ins_qual, dev_del_qual, devAlignmentScoresFloat, dev_offset_reads, dev_offset_haps, dev_read_len, dev_hap_len, dev_read_batches, dev_hap_batches, dev_offset_hap_batches,
        d_numIndicesPerPartitionPerBatch.data().get() + 0*num_batches, d_indicesPerPartitionPerBatch.data().get() + 0*num_reads,
        d_numWarpsPerBatchPerPartition.data().get() + 0*num_batches, d_numWarpsPerBatchInclusivePrefixSumPerPartition.data().get() + 0*num_batches,
        dev_offset_read_batches,  num_batches, d_resultOffsetsPerBatch.data().get());
    if (h_numWarpsPerPartition[1]) PairHMM_align_partition_half<8,12><<<h_numWarpsPerPartition[1],32,0,streams_part[1]>>>(dev_read_chars, dev_hap_chars, dev_base_qual, dev_ins_qual, dev_del_qual, devAlignmentScoresFloat, dev_offset_reads, dev_offset_haps, dev_read_len, dev_hap_len, dev_read_batches, dev_hap_batches, dev_offset_hap_batches,
        d_numIndicesPerPartitionPerBatch.data().get() + 1*num_batches, d_indicesPerPartitionPerBatch.data().get() + 1*num_reads,
        d_numWarpsPerBatchPerPartition.data().get() + 1*num_batches, d_numWarpsPerBatchInclusivePrefixSumPerPartition.data().get() + 1*num_batches,
        dev_offset_read_batches,  num_batches, d_resultOffsetsPerBatch.data().get());
    if (h_numWarpsPerPartition[2]) PairHMM_align_partition_half<16,8><<<h_numWarpsPerPartition[2],32,0,streams_part[2]>>>(dev_read_chars, dev_hap_chars, dev_base_qual, dev_ins_qual, dev_del_qual, devAlignmentScoresFloat, dev_offset_reads, dev_offset_haps, dev_read_len, dev_hap_len, dev_read_batches, dev_hap_batches, dev_offset_hap_batches,
        d_numIndicesPerPartitionPerBatch.data().get() + 2*num_batches, d_indicesPerPartitionPerBatch.data().get() + 2*num_reads,
        d_numWarpsPerBatchPerPartition.data().get() + 2*num_batches, d_numWarpsPerBatchInclusivePrefixSumPerPartition.data().get() + 2*num_batches,
        dev_offset_read_batches,  num_batches, d_resultOffsetsPerBatch.data().get());
    if (h_numWarpsPerPartition[3]) PairHMM_align_partition_half<8,20><<<h_numWarpsPerPartition[3],32,0,streams_part[3]>>>(dev_read_chars, dev_hap_chars, dev_base_qual, dev_ins_qual, dev_del_qual, devAlignmentScoresFloat, dev_offset_reads, dev_offset_haps, dev_read_len, dev_hap_len, dev_read_batches, dev_hap_batches, dev_offset_hap_batches,
        d_numIndicesPerPartitionPerBatch.data().get() + 3*num_batches, d_indicesPerPartitionPerBatch.data().get() + 3*num_reads,
        d_numWarpsPerBatchPerPartition.data().get() + 3*num_batches, d_numWarpsPerBatchInclusivePrefixSumPerPartition.data().get() + 3*num_batches,
        dev_offset_read_batches,  num_batches, d_resultOffsetsPerBatch.data().get());
    if (h_numWarpsPerPartition[4]) PairHMM_align_partition_half<16,16><<<h_numWarpsPerPartition[4],32,0,streams_part[4]>>>(dev_read_chars, dev_hap_chars, dev_base_qual, dev_ins_qual, dev_del_qual, devAlignmentScoresFloat, dev_offset_reads, dev_offset_haps, dev_read_len, dev_hap_len, dev_read_batches, dev_hap_batches, dev_offset_hap_batches,
        d_numIndicesPerPartitionPerBatch.data().get() + 4*num_batches, d_indicesPerPartitionPerBatch.data().get() + 4*num_reads,
        d_numWarpsPerBatchPerPartition.data().get() + 4*num_batches, d_numWarpsPerBatchInclusivePrefixSumPerPartition.data().get() + 4*num_batches,
        dev_offset_read_batches,  num_batches, d_resultOffsetsPerBatch.data().get());
    if (h_numWarpsPerPartition[5]) PairHMM_align_partition_half<16,20><<<h_numWarpsPerPartition[5],32,0,streams_part[5]>>>(dev_read_chars, dev_hap_chars, dev_base_qual, dev_ins_qual, dev_del_qual, devAlignmentScoresFloat, dev_offset_reads, dev_offset_haps, dev_read_len, dev_hap_len, dev_read_batches, dev_hap_batches, dev_offset_hap_batches,
        d_numIndicesPerPartitionPerBatch.data().get() + 5*num_batches, d_indicesPerPartitionPerBatch.data().get() + 5*num_reads,
        d_numWarpsPerBatchPerPartition.data().get() + 5*num_batches, d_numWarpsPerBatchInclusivePrefixSumPerPartition.data().get() + 5*num_batches,
        dev_offset_read_batches,  num_batches, d_resultOffsetsPerBatch.data().get());
    if (h_numWarpsPerPartition[6]) PairHMM_align_partition_half<16,24><<<h_numWarpsPerPartition[6],32,0,streams_part[6]>>>(dev_read_chars, dev_hap_chars, dev_base_qual, dev_ins_qual, dev_del_qual, devAlignmentScoresFloat, dev_offset_reads, dev_offset_haps, dev_read_len, dev_hap_len, dev_read_batches, dev_hap_batches, dev_offset_hap_batches,
        d_numIndicesPerPartitionPerBatch.data().get() + 6*num_batches, d_indicesPerPartitionPerBatch.data().get() + 6*num_reads,
        d_numWarpsPerBatchPerPartition.data().get() + 6*num_batches, d_numWarpsPerBatchInclusivePrefixSumPerPartition.data().get() + 6*num_batches,
        dev_offset_read_batches,  num_batches, d_resultOffsetsPerBatch.data().get());
//    if (h_numWarpsPerPartition[7]) PairHMM_align_partition<16,20><<<h_numWarpsPerPartition[7],32,0,streams_part[7]>>>(dev_read_chars, dev_hap_chars, dev_base_qual, dev_ins_qual, dev_del_qual, devAlignmentScoresFloat, dev_offset_reads, dev_offset_haps, dev_read_len, dev_hap_len, dev_read_batches, dev_hap_batches, dev_offset_hap_batches,
//        d_numIndicesPerPartitionPerBatch.data().get() + 7*num_batches, d_indicesPerPartitionPerBatch.data().get() + 7*num_reads,
//        d_numWarpsPerBatchPerPartition.data().get() + 7*num_batches, d_numWarpsPerBatchInclusivePrefixSumPerPartition.data().get() + 7*num_batches,
//        dev_offset_read_batches,  num_batches, d_resultOffsetsPerBatch.data().get());
//    if (h_numWarpsPerPartition[8]) PairHMM_align_partition<16,20><<<h_numWarpsPerPartition[8],32,0,streams_part[8]>>>(dev_read_chars, dev_hap_chars, dev_base_qual, dev_ins_qual, dev_del_qual, devAlignmentScoresFloat, dev_offset_reads, dev_offset_haps, dev_read_len, dev_hap_len, dev_read_batches, dev_hap_batches, dev_offset_hap_batches,
//        d_numIndicesPerPartitionPerBatch.data().get() + 8*num_batches, d_indicesPerPartitionPerBatch.data().get() + 8*num_reads,
//        d_numWarpsPerBatchPerPartition.data().get() + 8*num_batches, d_numWarpsPerBatchInclusivePrefixSumPerPartition.data().get() + 8*num_batches,
//        dev_offset_read_batches,  num_batches, d_resultOffsetsPerBatch.data().get());
    TIMERSTOP_CUDA(PAIR_HMM_PARTITIONED)
    cudaMemcpy(alignment_scores_float, devAlignmentScoresFloat, counter*sizeof(float), cudaMemcpyDeviceToHost);  CUERR
    res_off = 0;
    for (int i=0; i<1; i++) { // for (int i=0; i<num_batches; i++) {
        cout << "Batch:" << i << " Offset: " << res_off << " results: ";
        for(int j = 0; j < 8; j++) // for(int j = 0; j < read_batches[i]; j++)
            for(int k = 0; k < hap_batches[i]; k++)
                cout << " " << alignment_scores_float[res_off+j*hap_batches[i]+k];
        cout << " \n";
        res_off += read_batches[i] * hap_batches[i];
    }
#endif

    thrust::device_vector<int> d_numAlignmentsPerBatch(num_batches * numPartitions);
    thrust::device_vector<int> d_numAlignmentsPerBatchInclPrefixSum(num_batches * numPartitions);
    thrust::device_vector<int> d_numAlignmentsPerPartition(numPartitions);

    {
        int* d_numAlignmentsPerPartitionPerBatch = d_numAlignmentsPerBatchInclPrefixSum.data().get(); // reuse
        const int* d_numHaplotypesPerBatch = dev_hap_batches;
        computeAlignmentsPerPartitionPerBatch<<<dim3(SDIV(num_batches, 128), numPartitions), 128,0, (cudaStream_t)0>>>(
            d_numAlignmentsPerPartitionPerBatch,
            d_numIndicesPerPartitionPerBatch.data().get(),
            d_numHaplotypesPerBatch,
            numPartitions, 
            num_batches
        ); CUERR;

        auto offsets = thrust::make_transform_iterator(
            thrust::make_counting_iterator(0),
            [num_batches] __host__ __device__(int partition){
                return partition * num_batches;
            }
        );
        size_t temp_storage_bytes = 0;
        cub::DeviceSegmentedReduce::Sum(nullptr, temp_storage_bytes, d_numAlignmentsPerPartitionPerBatch, 
            d_numAlignmentsPerPartition.data().get(), numPartitions, offsets, offsets + 1, (cudaStream_t)0); CUERR;
        thrust::device_vector<char> d_temp(temp_storage_bytes);
        cub::DeviceSegmentedReduce::Sum(d_temp.data().get(), temp_storage_bytes, d_numAlignmentsPerPartitionPerBatch, 
            d_numAlignmentsPerPartition.data().get(), numPartitions, offsets, offsets + 1, (cudaStream_t)0); CUERR;
    }
    thrust::host_vector<int> h_numAlignmentsPerPartition = d_numAlignmentsPerPartition;

    std::cout << "h_numAlignmentsPerPartition: ";
    for(int i = 0; i< numPartitions; i++){
        std::cout << h_numAlignmentsPerPartition[i] << ", ";
    }
    std::cout << "\n";

    cudaMemset(devAlignmentScoresFloat, 0, totalNumberOfAlignments * sizeof(float)); CUERR;
    TIMERSTART_CUDA(PAIR_HMM_PARTITIONED_COMBINED)
    #define COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM(stream) \
        constexpr int groupsPerBlock = blocksize / group_size; \
        const int numAlignmentsInPartition = h_numAlignmentsPerPartition[partitionId]; \
        const int* d_numIndicesPerBatch = d_numIndicesPerPartitionPerBatch.data().get() + partitionId*num_batches; \
        thrust::transform( \
            thrust::cuda::par_nosync.on(stream), \
            d_numIndicesPerBatch, \
            d_numIndicesPerBatch + num_batches, \
            dev_hap_batches, \
            d_numAlignmentsPerBatch.begin() + partitionId * num_batches, \
            thrust::multiplies<int>{} \
        ); \
        thrust::inclusive_scan( \
            thrust::cuda::par_nosync(ThrustCudaMallocAsyncAllocator<int>(stream)).on(stream), \
            d_numAlignmentsPerBatch.begin() + partitionId * num_batches, \
            d_numAlignmentsPerBatch.begin() + partitionId * num_batches+ num_batches, \
            d_numAlignmentsPerBatchInclPrefixSum.begin() + partitionId * num_batches \
        );  \
        PairHMM_align_partition_half_allowMultipleBatchesPerWarp<group_size,numRegs><<<SDIV(numAlignmentsInPartition, groupsPerBlock), blocksize,0,stream>>>(dev_read_chars, dev_hap_chars, dev_base_qual, dev_ins_qual, dev_del_qual, devAlignmentScoresFloat, dev_offset_reads, dev_offset_haps, dev_read_len, dev_hap_len, dev_read_batches, dev_hap_batches, dev_offset_hap_batches,  \
            d_numIndicesPerBatch, d_indicesPerPartitionPerBatch.data().get() + partitionId*num_reads,  \
            dev_offset_read_batches,  num_batches, d_resultOffsetsPerBatch.data().get(), d_numAlignmentsPerBatch.data().get() + partitionId * num_batches, d_numAlignmentsPerBatchInclPrefixSum.data().get() + partitionId * num_batches, numAlignmentsInPartition); \


    if (h_numAlignmentsPerPartition[0]){
        constexpr int partitionId = 0;
        constexpr int group_size = 8;
        constexpr int numRegs = 8;
        constexpr int blocksize = 32;
        COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM(streams_part[0]);
    }
    if (h_numAlignmentsPerPartition[1]){
        constexpr int partitionId = 1;
        constexpr int group_size = 8;
        constexpr int numRegs = 12;
        constexpr int blocksize = 32;
        COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM(streams_part[1]);
    }
    if (h_numAlignmentsPerPartition[2]){
        constexpr int partitionId = 2;
        constexpr int group_size = 16;
        constexpr int numRegs = 8;
        constexpr int blocksize = 32;
        COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM(streams_part[2]);
    }
    if (h_numAlignmentsPerPartition[3]){
        constexpr int partitionId = 3;
        constexpr int group_size = 8;
        constexpr int numRegs = 20;
        constexpr int blocksize = 32;
        COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM(streams_part[3]);
    }
    if (h_numAlignmentsPerPartition[4]){
        constexpr int partitionId = 4;
        constexpr int group_size = 16;
        constexpr int numRegs = 16;
        constexpr int blocksize = 32;
        COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM(streams_part[4]);
    }
    if (h_numAlignmentsPerPartition[5]){
        constexpr int partitionId = 5;
        constexpr int group_size = 16;
        constexpr int numRegs = 20;
        constexpr int blocksize = 32;
        COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM(streams_part[5]);
    }
    if (h_numAlignmentsPerPartition[6]){
        constexpr int partitionId = 6;
        constexpr int group_size = 16;
        constexpr int numRegs = 24;
        constexpr int blocksize = 32;
        COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM(streams_part[6]);
    }

    TIMERSTOP_CUDA(PAIR_HMM_PARTITIONED_COMBINED)
    cudaMemcpy(alignment_scores_float, devAlignmentScoresFloat, counter*sizeof(float), cudaMemcpyDeviceToHost);  CUERR
    res_off = 0;
    for (int i=0; i<1; i++) { // for (int i=0; i<num_batches; i++) {
        cout << "Batch:" << i << " Offset: " << res_off << " results: ";
        for(int j = 0; j < 8; j++) // for(int j = 0; j < read_batches[i]; j++)
            for(int k = 0; k < hap_batches[i]; k++)
                cout << " " << alignment_scores_float[res_off+j*hap_batches[i]+k];
        cout << " \n";
        res_off += read_batches[i] * hap_batches[i];
    }

    cudaFree(dev_read_chars); CUERR
}
