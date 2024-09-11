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
#include "Context.h"

#include <omp.h>

#include <nvtx3/nvtx3.hpp>

using std::cout;
using std::copy;



// #define ENABLE_PEAK_BENCH_HALF
#define ENABLE_PEAK_BENCH_FLOAT



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



struct PartitionLimits{
    static constexpr int numPartitions(){
        return 10;
    }
    PartitionLimits() = default;
    #if 0
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
    #endif

    #if 1
    int boundaries[10]{
        48,
        64,
        80,
        96,
        128,
        160,
        192,
        256,
        320,
        384,
    };
    #endif
};
constexpr int numPartitions = PartitionLimits::numPartitions();

#if 1
#define LAUNCH_ALL_KERNELS \
    if (h_numAlignmentsPerPartition[0]){ \
        constexpr int partitionId = 0; \
        constexpr int group_size = 4; \
        constexpr int numRegs = 12; \
        constexpr int blocksize = 32; \
        COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM(streams_part[0]); \
    } \
    if (h_numAlignmentsPerPartition[1]){ \
        constexpr int partitionId = 1; \
        constexpr int group_size = 4; \
        constexpr int numRegs = 16; \
        constexpr int blocksize = 32; \
        COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM(streams_part[1]); \
    } \
    if (h_numAlignmentsPerPartition[2]){ \
        constexpr int partitionId = 2; \
        constexpr int group_size = 4; \
        constexpr int numRegs = 20; \
        constexpr int blocksize = 32; \
        COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM(streams_part[2]); \
    } \
    if (h_numAlignmentsPerPartition[3]){ \
        constexpr int partitionId = 3; \
        constexpr int group_size = 8; \
        constexpr int numRegs = 12; \
        constexpr int blocksize = 32; \
        COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM(streams_part[3]); \
    } \
    if (h_numAlignmentsPerPartition[4]){ \
        constexpr int partitionId = 4; \
        constexpr int group_size = 8; \
        constexpr int numRegs = 16; \
        constexpr int blocksize = 32; \
        COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM(streams_part[4]); \
    } \
    if (h_numAlignmentsPerPartition[5]){ \
        constexpr int partitionId = 5; \
        constexpr int group_size = 8; \
        constexpr int numRegs = 20; \
        constexpr int blocksize = 32; \
        COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM(streams_part[5]); \
    } \
    if (h_numAlignmentsPerPartition[6]){ \
        constexpr int partitionId = 6; \
        constexpr int group_size = 16; \
        constexpr int numRegs = 12; \
        constexpr int blocksize = 32; \
        COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM(streams_part[6]); \
    } \
    if (h_numAlignmentsPerPartition[7]){ \
        constexpr int partitionId = 7; \
        constexpr int group_size = 16; \
        constexpr int numRegs = 16; \
        constexpr int blocksize = 32; \
        COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM(streams_part[7]); \
    } \
    if (h_numAlignmentsPerPartition[8]){ \
        constexpr int partitionId = 8; \
        constexpr int group_size = 16; \
        constexpr int numRegs = 20; \
        constexpr int blocksize = 32; \
        COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM(streams_part[8]); \
    } \
    if (h_numAlignmentsPerPartition[9]){ \
        constexpr int partitionId = 9; \
        constexpr int group_size = 32; \
        constexpr int numRegs = 12; \
        constexpr int blocksize = 32; \
        COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM(streams_part[9]); \
    }
#endif


#if 0
#define LAUNCH_ALL_KERNELS \
    if (h_numAlignmentsPerPartition[0]){ \
        constexpr int partitionId = 0; \
        constexpr int group_size = 8; \
        constexpr int numRegs = 8; \
        constexpr int blocksize = 32; \
        COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM(streams_part[0]); \
    } \
    if (h_numAlignmentsPerPartition[1]){ \
        constexpr int partitionId = 1; \
        constexpr int group_size = 8; \
        constexpr int numRegs = 12; \
        constexpr int blocksize = 32; \
        COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM(streams_part[1]); \
    } \
    if (h_numAlignmentsPerPartition[2]){ \
        constexpr int partitionId = 2; \
        constexpr int group_size = 16; \
        constexpr int numRegs = 8; \
        constexpr int blocksize = 32; \
        COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM(streams_part[2]); \
    } \
    if (h_numAlignmentsPerPartition[3]){ \
        constexpr int partitionId = 3; \
        constexpr int group_size = 8; \
        constexpr int numRegs = 20; \
        constexpr int blocksize = 32; \
        COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM(streams_part[3]); \
    } \
    if (h_numAlignmentsPerPartition[4]){ \
        constexpr int partitionId = 4; \
        constexpr int group_size = 16; \
        constexpr int numRegs = 16; \
        constexpr int blocksize = 32; \
        COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM(streams_part[4]); \
    } \
    if (h_numAlignmentsPerPartition[5]){ \
        constexpr int partitionId = 5; \
        constexpr int group_size = 16; \
        constexpr int numRegs = 20; \
        constexpr int blocksize = 32; \
        COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM(streams_part[5]); \
    } \
    if (h_numAlignmentsPerPartition[6]){ \
        constexpr int partitionId = 6; \
        constexpr int group_size = 16; \
        constexpr int numRegs = 24; \
        constexpr int blocksize = 32; \
        COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM(streams_part[6]); \
    }

#endif





struct Options{
    std::string inputfile = "";
    std::string outputfile = "";
    int transferchunksize = 100000;
    bool checkResults = false;
    bool peakBenchFloat = false;
    bool peakBenchHalf = false;
};

template <class T>
struct PinnedAllocator {
    using value_type = T;

    // PinnedAllocator() = default;

    // template <class U>
    // constexpr Mallocator(const Mallocator<U>&) noexcept {}

    T* allocate(size_t elements){
        T* ptr{};
        cudaError_t err = cudaMallocHost(&ptr, elements * sizeof(T));
        if(err != cudaSuccess){
            std::cerr << "SimpleAllocator: Failed to allocate " << (elements) << " * " << sizeof(T)
                        << " = " << (elements * sizeof(T))
                        << " bytes using cudaMallocHost!\n";

            throw std::bad_alloc();
        }

        assert(ptr != nullptr);

        return ptr;
    }

    void deallocate(T* ptr, size_t /*n*/){
        cudaFreeHost(ptr);
    }
};


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
    const int* numReadsPerBatch,
    const int* numReadsPerBatchPrefixSum,
    int numBatches,
    int numReads
){
    const PartitionLimits partitionLimits;

    for(int batchId = blockIdx.x; batchId < numBatches; batchId += gridDim.x){
        const int offset = numReadsPerBatchPrefixSum[batchId];
        const int outputOffset = offset - numReadsPerBatchPrefixSum[0];
        const int numReadsInBatch = numReadsPerBatch[batchId];
        const int* readLengthsOfBatch = read_lengths + offset;

        for(int r = threadIdx.x; r < numReadsInBatch; r += blockDim.x){
            const int length = readLengthsOfBatch[r];
            for(int p = 0; p < numPartitions; p++){
                if(length <= partitionLimits.boundaries[p]){
                    const int pos = atomicAdd(&numIndicesPerPartitionPerBatch[p * numBatches + batchId], 1);
                    indicesPerPartitionPerBatch[p * numReads + outputOffset + pos] = r;
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
    
    // const int numGroupsInGrid = blockDim.x * gridDim.x / group_size;
    // for(int alignmentId = groupIdInGrid; alignmentId < numAlignments; alignmentId += numGroupsInGrid){
    const int alignmentId = groupIdInGrid;
    if(alignmentId < numAlignments){

        const int batchIdByGroupId = thrust::distance(
            numAlignmentsPerBatchInclusivePrefixSum,
            thrust::upper_bound(thrust::seq,
                numAlignmentsPerBatchInclusivePrefixSum,
                numAlignmentsPerBatchInclusivePrefixSum + numBatches,
                alignmentId
            )
        );
        const int batchId = min(batchIdByGroupId, numBatches-1);
        const int groupIdInBatch = alignmentId - (batchId == 0 ? 0 : numAlignmentsPerBatchInclusivePrefixSum[batchId-1]);
        const int hapToProcessInBatch = groupIdInBatch % haps_in_batch[batchId];
        const int readIndexToProcessInBatch = groupIdInBatch / haps_in_batch[batchId];

        const int offset_read_batches = numReadsPerBatchPrefixSum[batchId];
        const int offset_read_batches_inChunk = offset_read_batches - numReadsPerBatchPrefixSum[0];
        const int readToProcessInBatch = indicesPerBatch[offset_read_batches_inChunk + readIndexToProcessInBatch];

        const int read_nr = readToProcessInBatch;
        // const int global_read_id = read_nr + offset_read_batches;
        const int read_id_inChunk = read_nr + offset_read_batches_inChunk;


        const int byteOffsetForRead = read_offsets[read_id_inChunk];
        const int readLength = read_length[read_id_inChunk];

        const int b_h_off = offset_hap_batches[batchId];
        const int b_h_off_inChunk = b_h_off - offset_hap_batches[0];
        const int bytesOffsetForHap = hap_offsets[hapToProcessInBatch+b_h_off_inChunk];
        const char4* const HapsAsChar4 = reinterpret_cast<const char4*>(&hap_chars[bytesOffsetForHap]);
        const int haploLength = hap_length[hapToProcessInBatch+b_h_off_inChunk];

        const int resultOutputIndex = resultOffsetsPerBatch[batchId] + read_nr*haps_in_batch[batchId]+hapToProcessInBatch;
        // if(threadIdInGroup == 0){
        //     printf("group %d, batchIdByGroupId %d, batchId %d, groupIdInBatch %d, hapToProcessInBatch %d, readIndexToProcessInBatch %d, readToProcessInBatch %d, numAlignments %d\n"
        //         "resultOffsetsPerBatch %d, haps_in_batch %d, resultOutputIndex %d\n",
        //         alignmentId, batchIdByGroupId, batchId, groupIdInBatch, hapToProcessInBatch, readIndexToProcessInBatch, readToProcessInBatch, numAlignments,
        //         resultOffsetsPerBatch[batchId], haps_in_batch[batchId], resultOutputIndex);
        // }


        // if(alignmentId < 10 && group_size == 8 && numRegs == 8){
            // if(threadIdInGroup == 0){
            //     printf("group %d, batchId %d, groupIdInBatch %d, hapToProcessInBatch %d, readIndexToProcessInBatch %d, readToProcessInBatch %d, readLength %d, haploLength %d, numAlignments %d\n"
            //         "offset_read_batches %d, readToProcessInBatch %d, byteOffsetForRead %d, b_h_off %d, bytesOffsetForHap %d\n"
            //         "resultOffsetsPerBatch %d, haps_in_batch %d\n",
            //         alignmentId, batchId, groupIdInBatch, hapToProcessInBatch, readIndexToProcessInBatch, readToProcessInBatch, readLength, haploLength, numAlignments,
            //         offset_read_batches, readToProcessInBatch, byteOffsetForRead, b_h_off, bytesOffsetForHap,
            //         resultOffsetsPerBatch[batchId], haps_in_batch[batchId]);
            // }
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

                //init hap == A,C,G,T as mismatch
                for (int j=0; j<4; j++){
                    lambda_array[j][2*i+groupIdInBlock*(group_size*numRegs/2)] = temp_h3; // mismatch
                }
                //hap == N always matches
                lambda_array[4][2*i+groupIdInBlock*(group_size*numRegs/2)].x = one_half - temp_h2.x; // match
                lambda_array[4][2*i+groupIdInBlock*(group_size*numRegs/2)].y = one_half - temp_h2.y; // match

                if (temp1.x < 4){
                    // set hap == read
                    lambda_array[temp1.x][2*i+groupIdInBlock*(group_size*numRegs/2)].x = one_half - temp_h2.x; // match
                }else if (temp1.x == 4){
                    // read == N always matches
                    for (int j=0; j<4; j++){
                        lambda_array[j][2*i+groupIdInBlock*(group_size*numRegs/2)].x = one_half - temp_h2.x; // N always match
                    }
                }
                if (temp1.y < 4){
                    // set hap == read
                    lambda_array[temp1.y][2*i+groupIdInBlock*(group_size*numRegs/2)].y = one_half - temp_h2.y; // match
                }else if (temp1.y == 4){
                    // read == N always matches
                    for (int j=0; j<4; j++){
                        lambda_array[j][2*i+groupIdInBlock*(group_size*numRegs/2)].y = one_half - temp_h2.y; // N always match
                    }
                }

                temp_h2.x = cPH2PR[uint8_t(temp0.z)];
                temp_h2.y = cPH2PR[uint8_t(temp0.w)];
                temp_h3.x = temp_h2.x/three;
                temp_h3.y = temp_h2.y/three;

                //init hap == A,C,G,T as mismatch
                for (int j=0; j<4; j++){
                    lambda_array[j][2*i+1+groupIdInBlock*(group_size*numRegs/2)] = temp_h3; // mismatch
                }
                //hap == N always matches
                lambda_array[4][2*i+1+groupIdInBlock*(group_size*numRegs/2)].x = one_half - temp_h2.x; // match
                lambda_array[4][2*i+1+groupIdInBlock*(group_size*numRegs/2)].y = one_half - temp_h2.y; // match

                if (temp1.z < 4){
                    // set hap == read
                    lambda_array[temp1.z][2*i+1+groupIdInBlock*(group_size*numRegs/2)].x = one_half - temp_h2.x; // match
                }else if (temp1.z == 4){
                    // read == N always matches
                    for (int j=0; j<4; j++){
                        lambda_array[j][2*i+1+groupIdInBlock*(group_size*numRegs/2)].x = one_half - temp_h2.x; // N always match
                    }
                }
                if (temp1.w < 4){
                    // set hap == read
                    lambda_array[temp1.w][2*i+1+groupIdInBlock*(group_size*numRegs/2)].y = one_half - temp_h2.y; // match
                }else if (temp1.w == 4){
                    // read == N always matches
                    for (int j=0; j<4; j++){
                        lambda_array[j][2*i+1+groupIdInBlock*(group_size*numRegs/2)].y = one_half - temp_h2.y; // N always match
                    }
                }
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
            if (!threadIdInGroup) hap_letter = new_hap_letter4.z;
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
            devAlignmentScores[resultOutputIndex] = temp_res;
        }
    }


}



template <int group_size, int numRegs> 
__global__
void PairHMM_align_partition_half_allowMultipleBatchesPerWarp_coalesced_smem(
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
    static_assert(numRegs % 4 == 0);
    static_assert(group_size >= 2);

    constexpr int blocksize = 32;
    constexpr int warpsize = 32;
    constexpr int numGroupsPerBlock = blocksize / group_size;

    constexpr int rowsize = numGroupsPerBlock * group_size*numRegs;
    constexpr int rowsizePadded = SDIV(rowsize, 8) * 8; //pad to 16 bytes
    alignas(16) __shared__ half lambda_array_permuted[5][rowsizePadded];

    // constexpr int rowsize = numGroupsPerBlock * group_size*SDIV(numRegs, 8) * 8;
    // constexpr int rowsizePadded = SDIV(rowsize, 8) * 8; //pad to 16 bytes
    // alignas(16) __shared__ half lambda_array_permuted[5][rowsizePadded];


    float M[numRegs], I, D[numRegs];
    half2 alpha[numRegs/2], delta[numRegs/2], sigma[numRegs/2];
    alignas(8) float Results[numRegs]; //align to float2

    const int threadIdInGroup = threadIdx.x % group_size;
    // const int groupIdInBlock = threadIdx.x / group_size;
    const int threadIdInWarp = threadIdx.x % warpsize;
    const int groupIdInWarp = threadIdInWarp / group_size;
    const int groupIdInGrid = (threadIdx.x + blockIdx.x * blockDim.x) / group_size;
    const unsigned int myGroupMask = __match_any_sync(0xFFFFFFFF, groupIdInGrid); //compute mask for all threads with same groupIdInGrid
    
    // const int numGroupsInGrid = blockDim.x * gridDim.x / group_size;
    // for(int alignmentId = groupIdInGrid; alignmentId < numAlignments; alignmentId += numGroupsInGrid){
    const int alignmentId = groupIdInGrid;
    if(alignmentId < numAlignments){

        const int batchIdByGroupId = thrust::distance(
            numAlignmentsPerBatchInclusivePrefixSum,
            thrust::upper_bound(thrust::seq,
                numAlignmentsPerBatchInclusivePrefixSum,
                numAlignmentsPerBatchInclusivePrefixSum + numBatches,
                alignmentId
            )
        );
        const int batchId = min(batchIdByGroupId, numBatches-1);
        const int groupIdInBatch = alignmentId - (batchId == 0 ? 0 : numAlignmentsPerBatchInclusivePrefixSum[batchId-1]);
        const int hapToProcessInBatch = groupIdInBatch % haps_in_batch[batchId];
        const int readIndexToProcessInBatch = groupIdInBatch / haps_in_batch[batchId];

        const int offset_read_batches = numReadsPerBatchPrefixSum[batchId];
        const int offset_read_batches_inChunk = offset_read_batches - numReadsPerBatchPrefixSum[0];
        const int readToProcessInBatch = indicesPerBatch[offset_read_batches_inChunk + readIndexToProcessInBatch];

        const int read_nr = readToProcessInBatch;
        // const int global_read_id = read_nr + offset_read_batches;
        const int read_id_inChunk = read_nr + offset_read_batches_inChunk;


        const int byteOffsetForRead = read_offsets[read_id_inChunk];
        const int readLength = read_length[read_id_inChunk];

        const int b_h_off = offset_hap_batches[batchId];
        const int b_h_off_inChunk = b_h_off - offset_hap_batches[0];
        const int bytesOffsetForHap = hap_offsets[hapToProcessInBatch+b_h_off_inChunk];
        const char4* const HapsAsChar4 = reinterpret_cast<const char4*>(&hap_chars[bytesOffsetForHap]);
        const int haploLength = hap_length[hapToProcessInBatch+b_h_off_inChunk];

        const int resultOutputIndex = resultOffsetsPerBatch[batchId] + read_nr*haps_in_batch[batchId]+hapToProcessInBatch;
        // if(threadIdInGroup == 0){
        //     printf("group %d, batchIdByGroupId %d, batchId %d, groupIdInBatch %d, hapToProcessInBatch %d, readIndexToProcessInBatch %d, readToProcessInBatch %d, numAlignments %d\n"
        //         "resultOffsetsPerBatch %d, haps_in_batch %d, resultOutputIndex %d\n",
        //         alignmentId, batchIdByGroupId, batchId, groupIdInBatch, hapToProcessInBatch, readIndexToProcessInBatch, readToProcessInBatch, numAlignments,
        //         resultOffsetsPerBatch[batchId], haps_in_batch[batchId], resultOutputIndex);
        // }


        // if(alignmentId < 10 && group_size == 8 && numRegs == 8){
            // if(threadIdInGroup == 0){
            //     printf("group %d, batchId %d, groupIdInBatch %d, hapToProcessInBatch %d, readIndexToProcessInBatch %d, readToProcessInBatch %d, readLength %d, haploLength %d, numAlignments %d\n"
            //         "offset_read_batches %d, readToProcessInBatch %d, byteOffsetForRead %d, b_h_off %d, bytesOffsetForHap %d\n"
            //         "resultOffsetsPerBatch %d, haps_in_batch %d\n",
            //         alignmentId, batchId, groupIdInBatch, hapToProcessInBatch, readIndexToProcessInBatch, readToProcessInBatch, readLength, haploLength, numAlignments,
            //         offset_read_batches, readToProcessInBatch, byteOffsetForRead, b_h_off, bytesOffsetForHap,
            //         resultOffsetsPerBatch[batchId], haps_in_batch[batchId]);
            // }
        // }

        const float eps = 0.1;
        const float beta = 0.9;
        float M_l, D_l, M_ul, D_ul, I_ul;
        float penalty_temp0, penalty_temp1, penalty_temp2, penalty_temp3;
        float init_D;

        const float constant = ::cuda::std::numeric_limits<float>::max() / 16;

        auto construct_PSSM_warp_coalesced_float2 = [&](){
            const char4* QualsAsChar4 = reinterpret_cast<const char4*>(&base_quals[byteOffsetForRead]);
            const char4* ReadsAsChar4 = reinterpret_cast<const char4*>(&read_chars[byteOffsetForRead]);
            for (int i=threadIdInGroup; i<(readLength+3)/4; i+=group_size) {
                const char4 temp0 = QualsAsChar4[i];
                const char4 temp1 = ReadsAsChar4[i];
                alignas(4) char quals[4];
                memcpy(&quals[0], &temp0, sizeof(char4));
                alignas(4) char letters[4];
                memcpy(&letters[0], &temp1, sizeof(char4));

                float probs[4];
                #pragma unroll
                for(int c = 0; c < 4; c++){
                    probs[c] = cPH2PR[quals[c]];
                }

                float rowResult[5][4];

                #pragma unroll
                for(int c = 0; c < 4; c++){
                    //hap == N always matches
                    rowResult[4][c] = 1 - probs[c]; //match

                    if(letters[c] < 4){
                        // set hap == read to 1 - prob, hap != read to prob / 3
                        #pragma unroll
                        for (int j=0; j<4; j++){
                            rowResult[j][c] = (j == letters[c]) ? 1 - probs[c] : probs[c]/3.0f; //match or mismatch
                        }
                    }else{
                        // read == N always matches
                        #pragma unroll
                        for (int j=0; j<4; j++){
                            rowResult[j][c] = 1 - probs[c]; //match
                        }
                    }
                }


                alignas(8) half2 rowResultHalf2[5][2]; 

                #pragma unroll
                for(int r = 0; r < 5; r++){
                    #pragma unroll
                    for(int c = 0; c < 2; c++){
                        rowResultHalf2[r][c] =  __floats2half2_rn(rowResult[r][2*c+0], rowResult[r][2*c+1]);
                    }
                }

                //figure out where to save float2 in shared memory to allow coalesced read access to shared memory
                //read access should be coalesced within the whole warp, not only within the group

                constexpr int accessSize = sizeof(float2);
                static_assert(accessSize >= sizeof(half) * 4);
                constexpr int halfsPerAccess = accessSize / sizeof(half);
                constexpr int numAccesses = SDIV(numRegs, halfsPerAccess);
                constexpr int numBatchOfFoursPerAccess = halfsPerAccess / 4;

                const int accessChunk = i / numBatchOfFoursPerAccess;
                const int positionInAccessChunk = i % numBatchOfFoursPerAccess;
    
                const int accessChunkIdInThread = accessChunk % numAccesses;
                const int targetThreadIdInGroup = accessChunk / numAccesses;
                const int targetThreadIdInWarp = groupIdInWarp * group_size + targetThreadIdInGroup;
    
                const int outputAccessChunk = accessChunkIdInThread * warpsize + targetThreadIdInWarp;
                const int outputColFloat2 = outputAccessChunk * numBatchOfFoursPerAccess + positionInAccessChunk;
                // if(blockIdx.x == 0){
                //     printf("groupId %d, i %d, accessChunk %d, targetThreadIdInGroup %d, targetThreadIdInWarp %d, outputAccessChunk %d, outputColFloat2 %d\n", 
                //         groupIdInWarp, i, accessChunk, targetThreadIdInGroup, targetThreadIdInWarp, outputAccessChunk, outputColFloat2 );
                // }
                // assert(outputColFloat2 < rowsizePadded / 4);
                // assert(outputColFloat2 >= 0);

                #pragma unroll
                for (int j=0; j<5; j++){
                    float2* rowPtr = (float2*)(&lambda_array_permuted[j]);
                    rowPtr[outputColFloat2] = *((float2*)&rowResultHalf2[j][0]);
                }
            }

            __syncwarp(myGroupMask);

            // if(threadIdInGroup == 0){
            //     printf("half coalesced kernel permuted grouped pssm\n");
            //     for(int r = 0; r < 1; r++){
            //         for(int c = 0; c < rowsizePadded; c++){
            //             printf("%f ", float(lambda_array_permuted[r][c]));
            //         }
            //         printf("\n");
            //     }
            // }
        };

        auto construct_PSSM_warp_coalesced_float4 = [&](){
            const char4* QualsAsChar4 = reinterpret_cast<const char4*>(&base_quals[byteOffsetForRead]);
            const char4* ReadsAsChar4 = reinterpret_cast<const char4*>(&read_chars[byteOffsetForRead]);
            for (int i=threadIdInGroup; i<(readLength+3)/4; i+=group_size) {
                const char4 temp0 = QualsAsChar4[i];
                const char4 temp1 = ReadsAsChar4[i];
                alignas(4) char quals[4];
                memcpy(&quals[0], &temp0, sizeof(char4));
                alignas(4) char letters[4];
                memcpy(&letters[0], &temp1, sizeof(char4));

                float probs[4];
                #pragma unroll
                for(int c = 0; c < 4; c++){
                    probs[c] = cPH2PR[quals[c]];
                }

                float rowResult[5][4];

                #pragma unroll
                for(int c = 0; c < 4; c++){
                    //hap == N always matches
                    rowResult[4][c] = 1 - probs[c]; //match

                    if(letters[c] < 4){
                        // set hap == read to 1 - prob, hap != read to prob / 3
                        #pragma unroll
                        for (int j=0; j<4; j++){
                            rowResult[j][c] = (j == letters[c]) ? 1 - probs[c] : probs[c]/3.0f; //match or mismatch
                        }
                    }else{
                        // read == N always matches
                        #pragma unroll
                        for (int j=0; j<4; j++){
                            rowResult[j][c] = 1 - probs[c]; //match
                        }
                    }
                }


                alignas(8) half2 rowResultHalf2[5][2]; 

                #pragma unroll
                for(int r = 0; r < 5; r++){
                    #pragma unroll
                    for(int c = 0; c < 2; c++){
                        rowResultHalf2[r][c] =  __floats2half2_rn(rowResult[r][2*c+0], rowResult[r][2*c+1]);
                    }
                }

                //figure out where to save float2 in shared memory to allow coalesced read access to shared memory
                //read access should be coalesced within the whole warp, not only within the group

                //TODO this does not work correctly. need to pad registers of each thread to float4

                constexpr int accessSize = sizeof(float4);
                static_assert(accessSize >= sizeof(half) * 4);
                constexpr int halfsPerAccess = accessSize / sizeof(half);
                constexpr int numAccesses = SDIV(numRegs, halfsPerAccess);
                constexpr int numBatchOfFoursPerAccess = halfsPerAccess / 4;

                const int accessChunk = i / numBatchOfFoursPerAccess;
                const int positionInAccessChunk = i % numBatchOfFoursPerAccess;
    
                const int accessChunkIdInThread = accessChunk % numAccesses;
                const int targetThreadIdInGroup = accessChunk / numAccesses;
                const int targetThreadIdInWarp = groupIdInWarp * group_size + targetThreadIdInGroup;
    
                const int outputAccessChunk = accessChunkIdInThread * warpsize + targetThreadIdInWarp;
                const int outputColFloat2 = outputAccessChunk * numBatchOfFoursPerAccess + positionInAccessChunk;
                // if(blockIdx.x == 0){
                //     printf("groupId %d, i %d, accessChunk %d, targetThreadIdInGroup %d, targetThreadIdInWarp %d, outputAccessChunk %d, outputColFloat2 %d\n", 
                //         groupIdInWarp, i, accessChunk, targetThreadIdInGroup, targetThreadIdInWarp, outputAccessChunk, outputColFloat2 );
                // }
                // assert(outputColFloat2 < rowsizePadded / 4);
                // assert(outputColFloat2 >= 0);

                #pragma unroll
                for (int j=0; j<5; j++){
                    float2* rowPtr = (float2*)(&lambda_array_permuted[j]);
                    rowPtr[outputColFloat2] = *((float2*)&rowResultHalf2[j][0]);
                }
            }

            __syncwarp(myGroupMask);

            // if(threadIdInGroup == 0){
            //     printf("half coalesced kernel permuted grouped pssm\n");
            //     for(int r = 0; r < 1; r++){
            //         for(int c = 0; c < rowsizePadded; c++){
            //             printf("%f ", float(lambda_array_permuted[r][c]));
            //         }
            //         printf("\n");
            //     }
            // }
        };

        auto load_PSSM = [&](){
            construct_PSSM_warp_coalesced_float2();
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

        


        auto calc_DP_float_float4 = [&](){

            //warp coalesced
            float4* sbt_row = (float4*)(&lambda_array_permuted[hap_letter]);
            float4 foo = *((float4*)(&sbt_row[0 * warpsize + threadIdInWarp]));
            alignas(16) float fooArray[4];
            memcpy(&fooArray[0], &foo, sizeof(float4));
            //foo contains 8 half values. we know that numRegs >= 4 and numRegs % 4 == 0

            //process first four half values
            __half2 score2;
            memcpy(&score2, &fooArray[0], sizeof(__half2));

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

            memcpy(&score2, &fooArray[1], sizeof(__half2));
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

            //process halfs 5-8 if necessary
            if constexpr (numRegs > 4){
                constexpr int i = 1;
                memcpy(&score2, &fooArray[2], sizeof(__half2));
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

                memcpy(&score2, &fooArray[3], sizeof(__half2));
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

            #pragma unroll
            for (int oct = 1; oct < numRegs / 8; oct++) {
                float4 foo = *((float4*)(&sbt_row[oct * warpsize + threadIdInWarp]));
                memcpy(&fooArray[0], &foo, sizeof(float4));

                //process first four half values
                {
                    int i = 2*oct + 0;
                    memcpy(&score2, &fooArray[0], sizeof(__half2));
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

                    memcpy(&score2, &fooArray[1], sizeof(__half2));
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

                //process halfs 5-8 if necessary
                {
                    const int i = 2*oct + 1;
                    memcpy(&score2, &fooArray[2], sizeof(__half2));
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

                    memcpy(&score2, &fooArray[3], sizeof(__half2));
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
            }


            //process last float4 which is only partially used
            if constexpr(numRegs > 8 && numRegs % 8 != 0){
                static_assert(numRegs % 8 == 4);

                const int oct = numRegs / 8;
                float4 foo = *((float4*)(&sbt_row[oct * warpsize + threadIdInWarp]));
                memcpy(&fooArray[0], &foo, sizeof(float4));

                //process first four half values
                {
                    int i = 2*oct + 0;
                    memcpy(&score2, &fooArray[0], sizeof(__half2));
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

                    memcpy(&score2, &fooArray[1], sizeof(__half2));
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
            }

        };

        auto calc_DP_float_float2 = [&](){

            //warp coalesced
            float2* sbt_row = (float2*)(&lambda_array_permuted[hap_letter]);
            float2 foo = *((float2*)(&sbt_row[0 * warpsize + threadIdInWarp]));
            alignas(8) float fooArray[2];
            memcpy(&fooArray[0], &foo, sizeof(float2));
            //foo contains 8 half values. we know that numRegs >= 4 and numRegs % 4 == 0

            //process first four half values
            __half2 score2;
            memcpy(&score2, &fooArray[0], sizeof(__half2));

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

            memcpy(&score2, &fooArray[1], sizeof(__half2));
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
            for (int i = 1; i < numRegs / 4; i++) {
                float2 foo = *((float2*)(&sbt_row[i * warpsize + threadIdInWarp]));
                memcpy(&fooArray[0], &foo, sizeof(float2));
                memcpy(&score2, &fooArray[0], sizeof(__half2));
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

                memcpy(&score2, &fooArray[1], sizeof(__half2));
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

        auto calc_DP_float = [&](){
            //for float4 (which is not working currently), need to also change alignas of Result, and smem output buffering at the end
            calc_DP_float_float2();
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
            if (!threadIdInGroup) hap_letter = new_hap_letter4.z;
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


        //repurpose shared memory to stage output. 
        //since the output register index is computed at runtime, 
        //the compiler frequently stores the Results in local memory to be able to load the specific value at the end.
        //doing this once manually in shared memory avoids the atuomatic stores to local memory
        __syncwarp(myGroupMask);
        float* smemOutputBuffer = (float*)&lambda_array_permuted[0][0];


        if (threadIdInGroup == result_thread) {
            // float temp_res = Results[result_reg];

            float2* smemOutputBuffer2 = (float2*)smemOutputBuffer;
            #pragma unroll
            for(int i = 0; i < numRegs/2; i++){
                //need to ensure that we only access smem elements which are used by the group. here we use the same access pattern as during computations (warp striped)
                smemOutputBuffer2[i * warpsize + threadIdInWarp] = *((float2*)&Results[2*i]);
            }
            float temp_res = smemOutputBuffer[2*(result_reg/2 * warpsize + threadIdInWarp) + (result_reg % 2)];

            temp_res =  log10f(temp_res) - log10f(constant);
            devAlignmentScores[resultOutputIndex] = temp_res;
        }
    }


}


template <int group_size, int numRegs> 
__global__
void PairHMM_align_partition_float_allowMultipleBatchesPerWarp(
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

    alignas(8) __shared__ float2 lambda_array[5][16*numRegs];

    float M[numRegs], I, D[numRegs];
    float alpha[numRegs], delta[numRegs], sigma[numRegs];
    float Results[numRegs];

    const int threadIdInGroup = threadIdx.x % group_size;
    const int groupIdInBlock = threadIdx.x / group_size;
    const int groupIdInGrid = (threadIdx.x + blockIdx.x * blockDim.x) / group_size;
    const unsigned int myGroupMask = __match_any_sync(0xFFFFFFFF, groupIdInGrid); //compute mask for all threads with same groupIdInGrid
    
    // const int numGroupsInGrid = blockDim.x * gridDim.x / group_size;
    // for(int alignmentId = groupIdInGrid; alignmentId < numAlignments; alignmentId += numGroupsInGrid){
    const int alignmentId = groupIdInGrid;
    if(alignmentId < numAlignments){

        const int batchIdByGroupId = thrust::distance(
            numAlignmentsPerBatchInclusivePrefixSum,
            thrust::upper_bound(thrust::seq,
                numAlignmentsPerBatchInclusivePrefixSum,
                numAlignmentsPerBatchInclusivePrefixSum + numBatches,
                alignmentId
            )
        );
        const int batchId = min(batchIdByGroupId, numBatches-1);
        const int groupIdInBatch = alignmentId - (batchId == 0 ? 0 : numAlignmentsPerBatchInclusivePrefixSum[batchId-1]);
        const int hapToProcessInBatch = groupIdInBatch % haps_in_batch[batchId];
        const int readIndexToProcessInBatch = groupIdInBatch / haps_in_batch[batchId];

        const int offset_read_batches = numReadsPerBatchPrefixSum[batchId];
        const int offset_read_batches_inChunk = offset_read_batches - numReadsPerBatchPrefixSum[0];
        const int readToProcessInBatch = indicesPerBatch[offset_read_batches_inChunk + readIndexToProcessInBatch];

        const int read_nr = readToProcessInBatch;
        // const int global_read_id = read_nr + offset_read_batches;
        const int read_id_inChunk = read_nr + offset_read_batches_inChunk;


        const int byteOffsetForRead = read_offsets[read_id_inChunk];
        const int readLength = read_length[read_id_inChunk];

        const int b_h_off = offset_hap_batches[batchId];
        const int b_h_off_inChunk = b_h_off - offset_hap_batches[0];
        const int bytesOffsetForHap = hap_offsets[hapToProcessInBatch+b_h_off_inChunk];
        const char4* const HapsAsChar4 = reinterpret_cast<const char4*>(&hap_chars[bytesOffsetForHap]);
        const int haploLength = hap_length[hapToProcessInBatch+b_h_off_inChunk];

        const int resultOutputIndex = resultOffsetsPerBatch[batchId] + read_nr*haps_in_batch[batchId]+hapToProcessInBatch;
        // if(threadIdInGroup == 0){
        //     printf("group %d, batchIdByGroupId %d, batchId %d, groupIdInBatch %d, hapToProcessInBatch %d, readIndexToProcessInBatch %d, readToProcessInBatch %d, numAlignments %d\n"
        //         "resultOffsetsPerBatch %d, haps_in_batch %d, resultOutputIndex %d\n",
        //         alignmentId, batchIdByGroupId, batchId, groupIdInBatch, hapToProcessInBatch, readIndexToProcessInBatch, readToProcessInBatch, numAlignments,
        //         resultOffsetsPerBatch[batchId], haps_in_batch[batchId], resultOutputIndex);
        // }


        // if(alignmentId < 10 && group_size == 8 && numRegs == 8){
            // if(threadIdInGroup == 0){
            //     printf("group %d, batchId %d, groupIdInBatch %d, hapToProcessInBatch %d, readIndexToProcessInBatch %d, readToProcessInBatch %d, readLength %d, haploLength %d, numAlignments %d\n"
            //         "offset_read_batches %d, readToProcessInBatch %d, byteOffsetForRead %d, b_h_off %d, bytesOffsetForHap %d\n"
            //         "resultOffsetsPerBatch %d, haps_in_batch %d\n",
            //         alignmentId, batchId, groupIdInBatch, hapToProcessInBatch, readIndexToProcessInBatch, readToProcessInBatch, readLength, haploLength, numAlignments,
            //         offset_read_batches, readToProcessInBatch, byteOffsetForRead, b_h_off, bytesOffsetForHap,
            //         resultOffsetsPerBatch[batchId], haps_in_batch[batchId]);
            // }
        // }

        const float eps = 0.1;
        const float beta = 0.9;
        float M_l, D_l, M_ul, D_ul, I_ul;
        float penalty_temp0, penalty_temp1, penalty_temp2, penalty_temp3;
        float init_D;

        const float constant = ::cuda::std::numeric_limits<float>::max() / 16;

        auto load_PSSM = [&]() {

            char4 temp0, temp1;
            const float one = 1.0;
            const float three = 3.0;
            const char4* QualsAsChar4 = reinterpret_cast<const char4*>(&base_quals[byteOffsetForRead]);
            const char4* ReadsAsChar4 = reinterpret_cast<const char4*>(&read_chars[byteOffsetForRead]);
            for (int i=threadIdInGroup; i<(readLength+3)/4; i+=group_size) {
                float2 temp_h2, temp_h3;
                temp0 = QualsAsChar4[i];
                temp1 = ReadsAsChar4[i];
                temp_h2.x = cPH2PR[uint8_t(temp0.x)];
                temp_h2.y = cPH2PR[uint8_t(temp0.y)];
                temp_h3.x = temp_h2.x/three;
                temp_h3.y = temp_h2.y/three;

                //init hap == A,C,G,T as mismatch
                for (int j=0; j<4; j++){
                    lambda_array[j][2*i+groupIdInBlock*(group_size*numRegs/2)] = temp_h3; // mismatch
                }
                //hap == N always matches
                lambda_array[4][2*i+groupIdInBlock*(group_size*numRegs/2)].x = one - temp_h2.x; // match
                lambda_array[4][2*i+groupIdInBlock*(group_size*numRegs/2)].y = one - temp_h2.y; // match

                if (temp1.x < 4){
                    // set hap == read
                    lambda_array[temp1.x][2*i+groupIdInBlock*(group_size*numRegs/2)].x = one - temp_h2.x; // match
                }else if (temp1.x == 4){
                    // read == N always matches
                    for (int j=0; j<4; j++){
                        lambda_array[j][2*i+groupIdInBlock*(group_size*numRegs/2)].x = one - temp_h2.x; // N always match
                    }
                }
                if (temp1.y < 4){
                    // set hap == read
                    lambda_array[temp1.y][2*i+groupIdInBlock*(group_size*numRegs/2)].y = one - temp_h2.y; // match
                }else if (temp1.y == 4){
                    // read == N always matches
                    for (int j=0; j<4; j++){
                        lambda_array[j][2*i+groupIdInBlock*(group_size*numRegs/2)].y = one - temp_h2.y; // N always match
                    }
                }

                temp_h2.x = cPH2PR[uint8_t(temp0.z)];
                temp_h2.y = cPH2PR[uint8_t(temp0.w)];
                temp_h3.x = temp_h2.x/three;
                temp_h3.y = temp_h2.y/three;

                //init hap == A,C,G,T as mismatch
                for (int j=0; j<4; j++){
                    lambda_array[j][2*i+1+groupIdInBlock*(group_size*numRegs/2)] = temp_h3; // mismatch
                }
                //hap == N always matches
                lambda_array[4][2*i+1+groupIdInBlock*(group_size*numRegs/2)].x = one - temp_h2.x; // match
                lambda_array[4][2*i+1+groupIdInBlock*(group_size*numRegs/2)].y = one - temp_h2.y; // match

                if (temp1.z < 4){
                    // set hap == read
                    lambda_array[temp1.z][2*i+1+groupIdInBlock*(group_size*numRegs/2)].x = one - temp_h2.x; // match
                }else if (temp1.z == 4){
                    // read == N always matches
                    for (int j=0; j<4; j++){
                        lambda_array[j][2*i+1+groupIdInBlock*(group_size*numRegs/2)].x = one - temp_h2.x; // N always match
                    }
                }
                if (temp1.w < 4){
                    // set hap == read
                    lambda_array[temp1.w][2*i+1+groupIdInBlock*(group_size*numRegs/2)].y = one - temp_h2.y; // match
                }else if (temp1.w == 4){
                    // read == N always matches
                    for (int j=0; j<4; j++){
                        lambda_array[j][2*i+1+groupIdInBlock*(group_size*numRegs/2)].y = one - temp_h2.y; // N always match
                    }
                }
            }

            __syncwarp(myGroupMask);

            // if(threadIdInGroup == 0){
            //     printf("float kernel pssm\n");
            //     for(int r = 0; r < 5; r++){
            //         for(int c = 0; c < 16*numRegs; c++){
            //             printf("%f %f ", lambda_array[r][c].x, lambda_array[r][c].y);
            //         }
            //         printf("\n");
            //     }
            // }

        };

        auto load_probabilities = [&]() {
            char4 temp0, temp1;
            const char4* InsQualsAsChar4 = reinterpret_cast<const char4*>(&ins_quals[byteOffsetForRead]);
            const char4* DelQualsAsChar4 = reinterpret_cast<const char4*>(&del_quals[byteOffsetForRead]);
            for (int i=0; i<numRegs/4; i++) {
                if (threadIdInGroup*numRegs/4+i < (readLength+3)/4) {

                    temp0 = InsQualsAsChar4[threadIdInGroup*numRegs/4+i];
                    temp1 = DelQualsAsChar4[threadIdInGroup*numRegs/4+i];

                    delta[4*i] = cPH2PR[uint8_t(temp0.x)];
                    delta[4*i+1] = cPH2PR[uint8_t(temp0.y)];
                    delta[4*i+2] = cPH2PR[uint8_t(temp0.z)];
                    delta[4*i+3] = cPH2PR[uint8_t(temp0.w)];
            //        delta[2*i] = __floats2half2_rn(cPH2PR[uint8_t(temp0.x)],cPH2PR[uint8_t(temp0.y)]);
            //        delta[2*i+1] = __floats2half2_rn(cPH2PR[uint8_t(temp0.z)],cPH2PR[uint8_t(temp0.w)]);

                    sigma[4*i] = cPH2PR[uint8_t(temp1.x)];
                    sigma[4*i+1] = cPH2PR[uint8_t(temp1.y)];
                    sigma[4*i+2] = cPH2PR[uint8_t(temp1.z)];
                    sigma[4*i+3] = cPH2PR[uint8_t(temp1.w)];
            //        sigma[2*i] = __floats2half2_rn(cPH2PR[uint8_t(temp1.x)],cPH2PR[uint8_t(temp1.y)]);
            //        sigma[2*i+1] = __floats2half2_rn(cPH2PR[uint8_t(temp1.z)],cPH2PR[uint8_t(temp1.w)]);

                    alpha[4*i] = 1.0 - (delta[4*i] + sigma[4*i]);
                    alpha[4*i+1] = 1.0 - (delta[4*i+1] + sigma[4*i+1]);
                    alpha[4*i+2] = 1.0 - (delta[4*i+2] + sigma[4*i+2]);
                    alpha[4*i+3] = 1.0 - (delta[4*i+3] + sigma[4*i+3]);
                //    alpha[2*i] = __float2half2_rn(1.0) - __hadd2(delta[2*i], sigma[2*i]);
                //    alpha[2*i+1] = __float2half2_rn(1.0) - __hadd2(delta[2*i+1], sigma[2*i+1]);
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

        
        auto calc_DP_float = [&](int row){
            
            // __half2 score2;
            float2* sbt_row = lambda_array[hap_letter];
            float4 foo = *((float4*)&sbt_row[threadIdx.x*numRegs/2]);
            //memcpy(&score2, &foo.x, sizeof(float2));
            penalty_temp0 = M[0];
            penalty_temp1 = D[0];
            M[0] = foo.x * fmaf(alpha[0],M_ul,beta*(I_ul+D_ul));
            D[0] = fmaf(sigma[0],penalty_temp0,eps*D[0]);
            I = fmaf(delta[0],M_ul,eps*I_ul);
            Results[0] += M[0] + I;
            penalty_temp2 = M[1];
            penalty_temp3 = D[1];
            M[1] = foo.y * fmaf(alpha[1],penalty_temp0,beta*(I+penalty_temp1));
            D[1] = fmaf(sigma[1],penalty_temp2,eps*D[1]);
            I = fmaf(delta[1],penalty_temp0,eps*I);
            Results[1] += M[1] + I;

            //memcpy(&score2, &foo.z, sizeof(float2));
            penalty_temp0 = M[2];
            penalty_temp1 = D[2];
            M[2] = foo.z * fmaf(alpha[2],penalty_temp2,beta*(I+penalty_temp3));
            D[2] = fmaf(sigma[2],penalty_temp0,eps*D[2]);
            I = fmaf(delta[2],penalty_temp2,eps*I);
            Results[2] += M[2] + I;
            penalty_temp2 = M[3];
            penalty_temp3 = D[3];
            M[3] = foo.w * fmaf(alpha[3],penalty_temp0,beta*(I+penalty_temp1));
            D[3] = fmaf(sigma[3],penalty_temp2,eps*D[3]);
            I = fmaf(delta[3],penalty_temp0,eps*I);
            Results[3] += M[3] + I;

            // if(threadIdInGroup * numRegs + 0 < haploLength) printf("row %d, col %d, lambda %f\n", row, threadIdInGroup * numRegs + 0, foo.x);
            // if(threadIdInGroup * numRegs + 1 < haploLength) printf("row %d, col %d, lambda %f\n", row, threadIdInGroup * numRegs + 1, foo.y);
            // if(threadIdInGroup * numRegs + 2 < haploLength) printf("row %d, col %d, lambda %f\n", row, threadIdInGroup * numRegs + 2, foo.z);
            // if(threadIdInGroup * numRegs + 3 < haploLength) printf("row %d, col %d, lambda %f\n", row, threadIdInGroup * numRegs + 3, foo.w);

            #pragma unroll
            for (int i=1; i<numRegs/4; i++) {
                float4 foo = *((float4*)&sbt_row[threadIdx.x*numRegs/2+2*i]);
                //memcpy(&score2, &foo.x, sizeof(float2));
                penalty_temp0 = M[4*i];
                penalty_temp1 = D[4*i];
                M[4*i] = foo.x * fmaf(alpha[4*i],penalty_temp2,beta*(I+penalty_temp3));
                D[4*i] = fmaf(sigma[4*i],penalty_temp0,eps*D[4*i]);
                I = fmaf(delta[4*i],penalty_temp2,eps*I);
                Results[4*i] += M[4*i] + I;
                penalty_temp2 = M[4*i+1];
                penalty_temp3 = D[4*i+1];
                M[4*i+1] = foo.y * fmaf(alpha[4*i+1],penalty_temp0,beta*(I+penalty_temp1));
                D[4*i+1] = fmaf(sigma[4*i+1],penalty_temp2,eps*D[4*i+1]);
                I = fmaf(delta[4*i+1],penalty_temp0,eps*I);
                Results[4*i+1] += M[4*i+1] + I;

                //memcpy(&score2, &foo.z, sizeof(float2));
                penalty_temp0 = M[4*i+2];
                penalty_temp1 = D[4*i+2];
                M[4*i+2] = foo.z * fmaf(alpha[4*i+2],penalty_temp2,beta*(I+penalty_temp3));
                D[4*i+2] = fmaf(sigma[4*i+2],penalty_temp0,eps*D[4*i+2]);
                I = fmaf(delta[4*i+2],penalty_temp2,eps*I);
                Results[4*i+2] += M[4*i+2] + I;

                penalty_temp2 = M[4*i+3];
                penalty_temp3 = D[4*i+3];
                M[4*i+3] = foo.w * fmaf(alpha[4*i+3],penalty_temp0,beta*(I+penalty_temp1));
                D[4*i+3] = fmaf(sigma[4*i+3],penalty_temp2,eps*D[4*i+3]);
                I = fmaf(delta[4*i+3],penalty_temp0,eps*I);
                Results[4*i+3] += M[4*i+3] + I;

            //     if(threadIdInGroup * numRegs + 4*i + 0 < haploLength) printf("row %d, col %d, lambda %f\n", row, threadIdInGroup * numRegs + 4*i + 0, foo.x);
            //     if(threadIdInGroup * numRegs + 4*i + 1 < haploLength) printf("row %d, col %d, lambda %f\n", row, threadIdInGroup * numRegs + 4*i + 1, foo.y);
            //     if(threadIdInGroup * numRegs + 4*i + 2 < haploLength) printf("row %d, col %d, lambda %f\n", row, threadIdInGroup * numRegs + 4*i + 2, foo.z);
            //     if(threadIdInGroup * numRegs + 4*i + 3 < haploLength) printf("row %d, col %d, lambda %f\n", row, threadIdInGroup * numRegs + 4*i + 3, foo.w);
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
            calc_DP_float(k);
            shuffle_penalty();

            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.y;
            calc_DP_float(k);
            shuffle_penalty();

            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.z;
            calc_DP_float(k);
            shuffle_penalty();

            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.w;
            calc_DP_float(k);
            shuffle_penalty();
        }
        if (haploLength%4 >= 1) {
            new_hap_letter4 = HapsAsChar4[k/4];
            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.x;
            calc_DP_float(k);
            shuffle_penalty();
        }
        if (haploLength%4 >= 2) {
            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.y;
            calc_DP_float(k+1);
            shuffle_penalty();
        }
        if (haploLength%4 >= 3) {
            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.z;
            calc_DP_float(k+2);
            shuffle_penalty();
        }
        for (k=0; k<result_thread; k++) {
            // hap_letter = __shfl_up_sync(__activemask(), hap_letter, 1, 32);
            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            calc_DP_float(-1);
            shuffle_penalty(); // shuffle_penalty_active();
        }
        // adjust I values
        I = fmaf(delta[0],M_ul,eps*I_ul);
        Results[0] += I;
        I = fmaf(delta[1],M[0],eps*I);
        Results[1] += I;
        for (int p=1; p<numRegs/2; p++) {
            I = fmaf(delta[2*p],M[2*p-1],eps*I);
            Results[2*p] += I;
            I = fmaf(delta[2*p+1],M[2*p],eps*I);
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
            devAlignmentScores[resultOutputIndex] = temp_res;
        }
    }


}






template <int group_size, int numRegs> 
__global__
void PairHMM_align_partition_float_allowMultipleBatchesPerWarp_coalesced_smem(
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
    static_assert(numRegs % 4 == 0);

    constexpr int blocksize = 32;
    constexpr int warpsize = 32;
    constexpr int numGroupsPerBlock = blocksize / group_size;

    constexpr int rowsize = numGroupsPerBlock * group_size*numRegs;
    alignas(16) __shared__ float lambda_array_permuted[5][rowsize];

    float M[numRegs], I, D[numRegs];
    float alpha[numRegs], delta[numRegs], sigma[numRegs];
    alignas(16) float Results[numRegs];

    const int threadIdInGroup = threadIdx.x % group_size;
    // const int groupIdInBlock = threadIdx.x / group_size;
    const int threadIdInWarp = threadIdx.x % warpsize;
    const int groupIdInWarp = threadIdInWarp / group_size;
    const int groupIdInGrid = (threadIdx.x + blockIdx.x * blockDim.x) / group_size;
    const unsigned int myGroupMask = __match_any_sync(0xFFFFFFFF, groupIdInGrid); //compute mask for all threads with same groupIdInGrid
    
    // const int numGroupsInGrid = blockDim.x * gridDim.x / group_size;
    // for(int alignmentId = groupIdInGrid; alignmentId < numAlignments; alignmentId += numGroupsInGrid){
    const int alignmentId = groupIdInGrid;
    if(alignmentId < numAlignments){

        const int batchIdByGroupId = thrust::distance(
            numAlignmentsPerBatchInclusivePrefixSum,
            thrust::upper_bound(thrust::seq,
                numAlignmentsPerBatchInclusivePrefixSum,
                numAlignmentsPerBatchInclusivePrefixSum + numBatches,
                alignmentId
            )
        );
        const int batchId = min(batchIdByGroupId, numBatches-1);
        const int groupIdInBatch = alignmentId - (batchId == 0 ? 0 : numAlignmentsPerBatchInclusivePrefixSum[batchId-1]);
        const int hapToProcessInBatch = groupIdInBatch % haps_in_batch[batchId];
        const int readIndexToProcessInBatch = groupIdInBatch / haps_in_batch[batchId];

        const int offset_read_batches = numReadsPerBatchPrefixSum[batchId];
        const int offset_read_batches_inChunk = offset_read_batches - numReadsPerBatchPrefixSum[0];
        const int readToProcessInBatch = indicesPerBatch[offset_read_batches_inChunk + readIndexToProcessInBatch];

        const int read_nr = readToProcessInBatch;
        // const int global_read_id = read_nr + offset_read_batches;
        const int read_id_inChunk = read_nr + offset_read_batches_inChunk;


        const int byteOffsetForRead = read_offsets[read_id_inChunk];
        const int readLength = read_length[read_id_inChunk];

        const int b_h_off = offset_hap_batches[batchId];
        const int b_h_off_inChunk = b_h_off - offset_hap_batches[0];
        const int bytesOffsetForHap = hap_offsets[hapToProcessInBatch+b_h_off_inChunk];
        const char4* const HapsAsChar4 = reinterpret_cast<const char4*>(&hap_chars[bytesOffsetForHap]);
        const int haploLength = hap_length[hapToProcessInBatch+b_h_off_inChunk];

        const int resultOutputIndex = resultOffsetsPerBatch[batchId] + read_nr*haps_in_batch[batchId]+hapToProcessInBatch;

        const float eps = 0.1;
        const float beta = 0.9;
        float M_l, D_l, M_ul, D_ul, I_ul;
        float penalty_temp0, penalty_temp1, penalty_temp2, penalty_temp3;
        float init_D;

        const float constant = ::cuda::std::numeric_limits<float>::max() / 16;

        auto construct_PSSM_warp_coalesced = [&](){
            __syncwarp(myGroupMask);
            
            const char4* QualsAsChar4 = reinterpret_cast<const char4*>(&base_quals[byteOffsetForRead]);
            const char4* ReadsAsChar4 = reinterpret_cast<const char4*>(&read_chars[byteOffsetForRead]);
            for (int i=threadIdInGroup; i<(readLength+3)/4; i+=group_size) {
                const char4 temp0 = QualsAsChar4[i];
                const char4 temp1 = ReadsAsChar4[i];
                alignas(4) char quals[4];
                memcpy(&quals[0], &temp0, sizeof(char4));
                alignas(4) char letters[4];
                memcpy(&letters[0], &temp1, sizeof(char4));

                float probs[4];
                #pragma unroll
                for(int c = 0; c < 4; c++){
                    probs[c] = cPH2PR[quals[c]];
                }

                alignas(16) float rowResult[5][4];

                #pragma unroll
                for(int c = 0; c < 4; c++){
                    //hap == N always matches
                    rowResult[4][c] = 1 - probs[c]; //match

                    if(letters[c] < 4){
                        // set hap == read to 1 - prob, hap != read to prob / 3
                        #pragma unroll
                        for (int j=0; j<4; j++){
                            rowResult[j][c] = (j == letters[c]) ? 1 - probs[c] : probs[c]/3.0f; //match or mismatch
                        }
                    }else{
                        // read == N always matches
                        #pragma unroll
                        for (int j=0; j<4; j++){
                            rowResult[j][c] = 1 - probs[c]; //match
                        }
                    }
                }


                //figure out where to save float4 in shared memory to allow coalesced read access to shared memory
                //read access should be coalesced within the whole warp, not only within the group

                constexpr int numAccesses = numRegs/4;

                const int accessChunk = i;
                const int accessChunkIdInThread = accessChunk % numAccesses;
                const int targetThreadIdInGroup = accessChunk / numAccesses;
                const int targetThreadIdInWarp = groupIdInWarp * group_size + targetThreadIdInGroup;

                const int outputAccessChunk = accessChunkIdInThread * warpsize + targetThreadIdInWarp;
                const int outputCol = outputAccessChunk;

                // if(blockIdx.x == 0){
                //     printf("groupId %d, i %d, targetThreadIdInGroup %d, targetThreadIdInWarp %d, outputAccessChunk %d\n", 
                //         groupIdInWarp, i,targetThreadIdInGroup, targetThreadIdInWarp, outputAccessChunk );
                // }

                // if(threadIdInGroup == 0){
                //     printf("float kernel permuted grouped pssm\n");
                //     for(int r = 0; r < 5; r++){
                //         for(int c = 0; c < 16*numRegs; c++){
                //             printf("%f %f ", lambda_array[r][c].x, lambda_array[r][c].y);
                //         }
                //         printf("\n");
                //     }
                // }

                #pragma unroll
                for (int j=0; j<5; j++){
                    float4* rowPtr = (float4*)(&lambda_array_permuted[j]);
                    rowPtr[outputCol] = *((float4*)&rowResult[j][0]);
                }
            }

            __syncwarp(myGroupMask);
        };

        auto load_PSSM = [&](){
            construct_PSSM_warp_coalesced();
        };

        auto load_probabilities = [&]() {
            char4 temp0, temp1;
            const char4* InsQualsAsChar4 = reinterpret_cast<const char4*>(&ins_quals[byteOffsetForRead]);
            const char4* DelQualsAsChar4 = reinterpret_cast<const char4*>(&del_quals[byteOffsetForRead]);
            for (int i=0; i<numRegs/4; i++) {
                if (threadIdInGroup*numRegs/4+i < (readLength+3)/4) {

                    temp0 = InsQualsAsChar4[threadIdInGroup*numRegs/4+i];
                    temp1 = DelQualsAsChar4[threadIdInGroup*numRegs/4+i];

                    delta[4*i] = cPH2PR[uint8_t(temp0.x)];
                    delta[4*i+1] = cPH2PR[uint8_t(temp0.y)];
                    delta[4*i+2] = cPH2PR[uint8_t(temp0.z)];
                    delta[4*i+3] = cPH2PR[uint8_t(temp0.w)];
            //        delta[2*i] = __floats2half2_rn(cPH2PR[uint8_t(temp0.x)],cPH2PR[uint8_t(temp0.y)]);
            //        delta[2*i+1] = __floats2half2_rn(cPH2PR[uint8_t(temp0.z)],cPH2PR[uint8_t(temp0.w)]);

                    sigma[4*i] = cPH2PR[uint8_t(temp1.x)];
                    sigma[4*i+1] = cPH2PR[uint8_t(temp1.y)];
                    sigma[4*i+2] = cPH2PR[uint8_t(temp1.z)];
                    sigma[4*i+3] = cPH2PR[uint8_t(temp1.w)];
            //        sigma[2*i] = __floats2half2_rn(cPH2PR[uint8_t(temp1.x)],cPH2PR[uint8_t(temp1.y)]);
            //        sigma[2*i+1] = __floats2half2_rn(cPH2PR[uint8_t(temp1.z)],cPH2PR[uint8_t(temp1.w)]);

                    alpha[4*i] = 1.0 - (delta[4*i] + sigma[4*i]);
                    alpha[4*i+1] = 1.0 - (delta[4*i+1] + sigma[4*i+1]);
                    alpha[4*i+2] = 1.0 - (delta[4*i+2] + sigma[4*i+2]);
                    alpha[4*i+3] = 1.0 - (delta[4*i+3] + sigma[4*i+3]);
                //    alpha[2*i] = __float2half2_rn(1.0) - __hadd2(delta[2*i], sigma[2*i]);
                //    alpha[2*i+1] = __float2half2_rn(1.0) - __hadd2(delta[2*i+1], sigma[2*i+1]);
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

        
        auto calc_DP_float = [&](int row){
            
            //warp coalesced
            float4* sbt_row = (float4*)(&lambda_array_permuted[hap_letter]);
            float4 foo = *((float4*)(&sbt_row[0 * warpsize + threadIdInWarp]));

            // if(group_size == 8 && numRegs == 20){
            //     if(blockIdx.x == 0){
            //         printf("thread %d, load from %p, bank %lu\n", 
            //             threadIdInWarp, &sbt_row[0 * warpsize + threadIdInWarp], (size_t(&sbt_row[0 * warpsize + threadIdInWarp])/4) % 32);
            //     }
            // }
            
            //memcpy(&score2, &foo.x, sizeof(float2));
            penalty_temp0 = M[0];
            penalty_temp1 = D[0];
            M[0] = foo.x * fmaf(alpha[0],M_ul,beta*(I_ul+D_ul));
            D[0] = fmaf(sigma[0],penalty_temp0,eps*D[0]);
            I = fmaf(delta[0],M_ul,eps*I_ul);
            Results[0] += M[0] + I;
            penalty_temp2 = M[1];
            penalty_temp3 = D[1];
            M[1] = foo.y * fmaf(alpha[1],penalty_temp0,beta*(I+penalty_temp1));
            D[1] = fmaf(sigma[1],penalty_temp2,eps*D[1]);
            I = fmaf(delta[1],penalty_temp0,eps*I);
            Results[1] += M[1] + I;

            //memcpy(&score2, &foo.z, sizeof(float2));
            penalty_temp0 = M[2];
            penalty_temp1 = D[2];
            M[2] = foo.z * fmaf(alpha[2],penalty_temp2,beta*(I+penalty_temp3));
            D[2] = fmaf(sigma[2],penalty_temp0,eps*D[2]);
            I = fmaf(delta[2],penalty_temp2,eps*I);
            Results[2] += M[2] + I;
            penalty_temp2 = M[3];
            penalty_temp3 = D[3];
            M[3] = foo.w * fmaf(alpha[3],penalty_temp0,beta*(I+penalty_temp1));
            D[3] = fmaf(sigma[3],penalty_temp2,eps*D[3]);
            I = fmaf(delta[3],penalty_temp0,eps*I);
            Results[3] += M[3] + I;

            // if(threadIdInGroup * numRegs + 0 < haploLength) printf("row %d, col %d, lambda %f\n", row, threadIdInGroup * numRegs + 0, foo.x);
            // if(threadIdInGroup * numRegs + 1 < haploLength) printf("row %d, col %d, lambda %f\n", row, threadIdInGroup * numRegs + 1, foo.y);
            // if(threadIdInGroup * numRegs + 2 < haploLength) printf("row %d, col %d, lambda %f\n", row, threadIdInGroup * numRegs + 2, foo.z);
            // if(threadIdInGroup * numRegs + 3 < haploLength) printf("row %d, col %d, lambda %f\n", row, threadIdInGroup * numRegs + 3, foo.w);

            #pragma unroll
            for (int i=1; i<numRegs/4; i++) {
                float4 foo = *((float4*)(&sbt_row[i * warpsize + threadIdInWarp]));

                // if(group_size == 8 && numRegs == 20){
                //     if(blockIdx.x == 0){
                //         printf("thread %d, load from %p, bank %lu\n", 
                //             threadIdInWarp, &sbt_row[i * warpsize + threadIdInWarp], (size_t(&sbt_row[i * warpsize + threadIdInWarp])/4) % 32);
                //     }
                // }
                

                //memcpy(&score2, &foo.x, sizeof(float2));
                penalty_temp0 = M[4*i];
                penalty_temp1 = D[4*i];
                M[4*i] = foo.x * fmaf(alpha[4*i],penalty_temp2,beta*(I+penalty_temp3));
                D[4*i] = fmaf(sigma[4*i],penalty_temp0,eps*D[4*i]);
                I = fmaf(delta[4*i],penalty_temp2,eps*I);
                Results[4*i] += M[4*i] + I;
                penalty_temp2 = M[4*i+1];
                penalty_temp3 = D[4*i+1];
                M[4*i+1] = foo.y * fmaf(alpha[4*i+1],penalty_temp0,beta*(I+penalty_temp1));
                D[4*i+1] = fmaf(sigma[4*i+1],penalty_temp2,eps*D[4*i+1]);
                I = fmaf(delta[4*i+1],penalty_temp0,eps*I);
                Results[4*i+1] += M[4*i+1] + I;

                //memcpy(&score2, &foo.z, sizeof(float2));
                penalty_temp0 = M[4*i+2];
                penalty_temp1 = D[4*i+2];
                M[4*i+2] = foo.z * fmaf(alpha[4*i+2],penalty_temp2,beta*(I+penalty_temp3));
                D[4*i+2] = fmaf(sigma[4*i+2],penalty_temp0,eps*D[4*i+2]);
                I = fmaf(delta[4*i+2],penalty_temp2,eps*I);
                Results[4*i+2] += M[4*i+2] + I;

                penalty_temp2 = M[4*i+3];
                penalty_temp3 = D[4*i+3];
                M[4*i+3] = foo.w * fmaf(alpha[4*i+3],penalty_temp0,beta*(I+penalty_temp1));
                D[4*i+3] = fmaf(sigma[4*i+3],penalty_temp2,eps*D[4*i+3]);
                I = fmaf(delta[4*i+3],penalty_temp0,eps*I);
                Results[4*i+3] += M[4*i+3] + I;

            //     if(threadIdInGroup * numRegs + 4*i + 0 < haploLength) printf("row %d, col %d, lambda %f\n", row, threadIdInGroup * numRegs + 4*i + 0, foo.x);
            //     if(threadIdInGroup * numRegs + 4*i + 1 < haploLength) printf("row %d, col %d, lambda %f\n", row, threadIdInGroup * numRegs + 4*i + 1, foo.y);
            //     if(threadIdInGroup * numRegs + 4*i + 2 < haploLength) printf("row %d, col %d, lambda %f\n", row, threadIdInGroup * numRegs + 4*i + 2, foo.z);
            //     if(threadIdInGroup * numRegs + 4*i + 3 < haploLength) printf("row %d, col %d, lambda %f\n", row, threadIdInGroup * numRegs + 4*i + 3, foo.w);
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
            calc_DP_float(k);
            shuffle_penalty();

            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.y;
            calc_DP_float(k);
            shuffle_penalty();

            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.z;
            calc_DP_float(k);
            shuffle_penalty();

            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.w;
            calc_DP_float(k);
            shuffle_penalty();
        }
        if (haploLength%4 >= 1) {
            new_hap_letter4 = HapsAsChar4[k/4];
            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.x;
            calc_DP_float(k);
            shuffle_penalty();
        }
        if (haploLength%4 >= 2) {
            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.y;
            calc_DP_float(k+1);
            shuffle_penalty();
        }
        if (haploLength%4 >= 3) {
            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.z;
            calc_DP_float(k+2);
            shuffle_penalty();
        }
        for (k=0; k<result_thread; k++) {
            // hap_letter = __shfl_up_sync(__activemask(), hap_letter, 1, 32);
            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            calc_DP_float(-1);
            shuffle_penalty(); // shuffle_penalty_active();
        }
        // adjust I values
        I = fmaf(delta[0],M_ul,eps*I_ul);
        Results[0] += I;
        I = fmaf(delta[1],M[0],eps*I);
        Results[1] += I;
        for (int p=1; p<numRegs/2; p++) {
            I = fmaf(delta[2*p],M[2*p-1],eps*I);
            Results[2*p] += I;
            I = fmaf(delta[2*p+1],M[2*p],eps*I);
            Results[2*p+1] += I;
        }
        // adjust I values
        //I = fmaf(delta[0],M_ul,eps*I_ul);
        //Results[0] += I;
        //for (int p=1; p<numRegs; p++) {
        //    I = fmaf(delta[p],M[p-1],eps*I);
        //    Results[p] += I;
        //}

        //repurpose shared memory to stage output. 
        //since the output register index is computed at runtime, 
        //the compiler frequently stores the Results in local memory to be able to load the specific value at the end.
        //doing this once manually in shared memory avoids the atuomatic stores to local memory
        __syncwarp(myGroupMask);
        float* smemOutputBuffer = &lambda_array_permuted[0][0];


        if (threadIdInGroup == result_thread) {
            // float temp_res = Results[result_reg];

            float4* smemOutputBuffer4 = (float4*)smemOutputBuffer;
            #pragma unroll
            for(int i = 0; i < numRegs/4; i++){
                //need to ensure that we only access smem elements which are used by the group. here we use the same access pattern as during computations (warp striped)
                smemOutputBuffer4[i * warpsize + threadIdInWarp] = *((float4*)&Results[4*i]);
            }
            float temp_res = smemOutputBuffer[4*(result_reg/4 * warpsize + threadIdInWarp) + (result_reg % 4)];

            temp_res =  log10f(temp_res) - log10f(constant);
            devAlignmentScores[resultOutputIndex] = temp_res;
        }
    }


}





//###################################################################


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
            if (!group_id) hap_letter = new_hap_letter4.z;
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
            if (!group_id) hap_letter = new_hap_letter4.z;
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
            if (!group_id) hap_letter = new_hap_letter4.z;
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
public:
    template<class T>
    using Allocator = std::allocator<T>;
    // using Allocator = PinnedAllocator<T>;

    std::vector<uint8_t, Allocator<uint8_t>> reads;             //all reads padded
    std::vector<uint8_t, Allocator<uint8_t>> haps;              //all haplotypes padded

    std::vector<uint8_t, Allocator<uint8_t>> base_quals;        //base_qual - offset(33)
    std::vector<uint8_t, Allocator<uint8_t>> ins_quals;         //ins_qual - offset(33)
    std::vector<uint8_t, Allocator<uint8_t>> del_quals;         //del_qual - offset(33)
    std::vector<uint8_t, Allocator<uint8_t>> gcp_quals;         //gep_qual - offset(33)

    std::vector<int, Allocator<int>> batch_haps;            //number of haplotypes per batch
    std::vector<int, Allocator<int>> batch_reads;           //number of reads/qualities per batch

    std::vector<int, Allocator<int>> batch_haps_offsets;    //offsets of reads/qualities between batches
    std::vector<int, Allocator<int>> batch_reads_offsets;   //offsets of haplotypes between batches

    std::vector<int, Allocator<int>> readlen;               //length of each read/quality
    std::vector<int, Allocator<int>> haplen;                //length of each haplotype

    std::vector<int, Allocator<int>> read_offsets;          //offset between reads/qualities
    std::vector<int, Allocator<int>> hap_offsets;           //offset between hyplotypes

    int lastreadoffset;
    int lasthapoffset;


    /*
    std::vector<float> base_quals;
    std::vector<float> ins_quals;
    std::vector<float> del_quals;
    std::vector<float> gcp_quals;
    */

    int getTotalNumberOfAlignments() const{
        if(!totalNumberOfAlignments_opt.has_value()){
            initTotalNumberOfAlignmentsAndNumPerBatch();
        }
        return totalNumberOfAlignments_opt.value();
    }

    const std::vector<int>& getNumberOfAlignmentsPerBatch() const{
        if(!numberOfAlignmentsPerBatch_opt.has_value()){
            initTotalNumberOfAlignmentsAndNumPerBatch();
        }
        return numberOfAlignmentsPerBatch_opt.value();
    }

    const std::vector<int>& getNumberOfAlignmentsPerBatchInclusivePrefixSum() const{
        if(!numAlignmentsPerBatchInclusivePrefixSum_opt.has_value()){
            initNumPerBatchInclPrefixSum();
        }
        return numAlignmentsPerBatchInclusivePrefixSum_opt.value();
    }

    struct AlignmentInputInfo{
        int batchId{};
        int hapToProcessInBatch{};
        int readToProcessInBatch{};
        int alignmentOffset{};
    };

    AlignmentInputInfo getAlignmentInputInfo(int alignmentId) const {
        assert(alignmentId < getTotalNumberOfAlignments());

        const int numBatches = batch_haps.size();
        const int batchIdByAlignmentId = std::distance(
            getNumberOfAlignmentsPerBatchInclusivePrefixSum().begin(),
            std::upper_bound(
                getNumberOfAlignmentsPerBatchInclusivePrefixSum().begin(),
                getNumberOfAlignmentsPerBatchInclusivePrefixSum().begin() + numBatches,
                alignmentId
            )
        );
        AlignmentInputInfo inputInfo;
        const int batchId = min(batchIdByAlignmentId, numBatches-1);
        const int numHapsInBatch = batch_haps[batchId];
        const int alignmentOffset = (batchId == 0 ? 0 : getNumberOfAlignmentsPerBatchInclusivePrefixSum()[batchId-1]);
        const int alignmentIdInBatch = alignmentId - alignmentOffset;
        const int hapToProcessInBatch = alignmentIdInBatch % numHapsInBatch;
        const int readToProcessInBatch = alignmentIdInBatch / numHapsInBatch;

        inputInfo.batchId = batchId;
        inputInfo.hapToProcessInBatch = hapToProcessInBatch;
        inputInfo.readToProcessInBatch = readToProcessInBatch;
        inputInfo.alignmentOffset = alignmentOffset;

        return inputInfo;
    }

private:
    mutable std::optional<std::vector<int>> numberOfAlignmentsPerBatch_opt;
    mutable std::optional<std::vector<int>> numAlignmentsPerBatchInclusivePrefixSum_opt;
    mutable std::optional<int> totalNumberOfAlignments_opt;

    void initTotalNumberOfAlignmentsAndNumPerBatch() const{
        const int numBatches = batch_haps.size();
        std::vector<int> numberOfAlignmentsPerBatch(numBatches);

        int sum = 0;
        #pragma omp parallel for reduction(+:sum)
        for (int i=0; i < numBatches; i++){
            const int numReadsInBatch = batch_reads[i];
            const int numHapsInBatch = batch_haps[i];
            const int numAlignments = numReadsInBatch * numHapsInBatch;
            numberOfAlignmentsPerBatch[i] = numAlignments;
            sum += numAlignments;
        }
        totalNumberOfAlignments_opt = sum;
        numberOfAlignmentsPerBatch_opt = std::move(numberOfAlignmentsPerBatch);
    } 
    
    void initNumPerBatchInclPrefixSum() const{        
        const int numBatches = batch_haps.size();
        std::vector<int> numAlignmentsPerBatchInclusivePrefixSum(numBatches);
        numAlignmentsPerBatchInclusivePrefixSum[0] = getNumberOfAlignmentsPerBatch()[0];
        for (int i=1; i < numBatches; i++){
            numAlignmentsPerBatchInclusivePrefixSum[i] 
                = numAlignmentsPerBatchInclusivePrefixSum[i-1] + getNumberOfAlignmentsPerBatch()[i];
        }
        numAlignmentsPerBatchInclusivePrefixSum_opt = std::move(numAlignmentsPerBatchInclusivePrefixSum);
    }




};

struct pinned_batch{
    template<class T>
    using Allocator = PinnedAllocator<T>;

    pinned_batch() = default;
    pinned_batch(const pinned_batch&) = default;
    pinned_batch& operator=(const pinned_batch&) = default;
    
    pinned_batch(const batch& rhs)
    : reads(rhs.reads.begin(), rhs.reads.end()),
     haps(rhs.haps.begin(), rhs.haps.end()),
     base_quals(rhs.base_quals.begin(), rhs.base_quals.end()),
     ins_quals(rhs.ins_quals.begin(), rhs.ins_quals.end()),
     del_quals(rhs.del_quals.begin(), rhs.del_quals.end()),
     gcp_quals(rhs.gcp_quals.begin(), rhs.gcp_quals.end()),
     batch_haps(rhs.batch_haps.begin(), rhs.batch_haps.end()),
     batch_reads(rhs.batch_reads.begin(), rhs.batch_reads.end()),
     batch_haps_offsets(rhs.batch_haps_offsets.begin(), rhs.batch_haps_offsets.end()),
     batch_reads_offsets(rhs.batch_reads_offsets.begin(), rhs.batch_reads_offsets.end()),
     readlen(rhs.readlen.begin(), rhs.readlen.end()),
     haplen(rhs.haplen.begin(), rhs.haplen.end()),
     read_offsets(rhs.read_offsets.begin(), rhs.read_offsets.end()),
     hap_offsets(rhs.hap_offsets.begin(), rhs.hap_offsets.end()),
     lastreadoffset(rhs.lastreadoffset),
     lasthapoffset(rhs.lasthapoffset)
    {}

    std::vector<uint8_t, Allocator<uint8_t>> reads;             //all reads padded
    std::vector<uint8_t, Allocator<uint8_t>> haps;              //all haplotypes padded

    std::vector<uint8_t, Allocator<uint8_t>> base_quals;        //base_qual - offset(33)
    std::vector<uint8_t, Allocator<uint8_t>> ins_quals;         //ins_qual - offset(33)
    std::vector<uint8_t, Allocator<uint8_t>> del_quals;         //del_qual - offset(33)
    std::vector<uint8_t, Allocator<uint8_t>> gcp_quals;         //gep_qual - offset(33)

    std::vector<int, Allocator<int>> batch_haps;            //number of haplotypes per batch
    std::vector<int, Allocator<int>> batch_reads;           //number of reads/qualities per batch

    std::vector<int, Allocator<int>> batch_haps_offsets;    //offsets of reads/qualities between batches
    std::vector<int, Allocator<int>> batch_reads_offsets;   //offsets of haplotypes between batches

    std::vector<int, Allocator<int>> readlen;               //length of each read/quality
    std::vector<int, Allocator<int>> haplen;                //length of each haplotype

    std::vector<int, Allocator<int>> read_offsets;          //offset between reads/qualities
    std::vector<int, Allocator<int>> hap_offsets;           //offset between hyplotypes

    int lastreadoffset;
    int lasthapoffset;


    /*
    std::vector<float> base_quals;
    std::vector<float> ins_quals;
    std::vector<float> del_quals;
    std::vector<float> gcp_quals;
    */
};


std::int64_t computeNumberOfDPCells(const batch& batch){
    const int numBatches = batch.batch_haps.size();

    std::int64_t result = 0;

    for (int i=0; i<numBatches; i++) {
        const int numReadsInBatch = batch.batch_reads[i];
        const int numHapsInBatch = batch.batch_haps[i];
        for (int k=0; k < numReadsInBatch; k++){
            int read = batch.batch_reads_offsets[i]+k;
            int read_len = batch.readlen[read];
            for (int j=0; j < numHapsInBatch; j++){
                int hap = batch.batch_haps_offsets[i]+j;
                int hap_len = batch.haplen[hap];
                result += read_len * hap_len;
            }
        }
    }

    return result;
}



std::vector<std::int64_t> computeNumberOfDPCellsPerPartition(const batch& batch){
    PartitionLimits partitionLimits;
    std::vector<std::int64_t> dpCellsPerPartition(numPartitions, 0);

    auto getPartitionId = [&](int length){
        int id = -1;
        for(int i = 0; i < numPartitions; i++){
            if(length <= partitionLimits.boundaries[i]){
                id = i;
                break;
            }
        }
        assert(id != -1);
        return id;
    };

    const int numBatches = batch.batch_haps.size();

    for (int i=0; i<numBatches; i++) {
        const int numReadsInBatch = batch.batch_reads[i];
        const int numHapsInBatch = batch.batch_haps[i];
        for (int k=0; k < numReadsInBatch; k++){
            int read = batch.batch_reads_offsets[i]+k;
            int read_len = batch.readlen[read];
            const int p = getPartitionId(read_len);
            for (int j=0; j < numHapsInBatch; j++){
                int hap = batch.batch_haps_offsets[i]+j;
                int hap_len = batch.haplen[hap];
                dpCellsPerPartition[p] += read_len * hap_len;
            }
        }
    }

    return dpCellsPerPartition;
}


struct CountsOfDPCells{
    std::int64_t totalDPCells = 0;
    std::vector<std::int64_t> dpCellsPerPartition{};
};

CountsOfDPCells countDPCellsInBatch(const batch& batch){
    CountsOfDPCells result;
    result.dpCellsPerPartition = computeNumberOfDPCellsPerPartition(batch);
    result.totalDPCells = std::reduce(result.dpCellsPerPartition.begin(), result.dpCellsPerPartition.end(), std::int64_t(0));
    return result;
}




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

batch read_batch(std::ifstream& file_, const std::vector<float>& /*ph2pr*/, int nread, int nhapl, bool print=false, int lastreadoffset=0, int lasthapoffset=0){

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

        // uint8_t t = 0;
        // float score = 0;
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
    const uint8_t* hap_bases,
    const int hap_length,
    const uint8_t* read_bases,
    const int read_length,
    const uint8_t* base_quals,
    const uint8_t* ins_quals,
    const uint8_t* del_quals,
    const uint8_t gcp_qual,
    const float* ph2pr  // constant
) {
    double result = 0;
    Context<float> ctx;

    const double constant = std::numeric_limits<float>::max() / 16;

    std::vector<float> M(2*(hap_length+1));
    std::vector<float> I(2*(hap_length+1));
    std::vector<float> D(2*(hap_length+1));
    int target_row = 0;
    int source_row = 0;

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
            //transposed print because kernel has hap as outer loop
            // std::cout << "row " << (col-1) << ", col " << (row-1) << ", lambda " << lambda << "\n";
            //std::cout << "row " << (row-1) << ", col " << (col-1) << ", lambda " << lambda << "\n";
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
                // std::cout << counter << " Haplo " << hap << " len: " << hap_len <<  " Read " << read << " len: " << read_len << " score: " << score << "\n";
                //std::cout << counter << " Haplo " << hap << " len: " << hap_len <<  " offset: " << h_off <<  " Read " << read << " len: " << read_len << " offset: " << r_off << " score: " << score << "\n";
            }
    }
}

std::vector<float> processBatchCPU(const batch& batch_, const std::vector<float>& ph2pr){
    helpers::CpuTimer totalTimer("processBatchCPU");
    const uint64_t dp_cells = computeNumberOfDPCells(batch_);

    const int numBatches = batch_.batch_haps.size();

    std::vector<int> numberOfAlignmentsPerBatch(numBatches);
    int totalNumberOfAlignments = 0;
    #pragma omp parallel for reduction(+:totalNumberOfAlignments)
    for (int i=0; i < numBatches; i++){
        const int numReadsInBatch = batch_.batch_reads[i];
        const int numHapsInBatch = batch_.batch_haps[i];
        const int numAlignments = numReadsInBatch * numHapsInBatch;
        numberOfAlignmentsPerBatch[i] = numAlignments;
        totalNumberOfAlignments += numAlignments;
    }
    std::vector<int> numberOfAlignmentsPerBatchExclPrefixSum(numBatches);
    numberOfAlignmentsPerBatchExclPrefixSum[0] = 0;
    for (int i=1; i < numBatches; i++){
        numberOfAlignmentsPerBatchExclPrefixSum[i] 
            = numberOfAlignmentsPerBatchExclPrefixSum[i-1] + numberOfAlignmentsPerBatch[i-1];
    }

    std::vector<float> results(totalNumberOfAlignments);
    
    #pragma omp parallel for schedule(dynamic)
    for (int i=0; i<numBatches; i++) {

        const int numReadsInBatch = batch_.batch_reads[i];
        const int numHapsInBatch = batch_.batch_haps[i];
        for (int k=0; k < numReadsInBatch; k++){
            for (int j=0; j < numHapsInBatch; j++){
                int read = batch_.batch_reads_offsets[i]+k;
                int read_len = batch_.readlen[read];
                int hap = batch_.batch_haps_offsets[i]+j;
                int hap_len = batch_.haplen[hap];
                int h_off = batch_.hap_offsets[hap];
                int r_off = batch_.read_offsets[read];
                double score = align_host(&batch_.haps[h_off],hap_len,&batch_.reads[r_off],read_len,&batch_.base_quals[r_off],&batch_.ins_quals[r_off],&batch_.del_quals[r_off],batch_.gcp_quals[r_off],&ph2pr[0]);
                
                const int outputIndex = numberOfAlignmentsPerBatchExclPrefixSum[i] + k * numHapsInBatch + j;
                // const int outputIndex = numberOfAlignmentsPerBatchExclPrefixSum[i] + j * numReadsInBatch + k;
                results[outputIndex] = score;
            }
        }
    }

    totalTimer.stop();
    totalTimer.printGCUPS(dp_cells);

    return results;
}


std::vector<float> processBatchCPUFaster(const batch& batch_, const std::vector<float>& ph2pr){
    helpers::CpuTimer totalTimer("processBatchCPUFaster");
    const uint64_t dp_cells = computeNumberOfDPCells(batch_);

    const int numBatches = batch_.batch_haps.size();

    const int totalNumberOfAlignments = batch_.getTotalNumberOfAlignments();
    const auto& numAlignmentsPerBatchInclusivePrefixSum = batch_.getNumberOfAlignmentsPerBatchInclusivePrefixSum();

    std::vector<float> results(totalNumberOfAlignments);

    #pragma omp parallel for schedule(dynamic, 16)
    for(int alignmentId = 0; alignmentId < totalNumberOfAlignments; alignmentId++){
        const int batchIdByGroupId = std::distance(
            numAlignmentsPerBatchInclusivePrefixSum.begin(),
            std::upper_bound(
                numAlignmentsPerBatchInclusivePrefixSum.begin(),
                numAlignmentsPerBatchInclusivePrefixSum.begin() + numBatches,
                alignmentId
            )
        );
        const int batchId = min(batchIdByGroupId, numBatches-1);
        const int numHapsInBatch = batch_.batch_haps[batchId];
        const int alignmentOffset = (batchId == 0 ? 0 : numAlignmentsPerBatchInclusivePrefixSum[batchId-1]);
        const int alignmentIdInBatch = alignmentId - alignmentOffset;
        const int hapToProcessInBatch = alignmentIdInBatch % numHapsInBatch;
        const int readToProcessInBatch = alignmentIdInBatch / numHapsInBatch;

        int read = batch_.batch_reads_offsets[batchId]+readToProcessInBatch;
        int read_len = batch_.readlen[read];
        int hap = batch_.batch_haps_offsets[batchId]+hapToProcessInBatch;
        int hap_len = batch_.haplen[hap];
        int h_off = batch_.hap_offsets[hap];
        int r_off = batch_.read_offsets[read];
        double score = align_host(&batch_.haps[h_off],hap_len,&batch_.reads[r_off],read_len,&batch_.base_quals[r_off],&batch_.ins_quals[r_off],&batch_.del_quals[r_off],batch_.gcp_quals[r_off],&ph2pr[0]);
        
        const int outputIndex = alignmentOffset + readToProcessInBatch * numHapsInBatch + hapToProcessInBatch;
        // const int outputIndex = alignmentOffset + hapToProcessInBatch * numReadsInBatch + readToProcessInBatch;
        results[outputIndex] = score;
    }

    totalTimer.stop();
    totalTimer.printGCUPS(dp_cells);

    return results;
}


std::vector<float> processBatchAsWhole_half(
    const batch& fullBatch, 
    const Options& /*options*/, 
    const CountsOfDPCells& countsOfDPCells
){
    helpers::CpuTimer totalTimer("processBatchAsWhole_half");

    const uint8_t* read_chars       = fullBatch.reads.data(); //batch_2.chars.data();
    const uint read_bytes = fullBatch.reads.size();
    const uint8_t* hap_chars       = fullBatch.haps.data(); //batch_2.chars.data();
    const uint hap_bytes = fullBatch.haps.size();
    const uint8_t* base_qual       = fullBatch.base_quals.data(); //batch_2.chars.data();
    const uint8_t* ins_qual       = fullBatch.ins_quals.data(); //batch_2.chars.data();
    const uint8_t* del_qual       = fullBatch.del_quals.data(); //batch_2.chars.data();
    const int* offset_reads       = fullBatch.read_offsets.data(); //batch_2.chars.data();
    const int* offset_haps       = fullBatch.hap_offsets.data(); //batch_2.chars.data();
    const int* read_len       = fullBatch.readlen.data(); //batch_2.chars.data();
    const int* hap_len       = fullBatch.haplen.data(); //batch_2.chars.data();
    const int num_reads = fullBatch.readlen.size(); //batch_2.chars.data();
    const int num_haps = fullBatch.haplen.size(); //batch_2.chars.data();
    const int num_batches = fullBatch.batch_reads.size(); //batch_2.chars.data();
    const int* hap_batches       = fullBatch.batch_haps.data(); //batch_2.chars.data();
    const int* read_batches       = fullBatch.batch_reads.data(); //batch_2.chars.data();
    const int* offset_hap_batches       = fullBatch.batch_haps_offsets.data(); //batch_2.chars.data();
    const int* offset_read_batches       = fullBatch.batch_reads_offsets.data(); //batch_2.chars.data();

    const int totalNumberOfAlignments = fullBatch.getTotalNumberOfAlignments();

    std::vector<float> alignment_scores_float(totalNumberOfAlignments);

    cudaStream_t streams_part[numPartitions];
    for (int i=0; i<numPartitions; i++) cudaStreamCreate(&streams_part[i]);

    thrust::device_vector<uint8_t> dev_read_chars_vec(read_bytes);
    thrust::device_vector<uint8_t> dev_hap_chars_vec(hap_bytes);
    thrust::device_vector<uint8_t> dev_base_qual_vec(read_bytes);
    thrust::device_vector<uint8_t> dev_ins_qual_vec(read_bytes);
    thrust::device_vector<uint8_t> dev_del_qual_vec(read_bytes);

    thrust::device_vector<int> dev_offset_reads_vec(num_reads);
    thrust::device_vector<int> dev_offset_haps_vec(num_haps);
    thrust::device_vector<int> dev_read_len_vec(num_reads);
    thrust::device_vector<int> dev_hap_len_vec(num_haps);
    thrust::device_vector<int> dev_read_batches_vec(num_batches);
    thrust::device_vector<int> dev_hap_batches_vec(num_batches);
    thrust::device_vector<int> dev_offset_read_batches_vec(num_batches);
    thrust::device_vector<int> dev_offset_hap_batches_vec(num_batches);
    thrust::device_vector<float> devAlignmentScoresFloat_vec(totalNumberOfAlignments);

    uint8_t* const dev_read_chars = dev_read_chars_vec.data().get();
    uint8_t* const dev_hap_chars = dev_hap_chars_vec.data().get();
    uint8_t* const dev_base_qual = dev_base_qual_vec.data().get();
    uint8_t* const dev_ins_qual = dev_ins_qual_vec.data().get();
    uint8_t* const dev_del_qual = dev_del_qual_vec.data().get();
    int* const dev_offset_reads = dev_offset_reads_vec.data().get();
    int* const dev_offset_haps = dev_offset_haps_vec.data().get();
    int* const dev_read_len = dev_read_len_vec.data().get();
    int* const dev_hap_len = dev_hap_len_vec.data().get();
    int* const dev_read_batches = dev_read_batches_vec.data().get();
    int* const dev_hap_batches = dev_hap_batches_vec.data().get();
    int* const dev_offset_read_batches = dev_offset_read_batches_vec.data().get();
    int* const dev_offset_hap_batches = dev_offset_hap_batches_vec.data().get();
    float* const devAlignmentScoresFloat = devAlignmentScoresFloat_vec.data().get();


    helpers::CpuTimer transfertimer("DATA_TRANSFER");
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
    transfertimer.print();

    //print_batch(fullBatch); // prints first reads/qualities and haplotype per batch.

    convert_DNA<<<num_reads, 128>>>(dev_read_chars,read_bytes);
    convert_DNA<<<num_haps, 128>>>(dev_hap_chars,hap_bytes);

    thrust::device_vector<int> d_numIndicesPerPartitionPerBatch(num_batches * numPartitions, 0);
    thrust::device_vector<int> d_indicesPerPartitionPerBatch(num_reads * numPartitions, -1);
    thrust::device_vector<int> d_resultOffsetsPerBatch(num_batches);
    thrust::device_vector<int> d_numAlignmentsPerBatch(num_batches * numPartitions);
    thrust::device_vector<int> d_numAlignmentsPerBatchInclPrefixSum(num_batches * numPartitions);
    thrust::device_vector<int> d_numAlignmentsPerPartition(numPartitions);

    partitionIndicesKernel<<<num_batches, 128>>>(
        d_numIndicesPerPartitionPerBatch.data().get(),
        d_indicesPerPartitionPerBatch.data().get(),
        dev_read_len,
        dev_read_batches,
        dev_offset_read_batches,
        num_batches,
        num_reads
    );
    CUERR;

    thrust::transform(
        thrust::cuda::par_nosync.on((cudaStream_t)0),
        dev_read_batches,
        dev_read_batches + num_batches,
        dev_hap_batches,
        d_resultOffsetsPerBatch.begin(),
        thrust::multiplies<int>{}
    );
    thrust::exclusive_scan(
        thrust::cuda::par_nosync(ThrustCudaMallocAsyncAllocator<int>((cudaStream_t)0)).on((cudaStream_t)0),
        d_resultOffsetsPerBatch.begin(),
        d_resultOffsetsPerBatch.begin() + num_batches,
        d_resultOffsetsPerBatch.begin()
    );

    // for(int i = 0; i < std::min(10, num_batches); i++){
    //     std::cout << "batch " << i << ", num reads: " << read_batches[i]
    //         << ", num haps " << hap_batches[i] << ", product: " <<
    //         read_batches[i] * hap_batches[i]
    //         << ", offset : " << d_resultOffsetsPerBatch[i] << "\n";
    // }


    #if 0
        thrust::host_vector<int> h_numIndicesPerPartitionPerBatch = d_numIndicesPerPartitionPerBatch;
        thrust::host_vector<int> h_indicesPerPartitionPerBatch = d_indicesPerPartitionPerBatch;

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
            }
        }
    #endif

    {
        cudaStream_t stream = cudaStreamLegacy;

        int* d_numAlignmentsPerPartitionPerBatch = d_numAlignmentsPerBatchInclPrefixSum.data().get(); // reuse
        const int* d_numHaplotypesPerBatch = dev_hap_batches;
        computeAlignmentsPerPartitionPerBatch<<<dim3(SDIV(num_batches, 128), numPartitions), 128,0, stream>>>(
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
            d_numAlignmentsPerPartition.data().get(), numPartitions, offsets, offsets + 1, stream); CUERR;
        thrust::device_vector<char, ThrustCudaMallocAsyncAllocator<char>> d_temp(temp_storage_bytes, ThrustCudaMallocAsyncAllocator<char>(stream));
        cub::DeviceSegmentedReduce::Sum(d_temp.data().get(), temp_storage_bytes, d_numAlignmentsPerPartitionPerBatch, 
            d_numAlignmentsPerPartition.data().get(), numPartitions, offsets, offsets + 1, stream); CUERR;
    }
    thrust::host_vector<int> h_numAlignmentsPerPartition = d_numAlignmentsPerPartition;

    std::cout << "h_numAlignmentsPerPartition: ";
    for(int i = 0; i< numPartitions; i++){
        std::cout << h_numAlignmentsPerPartition[i] << ", ";
    }
    std::cout << "\n";

    cudaMemset(devAlignmentScoresFloat, 0, totalNumberOfAlignments * sizeof(float)); CUERR;
    helpers::GpuTimer computeTimer("pairhmm half kernels, total");
    std::vector<std::unique_ptr<helpers::GpuTimer>> perKernelTimers(numPartitions);
    for(int p = 0; p < numPartitions; p++){
        std::string name = "pairhmm half kernel, partition " + std::to_string(p);
        perKernelTimers[p] = std::make_unique<helpers::GpuTimer>(streams_part[p], name);
    }

    #define COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM(stream) \
        constexpr int groupsPerBlock = blocksize / group_size; \
        const int numAlignmentsInPartition = h_numAlignmentsPerPartition[partitionId]; \
        const int numBlocks = SDIV(numAlignmentsInPartition, groupsPerBlock); \
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
        perKernelTimers[partitionId]->start(); \
        PairHMM_align_partition_half_allowMultipleBatchesPerWarp<group_size,numRegs><<<numBlocks, blocksize,0,stream>>>(dev_read_chars, dev_hap_chars, dev_base_qual, dev_ins_qual, dev_del_qual, devAlignmentScoresFloat, dev_offset_reads, dev_offset_haps, dev_read_len, dev_hap_len, dev_read_batches, dev_hap_batches, dev_offset_hap_batches,  \
            d_numIndicesPerBatch, d_indicesPerPartitionPerBatch.data().get() + partitionId*num_reads,  \
            dev_offset_read_batches,  num_batches, d_resultOffsetsPerBatch.data().get(), d_numAlignmentsPerBatch.data().get() + partitionId * num_batches, d_numAlignmentsPerBatchInclPrefixSum.data().get() + partitionId * num_batches, numAlignmentsInPartition); \
        perKernelTimers[partitionId]->stop(); \


    LAUNCH_ALL_KERNELS

    #undef  COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM

    computeTimer.stop();

    for(int p = 0; p < numPartitions; p++){
        if(h_numAlignmentsPerPartition[p] > 0){
            perKernelTimers[p]->printGCUPS(countsOfDPCells.dpCellsPerPartition[p]);
        }
    }

    computeTimer.printGCUPS(countsOfDPCells.totalDPCells);

    cudaMemcpy(alignment_scores_float.data(), devAlignmentScoresFloat, totalNumberOfAlignments*sizeof(float), cudaMemcpyDeviceToHost);  CUERR

    perKernelTimers.clear();

    for (int i=0; i<numPartitions; i++) cudaStreamDestroy(streams_part[i]); CUERR;

    totalTimer.stop();
    totalTimer.printGCUPS(countsOfDPCells.totalDPCells);

    return alignment_scores_float;
}






std::vector<float> processBatch_overlapped_half(
    const batch& fullBatch_default, 
    const Options& options, 
    const CountsOfDPCells& countsOfDPCells
){
    helpers::CpuTimer totalTimer("processBatch_overlapped_half");

    // pinned_batch fullBatch(fullBatch_default);
    const auto& fullBatch = fullBatch_default;

    const uint8_t* read_chars       = fullBatch.reads.data(); //batch_2.chars.data();
    const uint read_bytes = fullBatch.reads.size();
    const uint8_t* hap_chars       = fullBatch.haps.data(); //batch_2.chars.data();
    const uint hap_bytes = fullBatch.haps.size();
    const uint8_t* base_qual       = fullBatch.base_quals.data(); //batch_2.chars.data();
    const uint8_t* ins_qual       = fullBatch.ins_quals.data(); //batch_2.chars.data();
    const uint8_t* del_qual       = fullBatch.del_quals.data(); //batch_2.chars.data();
    const int* offset_reads       = fullBatch.read_offsets.data(); //batch_2.chars.data();
    const int* offset_haps       = fullBatch.hap_offsets.data(); //batch_2.chars.data();
    const int* read_len       = fullBatch.readlen.data(); //batch_2.chars.data();
    const int* hap_len       = fullBatch.haplen.data(); //batch_2.chars.data();
    const int num_reads = fullBatch.readlen.size(); //batch_2.chars.data();
    const int num_haps = fullBatch.haplen.size(); //batch_2.chars.data();
    const int num_batches = fullBatch.batch_reads.size(); //batch_2.chars.data();
    const int* hap_batches       = fullBatch.batch_haps.data(); //batch_2.chars.data();
    const int* read_batches       = fullBatch.batch_reads.data(); //batch_2.chars.data();
    const int* offset_hap_batches       = fullBatch.batch_haps_offsets.data(); //batch_2.chars.data();
    const int* offset_read_batches       = fullBatch.batch_reads_offsets.data(); //batch_2.chars.data();

    const int totalNumberOfAlignments = fullBatch.getTotalNumberOfAlignments();

    std::vector<float> alignment_scores_float(totalNumberOfAlignments);

    cudaStream_t streams_part[numPartitions];
    for (int i=0; i<numPartitions; i++) cudaStreamCreate(&streams_part[i]);

    std::vector<cudaStream_t> transferStreams(2);
    for(auto& stream : transferStreams){
        cudaStreamCreate(&stream);
    }

    thrust::device_vector<float> devAlignmentScoresFloat_vec(totalNumberOfAlignments, 0);

    thrust::device_vector<uint8_t> dev_read_chars_vec(read_bytes);
    thrust::device_vector<uint8_t> dev_hap_chars_vec(hap_bytes);
    thrust::device_vector<uint8_t> dev_base_qual_vec(read_bytes);
    thrust::device_vector<uint8_t> dev_ins_qual_vec(read_bytes);
    thrust::device_vector<uint8_t> dev_del_qual_vec(read_bytes);

    thrust::device_vector<int> dev_offset_reads_vec(num_reads);
    thrust::device_vector<int> dev_offset_haps_vec(num_haps);
    thrust::device_vector<int> dev_read_len_vec(num_reads);
    thrust::device_vector<int> dev_hap_len_vec(num_haps);
    thrust::device_vector<int> dev_read_batches_vec(num_batches);
    thrust::device_vector<int> dev_hap_batches_vec(num_batches);
    thrust::device_vector<int> dev_offset_read_batches_vec(num_batches);
    thrust::device_vector<int> dev_offset_hap_batches_vec(num_batches);

    thrust::device_vector<int> d_numIndicesPerPartitionPerBatch(num_batches * numPartitions, 0);
    thrust::device_vector<int> d_indicesPerPartitionPerBatch(num_reads * numPartitions, -1);
    thrust::device_vector<int> d_resultOffsetsPerBatch(num_batches);
    thrust::device_vector<int> d_numAlignmentsPerBatch(num_batches * numPartitions);
    thrust::device_vector<int> d_numAlignmentsPerBatchInclPrefixSum(num_batches * numPartitions);
    thrust::device_vector<int> d_numAlignmentsPerPartition(numPartitions);
    std::vector<int, PinnedAllocator<int>> h_numAlignmentsPerPartition(numPartitions);

    uint8_t* const dev_read_chars = dev_read_chars_vec.data().get();
    uint8_t* const dev_hap_chars = dev_hap_chars_vec.data().get();
    uint8_t* const dev_base_qual = dev_base_qual_vec.data().get();
    uint8_t* const dev_ins_qual = dev_ins_qual_vec.data().get();
    uint8_t* const dev_del_qual = dev_del_qual_vec.data().get();
    int* const dev_offset_reads = dev_offset_reads_vec.data().get();
    int* const dev_offset_haps = dev_offset_haps_vec.data().get();
    int* const dev_read_len = dev_read_len_vec.data().get();
    int* const dev_hap_len = dev_hap_len_vec.data().get();
    int* const dev_read_batches = dev_read_batches_vec.data().get();
    int* const dev_hap_batches = dev_hap_batches_vec.data().get();
    int* const dev_offset_read_batches = dev_offset_read_batches_vec.data().get();
    int* const dev_offset_hap_batches = dev_offset_hap_batches_vec.data().get();
    float* const devAlignmentScoresFloat = devAlignmentScoresFloat_vec.data().get();

    int numProcessedAlignmentsByChunks = 0;
    int numProcessedBatchesByChunks = 0;

    const int numTransferChunks = SDIV(num_batches, options.transferchunksize);

    for(int computeChunk = 0, transferChunk = 0; computeChunk < numTransferChunks; computeChunk++){
        for(; transferChunk < numTransferChunks && transferChunk < (computeChunk + 2); transferChunk++){
            nvtx3::scoped_range sr1("transferChunk");
            cudaStream_t transferStream = transferStreams[transferChunk % 2];
            
            const int firstBatchId = transferChunk * options.transferchunksize;
            const int lastBatchId_excl = std::min((transferChunk+1)* options.transferchunksize, num_batches);
            const int numBatchesInChunk = lastBatchId_excl - firstBatchId;

            const int firstReadInChunk = offset_read_batches[firstBatchId];
            const int lastReadInChunk_excl = offset_read_batches[lastBatchId_excl];
            const int numReadsInChunk = lastReadInChunk_excl - firstReadInChunk;

            const int firstHapInChunk = offset_hap_batches[firstBatchId];
            const int lastHapInChunk_excl = offset_hap_batches[lastBatchId_excl];
            const int numHapsInChunk = lastHapInChunk_excl - firstHapInChunk;

            const size_t numReadBytesInChunk = offset_reads[lastReadInChunk_excl] - offset_reads[firstReadInChunk];
            const size_t numHapBytesInChunk = offset_haps[lastHapInChunk_excl] - offset_haps[firstHapInChunk];

            // std::cout << "transferChunk " << transferChunk << "\n";
            // std::cout << "firstBatchId " << firstBatchId << "\n";
            // std::cout << "lastBatchId_excl " << lastBatchId_excl << "\n";
            // std::cout << "numBatchesInChunk " << numBatchesInChunk << "\n";
            // std::cout << "firstReadInChunk " << firstReadInChunk << "\n";
            // std::cout << "lastReadInChunk_excl " << lastReadInChunk_excl << "\n";
            // std::cout << "numReadsInChunk " << numReadsInChunk << "\n";
            // std::cout << "firstHapInChunk " << firstHapInChunk << "\n";
            // std::cout << "lastHapInChunk_excl " << lastHapInChunk_excl << "\n";
            // std::cout << "numHapsInChunk " << numHapsInChunk << "\n";
            // std::cout << "numReadBytesInChunk " << numReadBytesInChunk << "\n";
            // std::cout << "numHapBytesInChunk " << numHapBytesInChunk << "\n";
            // std::cout << "----------------------------\n";



            cudaMemcpyAsync(dev_read_chars + offset_reads[firstReadInChunk], read_chars + offset_reads[firstReadInChunk], numReadBytesInChunk, cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_hap_chars + offset_haps[firstHapInChunk], hap_chars + offset_haps[firstHapInChunk], numHapBytesInChunk, cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_base_qual + offset_reads[firstReadInChunk], base_qual + offset_reads[firstReadInChunk], numReadBytesInChunk, cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_ins_qual + offset_reads[firstReadInChunk], ins_qual + offset_reads[firstReadInChunk], numReadBytesInChunk, cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_del_qual + offset_reads[firstReadInChunk], del_qual + offset_reads[firstReadInChunk], numReadBytesInChunk, cudaMemcpyHostToDevice, transferStream); CUERR

            cudaMemcpyAsync(dev_offset_reads + firstReadInChunk, offset_reads + firstReadInChunk, numReadsInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_offset_haps + firstHapInChunk, offset_haps + firstHapInChunk, numHapsInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_read_len + firstReadInChunk, read_len + firstReadInChunk, numReadsInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_hap_len + firstHapInChunk, hap_len + firstHapInChunk, numHapsInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_offset_read_batches + firstBatchId, offset_read_batches + firstBatchId, numBatchesInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_offset_hap_batches + firstBatchId, offset_hap_batches + firstBatchId, numBatchesInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_read_batches + firstBatchId, read_batches + firstBatchId, numBatchesInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_hap_batches + firstBatchId, hap_batches + firstBatchId, numBatchesInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream); CUERR
        }
        nvtx3::scoped_range sr2("computeChunk");
        cudaStream_t mainStream = transferStreams[computeChunk % 2];
        const int firstBatchId = computeChunk * options.transferchunksize;
        const int lastBatchId_excl = std::min((computeChunk+1)* options.transferchunksize, num_batches);
        const int numBatchesInChunk = lastBatchId_excl - firstBatchId;

        const int firstReadInChunk = offset_read_batches[firstBatchId];
        const int lastReadInChunk_excl = offset_read_batches[lastBatchId_excl];
        const int numReadsInChunk = lastReadInChunk_excl - firstReadInChunk;

        const int firstHapInChunk = offset_hap_batches[firstBatchId];
        const int lastHapInChunk_excl = offset_hap_batches[lastBatchId_excl];
        const int numHapsInChunk = lastHapInChunk_excl - firstHapInChunk;

        const size_t numReadBytesInChunk = offset_reads[lastReadInChunk_excl] - offset_reads[firstReadInChunk];
        const size_t numHapBytesInChunk = offset_haps[lastHapInChunk_excl] - offset_haps[firstHapInChunk];

        convert_DNA<<<numReadsInChunk, 128, 0, mainStream>>>(dev_read_chars + offset_reads[firstReadInChunk], numReadBytesInChunk);
        convert_DNA<<<numHapsInChunk, 128, 0, mainStream>>>(dev_hap_chars + offset_haps[firstHapInChunk], numHapBytesInChunk);

        //ensure buffers used by previous batch are no longer in use
        for(int i = 0; i < numPartitions; i++){
            cudaStreamSynchronize(streams_part[i]);
        }

        cudaMemsetAsync(d_numIndicesPerPartitionPerBatch.data().get(), 0, sizeof(int) * numPartitions * numBatchesInChunk, mainStream); CUERR;
        partitionIndicesKernel<<<numBatchesInChunk, 128, 0, mainStream>>>(
            d_numIndicesPerPartitionPerBatch.data().get(),
            d_indicesPerPartitionPerBatch.data().get(),
            dev_read_len,
            dev_read_batches + firstBatchId,
            dev_offset_read_batches + firstBatchId,
            numBatchesInChunk,
            numReadsInChunk
        );
        CUERR;

        #if 0
            thrust::host_vector<int> h_numIndicesPerPartitionPerBatch = d_numIndicesPerPartitionPerBatch;
            thrust::host_vector<int> h_indicesPerPartitionPerBatch = d_indicesPerPartitionPerBatch;

            for(int p = 0; p < numPartitions; p++){
                if(p <= 4){
                    std::cout << "Partition p = " << p << "\n";
                    std::cout << "numIndicesPerBatch: ";
                    for(int b = 0; b < numBatchesInChunk; b++){
                        std::cout << h_numIndicesPerPartitionPerBatch[p * numBatchesInChunk + b] << ", ";
                    }
                    std::cout << "\n";

                    std::cout << "indicesPerBatch: ";
                    for(int b = 0; b < numBatchesInChunk; b++){
                        const int num = h_numIndicesPerPartitionPerBatch[p * numBatchesInChunk + b];
                        for(int i = 0; i < num; i++){
                            const int outputOffset = offset_read_batches[firstBatchId + b] - offset_read_batches[firstBatchId];
                            std::cout << h_indicesPerPartitionPerBatch[p * numReadsInChunk + outputOffset + i];
                            if(i != num-1){
                                std::cout << ", ";
                            }
                        }
                        std::cout << " | ";
                    }
                    std::cout << "\n";
                }
            }
        #endif
    
        thrust::transform(
            thrust::cuda::par_nosync.on(mainStream),
            dev_read_batches + firstBatchId,
            dev_read_batches + firstBatchId + numBatchesInChunk,
            dev_hap_batches + firstBatchId,
            d_resultOffsetsPerBatch.begin(),
            thrust::multiplies<int>{}
        );
        thrust::exclusive_scan(
            thrust::cuda::par_nosync(ThrustCudaMallocAsyncAllocator<int>(mainStream)).on(mainStream),
            d_resultOffsetsPerBatch.begin(),
            d_resultOffsetsPerBatch.begin() + numBatchesInChunk,
            d_resultOffsetsPerBatch.begin()
        );

        // thrust::host_vector<int> h_resultOffsetsPerBatch = d_resultOffsetsPerBatch;
        // std::cout << "h_resultOffsetsPerBatch. numBatchesInChunk " << numBatchesInChunk << "\n";
        // for(auto& x : h_resultOffsetsPerBatch){
        //     std::cout << x << " ";
        // }
        // std::cout << "\n";

        {     
            int* d_numAlignmentsPerPartitionPerBatch = d_numAlignmentsPerBatchInclPrefixSum.data().get(); // reuse
            const int* d_numHaplotypesPerBatch = dev_hap_batches;
            computeAlignmentsPerPartitionPerBatch<<<dim3(SDIV(numBatchesInChunk, 128), numPartitions), 128,0, mainStream>>>(
                d_numAlignmentsPerPartitionPerBatch,
                d_numIndicesPerPartitionPerBatch.data().get(),
                d_numHaplotypesPerBatch + firstBatchId,
                numPartitions, 
                numBatchesInChunk
            ); CUERR;
    
            auto offsets = thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                [numBatchesInChunk] __host__ __device__(int partition){
                    return partition * numBatchesInChunk;
                }
            );
            size_t temp_storage_bytes = 0;
            cub::DeviceSegmentedReduce::Sum(nullptr, temp_storage_bytes, d_numAlignmentsPerPartitionPerBatch, 
                d_numAlignmentsPerPartition.data().get(), numPartitions, offsets, offsets + 1, mainStream); CUERR;
            thrust::device_vector<char, ThrustCudaMallocAsyncAllocator<char>> d_temp(temp_storage_bytes, ThrustCudaMallocAsyncAllocator<char>(mainStream));
            cub::DeviceSegmentedReduce::Sum(d_temp.data().get(), temp_storage_bytes, d_numAlignmentsPerPartitionPerBatch, 
                d_numAlignmentsPerPartition.data().get(), numPartitions, offsets, offsets + 1, mainStream); CUERR;
        }

        cudaMemcpyAsync(h_numAlignmentsPerPartition.data(), d_numAlignmentsPerPartition.data().get(), sizeof(int) * numPartitions, cudaMemcpyDeviceToHost, mainStream); CUERR;
        cudaStreamSynchronize(mainStream); CUERR;

        // std::cout << "h_numAlignmentsPerPartition: ";
        // for(int i = 0; i< numPartitions; i++){
        //     std::cout << h_numAlignmentsPerPartition[i] << ", ";
        // }
        // std::cout << "\n";

        // thrust::host_vector<int> h_offset_read_batches_vec = dev_offset_read_batches_vec;
        // for(auto& x : h_offset_read_batches_vec){
        //     std::cout << x << " ";
        // }
        // std::cout << "\n";

        int numProcessedAlignmentsByCurrentChunk = 0;

        #define COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM(stream) \
            nvtx3::scoped_range sr3("partition"); \
            constexpr int groupsPerBlock = blocksize / group_size; \
            const int numAlignmentsInPartition = h_numAlignmentsPerPartition[partitionId]; \
            const int numBlocks = SDIV(numAlignmentsInPartition, groupsPerBlock); \
            const int* d_numIndicesPerBatch = d_numIndicesPerPartitionPerBatch.data().get() + partitionId*numBatchesInChunk; \
            thrust::transform( \
                thrust::cuda::par_nosync.on(stream), \
                d_numIndicesPerBatch, \
                d_numIndicesPerBatch + numBatchesInChunk, \
                dev_hap_batches + numProcessedBatchesByChunks, \
                d_numAlignmentsPerBatch.begin() + partitionId * numBatchesInChunk, \
                thrust::multiplies<int>{} \
            ); \
            thrust::inclusive_scan( \
                thrust::cuda::par_nosync(ThrustCudaMallocAsyncAllocator<int>(stream)).on(stream), \
                d_numAlignmentsPerBatch.begin() + partitionId * numBatchesInChunk, \
                d_numAlignmentsPerBatch.begin() + partitionId * numBatchesInChunk+ numBatchesInChunk, \
                d_numAlignmentsPerBatchInclPrefixSum.begin() + partitionId * numBatchesInChunk \
            );  \
            PairHMM_align_partition_half_allowMultipleBatchesPerWarp<group_size,numRegs><<<numBlocks, blocksize,0,stream>>>( \
                dev_read_chars,  \
                dev_hap_chars,  \
                dev_base_qual,  \
                dev_ins_qual,  \
                dev_del_qual,  \
                devAlignmentScoresFloat + numProcessedAlignmentsByChunks,  \
                dev_offset_reads + firstReadInChunk,  \
                dev_offset_haps + firstHapInChunk,  \
                dev_read_len + firstReadInChunk,  \
                dev_hap_len + firstHapInChunk, \
                dev_read_batches + firstBatchId, \
                dev_hap_batches + firstBatchId,  \
                dev_offset_hap_batches + firstBatchId,  \
                d_numIndicesPerBatch,  \
                d_indicesPerPartitionPerBatch.data().get() + partitionId * numReadsInChunk,  \
                dev_offset_read_batches + firstBatchId,   \
                numBatchesInChunk,  \
                d_resultOffsetsPerBatch.data().get(),  \
                d_numAlignmentsPerBatch.data().get() + partitionId * numBatchesInChunk,  \
                d_numAlignmentsPerBatchInclPrefixSum.data().get() + partitionId * numBatchesInChunk,  \
                numAlignmentsInPartition \
            ); CUERR; \
            numProcessedAlignmentsByCurrentChunk += numAlignmentsInPartition;


        LAUNCH_ALL_KERNELS

        numProcessedAlignmentsByChunks += numProcessedAlignmentsByCurrentChunk;
        numProcessedBatchesByChunks += numBatchesInChunk;

        // for(int i = 0; i < numPartitions; i++){
        //     cudaStreamSynchronize(streams_part[i]);
        // }

            
    }
    #undef  COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM

    
    cudaMemcpy(alignment_scores_float.data(), devAlignmentScoresFloat, totalNumberOfAlignments*sizeof(float), cudaMemcpyDeviceToHost);  CUERR

    for (int i=0; i<numPartitions; i++) cudaStreamDestroy(streams_part[i]); CUERR;
    for(auto& stream : transferStreams){
        cudaStreamDestroy(stream);
    }

    totalTimer.stop();
    totalTimer.printGCUPS(countsOfDPCells.totalDPCells);

    return alignment_scores_float;
}





std::vector<float> processBatchAsWhole_half_coalesced_smem(
    const batch& fullBatch, 
    const Options& /*options*/, 
    const CountsOfDPCells& countsOfDPCells
){
    helpers::CpuTimer totalTimer("processBatchAsWhole_half_coalesced_smem");

    const uint8_t* read_chars       = fullBatch.reads.data(); //batch_2.chars.data();
    const uint read_bytes = fullBatch.reads.size();
    const uint8_t* hap_chars       = fullBatch.haps.data(); //batch_2.chars.data();
    const uint hap_bytes = fullBatch.haps.size();
    const uint8_t* base_qual       = fullBatch.base_quals.data(); //batch_2.chars.data();
    const uint8_t* ins_qual       = fullBatch.ins_quals.data(); //batch_2.chars.data();
    const uint8_t* del_qual       = fullBatch.del_quals.data(); //batch_2.chars.data();
    const int* offset_reads       = fullBatch.read_offsets.data(); //batch_2.chars.data();
    const int* offset_haps       = fullBatch.hap_offsets.data(); //batch_2.chars.data();
    const int* read_len       = fullBatch.readlen.data(); //batch_2.chars.data();
    const int* hap_len       = fullBatch.haplen.data(); //batch_2.chars.data();
    const int num_reads = fullBatch.readlen.size(); //batch_2.chars.data();
    const int num_haps = fullBatch.haplen.size(); //batch_2.chars.data();
    const int num_batches = fullBatch.batch_reads.size(); //batch_2.chars.data();
    const int* hap_batches       = fullBatch.batch_haps.data(); //batch_2.chars.data();
    const int* read_batches       = fullBatch.batch_reads.data(); //batch_2.chars.data();
    const int* offset_hap_batches       = fullBatch.batch_haps_offsets.data(); //batch_2.chars.data();
    const int* offset_read_batches       = fullBatch.batch_reads_offsets.data(); //batch_2.chars.data();

    const int totalNumberOfAlignments = fullBatch.getTotalNumberOfAlignments();

    std::vector<float> alignment_scores_float(totalNumberOfAlignments);

    cudaStream_t streams_part[numPartitions];
    for (int i=0; i<numPartitions; i++) cudaStreamCreate(&streams_part[i]);

    thrust::device_vector<uint8_t> dev_read_chars_vec(read_bytes);
    thrust::device_vector<uint8_t> dev_hap_chars_vec(hap_bytes);
    thrust::device_vector<uint8_t> dev_base_qual_vec(read_bytes);
    thrust::device_vector<uint8_t> dev_ins_qual_vec(read_bytes);
    thrust::device_vector<uint8_t> dev_del_qual_vec(read_bytes);

    thrust::device_vector<int> dev_offset_reads_vec(num_reads);
    thrust::device_vector<int> dev_offset_haps_vec(num_haps);
    thrust::device_vector<int> dev_read_len_vec(num_reads);
    thrust::device_vector<int> dev_hap_len_vec(num_haps);
    thrust::device_vector<int> dev_read_batches_vec(num_batches);
    thrust::device_vector<int> dev_hap_batches_vec(num_batches);
    thrust::device_vector<int> dev_offset_read_batches_vec(num_batches);
    thrust::device_vector<int> dev_offset_hap_batches_vec(num_batches);
    thrust::device_vector<float> devAlignmentScoresFloat_vec(totalNumberOfAlignments);

    uint8_t* const dev_read_chars = dev_read_chars_vec.data().get();
    uint8_t* const dev_hap_chars = dev_hap_chars_vec.data().get();
    uint8_t* const dev_base_qual = dev_base_qual_vec.data().get();
    uint8_t* const dev_ins_qual = dev_ins_qual_vec.data().get();
    uint8_t* const dev_del_qual = dev_del_qual_vec.data().get();
    int* const dev_offset_reads = dev_offset_reads_vec.data().get();
    int* const dev_offset_haps = dev_offset_haps_vec.data().get();
    int* const dev_read_len = dev_read_len_vec.data().get();
    int* const dev_hap_len = dev_hap_len_vec.data().get();
    int* const dev_read_batches = dev_read_batches_vec.data().get();
    int* const dev_hap_batches = dev_hap_batches_vec.data().get();
    int* const dev_offset_read_batches = dev_offset_read_batches_vec.data().get();
    int* const dev_offset_hap_batches = dev_offset_hap_batches_vec.data().get();
    float* const devAlignmentScoresFloat = devAlignmentScoresFloat_vec.data().get();


    helpers::CpuTimer transfertimer("DATA_TRANSFER");
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
    transfertimer.print();

    //print_batch(fullBatch); // prints first reads/qualities and haplotype per batch.

    convert_DNA<<<num_reads, 128>>>(dev_read_chars,read_bytes);
    convert_DNA<<<num_haps, 128>>>(dev_hap_chars,hap_bytes);

    thrust::device_vector<int> d_numIndicesPerPartitionPerBatch(num_batches * numPartitions, 0);
    thrust::device_vector<int> d_indicesPerPartitionPerBatch(num_reads * numPartitions, -1);
    thrust::device_vector<int> d_resultOffsetsPerBatch(num_batches);
    thrust::device_vector<int> d_numAlignmentsPerBatch(num_batches * numPartitions);
    thrust::device_vector<int> d_numAlignmentsPerBatchInclPrefixSum(num_batches * numPartitions);
    thrust::device_vector<int> d_numAlignmentsPerPartition(numPartitions);

    partitionIndicesKernel<<<num_batches, 128>>>(
        d_numIndicesPerPartitionPerBatch.data().get(),
        d_indicesPerPartitionPerBatch.data().get(),
        dev_read_len,
        dev_read_batches,
        dev_offset_read_batches,
        num_batches,
        num_reads
    );
    CUERR;

    thrust::transform(
        thrust::cuda::par_nosync.on((cudaStream_t)0),
        dev_read_batches,
        dev_read_batches + num_batches,
        dev_hap_batches,
        d_resultOffsetsPerBatch.begin(),
        thrust::multiplies<int>{}
    );
    thrust::exclusive_scan(
        thrust::cuda::par_nosync(ThrustCudaMallocAsyncAllocator<int>((cudaStream_t)0)).on((cudaStream_t)0),
        d_resultOffsetsPerBatch.begin(),
        d_resultOffsetsPerBatch.begin() + num_batches,
        d_resultOffsetsPerBatch.begin()
    );

    // for(int i = 0; i < std::min(10, num_batches); i++){
    //     std::cout << "batch " << i << ", num reads: " << read_batches[i]
    //         << ", num haps " << hap_batches[i] << ", product: " <<
    //         read_batches[i] * hap_batches[i]
    //         << ", offset : " << d_resultOffsetsPerBatch[i] << "\n";
    // }


    #if 0
        thrust::host_vector<int> h_numIndicesPerPartitionPerBatch = d_numIndicesPerPartitionPerBatch;
        thrust::host_vector<int> h_indicesPerPartitionPerBatch = d_indicesPerPartitionPerBatch;

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
            }
        }
    #endif

    {
        cudaStream_t stream = cudaStreamLegacy;

        int* d_numAlignmentsPerPartitionPerBatch = d_numAlignmentsPerBatchInclPrefixSum.data().get(); // reuse
        const int* d_numHaplotypesPerBatch = dev_hap_batches;
        computeAlignmentsPerPartitionPerBatch<<<dim3(SDIV(num_batches, 128), numPartitions), 128,0, stream>>>(
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
            d_numAlignmentsPerPartition.data().get(), numPartitions, offsets, offsets + 1, stream); CUERR;
        thrust::device_vector<char, ThrustCudaMallocAsyncAllocator<char>> d_temp(temp_storage_bytes, ThrustCudaMallocAsyncAllocator<char>(stream));
        cub::DeviceSegmentedReduce::Sum(d_temp.data().get(), temp_storage_bytes, d_numAlignmentsPerPartitionPerBatch, 
            d_numAlignmentsPerPartition.data().get(), numPartitions, offsets, offsets + 1, stream); CUERR;
    }
    thrust::host_vector<int> h_numAlignmentsPerPartition = d_numAlignmentsPerPartition;

    std::cout << "h_numAlignmentsPerPartition: ";
    for(int i = 0; i< numPartitions; i++){
        std::cout << h_numAlignmentsPerPartition[i] << ", ";
    }
    std::cout << "\n";

    cudaMemset(devAlignmentScoresFloat, 0, totalNumberOfAlignments * sizeof(float)); CUERR;
    helpers::GpuTimer computeTimer("pairhmm half kernels coalesced, total");
    std::vector<std::unique_ptr<helpers::GpuTimer>> perKernelTimers(numPartitions);
    for(int p = 0; p < numPartitions; p++){
        std::string name = "pairhmm half kernel coalesced, partition " + std::to_string(p);
        perKernelTimers[p] = std::make_unique<helpers::GpuTimer>(streams_part[p], name);
    }

    #define COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM(stream) \
        constexpr int groupsPerBlock = blocksize / group_size; \
        const int numAlignmentsInPartition = h_numAlignmentsPerPartition[partitionId]; \
        const int numBlocks = SDIV(numAlignmentsInPartition, groupsPerBlock); \
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
        perKernelTimers[partitionId]->start(); \
        PairHMM_align_partition_half_allowMultipleBatchesPerWarp_coalesced_smem<group_size,numRegs><<<numBlocks, blocksize,0,stream>>>(dev_read_chars, dev_hap_chars, dev_base_qual, dev_ins_qual, dev_del_qual, devAlignmentScoresFloat, dev_offset_reads, dev_offset_haps, dev_read_len, dev_hap_len, dev_read_batches, dev_hap_batches, dev_offset_hap_batches,  \
            d_numIndicesPerBatch, d_indicesPerPartitionPerBatch.data().get() + partitionId*num_reads,  \
            dev_offset_read_batches,  num_batches, d_resultOffsetsPerBatch.data().get(), d_numAlignmentsPerBatch.data().get() + partitionId * num_batches, d_numAlignmentsPerBatchInclPrefixSum.data().get() + partitionId * num_batches, numAlignmentsInPartition); \
        perKernelTimers[partitionId]->stop(); \


    LAUNCH_ALL_KERNELS

    #undef  COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM

    computeTimer.stop();

    for(int p = 0; p < numPartitions; p++){
        if(h_numAlignmentsPerPartition[p] > 0){
            perKernelTimers[p]->printGCUPS(countsOfDPCells.dpCellsPerPartition[p]);
        }
    }

    computeTimer.printGCUPS(countsOfDPCells.totalDPCells);

    cudaMemcpy(alignment_scores_float.data(), devAlignmentScoresFloat, totalNumberOfAlignments*sizeof(float), cudaMemcpyDeviceToHost);  CUERR

    perKernelTimers.clear();

    for (int i=0; i<numPartitions; i++) cudaStreamDestroy(streams_part[i]); CUERR;

    totalTimer.stop();
    totalTimer.printGCUPS(countsOfDPCells.totalDPCells);

    return alignment_scores_float;
}






std::vector<float> processBatch_overlapped_half_coalesced_smem(
    const batch& fullBatch_default, 
    const Options& options, 
    const CountsOfDPCells& countsOfDPCells
){
    helpers::CpuTimer totalTimer("processBatch_overlapped_half_coalesced_smem");

    // pinned_batch fullBatch(fullBatch_default);
    const auto& fullBatch = fullBatch_default;

    const uint8_t* read_chars       = fullBatch.reads.data(); //batch_2.chars.data();
    const uint read_bytes = fullBatch.reads.size();
    const uint8_t* hap_chars       = fullBatch.haps.data(); //batch_2.chars.data();
    const uint hap_bytes = fullBatch.haps.size();
    const uint8_t* base_qual       = fullBatch.base_quals.data(); //batch_2.chars.data();
    const uint8_t* ins_qual       = fullBatch.ins_quals.data(); //batch_2.chars.data();
    const uint8_t* del_qual       = fullBatch.del_quals.data(); //batch_2.chars.data();
    const int* offset_reads       = fullBatch.read_offsets.data(); //batch_2.chars.data();
    const int* offset_haps       = fullBatch.hap_offsets.data(); //batch_2.chars.data();
    const int* read_len       = fullBatch.readlen.data(); //batch_2.chars.data();
    const int* hap_len       = fullBatch.haplen.data(); //batch_2.chars.data();
    const int num_reads = fullBatch.readlen.size(); //batch_2.chars.data();
    const int num_haps = fullBatch.haplen.size(); //batch_2.chars.data();
    const int num_batches = fullBatch.batch_reads.size(); //batch_2.chars.data();
    const int* hap_batches       = fullBatch.batch_haps.data(); //batch_2.chars.data();
    const int* read_batches       = fullBatch.batch_reads.data(); //batch_2.chars.data();
    const int* offset_hap_batches       = fullBatch.batch_haps_offsets.data(); //batch_2.chars.data();
    const int* offset_read_batches       = fullBatch.batch_reads_offsets.data(); //batch_2.chars.data();

    const int totalNumberOfAlignments = fullBatch.getTotalNumberOfAlignments();

    std::vector<float> alignment_scores_float(totalNumberOfAlignments);

    cudaStream_t streams_part[numPartitions];
    for (int i=0; i<numPartitions; i++) cudaStreamCreate(&streams_part[i]);

    std::vector<cudaStream_t> transferStreams(2);
    for(auto& stream : transferStreams){
        cudaStreamCreate(&stream);
    }

    thrust::device_vector<float> devAlignmentScoresFloat_vec(totalNumberOfAlignments, 0);

    thrust::device_vector<uint8_t> dev_read_chars_vec(read_bytes);
    thrust::device_vector<uint8_t> dev_hap_chars_vec(hap_bytes);
    thrust::device_vector<uint8_t> dev_base_qual_vec(read_bytes);
    thrust::device_vector<uint8_t> dev_ins_qual_vec(read_bytes);
    thrust::device_vector<uint8_t> dev_del_qual_vec(read_bytes);

    thrust::device_vector<int> dev_offset_reads_vec(num_reads);
    thrust::device_vector<int> dev_offset_haps_vec(num_haps);
    thrust::device_vector<int> dev_read_len_vec(num_reads);
    thrust::device_vector<int> dev_hap_len_vec(num_haps);
    thrust::device_vector<int> dev_read_batches_vec(num_batches);
    thrust::device_vector<int> dev_hap_batches_vec(num_batches);
    thrust::device_vector<int> dev_offset_read_batches_vec(num_batches);
    thrust::device_vector<int> dev_offset_hap_batches_vec(num_batches);

    thrust::device_vector<int> d_numIndicesPerPartitionPerBatch(num_batches * numPartitions, 0);
    thrust::device_vector<int> d_indicesPerPartitionPerBatch(num_reads * numPartitions, -1);
    thrust::device_vector<int> d_resultOffsetsPerBatch(num_batches);
    thrust::device_vector<int> d_numAlignmentsPerBatch(num_batches * numPartitions);
    thrust::device_vector<int> d_numAlignmentsPerBatchInclPrefixSum(num_batches * numPartitions);
    thrust::device_vector<int> d_numAlignmentsPerPartition(numPartitions);
    std::vector<int, PinnedAllocator<int>> h_numAlignmentsPerPartition(numPartitions);

    uint8_t* const dev_read_chars = dev_read_chars_vec.data().get();
    uint8_t* const dev_hap_chars = dev_hap_chars_vec.data().get();
    uint8_t* const dev_base_qual = dev_base_qual_vec.data().get();
    uint8_t* const dev_ins_qual = dev_ins_qual_vec.data().get();
    uint8_t* const dev_del_qual = dev_del_qual_vec.data().get();
    int* const dev_offset_reads = dev_offset_reads_vec.data().get();
    int* const dev_offset_haps = dev_offset_haps_vec.data().get();
    int* const dev_read_len = dev_read_len_vec.data().get();
    int* const dev_hap_len = dev_hap_len_vec.data().get();
    int* const dev_read_batches = dev_read_batches_vec.data().get();
    int* const dev_hap_batches = dev_hap_batches_vec.data().get();
    int* const dev_offset_read_batches = dev_offset_read_batches_vec.data().get();
    int* const dev_offset_hap_batches = dev_offset_hap_batches_vec.data().get();
    float* const devAlignmentScoresFloat = devAlignmentScoresFloat_vec.data().get();

    int numProcessedAlignmentsByChunks = 0;
    int numProcessedBatchesByChunks = 0;

    const int numTransferChunks = SDIV(num_batches, options.transferchunksize);

    for(int computeChunk = 0, transferChunk = 0; computeChunk < numTransferChunks; computeChunk++){
        for(; transferChunk < numTransferChunks && transferChunk < (computeChunk + 2); transferChunk++){
            nvtx3::scoped_range sr1("transferChunk");
            cudaStream_t transferStream = transferStreams[transferChunk % 2];
            
            const int firstBatchId = transferChunk * options.transferchunksize;
            const int lastBatchId_excl = std::min((transferChunk+1)* options.transferchunksize, num_batches);
            const int numBatchesInChunk = lastBatchId_excl - firstBatchId;

            const int firstReadInChunk = offset_read_batches[firstBatchId];
            const int lastReadInChunk_excl = offset_read_batches[lastBatchId_excl];
            const int numReadsInChunk = lastReadInChunk_excl - firstReadInChunk;

            const int firstHapInChunk = offset_hap_batches[firstBatchId];
            const int lastHapInChunk_excl = offset_hap_batches[lastBatchId_excl];
            const int numHapsInChunk = lastHapInChunk_excl - firstHapInChunk;

            const size_t numReadBytesInChunk = offset_reads[lastReadInChunk_excl] - offset_reads[firstReadInChunk];
            const size_t numHapBytesInChunk = offset_haps[lastHapInChunk_excl] - offset_haps[firstHapInChunk];

            // std::cout << "transferChunk " << transferChunk << "\n";
            // std::cout << "firstBatchId " << firstBatchId << "\n";
            // std::cout << "lastBatchId_excl " << lastBatchId_excl << "\n";
            // std::cout << "numBatchesInChunk " << numBatchesInChunk << "\n";
            // std::cout << "firstReadInChunk " << firstReadInChunk << "\n";
            // std::cout << "lastReadInChunk_excl " << lastReadInChunk_excl << "\n";
            // std::cout << "numReadsInChunk " << numReadsInChunk << "\n";
            // std::cout << "firstHapInChunk " << firstHapInChunk << "\n";
            // std::cout << "lastHapInChunk_excl " << lastHapInChunk_excl << "\n";
            // std::cout << "numHapsInChunk " << numHapsInChunk << "\n";
            // std::cout << "numReadBytesInChunk " << numReadBytesInChunk << "\n";
            // std::cout << "numHapBytesInChunk " << numHapBytesInChunk << "\n";
            // std::cout << "----------------------------\n";



            cudaMemcpyAsync(dev_read_chars + offset_reads[firstReadInChunk], read_chars + offset_reads[firstReadInChunk], numReadBytesInChunk, cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_hap_chars + offset_haps[firstHapInChunk], hap_chars + offset_haps[firstHapInChunk], numHapBytesInChunk, cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_base_qual + offset_reads[firstReadInChunk], base_qual + offset_reads[firstReadInChunk], numReadBytesInChunk, cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_ins_qual + offset_reads[firstReadInChunk], ins_qual + offset_reads[firstReadInChunk], numReadBytesInChunk, cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_del_qual + offset_reads[firstReadInChunk], del_qual + offset_reads[firstReadInChunk], numReadBytesInChunk, cudaMemcpyHostToDevice, transferStream); CUERR

            cudaMemcpyAsync(dev_offset_reads + firstReadInChunk, offset_reads + firstReadInChunk, numReadsInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_offset_haps + firstHapInChunk, offset_haps + firstHapInChunk, numHapsInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_read_len + firstReadInChunk, read_len + firstReadInChunk, numReadsInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_hap_len + firstHapInChunk, hap_len + firstHapInChunk, numHapsInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_offset_read_batches + firstBatchId, offset_read_batches + firstBatchId, numBatchesInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_offset_hap_batches + firstBatchId, offset_hap_batches + firstBatchId, numBatchesInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_read_batches + firstBatchId, read_batches + firstBatchId, numBatchesInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_hap_batches + firstBatchId, hap_batches + firstBatchId, numBatchesInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream); CUERR
        }
        nvtx3::scoped_range sr2("computeChunk");
        cudaStream_t mainStream = transferStreams[computeChunk % 2];
        const int firstBatchId = computeChunk * options.transferchunksize;
        const int lastBatchId_excl = std::min((computeChunk+1)* options.transferchunksize, num_batches);
        const int numBatchesInChunk = lastBatchId_excl - firstBatchId;

        const int firstReadInChunk = offset_read_batches[firstBatchId];
        const int lastReadInChunk_excl = offset_read_batches[lastBatchId_excl];
        const int numReadsInChunk = lastReadInChunk_excl - firstReadInChunk;

        const int firstHapInChunk = offset_hap_batches[firstBatchId];
        const int lastHapInChunk_excl = offset_hap_batches[lastBatchId_excl];
        const int numHapsInChunk = lastHapInChunk_excl - firstHapInChunk;

        const size_t numReadBytesInChunk = offset_reads[lastReadInChunk_excl] - offset_reads[firstReadInChunk];
        const size_t numHapBytesInChunk = offset_haps[lastHapInChunk_excl] - offset_haps[firstHapInChunk];

        convert_DNA<<<numReadsInChunk, 128, 0, mainStream>>>(dev_read_chars + offset_reads[firstReadInChunk], numReadBytesInChunk);
        convert_DNA<<<numHapsInChunk, 128, 0, mainStream>>>(dev_hap_chars + offset_haps[firstHapInChunk], numHapBytesInChunk);

        //ensure buffers used by previous batch are no longer in use
        for(int i = 0; i < numPartitions; i++){
            cudaStreamSynchronize(streams_part[i]);
        }

        cudaMemsetAsync(d_numIndicesPerPartitionPerBatch.data().get(), 0, sizeof(int) * numPartitions * numBatchesInChunk, mainStream); CUERR;
        partitionIndicesKernel<<<numBatchesInChunk, 128, 0, mainStream>>>(
            d_numIndicesPerPartitionPerBatch.data().get(),
            d_indicesPerPartitionPerBatch.data().get(),
            dev_read_len,
            dev_read_batches + firstBatchId,
            dev_offset_read_batches + firstBatchId,
            numBatchesInChunk,
            numReadsInChunk
        );
        CUERR;

        #if 0
            thrust::host_vector<int> h_numIndicesPerPartitionPerBatch = d_numIndicesPerPartitionPerBatch;
            thrust::host_vector<int> h_indicesPerPartitionPerBatch = d_indicesPerPartitionPerBatch;

            for(int p = 0; p < numPartitions; p++){
                if(p <= 4){
                    std::cout << "Partition p = " << p << "\n";
                    std::cout << "numIndicesPerBatch: ";
                    for(int b = 0; b < numBatchesInChunk; b++){
                        std::cout << h_numIndicesPerPartitionPerBatch[p * numBatchesInChunk + b] << ", ";
                    }
                    std::cout << "\n";

                    std::cout << "indicesPerBatch: ";
                    for(int b = 0; b < numBatchesInChunk; b++){
                        const int num = h_numIndicesPerPartitionPerBatch[p * numBatchesInChunk + b];
                        for(int i = 0; i < num; i++){
                            const int outputOffset = offset_read_batches[firstBatchId + b] - offset_read_batches[firstBatchId];
                            std::cout << h_indicesPerPartitionPerBatch[p * numReadsInChunk + outputOffset + i];
                            if(i != num-1){
                                std::cout << ", ";
                            }
                        }
                        std::cout << " | ";
                    }
                    std::cout << "\n";
                }
            }
        #endif
    
        thrust::transform(
            thrust::cuda::par_nosync.on(mainStream),
            dev_read_batches + firstBatchId,
            dev_read_batches + firstBatchId + numBatchesInChunk,
            dev_hap_batches + firstBatchId,
            d_resultOffsetsPerBatch.begin(),
            thrust::multiplies<int>{}
        );
        thrust::exclusive_scan(
            thrust::cuda::par_nosync(ThrustCudaMallocAsyncAllocator<int>(mainStream)).on(mainStream),
            d_resultOffsetsPerBatch.begin(),
            d_resultOffsetsPerBatch.begin() + numBatchesInChunk,
            d_resultOffsetsPerBatch.begin()
        );

        // thrust::host_vector<int> h_resultOffsetsPerBatch = d_resultOffsetsPerBatch;
        // std::cout << "h_resultOffsetsPerBatch. numBatchesInChunk " << numBatchesInChunk << "\n";
        // for(auto& x : h_resultOffsetsPerBatch){
        //     std::cout << x << " ";
        // }
        // std::cout << "\n";

        {     
            int* d_numAlignmentsPerPartitionPerBatch = d_numAlignmentsPerBatchInclPrefixSum.data().get(); // reuse
            const int* d_numHaplotypesPerBatch = dev_hap_batches;
            computeAlignmentsPerPartitionPerBatch<<<dim3(SDIV(numBatchesInChunk, 128), numPartitions), 128,0, mainStream>>>(
                d_numAlignmentsPerPartitionPerBatch,
                d_numIndicesPerPartitionPerBatch.data().get(),
                d_numHaplotypesPerBatch + firstBatchId,
                numPartitions, 
                numBatchesInChunk
            ); CUERR;
    
            auto offsets = thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                [numBatchesInChunk] __host__ __device__(int partition){
                    return partition * numBatchesInChunk;
                }
            );
            size_t temp_storage_bytes = 0;
            cub::DeviceSegmentedReduce::Sum(nullptr, temp_storage_bytes, d_numAlignmentsPerPartitionPerBatch, 
                d_numAlignmentsPerPartition.data().get(), numPartitions, offsets, offsets + 1, mainStream); CUERR;
            thrust::device_vector<char, ThrustCudaMallocAsyncAllocator<char>> d_temp(temp_storage_bytes, ThrustCudaMallocAsyncAllocator<char>(mainStream));
            cub::DeviceSegmentedReduce::Sum(d_temp.data().get(), temp_storage_bytes, d_numAlignmentsPerPartitionPerBatch, 
                d_numAlignmentsPerPartition.data().get(), numPartitions, offsets, offsets + 1, mainStream); CUERR;
        }

        cudaMemcpyAsync(h_numAlignmentsPerPartition.data(), d_numAlignmentsPerPartition.data().get(), sizeof(int) * numPartitions, cudaMemcpyDeviceToHost, mainStream); CUERR;
        cudaStreamSynchronize(mainStream); CUERR;

        // std::cout << "h_numAlignmentsPerPartition: ";
        // for(int i = 0; i< numPartitions; i++){
        //     std::cout << h_numAlignmentsPerPartition[i] << ", ";
        // }
        // std::cout << "\n";

        // thrust::host_vector<int> h_offset_read_batches_vec = dev_offset_read_batches_vec;
        // for(auto& x : h_offset_read_batches_vec){
        //     std::cout << x << " ";
        // }
        // std::cout << "\n";

        int numProcessedAlignmentsByCurrentChunk = 0;

        #define COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM(stream) \
            nvtx3::scoped_range sr3("partition"); \
            constexpr int groupsPerBlock = blocksize / group_size; \
            const int numAlignmentsInPartition = h_numAlignmentsPerPartition[partitionId]; \
            const int numBlocks = SDIV(numAlignmentsInPartition, groupsPerBlock); \
            const int* d_numIndicesPerBatch = d_numIndicesPerPartitionPerBatch.data().get() + partitionId*numBatchesInChunk; \
            thrust::transform( \
                thrust::cuda::par_nosync.on(stream), \
                d_numIndicesPerBatch, \
                d_numIndicesPerBatch + numBatchesInChunk, \
                dev_hap_batches + numProcessedBatchesByChunks, \
                d_numAlignmentsPerBatch.begin() + partitionId * numBatchesInChunk, \
                thrust::multiplies<int>{} \
            ); \
            thrust::inclusive_scan( \
                thrust::cuda::par_nosync(ThrustCudaMallocAsyncAllocator<int>(stream)).on(stream), \
                d_numAlignmentsPerBatch.begin() + partitionId * numBatchesInChunk, \
                d_numAlignmentsPerBatch.begin() + partitionId * numBatchesInChunk+ numBatchesInChunk, \
                d_numAlignmentsPerBatchInclPrefixSum.begin() + partitionId * numBatchesInChunk \
            );  \
            PairHMM_align_partition_half_allowMultipleBatchesPerWarp_coalesced_smem<group_size,numRegs><<<numBlocks, blocksize,0,stream>>>( \
                dev_read_chars,  \
                dev_hap_chars,  \
                dev_base_qual,  \
                dev_ins_qual,  \
                dev_del_qual,  \
                devAlignmentScoresFloat + numProcessedAlignmentsByChunks,  \
                dev_offset_reads + firstReadInChunk,  \
                dev_offset_haps + firstHapInChunk,  \
                dev_read_len + firstReadInChunk,  \
                dev_hap_len + firstHapInChunk, \
                dev_read_batches + firstBatchId, \
                dev_hap_batches + firstBatchId,  \
                dev_offset_hap_batches + firstBatchId,  \
                d_numIndicesPerBatch,  \
                d_indicesPerPartitionPerBatch.data().get() + partitionId * numReadsInChunk,  \
                dev_offset_read_batches + firstBatchId,   \
                numBatchesInChunk,  \
                d_resultOffsetsPerBatch.data().get(),  \
                d_numAlignmentsPerBatch.data().get() + partitionId * numBatchesInChunk,  \
                d_numAlignmentsPerBatchInclPrefixSum.data().get() + partitionId * numBatchesInChunk,  \
                numAlignmentsInPartition \
            ); CUERR; \
            numProcessedAlignmentsByCurrentChunk += numAlignmentsInPartition;


        LAUNCH_ALL_KERNELS

        numProcessedAlignmentsByChunks += numProcessedAlignmentsByCurrentChunk;
        numProcessedBatchesByChunks += numBatchesInChunk;

        // for(int i = 0; i < numPartitions; i++){
        //     cudaStreamSynchronize(streams_part[i]);
        // }

            
    }
    #undef  COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM

    
    cudaMemcpy(alignment_scores_float.data(), devAlignmentScoresFloat, totalNumberOfAlignments*sizeof(float), cudaMemcpyDeviceToHost);  CUERR

    for (int i=0; i<numPartitions; i++) cudaStreamDestroy(streams_part[i]); CUERR;
    for(auto& stream : transferStreams){
        cudaStreamDestroy(stream);
    }

    totalTimer.stop();
    totalTimer.printGCUPS(countsOfDPCells.totalDPCells);

    return alignment_scores_float;
}




std::vector<float> processBatchAsWhole_float(
    const batch& fullBatch, 
    const Options& /*options*/, 
    const CountsOfDPCells& countsOfDPCells
){
    helpers::CpuTimer totalTimer("processBatchAsWhole_float");

    const uint8_t* read_chars       = fullBatch.reads.data(); //batch_2.chars.data();
    const uint read_bytes = fullBatch.reads.size();
    const uint8_t* hap_chars       = fullBatch.haps.data(); //batch_2.chars.data();
    const uint hap_bytes = fullBatch.haps.size();
    const uint8_t* base_qual       = fullBatch.base_quals.data(); //batch_2.chars.data();
    const uint8_t* ins_qual       = fullBatch.ins_quals.data(); //batch_2.chars.data();
    const uint8_t* del_qual       = fullBatch.del_quals.data(); //batch_2.chars.data();
    const int* offset_reads       = fullBatch.read_offsets.data(); //batch_2.chars.data();
    const int* offset_haps       = fullBatch.hap_offsets.data(); //batch_2.chars.data();
    const int* read_len       = fullBatch.readlen.data(); //batch_2.chars.data();
    const int* hap_len       = fullBatch.haplen.data(); //batch_2.chars.data();
    const int num_reads = fullBatch.readlen.size(); //batch_2.chars.data();
    const int num_haps = fullBatch.haplen.size(); //batch_2.chars.data();
    const int num_batches = fullBatch.batch_reads.size(); //batch_2.chars.data();
    const int* hap_batches       = fullBatch.batch_haps.data(); //batch_2.chars.data();
    const int* read_batches       = fullBatch.batch_reads.data(); //batch_2.chars.data();
    const int* offset_hap_batches       = fullBatch.batch_haps_offsets.data(); //batch_2.chars.data();
    const int* offset_read_batches       = fullBatch.batch_reads_offsets.data(); //batch_2.chars.data();

    const int totalNumberOfAlignments = fullBatch.getTotalNumberOfAlignments();

    std::vector<float> alignment_scores_float(totalNumberOfAlignments);

    cudaStream_t streams_part[numPartitions];
    for (int i=0; i<numPartitions; i++) cudaStreamCreate(&streams_part[i]);

    thrust::device_vector<uint8_t> dev_read_chars_vec(read_bytes);
    thrust::device_vector<uint8_t> dev_hap_chars_vec(hap_bytes);
    thrust::device_vector<uint8_t> dev_base_qual_vec(read_bytes);
    thrust::device_vector<uint8_t> dev_ins_qual_vec(read_bytes);
    thrust::device_vector<uint8_t> dev_del_qual_vec(read_bytes);

    thrust::device_vector<int> dev_offset_reads_vec(num_reads);
    thrust::device_vector<int> dev_offset_haps_vec(num_haps);
    thrust::device_vector<int> dev_read_len_vec(num_reads);
    thrust::device_vector<int> dev_hap_len_vec(num_haps);
    thrust::device_vector<int> dev_read_batches_vec(num_batches);
    thrust::device_vector<int> dev_hap_batches_vec(num_batches);
    thrust::device_vector<int> dev_offset_read_batches_vec(num_batches);
    thrust::device_vector<int> dev_offset_hap_batches_vec(num_batches);
    thrust::device_vector<float> devAlignmentScoresFloat_vec(totalNumberOfAlignments);

    uint8_t* const dev_read_chars = dev_read_chars_vec.data().get();
    uint8_t* const dev_hap_chars = dev_hap_chars_vec.data().get();
    uint8_t* const dev_base_qual = dev_base_qual_vec.data().get();
    uint8_t* const dev_ins_qual = dev_ins_qual_vec.data().get();
    uint8_t* const dev_del_qual = dev_del_qual_vec.data().get();
    int* const dev_offset_reads = dev_offset_reads_vec.data().get();
    int* const dev_offset_haps = dev_offset_haps_vec.data().get();
    int* const dev_read_len = dev_read_len_vec.data().get();
    int* const dev_hap_len = dev_hap_len_vec.data().get();
    int* const dev_read_batches = dev_read_batches_vec.data().get();
    int* const dev_hap_batches = dev_hap_batches_vec.data().get();
    int* const dev_offset_read_batches = dev_offset_read_batches_vec.data().get();
    int* const dev_offset_hap_batches = dev_offset_hap_batches_vec.data().get();
    float* const devAlignmentScoresFloat = devAlignmentScoresFloat_vec.data().get();


    helpers::CpuTimer transfertimer("DATA_TRANSFER");
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
    transfertimer.print();

    //print_batch(fullBatch); // prints first reads/qualities and haplotype per batch.

    convert_DNA<<<num_reads, 128>>>(dev_read_chars,read_bytes);
    convert_DNA<<<num_haps, 128>>>(dev_hap_chars,hap_bytes);

    thrust::device_vector<int> d_numIndicesPerPartitionPerBatch(num_batches * numPartitions, 0);
    thrust::device_vector<int> d_indicesPerPartitionPerBatch(num_reads * numPartitions, -1);
    thrust::device_vector<int> d_resultOffsetsPerBatch(num_batches);
    thrust::device_vector<int> d_numAlignmentsPerBatch(num_batches * numPartitions);
    thrust::device_vector<int> d_numAlignmentsPerBatchInclPrefixSum(num_batches * numPartitions);
    thrust::device_vector<int> d_numAlignmentsPerPartition(numPartitions);

    partitionIndicesKernel<<<num_batches, 128>>>(
        d_numIndicesPerPartitionPerBatch.data().get(),
        d_indicesPerPartitionPerBatch.data().get(),
        dev_read_len,
        dev_read_batches,
        dev_offset_read_batches,
        num_batches,
        num_reads
    );
    CUERR;

    thrust::transform(
        thrust::cuda::par_nosync.on((cudaStream_t)0),
        dev_read_batches,
        dev_read_batches + num_batches,
        dev_hap_batches,
        d_resultOffsetsPerBatch.begin(),
        thrust::multiplies<int>{}
    );
    thrust::exclusive_scan(
        thrust::cuda::par_nosync(ThrustCudaMallocAsyncAllocator<int>((cudaStream_t)0)).on((cudaStream_t)0),
        d_resultOffsetsPerBatch.begin(),
        d_resultOffsetsPerBatch.begin() + num_batches,
        d_resultOffsetsPerBatch.begin()
    );

    // for(int i = 0; i < std::min(10, num_batches); i++){
    //     std::cout << "batch " << i << ", num reads: " << read_batches[i]
    //         << ", num haps " << hap_batches[i] << ", product: " <<
    //         read_batches[i] * hap_batches[i]
    //         << ", offset : " << d_resultOffsetsPerBatch[i] << "\n";
    // }


    #if 0
        thrust::host_vector<int> h_numIndicesPerPartitionPerBatch = d_numIndicesPerPartitionPerBatch;
        thrust::host_vector<int> h_indicesPerPartitionPerBatch = d_indicesPerPartitionPerBatch;

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
            }
        }
    #endif

    {
        cudaStream_t stream = cudaStreamLegacy;

        int* d_numAlignmentsPerPartitionPerBatch = d_numAlignmentsPerBatchInclPrefixSum.data().get(); // reuse
        const int* d_numHaplotypesPerBatch = dev_hap_batches;
        computeAlignmentsPerPartitionPerBatch<<<dim3(SDIV(num_batches, 128), numPartitions), 128,0, stream>>>(
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
            d_numAlignmentsPerPartition.data().get(), numPartitions, offsets, offsets + 1, stream); CUERR;
        thrust::device_vector<char, ThrustCudaMallocAsyncAllocator<char>> d_temp(temp_storage_bytes, ThrustCudaMallocAsyncAllocator<char>(stream));
        cub::DeviceSegmentedReduce::Sum(d_temp.data().get(), temp_storage_bytes, d_numAlignmentsPerPartitionPerBatch, 
            d_numAlignmentsPerPartition.data().get(), numPartitions, offsets, offsets + 1, stream); CUERR;
    }
    thrust::host_vector<int> h_numAlignmentsPerPartition = d_numAlignmentsPerPartition;

    std::cout << "h_numAlignmentsPerPartition: ";
    for(int i = 0; i< numPartitions; i++){
        std::cout << h_numAlignmentsPerPartition[i] << ", ";
    }
    std::cout << "\n";

    cudaMemset(devAlignmentScoresFloat, 0, totalNumberOfAlignments * sizeof(float)); CUERR;
    helpers::GpuTimer computeTimer("pairhmm float kernels, total");
    std::vector<std::unique_ptr<helpers::GpuTimer>> perKernelTimers(numPartitions);
    for(int p = 0; p < numPartitions; p++){
        std::string name = "pairhmm float kernel, partition " + std::to_string(p);
        perKernelTimers[p] = std::make_unique<helpers::GpuTimer>(streams_part[p], name);
    }

    #define COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM(stream) \
        constexpr int groupsPerBlock = blocksize / group_size; \
        const int numAlignmentsInPartition = h_numAlignmentsPerPartition[partitionId]; \
        const int numBlocks = SDIV(numAlignmentsInPartition, groupsPerBlock); \
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
        perKernelTimers[partitionId]->start(); \
        PairHMM_align_partition_float_allowMultipleBatchesPerWarp<group_size,numRegs><<<numBlocks, blocksize,0,stream>>>(dev_read_chars, dev_hap_chars, dev_base_qual, dev_ins_qual, dev_del_qual, devAlignmentScoresFloat, dev_offset_reads, dev_offset_haps, dev_read_len, dev_hap_len, dev_read_batches, dev_hap_batches, dev_offset_hap_batches,  \
            d_numIndicesPerBatch, d_indicesPerPartitionPerBatch.data().get() + partitionId*num_reads,  \
            dev_offset_read_batches,  num_batches, d_resultOffsetsPerBatch.data().get(), d_numAlignmentsPerBatch.data().get() + partitionId * num_batches, d_numAlignmentsPerBatchInclPrefixSum.data().get() + partitionId * num_batches, numAlignmentsInPartition); \
        perKernelTimers[partitionId]->stop(); \


    LAUNCH_ALL_KERNELS

    #undef  COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM

    computeTimer.stop();

    for(int p = 0; p < numPartitions; p++){
        if(h_numAlignmentsPerPartition[p] > 0){
            perKernelTimers[p]->printGCUPS(countsOfDPCells.dpCellsPerPartition[p]);
        }
    }
    computeTimer.printGCUPS(countsOfDPCells.totalDPCells);

    cudaMemcpy(alignment_scores_float.data(), devAlignmentScoresFloat, totalNumberOfAlignments*sizeof(float), cudaMemcpyDeviceToHost);  CUERR

    perKernelTimers.clear();

    for (int i=0; i<numPartitions; i++) cudaStreamDestroy(streams_part[i]); CUERR;

    totalTimer.stop();
    totalTimer.printGCUPS(countsOfDPCells.totalDPCells);

    return alignment_scores_float;
}






std::vector<float> processBatch_overlapped_float(
    const batch& fullBatch_default, 
    const Options& options, 
    const CountsOfDPCells& countsOfDPCells
){
    helpers::CpuTimer totalTimer("processBatch_overlapped_float");

    // pinned_batch fullBatch(fullBatch_default);
    const auto& fullBatch = fullBatch_default;

    const uint8_t* read_chars       = fullBatch.reads.data(); //batch_2.chars.data();
    const uint read_bytes = fullBatch.reads.size();
    const uint8_t* hap_chars       = fullBatch.haps.data(); //batch_2.chars.data();
    const uint hap_bytes = fullBatch.haps.size();
    const uint8_t* base_qual       = fullBatch.base_quals.data(); //batch_2.chars.data();
    const uint8_t* ins_qual       = fullBatch.ins_quals.data(); //batch_2.chars.data();
    const uint8_t* del_qual       = fullBatch.del_quals.data(); //batch_2.chars.data();
    const int* offset_reads       = fullBatch.read_offsets.data(); //batch_2.chars.data();
    const int* offset_haps       = fullBatch.hap_offsets.data(); //batch_2.chars.data();
    const int* read_len       = fullBatch.readlen.data(); //batch_2.chars.data();
    const int* hap_len       = fullBatch.haplen.data(); //batch_2.chars.data();
    const int num_reads = fullBatch.readlen.size(); //batch_2.chars.data();
    const int num_haps = fullBatch.haplen.size(); //batch_2.chars.data();
    const int num_batches = fullBatch.batch_reads.size(); //batch_2.chars.data();
    const int* hap_batches       = fullBatch.batch_haps.data(); //batch_2.chars.data();
    const int* read_batches       = fullBatch.batch_reads.data(); //batch_2.chars.data();
    const int* offset_hap_batches       = fullBatch.batch_haps_offsets.data(); //batch_2.chars.data();
    const int* offset_read_batches       = fullBatch.batch_reads_offsets.data(); //batch_2.chars.data();

    const int totalNumberOfAlignments = fullBatch.getTotalNumberOfAlignments();

    std::vector<float> alignment_scores_float(totalNumberOfAlignments);

    cudaStream_t streams_part[numPartitions];
    for (int i=0; i<numPartitions; i++) cudaStreamCreate(&streams_part[i]);

    std::vector<cudaStream_t> transferStreams(2);
    for(auto& stream : transferStreams){
        cudaStreamCreate(&stream);
    }

    thrust::device_vector<float> devAlignmentScoresFloat_vec(totalNumberOfAlignments, 0);

    thrust::device_vector<uint8_t> dev_read_chars_vec(read_bytes);
    thrust::device_vector<uint8_t> dev_hap_chars_vec(hap_bytes);
    thrust::device_vector<uint8_t> dev_base_qual_vec(read_bytes);
    thrust::device_vector<uint8_t> dev_ins_qual_vec(read_bytes);
    thrust::device_vector<uint8_t> dev_del_qual_vec(read_bytes);

    thrust::device_vector<int> dev_offset_reads_vec(num_reads);
    thrust::device_vector<int> dev_offset_haps_vec(num_haps);
    thrust::device_vector<int> dev_read_len_vec(num_reads);
    thrust::device_vector<int> dev_hap_len_vec(num_haps);
    thrust::device_vector<int> dev_read_batches_vec(num_batches);
    thrust::device_vector<int> dev_hap_batches_vec(num_batches);
    thrust::device_vector<int> dev_offset_read_batches_vec(num_batches);
    thrust::device_vector<int> dev_offset_hap_batches_vec(num_batches);

    thrust::device_vector<int> d_numIndicesPerPartitionPerBatch(num_batches * numPartitions, 0);
    thrust::device_vector<int> d_indicesPerPartitionPerBatch(num_reads * numPartitions, -1);
    thrust::device_vector<int> d_resultOffsetsPerBatch(num_batches);
    thrust::device_vector<int> d_numAlignmentsPerBatch(num_batches * numPartitions);
    thrust::device_vector<int> d_numAlignmentsPerBatchInclPrefixSum(num_batches * numPartitions);
    thrust::device_vector<int> d_numAlignmentsPerPartition(numPartitions);
    std::vector<int, PinnedAllocator<int>> h_numAlignmentsPerPartition(numPartitions);

    uint8_t* const dev_read_chars = dev_read_chars_vec.data().get();
    uint8_t* const dev_hap_chars = dev_hap_chars_vec.data().get();
    uint8_t* const dev_base_qual = dev_base_qual_vec.data().get();
    uint8_t* const dev_ins_qual = dev_ins_qual_vec.data().get();
    uint8_t* const dev_del_qual = dev_del_qual_vec.data().get();
    int* const dev_offset_reads = dev_offset_reads_vec.data().get();
    int* const dev_offset_haps = dev_offset_haps_vec.data().get();
    int* const dev_read_len = dev_read_len_vec.data().get();
    int* const dev_hap_len = dev_hap_len_vec.data().get();
    int* const dev_read_batches = dev_read_batches_vec.data().get();
    int* const dev_hap_batches = dev_hap_batches_vec.data().get();
    int* const dev_offset_read_batches = dev_offset_read_batches_vec.data().get();
    int* const dev_offset_hap_batches = dev_offset_hap_batches_vec.data().get();
    float* const devAlignmentScoresFloat = devAlignmentScoresFloat_vec.data().get();

    int numProcessedAlignmentsByChunks = 0;
    int numProcessedBatchesByChunks = 0;

    const int numTransferChunks = SDIV(num_batches, options.transferchunksize);

    for(int computeChunk = 0, transferChunk = 0; computeChunk < numTransferChunks; computeChunk++){
        for(; transferChunk < numTransferChunks && transferChunk < (computeChunk + 2); transferChunk++){
            nvtx3::scoped_range sr1("transferChunk");
            cudaStream_t transferStream = transferStreams[transferChunk % 2];
            
            const int firstBatchId = transferChunk * options.transferchunksize;
            const int lastBatchId_excl = std::min((transferChunk+1)* options.transferchunksize, num_batches);
            const int numBatchesInChunk = lastBatchId_excl - firstBatchId;

            const int firstReadInChunk = offset_read_batches[firstBatchId];
            const int lastReadInChunk_excl = offset_read_batches[lastBatchId_excl];
            const int numReadsInChunk = lastReadInChunk_excl - firstReadInChunk;

            const int firstHapInChunk = offset_hap_batches[firstBatchId];
            const int lastHapInChunk_excl = offset_hap_batches[lastBatchId_excl];
            const int numHapsInChunk = lastHapInChunk_excl - firstHapInChunk;

            const size_t numReadBytesInChunk = offset_reads[lastReadInChunk_excl] - offset_reads[firstReadInChunk];
            const size_t numHapBytesInChunk = offset_haps[lastHapInChunk_excl] - offset_haps[firstHapInChunk];

            // std::cout << "transferChunk " << transferChunk << "\n";
            // std::cout << "firstBatchId " << firstBatchId << "\n";
            // std::cout << "lastBatchId_excl " << lastBatchId_excl << "\n";
            // std::cout << "numBatchesInChunk " << numBatchesInChunk << "\n";
            // std::cout << "firstReadInChunk " << firstReadInChunk << "\n";
            // std::cout << "lastReadInChunk_excl " << lastReadInChunk_excl << "\n";
            // std::cout << "numReadsInChunk " << numReadsInChunk << "\n";
            // std::cout << "firstHapInChunk " << firstHapInChunk << "\n";
            // std::cout << "lastHapInChunk_excl " << lastHapInChunk_excl << "\n";
            // std::cout << "numHapsInChunk " << numHapsInChunk << "\n";
            // std::cout << "numReadBytesInChunk " << numReadBytesInChunk << "\n";
            // std::cout << "numHapBytesInChunk " << numHapBytesInChunk << "\n";
            // std::cout << "----------------------------\n";



            cudaMemcpyAsync(dev_read_chars + offset_reads[firstReadInChunk], read_chars + offset_reads[firstReadInChunk], numReadBytesInChunk, cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_hap_chars + offset_haps[firstHapInChunk], hap_chars + offset_haps[firstHapInChunk], numHapBytesInChunk, cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_base_qual + offset_reads[firstReadInChunk], base_qual + offset_reads[firstReadInChunk], numReadBytesInChunk, cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_ins_qual + offset_reads[firstReadInChunk], ins_qual + offset_reads[firstReadInChunk], numReadBytesInChunk, cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_del_qual + offset_reads[firstReadInChunk], del_qual + offset_reads[firstReadInChunk], numReadBytesInChunk, cudaMemcpyHostToDevice, transferStream); CUERR

            cudaMemcpyAsync(dev_offset_reads + firstReadInChunk, offset_reads + firstReadInChunk, numReadsInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_offset_haps + firstHapInChunk, offset_haps + firstHapInChunk, numHapsInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_read_len + firstReadInChunk, read_len + firstReadInChunk, numReadsInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_hap_len + firstHapInChunk, hap_len + firstHapInChunk, numHapsInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_offset_read_batches + firstBatchId, offset_read_batches + firstBatchId, numBatchesInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_offset_hap_batches + firstBatchId, offset_hap_batches + firstBatchId, numBatchesInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_read_batches + firstBatchId, read_batches + firstBatchId, numBatchesInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_hap_batches + firstBatchId, hap_batches + firstBatchId, numBatchesInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream); CUERR
        }
        nvtx3::scoped_range sr2("computeChunk");
        cudaStream_t mainStream = transferStreams[computeChunk % 2];
        const int firstBatchId = computeChunk * options.transferchunksize;
        const int lastBatchId_excl = std::min((computeChunk+1)* options.transferchunksize, num_batches);
        const int numBatchesInChunk = lastBatchId_excl - firstBatchId;

        const int firstReadInChunk = offset_read_batches[firstBatchId];
        const int lastReadInChunk_excl = offset_read_batches[lastBatchId_excl];
        const int numReadsInChunk = lastReadInChunk_excl - firstReadInChunk;

        const int firstHapInChunk = offset_hap_batches[firstBatchId];
        const int lastHapInChunk_excl = offset_hap_batches[lastBatchId_excl];
        const int numHapsInChunk = lastHapInChunk_excl - firstHapInChunk;

        const size_t numReadBytesInChunk = offset_reads[lastReadInChunk_excl] - offset_reads[firstReadInChunk];
        const size_t numHapBytesInChunk = offset_haps[lastHapInChunk_excl] - offset_haps[firstHapInChunk];

        convert_DNA<<<numReadsInChunk, 128, 0, mainStream>>>(dev_read_chars + offset_reads[firstReadInChunk], numReadBytesInChunk);
        convert_DNA<<<numHapsInChunk, 128, 0, mainStream>>>(dev_hap_chars + offset_haps[firstHapInChunk], numHapBytesInChunk);

        //ensure buffers used by previous batch are no longer in use
        for(int i = 0; i < numPartitions; i++){
            cudaStreamSynchronize(streams_part[i]);
        }

        cudaMemsetAsync(d_numIndicesPerPartitionPerBatch.data().get(), 0, sizeof(int) * numPartitions * numBatchesInChunk, mainStream); CUERR;
        partitionIndicesKernel<<<numBatchesInChunk, 128, 0, mainStream>>>(
            d_numIndicesPerPartitionPerBatch.data().get(),
            d_indicesPerPartitionPerBatch.data().get(),
            dev_read_len,
            dev_read_batches + firstBatchId,
            dev_offset_read_batches + firstBatchId,
            numBatchesInChunk,
            numReadsInChunk
        );
        CUERR;

        #if 0
            thrust::host_vector<int> h_numIndicesPerPartitionPerBatch = d_numIndicesPerPartitionPerBatch;
            thrust::host_vector<int> h_indicesPerPartitionPerBatch = d_indicesPerPartitionPerBatch;

            for(int p = 0; p < numPartitions; p++){
                if(p <= 4){
                    std::cout << "Partition p = " << p << "\n";
                    std::cout << "numIndicesPerBatch: ";
                    for(int b = 0; b < numBatchesInChunk; b++){
                        std::cout << h_numIndicesPerPartitionPerBatch[p * numBatchesInChunk + b] << ", ";
                    }
                    std::cout << "\n";

                    std::cout << "indicesPerBatch: ";
                    for(int b = 0; b < numBatchesInChunk; b++){
                        const int num = h_numIndicesPerPartitionPerBatch[p * numBatchesInChunk + b];
                        for(int i = 0; i < num; i++){
                            const int outputOffset = offset_read_batches[firstBatchId + b] - offset_read_batches[firstBatchId];
                            std::cout << h_indicesPerPartitionPerBatch[p * numReadsInChunk + outputOffset + i];
                            if(i != num-1){
                                std::cout << ", ";
                            }
                        }
                        std::cout << " | ";
                    }
                    std::cout << "\n";
                }
            }
        #endif
    
        thrust::transform(
            thrust::cuda::par_nosync.on(mainStream),
            dev_read_batches + firstBatchId,
            dev_read_batches + firstBatchId + numBatchesInChunk,
            dev_hap_batches + firstBatchId,
            d_resultOffsetsPerBatch.begin(),
            thrust::multiplies<int>{}
        );
        thrust::exclusive_scan(
            thrust::cuda::par_nosync(ThrustCudaMallocAsyncAllocator<int>(mainStream)).on(mainStream),
            d_resultOffsetsPerBatch.begin(),
            d_resultOffsetsPerBatch.begin() + numBatchesInChunk,
            d_resultOffsetsPerBatch.begin()
        );

        // thrust::host_vector<int> h_resultOffsetsPerBatch = d_resultOffsetsPerBatch;
        // std::cout << "h_resultOffsetsPerBatch. numBatchesInChunk " << numBatchesInChunk << "\n";
        // for(auto& x : h_resultOffsetsPerBatch){
        //     std::cout << x << " ";
        // }
        // std::cout << "\n";

        {     
            int* d_numAlignmentsPerPartitionPerBatch = d_numAlignmentsPerBatchInclPrefixSum.data().get(); // reuse
            const int* d_numHaplotypesPerBatch = dev_hap_batches;
            computeAlignmentsPerPartitionPerBatch<<<dim3(SDIV(numBatchesInChunk, 128), numPartitions), 128,0, mainStream>>>(
                d_numAlignmentsPerPartitionPerBatch,
                d_numIndicesPerPartitionPerBatch.data().get(),
                d_numHaplotypesPerBatch + firstBatchId,
                numPartitions, 
                numBatchesInChunk
            ); CUERR;
    
            auto offsets = thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                [numBatchesInChunk] __host__ __device__(int partition){
                    return partition * numBatchesInChunk;
                }
            );
            size_t temp_storage_bytes = 0;
            cub::DeviceSegmentedReduce::Sum(nullptr, temp_storage_bytes, d_numAlignmentsPerPartitionPerBatch, 
                d_numAlignmentsPerPartition.data().get(), numPartitions, offsets, offsets + 1, mainStream); CUERR;
            thrust::device_vector<char, ThrustCudaMallocAsyncAllocator<char>> d_temp(temp_storage_bytes, ThrustCudaMallocAsyncAllocator<char>(mainStream));
            cub::DeviceSegmentedReduce::Sum(d_temp.data().get(), temp_storage_bytes, d_numAlignmentsPerPartitionPerBatch, 
                d_numAlignmentsPerPartition.data().get(), numPartitions, offsets, offsets + 1, mainStream); CUERR;
        }

        cudaMemcpyAsync(h_numAlignmentsPerPartition.data(), d_numAlignmentsPerPartition.data().get(), sizeof(int) * numPartitions, cudaMemcpyDeviceToHost, mainStream); CUERR;
        cudaStreamSynchronize(mainStream); CUERR;

        // std::cout << "h_numAlignmentsPerPartition: ";
        // for(int i = 0; i< numPartitions; i++){
        //     std::cout << h_numAlignmentsPerPartition[i] << ", ";
        // }
        // std::cout << "\n";

        // thrust::host_vector<int> h_offset_read_batches_vec = dev_offset_read_batches_vec;
        // for(auto& x : h_offset_read_batches_vec){
        //     std::cout << x << " ";
        // }
        // std::cout << "\n";

        int numProcessedAlignmentsByCurrentChunk = 0;

        #define COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM(stream) \
            nvtx3::scoped_range sr3("partition"); \
            constexpr int groupsPerBlock = blocksize / group_size; \
            const int numAlignmentsInPartition = h_numAlignmentsPerPartition[partitionId]; \
            const int numBlocks = SDIV(numAlignmentsInPartition, groupsPerBlock); \
            const int* d_numIndicesPerBatch = d_numIndicesPerPartitionPerBatch.data().get() + partitionId*numBatchesInChunk; \
            thrust::transform( \
                thrust::cuda::par_nosync.on(stream), \
                d_numIndicesPerBatch, \
                d_numIndicesPerBatch + numBatchesInChunk, \
                dev_hap_batches + numProcessedBatchesByChunks, \
                d_numAlignmentsPerBatch.begin() + partitionId * numBatchesInChunk, \
                thrust::multiplies<int>{} \
            ); \
            thrust::inclusive_scan( \
                thrust::cuda::par_nosync(ThrustCudaMallocAsyncAllocator<int>(stream)).on(stream), \
                d_numAlignmentsPerBatch.begin() + partitionId * numBatchesInChunk, \
                d_numAlignmentsPerBatch.begin() + partitionId * numBatchesInChunk+ numBatchesInChunk, \
                d_numAlignmentsPerBatchInclPrefixSum.begin() + partitionId * numBatchesInChunk \
            );  \
            PairHMM_align_partition_float_allowMultipleBatchesPerWarp<group_size,numRegs><<<numBlocks, blocksize,0,stream>>>( \
                dev_read_chars,  \
                dev_hap_chars,  \
                dev_base_qual,  \
                dev_ins_qual,  \
                dev_del_qual,  \
                devAlignmentScoresFloat + numProcessedAlignmentsByChunks,  \
                dev_offset_reads + firstReadInChunk,  \
                dev_offset_haps + firstHapInChunk,  \
                dev_read_len + firstReadInChunk,  \
                dev_hap_len + firstHapInChunk, \
                dev_read_batches + firstBatchId, \
                dev_hap_batches + firstBatchId,  \
                dev_offset_hap_batches + firstBatchId,  \
                d_numIndicesPerBatch,  \
                d_indicesPerPartitionPerBatch.data().get() + partitionId * numReadsInChunk,  \
                dev_offset_read_batches + firstBatchId,   \
                numBatchesInChunk,  \
                d_resultOffsetsPerBatch.data().get(),  \
                d_numAlignmentsPerBatch.data().get() + partitionId * numBatchesInChunk,  \
                d_numAlignmentsPerBatchInclPrefixSum.data().get() + partitionId * numBatchesInChunk,  \
                numAlignmentsInPartition \
            ); CUERR; \
            numProcessedAlignmentsByCurrentChunk += numAlignmentsInPartition;


        LAUNCH_ALL_KERNELS

        numProcessedAlignmentsByChunks += numProcessedAlignmentsByCurrentChunk;
        numProcessedBatchesByChunks += numBatchesInChunk;

        // for(int i = 0; i < numPartitions; i++){
        //     cudaStreamSynchronize(streams_part[i]);
        // }

            
    }
    #undef  COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM

    
    cudaMemcpy(alignment_scores_float.data(), devAlignmentScoresFloat, totalNumberOfAlignments*sizeof(float), cudaMemcpyDeviceToHost);  CUERR

    for (int i=0; i<numPartitions; i++) cudaStreamDestroy(streams_part[i]); CUERR;
    for(auto& stream : transferStreams){
        cudaStreamDestroy(stream);
    }

    totalTimer.stop();
    totalTimer.printGCUPS(countsOfDPCells.totalDPCells);

    return alignment_scores_float;
}





std::vector<float> processBatchAsWhole_float_coalesced_smem(
    const batch& fullBatch, 
    const Options& /*options*/, 
    const CountsOfDPCells& countsOfDPCells
){
    helpers::CpuTimer totalTimer("processBatchAsWhole_float_coalesced_smem");

    const uint8_t* read_chars       = fullBatch.reads.data(); //batch_2.chars.data();
    const uint read_bytes = fullBatch.reads.size();
    const uint8_t* hap_chars       = fullBatch.haps.data(); //batch_2.chars.data();
    const uint hap_bytes = fullBatch.haps.size();
    const uint8_t* base_qual       = fullBatch.base_quals.data(); //batch_2.chars.data();
    const uint8_t* ins_qual       = fullBatch.ins_quals.data(); //batch_2.chars.data();
    const uint8_t* del_qual       = fullBatch.del_quals.data(); //batch_2.chars.data();
    const int* offset_reads       = fullBatch.read_offsets.data(); //batch_2.chars.data();
    const int* offset_haps       = fullBatch.hap_offsets.data(); //batch_2.chars.data();
    const int* read_len       = fullBatch.readlen.data(); //batch_2.chars.data();
    const int* hap_len       = fullBatch.haplen.data(); //batch_2.chars.data();
    const int num_reads = fullBatch.readlen.size(); //batch_2.chars.data();
    const int num_haps = fullBatch.haplen.size(); //batch_2.chars.data();
    const int num_batches = fullBatch.batch_reads.size(); //batch_2.chars.data();
    const int* hap_batches       = fullBatch.batch_haps.data(); //batch_2.chars.data();
    const int* read_batches       = fullBatch.batch_reads.data(); //batch_2.chars.data();
    const int* offset_hap_batches       = fullBatch.batch_haps_offsets.data(); //batch_2.chars.data();
    const int* offset_read_batches       = fullBatch.batch_reads_offsets.data(); //batch_2.chars.data();

    const int totalNumberOfAlignments = fullBatch.getTotalNumberOfAlignments();

    std::vector<float> alignment_scores_float(totalNumberOfAlignments);

    cudaStream_t streams_part[numPartitions];
    for (int i=0; i<numPartitions; i++) cudaStreamCreate(&streams_part[i]);

    thrust::device_vector<uint8_t> dev_read_chars_vec(read_bytes);
    thrust::device_vector<uint8_t> dev_hap_chars_vec(hap_bytes);
    thrust::device_vector<uint8_t> dev_base_qual_vec(read_bytes);
    thrust::device_vector<uint8_t> dev_ins_qual_vec(read_bytes);
    thrust::device_vector<uint8_t> dev_del_qual_vec(read_bytes);

    thrust::device_vector<int> dev_offset_reads_vec(num_reads);
    thrust::device_vector<int> dev_offset_haps_vec(num_haps);
    thrust::device_vector<int> dev_read_len_vec(num_reads);
    thrust::device_vector<int> dev_hap_len_vec(num_haps);
    thrust::device_vector<int> dev_read_batches_vec(num_batches);
    thrust::device_vector<int> dev_hap_batches_vec(num_batches);
    thrust::device_vector<int> dev_offset_read_batches_vec(num_batches);
    thrust::device_vector<int> dev_offset_hap_batches_vec(num_batches);
    thrust::device_vector<float> devAlignmentScoresFloat_vec(totalNumberOfAlignments);

    uint8_t* const dev_read_chars = dev_read_chars_vec.data().get();
    uint8_t* const dev_hap_chars = dev_hap_chars_vec.data().get();
    uint8_t* const dev_base_qual = dev_base_qual_vec.data().get();
    uint8_t* const dev_ins_qual = dev_ins_qual_vec.data().get();
    uint8_t* const dev_del_qual = dev_del_qual_vec.data().get();
    int* const dev_offset_reads = dev_offset_reads_vec.data().get();
    int* const dev_offset_haps = dev_offset_haps_vec.data().get();
    int* const dev_read_len = dev_read_len_vec.data().get();
    int* const dev_hap_len = dev_hap_len_vec.data().get();
    int* const dev_read_batches = dev_read_batches_vec.data().get();
    int* const dev_hap_batches = dev_hap_batches_vec.data().get();
    int* const dev_offset_read_batches = dev_offset_read_batches_vec.data().get();
    int* const dev_offset_hap_batches = dev_offset_hap_batches_vec.data().get();
    float* const devAlignmentScoresFloat = devAlignmentScoresFloat_vec.data().get();


    helpers::CpuTimer transfertimer("DATA_TRANSFER");
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
    transfertimer.print();

    //print_batch(fullBatch); // prints first reads/qualities and haplotype per batch.

    convert_DNA<<<num_reads, 128>>>(dev_read_chars,read_bytes);
    convert_DNA<<<num_haps, 128>>>(dev_hap_chars,hap_bytes);

    thrust::device_vector<int> d_numIndicesPerPartitionPerBatch(num_batches * numPartitions, 0);
    thrust::device_vector<int> d_indicesPerPartitionPerBatch(num_reads * numPartitions, -1);
    thrust::device_vector<int> d_resultOffsetsPerBatch(num_batches);
    thrust::device_vector<int> d_numAlignmentsPerBatch(num_batches * numPartitions);
    thrust::device_vector<int> d_numAlignmentsPerBatchInclPrefixSum(num_batches * numPartitions);
    thrust::device_vector<int> d_numAlignmentsPerPartition(numPartitions);

    partitionIndicesKernel<<<num_batches, 128>>>(
        d_numIndicesPerPartitionPerBatch.data().get(),
        d_indicesPerPartitionPerBatch.data().get(),
        dev_read_len,
        dev_read_batches,
        dev_offset_read_batches,
        num_batches,
        num_reads
    );
    CUERR;

    thrust::transform(
        thrust::cuda::par_nosync.on((cudaStream_t)0),
        dev_read_batches,
        dev_read_batches + num_batches,
        dev_hap_batches,
        d_resultOffsetsPerBatch.begin(),
        thrust::multiplies<int>{}
    );
    thrust::exclusive_scan(
        thrust::cuda::par_nosync(ThrustCudaMallocAsyncAllocator<int>((cudaStream_t)0)).on((cudaStream_t)0),
        d_resultOffsetsPerBatch.begin(),
        d_resultOffsetsPerBatch.begin() + num_batches,
        d_resultOffsetsPerBatch.begin()
    );

    // for(int i = 0; i < std::min(10, num_batches); i++){
    //     std::cout << "batch " << i << ", num reads: " << read_batches[i]
    //         << ", num haps " << hap_batches[i] << ", product: " <<
    //         read_batches[i] * hap_batches[i]
    //         << ", offset : " << d_resultOffsetsPerBatch[i] << "\n";
    // }


    #if 0
        thrust::host_vector<int> h_numIndicesPerPartitionPerBatch = d_numIndicesPerPartitionPerBatch;
        thrust::host_vector<int> h_indicesPerPartitionPerBatch = d_indicesPerPartitionPerBatch;

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
            }
        }
    #endif

    {
        cudaStream_t stream = cudaStreamLegacy;

        int* d_numAlignmentsPerPartitionPerBatch = d_numAlignmentsPerBatchInclPrefixSum.data().get(); // reuse
        const int* d_numHaplotypesPerBatch = dev_hap_batches;
        computeAlignmentsPerPartitionPerBatch<<<dim3(SDIV(num_batches, 128), numPartitions), 128,0, stream>>>(
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
            d_numAlignmentsPerPartition.data().get(), numPartitions, offsets, offsets + 1, stream); CUERR;
        thrust::device_vector<char, ThrustCudaMallocAsyncAllocator<char>> d_temp(temp_storage_bytes, ThrustCudaMallocAsyncAllocator<char>(stream));
        cub::DeviceSegmentedReduce::Sum(d_temp.data().get(), temp_storage_bytes, d_numAlignmentsPerPartitionPerBatch, 
            d_numAlignmentsPerPartition.data().get(), numPartitions, offsets, offsets + 1, stream); CUERR;
    }
    thrust::host_vector<int> h_numAlignmentsPerPartition = d_numAlignmentsPerPartition;

    std::cout << "h_numAlignmentsPerPartition: ";
    for(int i = 0; i< numPartitions; i++){
        std::cout << h_numAlignmentsPerPartition[i] << ", ";
    }
    std::cout << "\n";

    cudaMemset(devAlignmentScoresFloat, 0, totalNumberOfAlignments * sizeof(float)); CUERR;
    helpers::GpuTimer computeTimer("pairhmm float kernels coalesced, total");
    std::vector<std::unique_ptr<helpers::GpuTimer>> perKernelTimers(numPartitions);
    for(int p = 0; p < numPartitions; p++){
        std::string name = "pairhmm float kernel coalesced, partition " + std::to_string(p);
        perKernelTimers[p] = std::make_unique<helpers::GpuTimer>(streams_part[p], name);
    }

    #define COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM(stream) \
        constexpr int groupsPerBlock = blocksize / group_size; \
        const int numAlignmentsInPartition = h_numAlignmentsPerPartition[partitionId]; \
        const int numBlocks = SDIV(numAlignmentsInPartition, groupsPerBlock); \
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
        perKernelTimers[partitionId]->start(); \
        PairHMM_align_partition_float_allowMultipleBatchesPerWarp_coalesced_smem<group_size,numRegs><<<numBlocks, blocksize,0,stream>>>(dev_read_chars, dev_hap_chars, dev_base_qual, dev_ins_qual, dev_del_qual, devAlignmentScoresFloat, dev_offset_reads, dev_offset_haps, dev_read_len, dev_hap_len, dev_read_batches, dev_hap_batches, dev_offset_hap_batches,  \
            d_numIndicesPerBatch, d_indicesPerPartitionPerBatch.data().get() + partitionId*num_reads,  \
            dev_offset_read_batches,  num_batches, d_resultOffsetsPerBatch.data().get(), d_numAlignmentsPerBatch.data().get() + partitionId * num_batches, d_numAlignmentsPerBatchInclPrefixSum.data().get() + partitionId * num_batches, numAlignmentsInPartition); \
        perKernelTimers[partitionId]->stop(); \


    LAUNCH_ALL_KERNELS

    #undef  COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM

    computeTimer.stop();

    for(int p = 0; p < numPartitions; p++){
        if(h_numAlignmentsPerPartition[p] > 0){
            perKernelTimers[p]->printGCUPS(countsOfDPCells.dpCellsPerPartition[p]);
        }
    }
    computeTimer.printGCUPS(countsOfDPCells.totalDPCells);

    cudaMemcpy(alignment_scores_float.data(), devAlignmentScoresFloat, totalNumberOfAlignments*sizeof(float), cudaMemcpyDeviceToHost);  CUERR

    perKernelTimers.clear();

    for (int i=0; i<numPartitions; i++) cudaStreamDestroy(streams_part[i]); CUERR;

    totalTimer.stop();
    totalTimer.printGCUPS(countsOfDPCells.totalDPCells);

    return alignment_scores_float;
}



std::vector<float> processBatch_overlapped_float_coalesced_smem(
    const batch& fullBatch_default, 
    const Options& options, 
    const CountsOfDPCells& countsOfDPCells
){
    helpers::CpuTimer totalTimer("processBatch_overlapped_float_coalesced_smem");

    // pinned_batch fullBatch(fullBatch_default);
    const auto& fullBatch = fullBatch_default;

    const uint8_t* read_chars       = fullBatch.reads.data(); //batch_2.chars.data();
    const uint read_bytes = fullBatch.reads.size();
    const uint8_t* hap_chars       = fullBatch.haps.data(); //batch_2.chars.data();
    const uint hap_bytes = fullBatch.haps.size();
    const uint8_t* base_qual       = fullBatch.base_quals.data(); //batch_2.chars.data();
    const uint8_t* ins_qual       = fullBatch.ins_quals.data(); //batch_2.chars.data();
    const uint8_t* del_qual       = fullBatch.del_quals.data(); //batch_2.chars.data();
    const int* offset_reads       = fullBatch.read_offsets.data(); //batch_2.chars.data();
    const int* offset_haps       = fullBatch.hap_offsets.data(); //batch_2.chars.data();
    const int* read_len       = fullBatch.readlen.data(); //batch_2.chars.data();
    const int* hap_len       = fullBatch.haplen.data(); //batch_2.chars.data();
    const int num_reads = fullBatch.readlen.size(); //batch_2.chars.data();
    const int num_haps = fullBatch.haplen.size(); //batch_2.chars.data();
    const int num_batches = fullBatch.batch_reads.size(); //batch_2.chars.data();
    const int* hap_batches       = fullBatch.batch_haps.data(); //batch_2.chars.data();
    const int* read_batches       = fullBatch.batch_reads.data(); //batch_2.chars.data();
    const int* offset_hap_batches       = fullBatch.batch_haps_offsets.data(); //batch_2.chars.data();
    const int* offset_read_batches       = fullBatch.batch_reads_offsets.data(); //batch_2.chars.data();

    const int totalNumberOfAlignments = fullBatch.getTotalNumberOfAlignments();

    std::vector<float> alignment_scores_float(totalNumberOfAlignments);

    cudaStream_t streams_part[numPartitions];
    for (int i=0; i<numPartitions; i++) cudaStreamCreate(&streams_part[i]);

    std::vector<cudaStream_t> transferStreams(2);
    for(auto& stream : transferStreams){
        cudaStreamCreate(&stream);
    }

    thrust::device_vector<float> devAlignmentScoresFloat_vec(totalNumberOfAlignments, 0);

    thrust::device_vector<uint8_t> dev_read_chars_vec(read_bytes);
    thrust::device_vector<uint8_t> dev_hap_chars_vec(hap_bytes);
    thrust::device_vector<uint8_t> dev_base_qual_vec(read_bytes);
    thrust::device_vector<uint8_t> dev_ins_qual_vec(read_bytes);
    thrust::device_vector<uint8_t> dev_del_qual_vec(read_bytes);

    thrust::device_vector<int> dev_offset_reads_vec(num_reads);
    thrust::device_vector<int> dev_offset_haps_vec(num_haps);
    thrust::device_vector<int> dev_read_len_vec(num_reads);
    thrust::device_vector<int> dev_hap_len_vec(num_haps);
    thrust::device_vector<int> dev_read_batches_vec(num_batches);
    thrust::device_vector<int> dev_hap_batches_vec(num_batches);
    thrust::device_vector<int> dev_offset_read_batches_vec(num_batches);
    thrust::device_vector<int> dev_offset_hap_batches_vec(num_batches);

    thrust::device_vector<int> d_numIndicesPerPartitionPerBatch(num_batches * numPartitions, 0);
    thrust::device_vector<int> d_indicesPerPartitionPerBatch(num_reads * numPartitions, -1);
    thrust::device_vector<int> d_resultOffsetsPerBatch(num_batches);
    thrust::device_vector<int> d_numAlignmentsPerBatch(num_batches * numPartitions);
    thrust::device_vector<int> d_numAlignmentsPerBatchInclPrefixSum(num_batches * numPartitions);
    thrust::device_vector<int> d_numAlignmentsPerPartition(numPartitions);
    std::vector<int, PinnedAllocator<int>> h_numAlignmentsPerPartition(numPartitions);

    uint8_t* const dev_read_chars = dev_read_chars_vec.data().get();
    uint8_t* const dev_hap_chars = dev_hap_chars_vec.data().get();
    uint8_t* const dev_base_qual = dev_base_qual_vec.data().get();
    uint8_t* const dev_ins_qual = dev_ins_qual_vec.data().get();
    uint8_t* const dev_del_qual = dev_del_qual_vec.data().get();
    int* const dev_offset_reads = dev_offset_reads_vec.data().get();
    int* const dev_offset_haps = dev_offset_haps_vec.data().get();
    int* const dev_read_len = dev_read_len_vec.data().get();
    int* const dev_hap_len = dev_hap_len_vec.data().get();
    int* const dev_read_batches = dev_read_batches_vec.data().get();
    int* const dev_hap_batches = dev_hap_batches_vec.data().get();
    int* const dev_offset_read_batches = dev_offset_read_batches_vec.data().get();
    int* const dev_offset_hap_batches = dev_offset_hap_batches_vec.data().get();
    float* const devAlignmentScoresFloat = devAlignmentScoresFloat_vec.data().get();

    int numProcessedAlignmentsByChunks = 0;
    int numProcessedBatchesByChunks = 0;

    const int numTransferChunks = SDIV(num_batches, options.transferchunksize);

    for(int computeChunk = 0, transferChunk = 0; computeChunk < numTransferChunks; computeChunk++){
        for(; transferChunk < numTransferChunks && transferChunk < (computeChunk + 2); transferChunk++){
            nvtx3::scoped_range sr1("transferChunk");
            cudaStream_t transferStream = transferStreams[transferChunk % 2];
            
            const int firstBatchId = transferChunk * options.transferchunksize;
            const int lastBatchId_excl = std::min((transferChunk+1)* options.transferchunksize, num_batches);
            const int numBatchesInChunk = lastBatchId_excl - firstBatchId;

            const int firstReadInChunk = offset_read_batches[firstBatchId];
            const int lastReadInChunk_excl = offset_read_batches[lastBatchId_excl];
            const int numReadsInChunk = lastReadInChunk_excl - firstReadInChunk;

            const int firstHapInChunk = offset_hap_batches[firstBatchId];
            const int lastHapInChunk_excl = offset_hap_batches[lastBatchId_excl];
            const int numHapsInChunk = lastHapInChunk_excl - firstHapInChunk;

            const size_t numReadBytesInChunk = offset_reads[lastReadInChunk_excl] - offset_reads[firstReadInChunk];
            const size_t numHapBytesInChunk = offset_haps[lastHapInChunk_excl] - offset_haps[firstHapInChunk];

            // std::cout << "transferChunk " << transferChunk << "\n";
            // std::cout << "firstBatchId " << firstBatchId << "\n";
            // std::cout << "lastBatchId_excl " << lastBatchId_excl << "\n";
            // std::cout << "numBatchesInChunk " << numBatchesInChunk << "\n";
            // std::cout << "firstReadInChunk " << firstReadInChunk << "\n";
            // std::cout << "lastReadInChunk_excl " << lastReadInChunk_excl << "\n";
            // std::cout << "numReadsInChunk " << numReadsInChunk << "\n";
            // std::cout << "firstHapInChunk " << firstHapInChunk << "\n";
            // std::cout << "lastHapInChunk_excl " << lastHapInChunk_excl << "\n";
            // std::cout << "numHapsInChunk " << numHapsInChunk << "\n";
            // std::cout << "numReadBytesInChunk " << numReadBytesInChunk << "\n";
            // std::cout << "numHapBytesInChunk " << numHapBytesInChunk << "\n";
            // std::cout << "----------------------------\n";



            cudaMemcpyAsync(dev_read_chars + offset_reads[firstReadInChunk], read_chars + offset_reads[firstReadInChunk], numReadBytesInChunk, cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_hap_chars + offset_haps[firstHapInChunk], hap_chars + offset_haps[firstHapInChunk], numHapBytesInChunk, cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_base_qual + offset_reads[firstReadInChunk], base_qual + offset_reads[firstReadInChunk], numReadBytesInChunk, cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_ins_qual + offset_reads[firstReadInChunk], ins_qual + offset_reads[firstReadInChunk], numReadBytesInChunk, cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_del_qual + offset_reads[firstReadInChunk], del_qual + offset_reads[firstReadInChunk], numReadBytesInChunk, cudaMemcpyHostToDevice, transferStream); CUERR

            cudaMemcpyAsync(dev_offset_reads + firstReadInChunk, offset_reads + firstReadInChunk, numReadsInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_offset_haps + firstHapInChunk, offset_haps + firstHapInChunk, numHapsInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_read_len + firstReadInChunk, read_len + firstReadInChunk, numReadsInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_hap_len + firstHapInChunk, hap_len + firstHapInChunk, numHapsInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_offset_read_batches + firstBatchId, offset_read_batches + firstBatchId, numBatchesInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_offset_hap_batches + firstBatchId, offset_hap_batches + firstBatchId, numBatchesInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_read_batches + firstBatchId, read_batches + firstBatchId, numBatchesInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream); CUERR
            cudaMemcpyAsync(dev_hap_batches + firstBatchId, hap_batches + firstBatchId, numBatchesInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream); CUERR
        }
        nvtx3::scoped_range sr2("computeChunk");
        cudaStream_t mainStream = transferStreams[computeChunk % 2];
        const int firstBatchId = computeChunk * options.transferchunksize;
        const int lastBatchId_excl = std::min((computeChunk+1)* options.transferchunksize, num_batches);
        const int numBatchesInChunk = lastBatchId_excl - firstBatchId;

        const int firstReadInChunk = offset_read_batches[firstBatchId];
        const int lastReadInChunk_excl = offset_read_batches[lastBatchId_excl];
        const int numReadsInChunk = lastReadInChunk_excl - firstReadInChunk;

        const int firstHapInChunk = offset_hap_batches[firstBatchId];
        const int lastHapInChunk_excl = offset_hap_batches[lastBatchId_excl];
        const int numHapsInChunk = lastHapInChunk_excl - firstHapInChunk;

        const size_t numReadBytesInChunk = offset_reads[lastReadInChunk_excl] - offset_reads[firstReadInChunk];
        const size_t numHapBytesInChunk = offset_haps[lastHapInChunk_excl] - offset_haps[firstHapInChunk];

        convert_DNA<<<numReadsInChunk, 128, 0, mainStream>>>(dev_read_chars + offset_reads[firstReadInChunk], numReadBytesInChunk);
        convert_DNA<<<numHapsInChunk, 128, 0, mainStream>>>(dev_hap_chars + offset_haps[firstHapInChunk], numHapBytesInChunk);

        //ensure buffers used by previous batch are no longer in use
        for(int i = 0; i < numPartitions; i++){
            cudaStreamSynchronize(streams_part[i]);
        }

        cudaMemsetAsync(d_numIndicesPerPartitionPerBatch.data().get(), 0, sizeof(int) * numPartitions * numBatchesInChunk, mainStream); CUERR;
        partitionIndicesKernel<<<numBatchesInChunk, 128, 0, mainStream>>>(
            d_numIndicesPerPartitionPerBatch.data().get(),
            d_indicesPerPartitionPerBatch.data().get(),
            dev_read_len,
            dev_read_batches + firstBatchId,
            dev_offset_read_batches + firstBatchId,
            numBatchesInChunk,
            numReadsInChunk
        );
        CUERR;

        #if 0
            thrust::host_vector<int> h_numIndicesPerPartitionPerBatch = d_numIndicesPerPartitionPerBatch;
            thrust::host_vector<int> h_indicesPerPartitionPerBatch = d_indicesPerPartitionPerBatch;

            for(int p = 0; p < numPartitions; p++){
                if(p <= 4){
                    std::cout << "Partition p = " << p << "\n";
                    std::cout << "numIndicesPerBatch: ";
                    for(int b = 0; b < numBatchesInChunk; b++){
                        std::cout << h_numIndicesPerPartitionPerBatch[p * numBatchesInChunk + b] << ", ";
                    }
                    std::cout << "\n";

                    std::cout << "indicesPerBatch: ";
                    for(int b = 0; b < numBatchesInChunk; b++){
                        const int num = h_numIndicesPerPartitionPerBatch[p * numBatchesInChunk + b];
                        for(int i = 0; i < num; i++){
                            const int outputOffset = offset_read_batches[firstBatchId + b] - offset_read_batches[firstBatchId];
                            std::cout << h_indicesPerPartitionPerBatch[p * numReadsInChunk + outputOffset + i];
                            if(i != num-1){
                                std::cout << ", ";
                            }
                        }
                        std::cout << " | ";
                    }
                    std::cout << "\n";
                }
            }
        #endif
    
        thrust::transform(
            thrust::cuda::par_nosync.on(mainStream),
            dev_read_batches + firstBatchId,
            dev_read_batches + firstBatchId + numBatchesInChunk,
            dev_hap_batches + firstBatchId,
            d_resultOffsetsPerBatch.begin(),
            thrust::multiplies<int>{}
        );
        thrust::exclusive_scan(
            thrust::cuda::par_nosync(ThrustCudaMallocAsyncAllocator<int>(mainStream)).on(mainStream),
            d_resultOffsetsPerBatch.begin(),
            d_resultOffsetsPerBatch.begin() + numBatchesInChunk,
            d_resultOffsetsPerBatch.begin()
        );

        // thrust::host_vector<int> h_resultOffsetsPerBatch = d_resultOffsetsPerBatch;
        // std::cout << "h_resultOffsetsPerBatch. numBatchesInChunk " << numBatchesInChunk << "\n";
        // for(auto& x : h_resultOffsetsPerBatch){
        //     std::cout << x << " ";
        // }
        // std::cout << "\n";

        {     
            int* d_numAlignmentsPerPartitionPerBatch = d_numAlignmentsPerBatchInclPrefixSum.data().get(); // reuse
            const int* d_numHaplotypesPerBatch = dev_hap_batches;
            computeAlignmentsPerPartitionPerBatch<<<dim3(SDIV(numBatchesInChunk, 128), numPartitions), 128,0, mainStream>>>(
                d_numAlignmentsPerPartitionPerBatch,
                d_numIndicesPerPartitionPerBatch.data().get(),
                d_numHaplotypesPerBatch + firstBatchId,
                numPartitions, 
                numBatchesInChunk
            ); CUERR;
    
            auto offsets = thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                [numBatchesInChunk] __host__ __device__(int partition){
                    return partition * numBatchesInChunk;
                }
            );
            size_t temp_storage_bytes = 0;
            cub::DeviceSegmentedReduce::Sum(nullptr, temp_storage_bytes, d_numAlignmentsPerPartitionPerBatch, 
                d_numAlignmentsPerPartition.data().get(), numPartitions, offsets, offsets + 1, mainStream); CUERR;
            thrust::device_vector<char, ThrustCudaMallocAsyncAllocator<char>> d_temp(temp_storage_bytes, ThrustCudaMallocAsyncAllocator<char>(mainStream));
            cub::DeviceSegmentedReduce::Sum(d_temp.data().get(), temp_storage_bytes, d_numAlignmentsPerPartitionPerBatch, 
                d_numAlignmentsPerPartition.data().get(), numPartitions, offsets, offsets + 1, mainStream); CUERR;
        }

        cudaMemcpyAsync(h_numAlignmentsPerPartition.data(), d_numAlignmentsPerPartition.data().get(), sizeof(int) * numPartitions, cudaMemcpyDeviceToHost, mainStream); CUERR;
        cudaStreamSynchronize(mainStream); CUERR;

        // std::cout << "h_numAlignmentsPerPartition: ";
        // for(int i = 0; i< numPartitions; i++){
        //     std::cout << h_numAlignmentsPerPartition[i] << ", ";
        // }
        // std::cout << "\n";

        // thrust::host_vector<int> h_offset_read_batches_vec = dev_offset_read_batches_vec;
        // for(auto& x : h_offset_read_batches_vec){
        //     std::cout << x << " ";
        // }
        // std::cout << "\n";

        int numProcessedAlignmentsByCurrentChunk = 0;

        #define COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM(stream) \
            nvtx3::scoped_range sr3("partition"); \
            constexpr int groupsPerBlock = blocksize / group_size; \
            const int numAlignmentsInPartition = h_numAlignmentsPerPartition[partitionId]; \
            const int numBlocks = SDIV(numAlignmentsInPartition, groupsPerBlock); \
            const int* d_numIndicesPerBatch = d_numIndicesPerPartitionPerBatch.data().get() + partitionId*numBatchesInChunk; \
            thrust::transform( \
                thrust::cuda::par_nosync.on(stream), \
                d_numIndicesPerBatch, \
                d_numIndicesPerBatch + numBatchesInChunk, \
                dev_hap_batches + numProcessedBatchesByChunks, \
                d_numAlignmentsPerBatch.begin() + partitionId * numBatchesInChunk, \
                thrust::multiplies<int>{} \
            ); \
            thrust::inclusive_scan( \
                thrust::cuda::par_nosync(ThrustCudaMallocAsyncAllocator<int>(stream)).on(stream), \
                d_numAlignmentsPerBatch.begin() + partitionId * numBatchesInChunk, \
                d_numAlignmentsPerBatch.begin() + partitionId * numBatchesInChunk+ numBatchesInChunk, \
                d_numAlignmentsPerBatchInclPrefixSum.begin() + partitionId * numBatchesInChunk \
            );  \
            PairHMM_align_partition_float_allowMultipleBatchesPerWarp_coalesced_smem<group_size,numRegs><<<numBlocks, blocksize,0,stream>>>( \
                dev_read_chars,  \
                dev_hap_chars,  \
                dev_base_qual,  \
                dev_ins_qual,  \
                dev_del_qual,  \
                devAlignmentScoresFloat + numProcessedAlignmentsByChunks,  \
                dev_offset_reads + firstReadInChunk,  \
                dev_offset_haps + firstHapInChunk,  \
                dev_read_len + firstReadInChunk,  \
                dev_hap_len + firstHapInChunk, \
                dev_read_batches + firstBatchId, \
                dev_hap_batches + firstBatchId,  \
                dev_offset_hap_batches + firstBatchId,  \
                d_numIndicesPerBatch,  \
                d_indicesPerPartitionPerBatch.data().get() + partitionId * numReadsInChunk,  \
                dev_offset_read_batches + firstBatchId,   \
                numBatchesInChunk,  \
                d_resultOffsetsPerBatch.data().get(),  \
                d_numAlignmentsPerBatch.data().get() + partitionId * numBatchesInChunk,  \
                d_numAlignmentsPerBatchInclPrefixSum.data().get() + partitionId * numBatchesInChunk,  \
                numAlignmentsInPartition \
            ); CUERR; \
            numProcessedAlignmentsByCurrentChunk += numAlignmentsInPartition;


        LAUNCH_ALL_KERNELS

        numProcessedAlignmentsByChunks += numProcessedAlignmentsByCurrentChunk;
        numProcessedBatchesByChunks += numBatchesInChunk;

        // for(int i = 0; i < numPartitions; i++){
        //     cudaStreamSynchronize(streams_part[i]);
        // }

            
    }
    #undef  COMPUTE_NUM_ALIGNMENTS_AND_PAIRHMM

    
    cudaMemcpy(alignment_scores_float.data(), devAlignmentScoresFloat, totalNumberOfAlignments*sizeof(float), cudaMemcpyDeviceToHost);  CUERR

    for (int i=0; i<numPartitions; i++) cudaStreamDestroy(streams_part[i]); CUERR;
    for(auto& stream : transferStreams){
        cudaStreamDestroy(stream);
    }

    totalTimer.stop();
    totalTimer.printGCUPS(countsOfDPCells.totalDPCells);

    return alignment_scores_float;
}




struct ScoreComparisonResult{

};


void computeAbsoluteErrorStatistics(const std::vector<float>& scoresA, const std::vector<float>& scoresB){
    assert(scoresA.size() == scoresB.size());

    std::vector<float> bins{0.0, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001, 0.00001, 0.000001};

    std::vector<int> numErrorsPerBin(bins.size(), 0);

    for(int b = 0; b < int(bins.size()); b++){
        for(int i = 0; i < int(scoresA.size()); i++){
            if(std::abs(scoresA[i] - scoresB[i]) > bins[b]){
                numErrorsPerBin[b]++;
            }
        }
    }

    for(int b = 0; b < int(bins.size()); b++){
        std::cout << "num alignments with absolut error > " << bins[b] << " : " << numErrorsPerBin[b] << " (" << double(numErrorsPerBin[b]) / scoresA.size() * 100.0 << " %)\n";
    }
}

void computeRelativeErrorStatistics(const std::vector<float>& scoresA, const std::vector<float>& scoresB){
    assert(scoresA.size() == scoresB.size());

    std::vector<float> bins{0.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001};

    std::vector<int> numErrorsPerBin(bins.size(), 0);

    for(int b = 0; b < int(bins.size()); b++){
        for(int i = 0; i < int(scoresA.size()); i++){
            const float error = std::abs(scoresA[i] - scoresB[i]);
            const float relError = error / std::abs(scoresA[i]);
            if(relError > bins[b]){
                numErrorsPerBin[b]++;
            }
        }
    }

    for(int b = 0; b < int(bins.size()); b++){
        std::cout << "num alignments with relative error > " << bins[b] << " : " << numErrorsPerBin[b] << " (" << double(numErrorsPerBin[b]) / scoresA.size() * 100.0 << " %)\n";
    }
}



batch parseInputFile(const std::string& filename, const std::vector<float>& ph2pr){
    std::ifstream file_(filename);
    if(!file_){
        throw std::runtime_error("Could not open file" + filename);
    }
    batch fullBatch;
    fullBatch.read_offsets.push_back(0);
    fullBatch.hap_offsets.push_back(0);
    int lastread = 0;
    int lasthap = 0;
    std::string linebuffer_;
    while (!file_.eof()){

        getline(file_, linebuffer_);
        if (file_.good()){
            const size_t split = linebuffer_.find(" ");
            const int nread = stoi(linebuffer_.substr(0,split));
            const int nhapl = stoi(linebuffer_.substr(split));

            batch test = read_batch(file_, ph2pr, nread, nhapl, false, lastread, lasthap);
            lasthap = test.lasthapoffset;
            lastread = test.lastreadoffset;
            concat_batch(fullBatch, test);
        }
    }
    generate_batch_offsets(fullBatch);

    return fullBatch;
}


#ifdef ENABLE_PEAK_BENCH_HALF

template<int group_size, int numRegs>
void runPeakBenchHalfImpl(int sequencelength){
    const int readLength = sequencelength;
    const int hapLength = sequencelength;
    const int paddedReadLength = SDIV(readLength, 4) * 4;
    const int paddedHapLength = SDIV(hapLength, 4) * 4;
    const int numReadsInBatch = 32;
    const int numHapsInBatch = 32;

    const size_t readBytes = 256 * 1024 * 1024;
    const size_t hapBytes = 256 * 1024 * 1024;

    if(readLength > group_size * numRegs){
        std::cout << "read length " << readLength << " > " << group_size << " * " << numRegs << ". Skipping\n";
        return;
    }

    
    std::vector<char> readData(readBytes);
    std::vector<char> hapData(hapBytes);
    const char* letters = "ACGT";
    
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dist(0,3);
    for(size_t i = 0; i < readBytes; i++){
        readData[i] = letters[dist(gen)];
    }
    hapData = readData;
    
    const int numReadsInReadData = readBytes / paddedReadLength;
    const int numHapsInHapData = hapBytes / paddedHapLength;
    const int maxNumBatchesInGeneratedData = std::min(numReadsInReadData / numReadsInBatch, numHapsInHapData / numHapsInBatch);
    const int numBatches = maxNumBatchesInGeneratedData;
    const int totalNumReads = maxNumBatchesInGeneratedData * numReadsInBatch;
    const int totalNumHaps = maxNumBatchesInGeneratedData * numHapsInBatch;
    const int totalNumAlignments = numReadsInBatch * numHapsInBatch * numBatches;
    const size_t totalDPCells = size_t(readLength) * size_t(hapLength) * size_t(numReadsInBatch) * size_t(numHapsInBatch) * size_t(numBatches);

    // std::cout << "numReadsInReadData " << numReadsInReadData
    //     << ", numHapsInHapData " << numHapsInHapData
    //     << ", numBatches " << numBatches
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


    std::vector<int> numReadsPerBatch(numBatches, numReadsInBatch);
    std::vector<int> numHapsPerBatch(numBatches, numHapsInBatch);
    std::vector<int> numAlignmentsPerBatch(numBatches, numReadsInBatch * numHapsInBatch);
    std::vector<int> numReadsPerBatchPrefixSum(numBatches);
    std::vector<int> numHapsPerBatchPrefixSum(numBatches);
    std::vector<int> numAlignmentsPerBatchInclusivePrefixSum(numBatches);
    std::exclusive_scan(numReadsPerBatch.begin(), numReadsPerBatch.end(), numReadsPerBatchPrefixSum.begin(), int(0));
    std::exclusive_scan(numHapsPerBatch.begin(), numHapsPerBatch.end(), numHapsPerBatchPrefixSum.begin(), int(0));
    std::inclusive_scan(numAlignmentsPerBatch.begin(), numAlignmentsPerBatch.end(), numAlignmentsPerBatchInclusivePrefixSum.begin());


    thrust::device_vector<uint8_t> d_readData = readData;
    thrust::device_vector<uint8_t> d_hapData = hapData;
    thrust::device_vector<int> d_readLengths = readLengths;
    thrust::device_vector<int> d_hapLengths = hapLengths;
    thrust::device_vector<uint8_t> d_baseQuals = baseQuals;
    thrust::device_vector<uint8_t> d_insQuals = insQuals;
    thrust::device_vector<uint8_t> d_delQuals = delQuals;
    thrust::device_vector<int> d_readBeginOffsets = readBeginOffsets;
    thrust::device_vector<int> d_hapBeginOffsets = hapBeginOffsets;
    thrust::device_vector<int> d_numReadsPerBatch = numReadsPerBatch;
    thrust::device_vector<int> d_numHapsPerBatch = numHapsPerBatch;
    thrust::device_vector<int> d_numAlignmentsPerBatch = numAlignmentsPerBatch;
    thrust::device_vector<int> d_numReadsPerBatchPrefixSum = numReadsPerBatchPrefixSum;
    thrust::device_vector<int> d_numHapsPerBatchPrefixSum = numHapsPerBatchPrefixSum;
    thrust::device_vector<int> d_numAlignmentsPerBatchInclusivePrefixSum = numAlignmentsPerBatchInclusivePrefixSum;

    thrust::device_vector<int> d_numIndicesPerBatch = d_numReadsPerBatch;
    thrust::device_vector<int> d_indicesPerBatch(totalNumReads);
    thrust::transform(
        thrust::make_counting_iterator(0), 
        thrust::make_counting_iterator(totalNumReads),
        d_indicesPerBatch.begin(),
        cuda::proclaim_return_type<int>([numReadsInBatch] __device__ (int i){ return i % numReadsInBatch; })
    );

    thrust::device_vector<int> d_resultOffsetsPerBatch(totalNumAlignments);
    thrust::sequence(d_resultOffsetsPerBatch.begin(), d_resultOffsetsPerBatch.end(), 0);

    thrust::device_vector<float> d_results(totalNumAlignments,-42);

    convert_DNA<<<SDIV(d_readData.size(), 512), 512>>>(d_readData.data().get(), d_readData.size());
    convert_DNA<<<SDIV(d_hapData.size(), 512), 512>>>(d_hapData.data().get(), d_hapData.size());

    cudaStream_t stream = cudaStreamLegacy;
    std::string name = "PairHMM_align_partition_half_allowMultipleBatchesPerWarp_coalesced_smem " + std::to_string(group_size) + " " + std::to_string(numRegs);
    constexpr int groupsPerBlock = 32 / group_size;
    const int numBlocks = SDIV(totalNumAlignments, groupsPerBlock);
    helpers::GpuTimer timer(stream, name);
    PairHMM_align_partition_half_allowMultipleBatchesPerWarp_coalesced_smem<group_size,numRegs><<<numBlocks, 32, 0, stream>>>(
        d_readData.data().get(), 
        d_hapData.data().get(), 
        d_baseQuals.data().get(), 
        d_insQuals.data().get(),  
        d_delQuals.data().get(), 
        d_results.data().get(), 
        d_readBeginOffsets.data().get(), 
        d_hapBeginOffsets.data().get(), 
        d_readLengths.data().get(), 
        d_hapLengths.data().get(), 
        d_numReadsPerBatch.data().get(), 
        d_numHapsPerBatch.data().get(), 
        d_numHapsPerBatchPrefixSum.data().get(),
        d_numIndicesPerBatch.data().get(), 
        d_indicesPerBatch.data().get(),
        d_numReadsPerBatchPrefixSum.data().get(), 
        numBatches, 
        d_resultOffsetsPerBatch.data().get(), 
        d_numAlignmentsPerBatch.data().get(), 
        d_numAlignmentsPerBatchInclusivePrefixSum.data().get(), 
        totalNumAlignments
    );
    CUERR;
    timer.stop();
    timer.printGCUPS(totalDPCells);
    cudaDeviceSynchronize(); CUERR;
}

void runPeakBenchHalf(){
    std::cout << "runPeakBenchHalf\n";

    #define RUN(group_size, numRegs){ \
        const int sequencelength = group_size * numRegs; \
        runPeakBenchHalfImpl<group_size, numRegs>(sequencelength); \
    }


    RUN(4,4);
    RUN(4,8);
    RUN(4,12);
    RUN(4,16);
    RUN(4,20);
    RUN(4,24);
    RUN(4,28);
    RUN(4,32);

    RUN(8,4);
    RUN(8,8);
    RUN(8,12);
    RUN(8,16);
    RUN(8,20);
    RUN(8,24);
    RUN(8,28);
    RUN(8,32);

    RUN(16,4);
    RUN(16,8);
    RUN(16,12);
    RUN(16,16);
    RUN(16,20);
    RUN(16,24);
    RUN(16,28);
    RUN(16,32);

    RUN(32,4);
    RUN(32,8);
    RUN(32,12);
    RUN(32,16);
    RUN(32,20);
    RUN(32,24);
    RUN(32,28);
    RUN(32,32);

    #undef RUN
}

#endif

#ifdef ENABLE_PEAK_BENCH_FLOAT

template<int group_size, int numRegs>
void runPeakBenchFloatImpl(int sequencelength){
    const int readLength = sequencelength;
    const int hapLength = sequencelength;
    const int paddedReadLength = SDIV(readLength, 4) * 4;
    const int paddedHapLength = SDIV(hapLength, 4) * 4;
    const int numReadsInBatch = 32;
    const int numHapsInBatch = 32;

    const size_t readBytes = 256 * 1024 * 1024;
    const size_t hapBytes = 256 * 1024 * 1024;

    if(readLength > group_size * numRegs){
        std::cout << "read length " << readLength << " > " << group_size << " * " << numRegs << ". Skipping\n";
        return;
    }

    
    std::vector<char> readData(readBytes);
    std::vector<char> hapData(hapBytes);
    const char* letters = "ACGT";
    
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dist(0,3);
    for(size_t i = 0; i < readBytes; i++){
        readData[i] = letters[dist(gen)];
    }
    hapData = readData;
    
    const int numReadsInReadData = readBytes / paddedReadLength;
    const int numHapsInHapData = hapBytes / paddedHapLength;
    const int maxNumBatchesInGeneratedData = std::min(numReadsInReadData / numReadsInBatch, numHapsInHapData / numHapsInBatch);
    const int numBatches = maxNumBatchesInGeneratedData;
    const int totalNumReads = maxNumBatchesInGeneratedData * numReadsInBatch;
    const int totalNumHaps = maxNumBatchesInGeneratedData * numHapsInBatch;
    const int totalNumAlignments = numReadsInBatch * numHapsInBatch * numBatches;
    const size_t totalDPCells = size_t(readLength) * size_t(hapLength) * size_t(numReadsInBatch) * size_t(numHapsInBatch) * size_t(numBatches);

    // std::cout << "numReadsInReadData " << numReadsInReadData
    //     << ", numHapsInHapData " << numHapsInHapData
    //     << ", numBatches " << numBatches
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


    std::vector<int> numReadsPerBatch(numBatches, numReadsInBatch);
    std::vector<int> numHapsPerBatch(numBatches, numHapsInBatch);
    std::vector<int> numAlignmentsPerBatch(numBatches, numReadsInBatch * numHapsInBatch);
    std::vector<int> numReadsPerBatchPrefixSum(numBatches);
    std::vector<int> numHapsPerBatchPrefixSum(numBatches);
    std::vector<int> numAlignmentsPerBatchInclusivePrefixSum(numBatches);
    std::exclusive_scan(numReadsPerBatch.begin(), numReadsPerBatch.end(), numReadsPerBatchPrefixSum.begin(), int(0));
    std::exclusive_scan(numHapsPerBatch.begin(), numHapsPerBatch.end(), numHapsPerBatchPrefixSum.begin(), int(0));
    std::inclusive_scan(numAlignmentsPerBatch.begin(), numAlignmentsPerBatch.end(), numAlignmentsPerBatchInclusivePrefixSum.begin());


    thrust::device_vector<uint8_t> d_readData = readData;
    thrust::device_vector<uint8_t> d_hapData = hapData;
    thrust::device_vector<int> d_readLengths = readLengths;
    thrust::device_vector<int> d_hapLengths = hapLengths;
    thrust::device_vector<uint8_t> d_baseQuals = baseQuals;
    thrust::device_vector<uint8_t> d_insQuals = insQuals;
    thrust::device_vector<uint8_t> d_delQuals = delQuals;
    thrust::device_vector<int> d_readBeginOffsets = readBeginOffsets;
    thrust::device_vector<int> d_hapBeginOffsets = hapBeginOffsets;
    thrust::device_vector<int> d_numReadsPerBatch = numReadsPerBatch;
    thrust::device_vector<int> d_numHapsPerBatch = numHapsPerBatch;
    thrust::device_vector<int> d_numAlignmentsPerBatch = numAlignmentsPerBatch;
    thrust::device_vector<int> d_numReadsPerBatchPrefixSum = numReadsPerBatchPrefixSum;
    thrust::device_vector<int> d_numHapsPerBatchPrefixSum = numHapsPerBatchPrefixSum;
    thrust::device_vector<int> d_numAlignmentsPerBatchInclusivePrefixSum = numAlignmentsPerBatchInclusivePrefixSum;

    thrust::device_vector<int> d_numIndicesPerBatch = d_numReadsPerBatch;
    thrust::device_vector<int> d_indicesPerBatch(totalNumReads);
    thrust::transform(
        thrust::make_counting_iterator(0), 
        thrust::make_counting_iterator(totalNumReads),
        d_indicesPerBatch.begin(),
        cuda::proclaim_return_type<int>([numReadsInBatch] __device__ (int i){ return i % numReadsInBatch; })
    );

    thrust::device_vector<int> d_resultOffsetsPerBatch(totalNumAlignments);
    thrust::sequence(d_resultOffsetsPerBatch.begin(), d_resultOffsetsPerBatch.end(), 0);

    thrust::device_vector<float> d_results(totalNumAlignments,-42);

    convert_DNA<<<SDIV(d_readData.size(), 512), 512>>>(d_readData.data().get(), d_readData.size());
    convert_DNA<<<SDIV(d_hapData.size(), 512), 512>>>(d_hapData.data().get(), d_hapData.size());

    cudaStream_t stream = cudaStreamLegacy;
    std::string name = "PairHMM_align_partition_float_allowMultipleBatchesPerWarp_coalesced_smem " + std::to_string(group_size) + " " + std::to_string(numRegs);
    constexpr int groupsPerBlock = 32 / group_size;
    const int numBlocks = SDIV(totalNumAlignments, groupsPerBlock);
    helpers::GpuTimer timer(stream, name);
    PairHMM_align_partition_float_allowMultipleBatchesPerWarp_coalesced_smem<group_size,numRegs><<<numBlocks, 32, 0, stream>>>(
        d_readData.data().get(), 
        d_hapData.data().get(), 
        d_baseQuals.data().get(), 
        d_insQuals.data().get(),  
        d_delQuals.data().get(), 
        d_results.data().get(), 
        d_readBeginOffsets.data().get(), 
        d_hapBeginOffsets.data().get(), 
        d_readLengths.data().get(), 
        d_hapLengths.data().get(), 
        d_numReadsPerBatch.data().get(), 
        d_numHapsPerBatch.data().get(), 
        d_numHapsPerBatchPrefixSum.data().get(),
        d_numIndicesPerBatch.data().get(), 
        d_indicesPerBatch.data().get(),
        d_numReadsPerBatchPrefixSum.data().get(), 
        numBatches, 
        d_resultOffsetsPerBatch.data().get(), 
        d_numAlignmentsPerBatch.data().get(), 
        d_numAlignmentsPerBatchInclusivePrefixSum.data().get(), 
        totalNumAlignments
    );
    CUERR;
    timer.stop();
    timer.printGCUPS(totalDPCells);
    cudaDeviceSynchronize(); CUERR;
}




void runPeakBenchFloat(){
    std::cout << "runPeakBenchFloat\n";

    #define RUN(group_size, numRegs){ \
        const int sequencelength = group_size * numRegs; \
        runPeakBenchFloatImpl<group_size, numRegs>(sequencelength); \
    }


    RUN(4,4);
    RUN(4,8);
    RUN(4,12);
    RUN(4,16);
    RUN(4,20);
    RUN(4,24);
    RUN(4,28);
    RUN(4,32);

    RUN(8,4);
    RUN(8,8);
    RUN(8,12);
    RUN(8,16);
    RUN(8,20);
    RUN(8,24);
    RUN(8,28);
    RUN(8,32);

    RUN(16,4);
    RUN(16,8);
    RUN(16,12);
    RUN(16,16);
    RUN(16,20);
    RUN(16,24);
    RUN(16,28);
    RUN(16,32);

    RUN(32,4);
    RUN(32,8);
    RUN(32,12);
    RUN(32,16);
    RUN(32,20);
    RUN(32,24);
    RUN(32,28);
    RUN(32,32);

    #undef RUN
}
#endif


int main(const int argc, char const * const argv[])
{

    const int MAX_PH2PR_INDEX = 128;

    std::vector<float> ph2pr = generate_ph2pr(MAX_PH2PR_INDEX);

    Options options;

    for(int x = 1; x < argc; x++){
        std::string argstring = argv[x];
        if(argstring == "--inputfile"){
            options.inputfile = argv[x+1];
            x++;
        }
        if(argstring == "--outputfile"){
            options.outputfile = argv[x+1];
            x++;
        }        
        if(argstring == "--transferchunksize"){
            options.transferchunksize = std::atoi(argv[x+1]);
            x++;
        }
        if(argstring == "--checkResults"){
            options.checkResults = true;
        }
        if(argstring == "--peakBenchHalf"){
            options.peakBenchHalf = true;
        }
        if(argstring == "--peakBenchFloat"){
            options.peakBenchFloat = true;
        }
    }

    
    
    if(options.peakBenchHalf){
        #ifdef ENABLE_PEAK_BENCH_HALF
        runPeakBenchHalf();
        #else
        std::cout << "Need to define ENABLE_PEAK_BENCH_HALF to run runPeakBenchHalf\n";
        #endif
    }
    
    if(options.peakBenchFloat){
        #ifdef ENABLE_PEAK_BENCH_FLOAT
        runPeakBenchFloat();
        #else
        std::cout << "Need to define ENABLE_PEAK_BENCH_FLOAT to run peakBenchFloat\n";
        #endif
    }
    if(options.peakBenchFloat || options.peakBenchHalf){
        return 0;
    }
    
    std::cout << "options.inputfile = " << options.inputfile << "\n";
    // std::cout << "options.outputfile = " << options.outputfile << "\n";
    // std::cout << "options.transferchunksize = " << options.transferchunksize << "\n";
    std::cout << "options.checkResults = " << options.checkResults << "\n";
    // std::cout << "options.peakBenchHalf = " << options.peakBenchHalf << "\n";
    // std::cout << "options.peakBenchFloat = " << options.peakBenchFloat << "\n";
    
    if(options.inputfile == ""){
        throw std::runtime_error("Input file not specified");
    }

    helpers::CpuTimer timerParseInputFile("parse input file");
    batch fullBatch = parseInputFile(options.inputfile, ph2pr);
    timerParseInputFile.stop();
    timerParseInputFile.print();

    CountsOfDPCells countsOfDPCells = countDPCellsInBatch(fullBatch);



    const size_t read_bytes = fullBatch.reads.size();
    const size_t hap_bytes = fullBatch.haps.size();   
    const int numBatches = fullBatch.batch_reads.size();
    const int numReads = fullBatch.readlen.size();
    const int numHaps = fullBatch.haplen.size();


    std::cout << "Calculating:  " << fullBatch.getTotalNumberOfAlignments() << " alignments in " << numBatches << " batches \n";
    std::cout << "read_bytes:  " << read_bytes << ", hap_bytes  " << hap_bytes << " \n";
    std::cout << "num_reads:  " << numReads << ", num_haps  " << numHaps << " \n";

    {
        int minLength = *std::min_element(fullBatch.readlen.begin(), fullBatch.readlen.end());
        int maxLength = *std::max_element(fullBatch.readlen.begin(), fullBatch.readlen.end());
        size_t sumOfReadLengths = std::reduce(fullBatch.readlen.begin(), fullBatch.readlen.end(), size_t(0));
        int avgLength = sumOfReadLengths / fullBatch.readlen.size();
        std::cout << "minLength: " << minLength << ", maxLength: " << maxLength << ", avgLength: " << avgLength << "\n";
    }


    #if 0
    {
        
        std::cout << "batch_haps:\n";
        for(const auto& x : fullBatch.batch_haps) std::cout << x << " ";
        std::cout << "\n";
        std::cout << "batch_reads:\n";
        for(const auto& x : fullBatch.batch_reads) std::cout << x << " ";
        std::cout << "\n";
        std::cout << "batch_haps_offsets:\n";
        for(const auto& x : fullBatch.batch_haps_offsets) std::cout << x << " ";
        std::cout << "\n";
        std::cout << "batch_reads_offsets:\n";
        for(const auto& x : fullBatch.batch_reads_offsets) std::cout << x << " ";
        std::cout << "\n";
        std::cout << "readlen:\n";
        for(const auto& x : fullBatch.readlen) std::cout << x << " ";
        std::cout << "\n";
        std::cout << "haplen:\n";
        for(const auto& x : fullBatch.haplen) std::cout << x << " ";
        std::cout << "\n";
        std::cout << "read_offsets:\n";
        for(const auto& x : fullBatch.read_offsets) std::cout << x << " ";
        std::cout << "\n";
        std::cout << "hap_offsets:\n";
        for(const auto& x : fullBatch.hap_offsets) std::cout << x << " ";
        std::cout << "\n";
    }

    #endif


    //print_batch(fullBatch); // prints first reads/qualities and haplotype per batch.
    // align_all_host(fullBatch,ph2pr);

    const int deviceId = 0;
    cudaSetDevice(deviceId); CUERR;
    cudaMemcpyToSymbol(cPH2PR,ph2pr.data(),MAX_PH2PR_INDEX*sizeof(float));

    size_t releaseThreshold = UINT64_MAX;
    cudaMemPool_t memPool;
    cudaDeviceGetDefaultMemPool(&memPool, deviceId);
    cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &releaseThreshold); CUERR;



    std::vector<float> resultsBatchAsWhole_half = processBatchAsWhole_half(fullBatch, options, countsOfDPCells);    
    std::vector<float> resultsBatchOverlapped_half = processBatch_overlapped_half(fullBatch, options, countsOfDPCells);
    if(resultsBatchAsWhole_half != resultsBatchOverlapped_half){
        std::cout << "ERROR: resultsBatchAsWhole_half != resultsBatchOverlapped_half\n";
    }



    std::vector<float> resultsBatchAsWhole_half_coalesced_smem = processBatchAsWhole_half_coalesced_smem(fullBatch, options, countsOfDPCells);
    std::vector<float> resultsBatchOverlapped_half_coalesced_smem = processBatch_overlapped_half_coalesced_smem(fullBatch, options, countsOfDPCells);
    if(resultsBatchAsWhole_half_coalesced_smem != resultsBatchOverlapped_half_coalesced_smem){
        std::cout << "ERROR: resultsBatchAsWhole_half_coalesced_smem != resultsBatchOverlapped_half_coalesced_smem\n";
    }


    std::vector<float> resultsBatchAsWhole_float = processBatchAsWhole_float(fullBatch, options, countsOfDPCells);
    std::vector<float> resultsBatchOverlapped_float = processBatch_overlapped_float(fullBatch, options, countsOfDPCells);
    if(resultsBatchAsWhole_float != resultsBatchOverlapped_float){
        std::cout << "ERROR: resultsBatchAsWhole_float != resultsBatchOverlapped_float\n";
    }

    // for(auto x : resultsBatchAsWhole_float){
    //     std::cout << x << "\n";
    // }

    std::vector<float> resultsBatchAsWhole_float_coalesced_smem = processBatchAsWhole_float_coalesced_smem(fullBatch, options, countsOfDPCells);
    std::vector<float> resultsBatchOverlapped_float_coalesced_smem = processBatch_overlapped_float_coalesced_smem(fullBatch, options, countsOfDPCells);
    if(resultsBatchAsWhole_float_coalesced_smem != resultsBatchOverlapped_float_coalesced_smem){
        std::cout << "ERROR: resultsBatchAsWhole_float_coalesced_smem != resultsBatchOverlapped_float_coalesced_smem\n";
    }
    if(resultsBatchAsWhole_float != resultsBatchAsWhole_float_coalesced_smem){
        std::cout << "ERROR: resultsBatchAsWhole_float != resultsBatchAsWhole_float_coalesced_smem\n";
    }

    // int numErrors = 0;
    // for(int i = 0; i < int(resultsBatchAsWhole_half.size()); i++){
    //     if(resultsBatchAsWhole_half[i] != resultsBatchOverlapped_half[i]){
    //         std::cout << "error i " << i << " : " << resultsBatchAsWhole_half[i] << " " <<  resultsBatchOverlapped_half[i] << "\n";
    //         numErrors++;
    //         if(numErrors > 10) break;
    //     }
    // }
#if 1
    if(options.checkResults){
        const int numBatches = fullBatch.batch_haps.size();
        const int totalNumberOfAlignments = fullBatch.getTotalNumberOfAlignments();
        const auto& numAlignmentsPerBatchInclusivePrefixSum = fullBatch.getNumberOfAlignmentsPerBatchInclusivePrefixSum();

        // std::vector<float> resultsCPU2 = processBatchCPU(fullBatch, ph2pr);
        std::vector<float> resultsCPU = processBatchCPUFaster(fullBatch, ph2pr);
        // assert(resultsCPU == resultsCPU2);



        std::cout << "comparing half:\n";
        computeAbsoluteErrorStatistics(resultsCPU, resultsBatchOverlapped_half);
        computeRelativeErrorStatistics(resultsCPU, resultsBatchOverlapped_half);

        std::cout << "comparing float:\n";
        computeAbsoluteErrorStatistics(resultsCPU, resultsBatchOverlapped_float);
        computeRelativeErrorStatistics(resultsCPU, resultsBatchOverlapped_float);

        std::cout << "comparing half coalesced smem:\n";
        computeAbsoluteErrorStatistics(resultsCPU, resultsBatchOverlapped_half_coalesced_smem);
        computeRelativeErrorStatistics(resultsCPU, resultsBatchOverlapped_half_coalesced_smem);

        std::cout << "comparing float coalesced smem:\n";
        computeAbsoluteErrorStatistics(resultsCPU, resultsBatchOverlapped_float_coalesced_smem);
        computeRelativeErrorStatistics(resultsCPU, resultsBatchOverlapped_float_coalesced_smem);

        {
            constexpr double checklimit = 0.05;

            for(int i = 0, numErrors = 0; i < int(resultsBatchOverlapped_half.size()); i++){
                const float absError = std::abs(resultsCPU[i] - resultsBatchOverlapped_half[i]);
                if(absError > checklimit){
                    if(numErrors == 0){
                        std::cout << "some half error inputs:\n";
                    }
                    if(numErrors < 5){
                        std::cout << "i " << i << " : " << resultsCPU[i] << " "  <<  resultsBatchOverlapped_half[i] << ", abs error " << absError << "\n";
                    }
                    numErrors++;
                }
            }

            for(int i = 0, numErrors = 0; i < int(resultsBatchOverlapped_float.size()); i++){
                const float absError = std::abs(resultsCPU[i] - resultsBatchOverlapped_float[i]);
                if(absError > checklimit){
                    if(numErrors == 0){
                        std::cout << "some float error inputs:\n";
                    }
                    if(numErrors < 5){
                        std::cout << "i " << i << " : " << resultsCPU[i] << " "  <<  resultsBatchOverlapped_half[i] << ", abs error " << absError << "\n";

                        const int batchIdByGroupId = std::distance(
                            numAlignmentsPerBatchInclusivePrefixSum.begin(),
                            std::upper_bound(
                                numAlignmentsPerBatchInclusivePrefixSum.begin(),
                                numAlignmentsPerBatchInclusivePrefixSum.begin() + numBatches,
                                i
                            )
                        );
                        const int batchId = min(batchIdByGroupId, numBatches-1);
                        const int numHapsInBatch = fullBatch.batch_haps[batchId];
                        const int alignmentOffset = (batchId == 0 ? 0 : numAlignmentsPerBatchInclusivePrefixSum[batchId-1]);
                        const int alignmentIdInBatch = i - alignmentOffset;
                        const int hapToProcessInBatch = alignmentIdInBatch % numHapsInBatch;
                        const int readToProcessInBatch = alignmentIdInBatch / numHapsInBatch;

                        int read = fullBatch.batch_reads_offsets[batchId]+readToProcessInBatch;
                        int read_len = fullBatch.readlen[read];
                        int hap = fullBatch.batch_haps_offsets[batchId]+hapToProcessInBatch;
                        int hap_len = fullBatch.haplen[hap];
                        int h_off = fullBatch.hap_offsets[hap];
                        int r_off = fullBatch.read_offsets[read];
                    
                        std::cout << "batchId " << batchId << ", read nr" << readToProcessInBatch << ", hap nr" << hapToProcessInBatch << "\n";
                        for(int k = 0; k < read_len; k++){
                            std::cout << fullBatch.reads[r_off + k];
                        }
                        std::cout << "\n";
                        for(int k = 0; k < hap_len; k++){
                            std::cout << fullBatch.haps[h_off + k];
                        }
                        std::cout << "\n";
                    }
                    numErrors++;
                }
            }
        }
    }
#endif
    // int res_off = 0;
    // for (int i=0; i<1; i++) { // for (int i=0; i<num_batches; i++) {
    //     cout << "Batch:" << i << " Offset: " << res_off << " results: ";
    //     for(int j = 0; j < 8; j++) // for(int j = 0; j < read_batches[i]; j++)
    //         for(int k = 0; k < fullBatch.batch_haps[i]; k++)
    //             cout << " " << resultsBatchAsWhole_half[res_off+j*fullBatch.batch_haps[i]+k];
    //     cout << " \n";
    //     res_off += fullBatch.batch_reads[i] * fullBatch.batch_haps[i];
    // }

}
