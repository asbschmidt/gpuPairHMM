#ifndef UTILITY_KERNELS_CUH
#define UTILITY_KERNELS_CUH

#include <cuda/std/cstdint>

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


template<class TileSizesArray>
__global__
void partitionIndicesKernel(
    int* numIndicesPerPartitionPerBatch,
    int* indicesPerPartitionPerBatch,
    const int* read_lengths,
    const int* numReadsPerBatch,
    const int* numReadsPerBatchPrefixSum,
    int numBatches,
    int numReads,
    const TileSizesArray tileSizesArray
){

    for(int batchId = blockIdx.x; batchId < numBatches; batchId += gridDim.x){
        const int offset = numReadsPerBatchPrefixSum[batchId];
        const int outputOffset = offset - numReadsPerBatchPrefixSum[0];
        const int numReadsInBatch = numReadsPerBatch[batchId];
        const int* readLengthsOfBatch = read_lengths + offset;

        for(int r = threadIdx.x; r < numReadsInBatch; r += blockDim.x){
            const int length = readLengthsOfBatch[r];
            int partitionId = tileSizesArray.size();

            for(int p = 0; p < int(tileSizesArray.size()); p++){
                if(length <= tileSizesArray[p]){
                    partitionId = p;
                    break;
                }
            }

            const int pos = atomicAdd(&numIndicesPerPartitionPerBatch[partitionId * numBatches + batchId], 1);
            indicesPerPartitionPerBatch[partitionId * numReads + outputOffset + pos] = r;
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




#endif