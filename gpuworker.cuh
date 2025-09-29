#include "execution_pipeline.cuh"
#include "common.cuh"
#include "cuda_helpers.cuh"
#include "timers.cuh"
#include "cuda_errorcheck.cuh"

#include "utility_kernels.cuh"
#include "pairhmm_kernels.cuh"

#include <algorithm>
#include <future>
#include <numeric>
#include <string>
#include <vector>
#include <fstream>

#ifdef ENABLE_NVTX3
#include <nvtx3/nvtx3.hpp>
#endif

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/equal.h>
#include <thrust/extrema.h>

#include <cub/cub.cuh>

#include <cuda/std/array>

//#define DEBUGSYNC CUDACHECK(cudaDeviceSynchronize());
#define DEBUGSYNC

__inline__
void print_batch(const BatchOfSequenceGroups& batch_){

    int numSequenceGroupsInSequenceGroup = batch_.numReadsPerSequenceGroup.size();
    int batchreadoffset = 0;
    int batchhapoffset = 0;

    std::cout << numSequenceGroupsInSequenceGroup << std::endl;

    for (int i=0; i<numSequenceGroupsInSequenceGroup; i++){
        std::cout << "group " <<i << "\n";
        batchreadoffset = batch_.numReadsPerSequenceGroupPrefixSum[i];
        batchhapoffset = batch_.numHapsPerSequenceGroupPrefixSum[i];

        const int numReadsInGroup = batch_.numReadsPerSequenceGroup[i];
        const int numHapsInGroup = batch_.numHapsPerSequenceGroup[i];

        for(int r = 0; r < numReadsInGroup; r++){
            std::cout << batch_.readBeginOffsets[batchreadoffset + r] << " " << batch_.readBeginOffsets[batchreadoffset + r+1] << std::endl;
            std::cout << "read: ";
            for (int j=batch_.readBeginOffsets[batchreadoffset + r]; j < batch_.readBeginOffsets[batchreadoffset + r+1]; j++ )
                std::cout << (char)(batch_.readData[j]);
            std::cout << std::endl;

            std::cout << "base_qual: ";
            for (int j=batch_.readBeginOffsets[batchreadoffset + r]; j < batch_.readBeginOffsets[batchreadoffset + r+1]; j++ )
                std::cout << (char)(batch_.base_quals[j]+33);
            std::cout << std::endl;

            std::cout << "ins_qual: ";
            for (int j=batch_.readBeginOffsets[batchreadoffset + r]; j < batch_.readBeginOffsets[batchreadoffset + r+1]; j++ )
                std::cout << (char)(batch_.ins_quals[j]+33);
            std::cout << std::endl;

            std::cout << "del_qual: ";
            for (int j=batch_.readBeginOffsets[batchreadoffset + r]; j < batch_.readBeginOffsets[batchreadoffset + r+1]; j++ )
                std::cout << (char)(batch_.del_quals[j]+33);
            std::cout << std::endl;

            std::cout << "gcp_qual: ";
            for (int j=batch_.readBeginOffsets[batchreadoffset + r]; j < batch_.readBeginOffsets[batchreadoffset + r+1]; j++ )
                std::cout << (char)(batch_.gcp_quals[j]+33);
            std::cout << std::endl;
        }

        for(int h = 0; h < numHapsInGroup; h++){
            std::cout << batch_.hapBeginOffsets[batchhapoffset + h] << " " << batch_.hapBeginOffsets[batchhapoffset + h+1] << std::endl;
            std::cout << "hap: ";
            for (int j=batch_.hapBeginOffsets[batchhapoffset + h]; j < batch_.hapBeginOffsets[batchhapoffset + h+1]; j++ )
                std::cout << (char)(batch_.hapData[j]);
            std::cout << std::endl;
        }
    }

}



template<class ConfigTuple>
struct ArrayOfTileSizes;

template<class... Configs>
struct ArrayOfTileSizes<std::tuple<Configs...>>{
    static constexpr std::array<int, sizeof...(Configs)> array{Configs::tileSize ...};
};



struct GpuPairHMMI{
public:
    virtual void launchPairHMMSingleTileKernel(
        float* d_resultoutput,
        const uint8_t* d_readData,
        const uint8_t* d_hapData,
        const uint8_t* d_base_quals,
        const uint8_t* d_ins_quals,
        const uint8_t* d_del_quals,
        const int* d_readBeginOffsets,
        const int* d_hapBeginOffsets,
        const int* d_readLengths,
        const int* d_hapLengths,
        const int* d_numHapsPerSequenceGroup,
        const int* d_numHapsPerSequenceGroupPrefixSum,
        const int* d_indicesPerSequenceGroup,
        const int* d_numReadsPerSequenceGroupPrefixSum,
        const int numSequenceGroups,
        const int* d_resultOffsetsPerBatch,
        const int* d_numAlignmentsPerBatch,
        const int* d_numAlignmentsPerBatchInclusivePrefixSum,
        const int numAlignments,
        cudaStream_t stream
    ) = 0;
};

template<int blocksize, int group_size, int numRegs>
struct GpuPairHMM : public GpuPairHMMI{
    static_assert(blocksize == 32);
    static_assert(numRegs % 4 == 0);

    void launchPairHMMSingleTileKernel(
        float* d_resultoutput,
        const uint8_t* d_readData,
        const uint8_t* d_hapData,
        const uint8_t* d_base_quals,
        const uint8_t* d_ins_quals,
        const uint8_t* d_del_quals,
        const int* d_readBeginOffsets,
        const int* d_hapBeginOffsets,
        const int* d_readLengths,
        const int* d_hapLengths,
        const int* d_numHapsPerSequenceGroup,
        const int* d_numHapsPerSequenceGroupPrefixSum,
        const int* d_indicesPerSequenceGroup,
        const int* d_numReadsPerSequenceGroupPrefixSum,
        const int numSequenceGroups,
        const int* d_resultOffsetsPerBatch,
        const int* d_numAlignmentsPerBatch,
        const int* d_numAlignmentsPerBatchInclusivePrefixSum,
        const int numAlignments,
        cudaStream_t stream
    ) override {
        constexpr int groupsPerBlock = blocksize / group_size;
        const int numBlocks = SDIV(numAlignments, groupsPerBlock);

        PairHMM_float_kernel<group_size,numRegs><<<numBlocks, blocksize,0,stream>>>( \
            d_resultoutput, 
            d_readData,
            d_hapData,
            d_base_quals,
            d_ins_quals,
            d_del_quals,
            d_readBeginOffsets,
            d_hapBeginOffsets,
            d_readLengths,
            d_hapLengths,
            d_numHapsPerSequenceGroup,
            d_numHapsPerSequenceGroupPrefixSum,
            d_indicesPerSequenceGroup,
            d_numReadsPerSequenceGroupPrefixSum,
            numSequenceGroups,
            d_resultOffsetsPerBatch,
            d_numAlignmentsPerBatch,
            d_numAlignmentsPerBatchInclusivePrefixSum,
            numAlignments
        );
        CUDACHECKASYNC;
    }
};

template<class KernelConfigsSingleTile>
struct MultiConfigGpuPairHMM {
    std::vector<std::unique_ptr<GpuPairHMMI>> singleTileAligners;

    int deviceId = 0;
    int numSMs = 0;

    static constexpr int multiTileGroupSize = 32;
    static constexpr int multiTileNumRegs = 16;
    static constexpr int multiTileBlockSize = 32;
    static_assert(multiTileBlockSize == 32);

    static constexpr auto multiTileKernel = PairHMM_float_multitile_kernel<multiTileGroupSize, multiTileNumRegs>;

    MultiConfigGpuPairHMM(){

        CUDACHECK(cudaGetDevice(&deviceId));
        CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));

        auto addAligner = [&](auto config){
            using Config = decltype(config);
            singleTileAligners.emplace_back(std::make_unique<GpuPairHMM<Config::blocksize, Config::groupsize, Config::numItems>>());
        };

        std::apply(
            [&](auto ... x){
                (addAligner(x), ...);
            },
            typename KernelConfigsSingleTile::type{}
        );
    }

    static constexpr auto getArrayOfSingleTileTileSizes(){
        constexpr auto array = ArrayOfTileSizes<typename KernelConfigsSingleTile::type>::array;
        static_assert(array.size() > 0);
        return array;
    }

    void launchPairHMMSingleTileKernel(
        int maximumReadLength,
        float* d_resultoutput,
        const uint8_t* d_readData,
        const uint8_t* d_hapData,
        const uint8_t* d_base_quals,
        const uint8_t* d_ins_quals,
        const uint8_t* d_del_quals,
        const int* d_readBeginOffsets,
        const int* d_hapBeginOffsets,
        const int* d_readLengths,
        const int* d_hapLengths,
        const int* d_numHapsPerSequenceGroup,
        const int* d_numHapsPerSequenceGroupPrefixSum,
        const int* d_indicesPerSequenceGroup,
        const int* d_numReadsPerSequenceGroupPrefixSum,
        const int numSequenceGroups,
        const int* d_resultOffsetsPerBatch,
        const int* d_numAlignmentsPerBatch,
        const int* d_numAlignmentsPerBatchInclusivePrefixSum,
        const int numAlignments,
        cudaStream_t stream
    ){
        constexpr auto tilesizes = getArrayOfSingleTileTileSizes();
        if(maximumReadLength > tilesizes.back()){
            throw std::runtime_error("maximumReadLength too large for single tile kernel");
        }
        for(int i = 0; i < int(tilesizes.size()); i++){
            if(maximumReadLength <= tilesizes[i]){
                singleTileAligners[i]->launchPairHMMSingleTileKernel(
                    d_resultoutput, 
                    d_readData,
                    d_hapData,
                    d_base_quals,
                    d_ins_quals,
                    d_del_quals,
                    d_readBeginOffsets,
                    d_hapBeginOffsets,
                    d_readLengths,
                    d_hapLengths,
                    d_numHapsPerSequenceGroup,
                    d_numHapsPerSequenceGroupPrefixSum,
                    d_indicesPerSequenceGroup,
                    d_numReadsPerSequenceGroupPrefixSum,
                    numSequenceGroups,
                    d_resultOffsetsPerBatch,
                    d_numAlignmentsPerBatch,
                    d_numAlignmentsPerBatchInclusivePrefixSum,
                    numAlignments,
                    stream
                );
                break;
            }
        }
    }

    size_t getTileTempBytesPerGroup(int maximumHaplotypeLength) const{
        return sizeof(float) * 3 * (maximumHaplotypeLength + 1 + multiTileGroupSize);
    }

    size_t getMinimumSuggestedTempBytesForMultiTile(int maximumHaplotypeLength) const{
        constexpr int groupsPerBlock = (multiTileBlockSize / multiTileGroupSize);
        const size_t tileTempBytesPerGroup = getTileTempBytesPerGroup(maximumHaplotypeLength);        
        size_t tempBytes1BlockPerSM = tileTempBytesPerGroup * groupsPerBlock * numSMs;

        return tempBytes1BlockPerSM;
    }

    size_t getSuggestedTempBytesForMultiTile(int maximumHaplotypeLength) const{    
        const size_t tempBytes1BlockPerSM = getMinimumSuggestedTempBytesForMultiTile(maximumHaplotypeLength);

        int smem = 0;
        int maxBlocksPerSM = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSM,
            multiTileKernel,
            multiTileBlockSize, 
            smem
        );

        return tempBytes1BlockPerSM * maxBlocksPerSM;
    }

    void launchPairHMMMultiTileKernel(
        char* d_temp,
        size_t tempBytes,
        int maximumHaplotypeLength,
        float* d_resultoutput,
        const uint8_t* d_readData,
        const uint8_t* d_hapData,
        const uint8_t* d_base_quals,
        const uint8_t* d_ins_quals,
        const uint8_t* d_del_quals,
        const int* d_readBeginOffsets,
        const int* d_hapBeginOffsets,
        const int* d_readLengths,
        const int* d_hapLengths,
        const int* d_numHapsPerSequenceGroup,
        const int* d_numHapsPerSequenceGroupPrefixSum,
        const int* d_indicesPerSequenceGroup,
        const int* d_numReadsPerSequenceGroupPrefixSum,
        const int numSequenceGroups,
        const int* d_resultOffsetsPerBatch,
        const int* d_numAlignmentsPerBatch,
        const int* d_numAlignmentsPerBatchInclusivePrefixSum,
        const int numAlignments,
        cudaStream_t stream
    ){
        if(d_temp == nullptr || tempBytes == 0) throw std::runtime_error("tempstorage is 0");

        int smem = 0;
        int maxBlocksPerSM = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSM,
            multiTileKernel,
            multiTileBlockSize, 
            smem
        );
        const size_t tileTempBytesPerGroup = getTileTempBytesPerGroup(maximumHaplotypeLength);

        constexpr int groupsPerBlock = (multiTileBlockSize / multiTileGroupSize);
        constexpr int alignmentsPerBlock = groupsPerBlock;
        const int maxNumBlocksByInputSize = (numAlignments + alignmentsPerBlock - 1) / alignmentsPerBlock;
        const int maxNumBlocksByOccupancy = maxBlocksPerSM * numSMs;
        const int maxNumBlocksByTempBytes = tempBytes / (tileTempBytesPerGroup * groupsPerBlock);

        const int numBlocks = std::min(maxNumBlocksByTempBytes, std::min(maxNumBlocksByInputSize, maxNumBlocksByOccupancy));
        if(numBlocks <= 0){
            throw std::runtime_error("could not launch kernel. numBlocks <= 0");
        }

        multiTileKernel<<<numBlocks, multiTileBlockSize, 0, stream>>>(
            d_resultoutput,
            d_readData,
            d_hapData,
            d_base_quals,
            d_ins_quals,
            d_del_quals,
            d_readBeginOffsets,
            d_hapBeginOffsets,
            d_readLengths,
            d_hapLengths,
            d_numHapsPerSequenceGroup,
            d_numHapsPerSequenceGroupPrefixSum,
            d_indicesPerSequenceGroup,
            d_numReadsPerSequenceGroupPrefixSum,
            numSequenceGroups,
            d_resultOffsetsPerBatch,
            d_numAlignmentsPerBatch,
            d_numAlignmentsPerBatchInclusivePrefixSum,
            numAlignments,
            d_temp,
            tileTempBytesPerGroup,
            maximumHaplotypeLength
        );
    }

};



template<class MultiConfigAligner>
struct GpuPairHMMWorker{
    const Options* optionsPtr;
    PipelineDataQueue* pipelineDataQueueIn;
    PipelineDataQueue* pipelineDataQueueOut;

    std::vector<float> ph2pr;

    MultiConfigAligner multiConfigGpuPairHMM;
    std::vector<cudaStream_t> computeStreams;
    std::vector<cudaStream_t> transferStreams;

    float totalSecondsAllBatches = 0;
    std::int64_t totalDPCellsAllBatches = 0;

    struct CountsOfDPCells{
        std::int64_t totalDPCells = 0;
        std::vector<std::int64_t> dpCellsPerPartition{};
    };
    
    GpuPairHMMWorker(const Options* optionsPtr_, PipelineDataQueue* pipelineDataQueueIn_, PipelineDataQueue* pipelineDataQueueOut_)
        : optionsPtr(optionsPtr_), pipelineDataQueueIn(pipelineDataQueueIn_), pipelineDataQueueOut(pipelineDataQueueOut_)
    {
        const int MAX_PH2PR_INDEX = 128;
        ph2pr = generate_ph2pr(MAX_PH2PR_INDEX);
        CUDACHECK(cudaMemcpyToSymbol(cPH2PR,ph2pr.data(),MAX_PH2PR_INDEX*sizeof(float)));
        
        computeStreams.resize(8);
        for(auto& stream : computeStreams){
            CUDACHECK(cudaStreamCreate(&stream));
        }
        transferStreams.resize(2);
        for(auto& stream : transferStreams){
            CUDACHECK(cudaStreamCreate(&stream));
        }
    }

    ~GpuPairHMMWorker(){
        for(auto& stream : computeStreams){
            cudaStreamDestroy(stream);
        }
        for(auto& stream : transferStreams){
            cudaStreamDestroy(stream);
        }

        double gcups = totalDPCellsAllBatches / 1000. / 1000. / 1000. / totalSecondsAllBatches;
        std::cout << "Batch processing time: " << totalSecondsAllBatches << "s (" << gcups << " GCUPS)\n";
    }

    GpuPairHMMWorker(const GpuPairHMMWorker&) = delete;
    GpuPairHMMWorker(GpuPairHMMWorker&&) = delete;
    GpuPairHMMWorker& operator=(const GpuPairHMMWorker&) = delete;
    GpuPairHMMWorker& operator=(GpuPairHMMWorker&&) = delete;

    void run(){
        #ifdef ENABLE_NVTX3
        nvtx3::scoped_range sr("GpuPairHMMWorker::run");
        #endif
        helpers::CpuTimer cputimer("GpuPairHMMWorker::run");
        const auto& options = *optionsPtr;

        PipelineData* pipelineDataPtr = pipelineDataQueueIn->pop();

        while(pipelineDataPtr != nullptr){
            //auto& pipelineData = *pipelineDataPtr;
            //auto& batch = pipelineDataPtr->batch;

            processBatch_overlapped_float_coalesced_smem(pipelineDataPtr);

            // auto cpuresults = processBatchCPU(pipelineDataPtr->batch, ph2pr);
            // std::cout << "cpu\n";
            // for(auto x : cpuresults){
            //     printf("%.20f\n", x);
            // }
            // std::cout << "gpu\n";
            // for(auto x : pipelineDataPtr->h_scores){
            //     printf("%.20f\n", x);
            // }

            pipelineDataQueueOut->push(pipelineDataPtr);
            pipelineDataPtr = pipelineDataQueueIn->pop();
        }
        //notify writer
        pipelineDataQueueOut->push(nullptr);

        cputimer.stop();
    
        if(options.verbose){
            cputimer.print();
        }
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

    CountsOfDPCells countDPCellsInSequenceGroups(const BatchOfSequenceGroups& batch){
        CountsOfDPCells result;
        result.dpCellsPerPartition = computeNumberOfDPCellsPerPartition(batch);
        result.totalDPCells = std::reduce(result.dpCellsPerPartition.begin(), result.dpCellsPerPartition.end(), std::int64_t(0));
        return result;
    }

    std::int64_t computeNumberOfDPCells(const BatchOfSequenceGroups& batch){
        #ifdef ENABLE_NVTX3
        nvtx3::scoped_range sr1("computeNumberOfDPCells");
        #endif
        std::int64_t result = 0;

        const int numSequenceGroups = batch.numHapsPerSequenceGroup.size();
    
        for (int i=0; i<numSequenceGroups; i++) {
            const int numReadsInSequenceGroup = batch.numReadsPerSequenceGroup[i];
            const int numHapsInSequenceGroup = batch.numHapsPerSequenceGroup[i];
            std::int64_t sumOfReadLengths = 0;
            for (int k=0; k < numReadsInSequenceGroup; k++){
                const int read = batch.numReadsPerSequenceGroupPrefixSum[i]+k;
                sumOfReadLengths += batch.readLengths[read];
            }
            std::int64_t sumOfHapLengths = 0;
            for (int k=0; k < numHapsInSequenceGroup; k++){
                const int hap = batch.numHapsPerSequenceGroupPrefixSum[i]+k;
                sumOfHapLengths += batch.hapLengths[hap];
            }
            result += sumOfReadLengths * sumOfHapLengths;
        }

        return result;
    }

    std::vector<std::int64_t> computeNumberOfDPCellsPerPartition(const BatchOfSequenceGroups& batch){
        auto singleTileTilesizes = MultiConfigAligner::getArrayOfSingleTileTileSizes();

        //single tiles + 1 multitile
        std::vector<std::int64_t> dpCellsPerPartition(singleTileTilesizes.size() + 1, 0);
    
        auto getPartitionId = [&](int length){
            int id = singleTileTilesizes.size();
            for(int i = 0; i < int(singleTileTilesizes.size()); i++){
                if(length <= singleTileTilesizes[i]){
                    id = i;
                    break;
                }
            }
            return id;
        };
    
        const int numSequenceGroups = batch.numHapsPerSequenceGroup.size();
    
        for (int i=0; i<numSequenceGroups; i++) {
            const int numReadsInSequenceGroup = batch.numReadsPerSequenceGroup[i];
            const int numHapsInSequenceGroup = batch.numHapsPerSequenceGroup[i];
            for (int k=0; k < numReadsInSequenceGroup; k++){
                int read = batch.numReadsPerSequenceGroupPrefixSum[i]+k;
                int readLengths = batch.readLengths[read];
                const int p = getPartitionId(readLengths);
                for (int j=0; j < numHapsInSequenceGroup; j++){
                    int hap = batch.numHapsPerSequenceGroupPrefixSum[i]+j;
                    int hapLengths = batch.hapLengths[hap];
                    dpCellsPerPartition[p] += readLengths * hapLengths;
                }
            }
        }
    
        return dpCellsPerPartition;
    }

    void processBatch_overlapped_float_coalesced_smem(
        PipelineData* pipelineDataPtr
    ){
        helpers::CpuTimer totalTimer("processBatch_overlapped_float_coalesced_smem");

        const auto& options = *optionsPtr;
        auto& fullBatch = pipelineDataPtr->batch;
        // PinnedBatchOfSequenceGroups fullBatch(fullBatch_default);

        // print_batch(fullBatch);
    
        const uint8_t* readData = fullBatch.readData.data();
        const uint readDataBytes = fullBatch.readData.size();
        const uint8_t* hapData = fullBatch.hapData.data();
        const uint hapDataBytes = fullBatch.hapData.size();
        const uint8_t* base_qual = fullBatch.base_quals.data();
        const uint8_t* ins_qual = fullBatch.ins_quals.data();
        const uint8_t* del_qual = fullBatch.del_quals.data();
        const int* readBeginOffsets = fullBatch.readBeginOffsets.data();
        const int* hapBeginOffsets = fullBatch.hapBeginOffsets.data();
        const int* readLengths = fullBatch.readLengths.data();
        const int* hapLengths = fullBatch.hapLengths.data();
        const int numReadsInSequenceGroup = fullBatch.readLengths.size();
        const int numHapsInSequenceGroup = fullBatch.hapLengths.size();
        const int numSequenceGroupsInSequenceGroup = fullBatch.numReadsPerSequenceGroup.size();
        const int* numHapsPerSequenceGroup = fullBatch.numHapsPerSequenceGroup.data();
        const int* numReadsPerSequenceGroup = fullBatch.numReadsPerSequenceGroup.data();
        const int* numHapsPerSequenceGroupPrefixSum = fullBatch.numHapsPerSequenceGroupPrefixSum.data();
        const int* numReadsPerSequenceGroupPrefixSum = fullBatch.numReadsPerSequenceGroupPrefixSum.data();
        if(numHapsInSequenceGroup == 0 || numReadsInSequenceGroup == 0){
            return;
        }

        DEBUGSYNC
    
        const int totalNumberOfAlignments = fullBatch.getTotalNumberOfAlignments();     
     
        constexpr auto singleTileTilesizes = MultiConfigAligner::getArrayOfSingleTileTileSizes();
        constexpr int numSingleTileTileSizes = singleTileTilesizes.size();
        constexpr int numMultiTileTileSizes = 1;
        constexpr int numLengthPartitions = (numSingleTileTileSizes + numMultiTileTileSizes);

        const int maximumHaplotypeLength = *std::max_element(hapLengths, hapLengths + numHapsInSequenceGroup);
    

        ThrustCudaMallocAsyncAllocator<int> defaultAllocInt(cudaStreamPerThread);
        ThrustCudaMallocAsyncAllocator<float> defaultAllocFloat(cudaStreamPerThread);
        ThrustCudaMallocAsyncAllocator<uint8_t> defaultAllocUint8(cudaStreamPerThread);

        thrust::device_vector<float, ThrustCudaMallocAsyncAllocator<float>> devAlignmentScoresFloat_vec(totalNumberOfAlignments, 0, defaultAllocFloat);
        

        thrust::device_vector<uint8_t, ThrustCudaMallocAsyncAllocator<uint8_t>> d_readData_vec(readDataBytes, defaultAllocUint8);
        thrust::device_vector<uint8_t, ThrustCudaMallocAsyncAllocator<uint8_t>> d_hapData_vec(hapDataBytes, defaultAllocUint8);
        thrust::device_vector<uint8_t, ThrustCudaMallocAsyncAllocator<uint8_t>> d_base_qual_vec(readDataBytes, defaultAllocUint8);
        thrust::device_vector<uint8_t, ThrustCudaMallocAsyncAllocator<uint8_t>> d_ins_qual_vec(readDataBytes, defaultAllocUint8);
        thrust::device_vector<uint8_t, ThrustCudaMallocAsyncAllocator<uint8_t>> d_del_qual_vec(readDataBytes, defaultAllocUint8);    
        thrust::device_vector<int, ThrustCudaMallocAsyncAllocator<int>> d_readBeginOffsets_vec(numReadsInSequenceGroup, defaultAllocInt);
        thrust::device_vector<int, ThrustCudaMallocAsyncAllocator<int>> d_hapBeginOffsets_vec(numHapsInSequenceGroup, defaultAllocInt);
        thrust::device_vector<int, ThrustCudaMallocAsyncAllocator<int>> d_readLengths_vec(numReadsInSequenceGroup, defaultAllocInt);
        thrust::device_vector<int, ThrustCudaMallocAsyncAllocator<int>> d_hapLengths_vec(numHapsInSequenceGroup, defaultAllocInt);
        thrust::device_vector<int, ThrustCudaMallocAsyncAllocator<int>> d_numReadsPerSequenceGroup_vec(numSequenceGroupsInSequenceGroup, defaultAllocInt);
        thrust::device_vector<int, ThrustCudaMallocAsyncAllocator<int>> d_numHapsPerSequenceGroup_vec(numSequenceGroupsInSequenceGroup, defaultAllocInt);
        thrust::device_vector<int, ThrustCudaMallocAsyncAllocator<int>> d_numReadsPerSequenceGroupPrefixSum_vec(numSequenceGroupsInSequenceGroup, defaultAllocInt);
        thrust::device_vector<int, ThrustCudaMallocAsyncAllocator<int>> d_numHapsPerSequenceGroupPrefixSum_vec(numSequenceGroupsInSequenceGroup, defaultAllocInt);    
        thrust::device_vector<int, ThrustCudaMallocAsyncAllocator<int>> d_numIndicesPerPartitionPerBatch(numSequenceGroupsInSequenceGroup * numLengthPartitions, 0, defaultAllocInt);
        thrust::device_vector<int, ThrustCudaMallocAsyncAllocator<int>> d_indicesPerPartitionPerBatch(numReadsInSequenceGroup * numLengthPartitions, -1, defaultAllocInt);
        thrust::device_vector<int, ThrustCudaMallocAsyncAllocator<int>> d_resultOffsetsPerBatch(numSequenceGroupsInSequenceGroup, defaultAllocInt);
        thrust::device_vector<int, ThrustCudaMallocAsyncAllocator<int>> d_numAlignmentsPerBatch(numSequenceGroupsInSequenceGroup * numLengthPartitions, defaultAllocInt);
        thrust::device_vector<int, ThrustCudaMallocAsyncAllocator<int>> d_numAlignmentsPerBatchInclPrefixSum(numSequenceGroupsInSequenceGroup * numLengthPartitions, defaultAllocInt);
        thrust::device_vector<int, ThrustCudaMallocAsyncAllocator<int>> d_numAlignmentsPerPartition(numLengthPartitions, defaultAllocInt);
        std::vector<int, PinnedAllocator<int>> h_numAlignmentsPerPartition(numLengthPartitions);
        //wait for allocations
        CUDACHECK(cudaStreamSynchronize(cudaStreamPerThread));
    
        uint8_t* const d_readData = d_readData_vec.data().get();
        uint8_t* const d_hapData = d_hapData_vec.data().get();
        uint8_t* const d_base_qual = d_base_qual_vec.data().get();
        uint8_t* const d_ins_qual = d_ins_qual_vec.data().get();
        uint8_t* const d_del_qual = d_del_qual_vec.data().get();
        int* const d_readBeginOffsets = d_readBeginOffsets_vec.data().get();
        int* const d_hapBeginOffsets = d_hapBeginOffsets_vec.data().get();
        int* const d_readLengths = d_readLengths_vec.data().get();
        int* const d_hapLengths = d_hapLengths_vec.data().get();
        int* const d_numReadsPerSequenceGroup = d_numReadsPerSequenceGroup_vec.data().get();
        int* const d_numHapsPerSequenceGroup = d_numHapsPerSequenceGroup_vec.data().get();
        int* const d_numReadsPerSequenceGroupPrefixSum = d_numReadsPerSequenceGroupPrefixSum_vec.data().get();
        int* const d_numHapsPerSequenceGroupPrefixSum = d_numHapsPerSequenceGroupPrefixSum_vec.data().get();
        float* const devAlignmentScoresFloat = devAlignmentScoresFloat_vec.data().get();

        DEBUGSYNC
    
        int numProcessedAlignmentsByChunks = 0;
        int numProcessedSequenceGroupsByChunks = 0;

        struct ChunkInfo{
            int firstSequenceGroup = 0;
            int lastSequenceGroup_excl = 0;
        };
        std::vector<ChunkInfo> chunkInfoVec(1);
        for(int i = 0, numAlignments = 0; i < numSequenceGroupsInSequenceGroup; i++){
            chunkInfoVec.back().lastSequenceGroup_excl++;
            numAlignments += fullBatch.numAlignmentsPerSequenceGroup[i];
            if(numAlignments >= options.transferBatchsize){
                numAlignments = 0;
                if(i < numSequenceGroupsInSequenceGroup-1){
                    ChunkInfo next{i+1, i+1};
                    chunkInfoVec.push_back(next);
                }
            }
        }
        
        const int numTransferChunks = chunkInfoVec.size();

        for(int computeChunk = 0, transferChunk = 0; computeChunk < numTransferChunks; computeChunk++){
            for(; transferChunk < numTransferChunks && transferChunk < (computeChunk + 2); transferChunk++){
                #ifdef ENABLE_NVTX3
                nvtx3::scoped_range sr1("transferChunk");
                #endif
                cudaStream_t transferStream = transferStreams[transferChunk % transferStreams.size()];
                
                const int firstBatchId = chunkInfoVec[transferChunk].firstSequenceGroup;
                const int lastBatchId_excl = chunkInfoVec[transferChunk].lastSequenceGroup_excl;
                const int numSequenceGroupsInChunk = lastBatchId_excl - firstBatchId;
    
                const int firstReadInChunk = numReadsPerSequenceGroupPrefixSum[firstBatchId];
                const int lastReadInChunk_excl = numReadsPerSequenceGroupPrefixSum[lastBatchId_excl];
                const int numReadsInChunk = lastReadInChunk_excl - firstReadInChunk;
    
                const int firstHapInChunk = numHapsPerSequenceGroupPrefixSum[firstBatchId];
                const int lastHapInChunk_excl = numHapsPerSequenceGroupPrefixSum[lastBatchId_excl];
                const int numHapsInChunk = lastHapInChunk_excl - firstHapInChunk;
    
                const size_t numReadBytesInChunk = readBeginOffsets[lastReadInChunk_excl] - readBeginOffsets[firstReadInChunk];
                const size_t numHapBytesInChunk = hapBeginOffsets[lastHapInChunk_excl] - hapBeginOffsets[firstHapInChunk];
    
                // std::cout << "transferChunk " << transferChunk << "\n";
                // std::cout << "firstBatchId " << firstBatchId << "\n";
                // std::cout << "lastBatchId_excl " << lastBatchId_excl << "\n";
                // std::cout << "numSequenceGroupsInChunk " << numSequenceGroupsInChunk << "\n";
                // std::cout << "firstReadInChunk " << firstReadInChunk << "\n";
                // std::cout << "lastReadInChunk_excl " << lastReadInChunk_excl << "\n";
                // std::cout << "numReadsInChunk " << numReadsInChunk << "\n";
                // std::cout << "firstHapInChunk " << firstHapInChunk << "\n";
                // std::cout << "lastHapInChunk_excl " << lastHapInChunk_excl << "\n";
                // std::cout << "numHapsInChunk " << numHapsInChunk << "\n";
                // std::cout << "numReadBytesInChunk " << numReadBytesInChunk << "\n";
                // std::cout << "numHapBytesInChunk " << numHapBytesInChunk << "\n";
                // std::cout << "----------------------------\n";
    
    
    
                CUDACHECK(cudaMemcpyAsync(d_readData + readBeginOffsets[firstReadInChunk], readData + readBeginOffsets[firstReadInChunk], numReadBytesInChunk, cudaMemcpyHostToDevice, transferStream));
                CUDACHECK(cudaMemcpyAsync(d_hapData + hapBeginOffsets[firstHapInChunk], hapData + hapBeginOffsets[firstHapInChunk], numHapBytesInChunk, cudaMemcpyHostToDevice, transferStream));
                CUDACHECK(cudaMemcpyAsync(d_base_qual + readBeginOffsets[firstReadInChunk], base_qual + readBeginOffsets[firstReadInChunk], numReadBytesInChunk, cudaMemcpyHostToDevice, transferStream));
                CUDACHECK(cudaMemcpyAsync(d_ins_qual + readBeginOffsets[firstReadInChunk], ins_qual + readBeginOffsets[firstReadInChunk], numReadBytesInChunk, cudaMemcpyHostToDevice, transferStream));
                CUDACHECK(cudaMemcpyAsync(d_del_qual + readBeginOffsets[firstReadInChunk], del_qual + readBeginOffsets[firstReadInChunk], numReadBytesInChunk, cudaMemcpyHostToDevice, transferStream));
    
                CUDACHECK(cudaMemcpyAsync(d_readBeginOffsets + firstReadInChunk, readBeginOffsets + firstReadInChunk, numReadsInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream));
                CUDACHECK(cudaMemcpyAsync(d_hapBeginOffsets + firstHapInChunk, hapBeginOffsets + firstHapInChunk, numHapsInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream));
                CUDACHECK(cudaMemcpyAsync(d_readLengths + firstReadInChunk, readLengths + firstReadInChunk, numReadsInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream));
                CUDACHECK(cudaMemcpyAsync(d_hapLengths + firstHapInChunk, hapLengths + firstHapInChunk, numHapsInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream));
                CUDACHECK(cudaMemcpyAsync(d_numReadsPerSequenceGroupPrefixSum + firstBatchId, numReadsPerSequenceGroupPrefixSum + firstBatchId, numSequenceGroupsInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream));
                CUDACHECK(cudaMemcpyAsync(d_numHapsPerSequenceGroupPrefixSum + firstBatchId, numHapsPerSequenceGroupPrefixSum + firstBatchId, numSequenceGroupsInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream));
                CUDACHECK(cudaMemcpyAsync(d_numReadsPerSequenceGroup + firstBatchId, numReadsPerSequenceGroup + firstBatchId, numSequenceGroupsInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream));
                CUDACHECK(cudaMemcpyAsync(d_numHapsPerSequenceGroup + firstBatchId, numHapsPerSequenceGroup + firstBatchId, numSequenceGroupsInChunk*sizeof(int), cudaMemcpyHostToDevice, transferStream));

                DEBUGSYNC
            }
            #ifdef ENABLE_NVTX3
            nvtx3::scoped_range sr2("computeChunk");
            #endif
            cudaStream_t mainStream = transferStreams[computeChunk % transferStreams.size()];
            const int firstBatchId = chunkInfoVec[computeChunk].firstSequenceGroup;
            const int lastBatchId_excl = chunkInfoVec[computeChunk].lastSequenceGroup_excl;
            const int numSequenceGroupsInChunk = lastBatchId_excl - firstBatchId;
    
            const int firstReadInChunk = numReadsPerSequenceGroupPrefixSum[firstBatchId];
            const int lastReadInChunk_excl = numReadsPerSequenceGroupPrefixSum[lastBatchId_excl];
            const int numReadsInChunk = lastReadInChunk_excl - firstReadInChunk;
    
            const int firstHapInChunk = numHapsPerSequenceGroupPrefixSum[firstBatchId];
            const int lastHapInChunk_excl = numHapsPerSequenceGroupPrefixSum[lastBatchId_excl];
            const int numHapsInChunk = lastHapInChunk_excl - firstHapInChunk;
    
            const size_t numReadBytesInChunk = readBeginOffsets[lastReadInChunk_excl] - readBeginOffsets[firstReadInChunk];
            const size_t numHapBytesInChunk = hapBeginOffsets[lastHapInChunk_excl] - hapBeginOffsets[firstHapInChunk];
    
            convert_DNA<<<numReadsInChunk, 128, 0, mainStream>>>(d_readData + readBeginOffsets[firstReadInChunk], numReadBytesInChunk);
            convert_DNA<<<numHapsInChunk, 128, 0, mainStream>>>(d_hapData + hapBeginOffsets[firstHapInChunk], numHapBytesInChunk);

            DEBUGSYNC
    
            //ensure buffers used by previous batch are no longer in use
            for(auto stream : computeStreams){
                CUDACHECK(cudaStreamSynchronize(stream));
            }
    
            CUDACHECK(cudaMemsetAsync(d_numIndicesPerPartitionPerBatch.data().get(), 0, sizeof(int) * numLengthPartitions * numSequenceGroupsInChunk, mainStream));
            partitionIndicesKernel<<<numSequenceGroupsInChunk, 128, 0, mainStream>>>(
                d_numIndicesPerPartitionPerBatch.data().get(),
                d_indicesPerPartitionPerBatch.data().get(),
                d_readLengths,
                d_numReadsPerSequenceGroup + firstBatchId,
                d_numReadsPerSequenceGroupPrefixSum + firstBatchId,
                numSequenceGroupsInChunk,
                numReadsInChunk,
                MultiConfigAligner::getArrayOfSingleTileTileSizes()
            );
            CUDACHECKASYNC;

            DEBUGSYNC
    
            #if 0
                thrust::host_vector<int> h_numIndicesPerPartitionPerBatch = d_numIndicesPerPartitionPerBatch;
                thrust::host_vector<int> h_indicesPerPartitionPerBatch = d_indicesPerPartitionPerBatch;
    
                for(int p = 0; p < numLengthPartitions; p++){
                    if(p <= 4){
                        std::cout << "Partition p = " << p << "\n";
                        std::cout << "numIndicesPerBatch: ";
                        for(int b = 0; b < numSequenceGroupsInChunk; b++){
                            std::cout << h_numIndicesPerPartitionPerBatch[p * numSequenceGroupsInChunk + b] << ", ";
                        }
                        std::cout << "\n";
    
                        std::cout << "indicesPerBatch: ";
                        for(int b = 0; b < numSequenceGroupsInChunk; b++){
                            const int num = h_numIndicesPerPartitionPerBatch[p * numSequenceGroupsInChunk + b];
                            for(int i = 0; i < num; i++){
                                const int outputOffset = numReadsPerSequenceGroupPrefixSum[firstBatchId + b] - numReadsPerSequenceGroupPrefixSum[firstBatchId];
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
                d_numReadsPerSequenceGroup + firstBatchId,
                d_numReadsPerSequenceGroup + firstBatchId + numSequenceGroupsInChunk,
                d_numHapsPerSequenceGroup + firstBatchId,
                d_resultOffsetsPerBatch.begin(),
                thrust::multiplies<int>{}
            );
            thrust::exclusive_scan(
                thrust::cuda::par_nosync(ThrustCudaMallocAsyncAllocator<int>(mainStream)).on(mainStream),
                d_resultOffsetsPerBatch.begin(),
                d_resultOffsetsPerBatch.begin() + numSequenceGroupsInChunk,
                d_resultOffsetsPerBatch.begin()
            );

            DEBUGSYNC
    
            {     
                int* d_numAlignmentsPerPartitionPerBatch = d_numAlignmentsPerBatchInclPrefixSum.data().get(); // reuse
                const int* d_numHaplotypesPerBatch = d_numHapsPerSequenceGroup;
                computeAlignmentsPerPartitionPerBatch<<<dim3(SDIV(numSequenceGroupsInChunk, 128), numLengthPartitions), 128,0, mainStream>>>(
                    d_numAlignmentsPerPartitionPerBatch,
                    d_numIndicesPerPartitionPerBatch.data().get(),
                    d_numHaplotypesPerBatch + firstBatchId,
                    numLengthPartitions, 
                    numSequenceGroupsInChunk
                );
                CUDACHECKASYNC;
        
                auto offsets = thrust::make_transform_iterator(
                    thrust::make_counting_iterator(0),
                    [numSequenceGroupsInChunk] __host__ __device__(int partition){
                        return partition * numSequenceGroupsInChunk;
                    }
                );
                size_t temp_storage_bytes = 0;
                CUDACHECK(cub::DeviceSegmentedReduce::Sum(nullptr, temp_storage_bytes, d_numAlignmentsPerPartitionPerBatch, 
                    d_numAlignmentsPerPartition.data().get(), numLengthPartitions, offsets, offsets + 1, mainStream));
                char* d_temp_ptr;
                //thrust::device_vector<char, ThrustCudaMallocAsyncAllocator<char>> d_temp(temp_storage_bytes, ThrustCudaMallocAsyncAllocator<char>(mainStream));
                //d_temp_ptr = d_temp.data().get();
                //Don't use thrust::device_vector, skips internal synchronization
                CUDACHECK(cudaMallocAsync(&d_temp_ptr, temp_storage_bytes, mainStream));
                CUDACHECK(cub::DeviceSegmentedReduce::Sum(d_temp_ptr, temp_storage_bytes, d_numAlignmentsPerPartitionPerBatch, 
                    d_numAlignmentsPerPartition.data().get(), numLengthPartitions, offsets, offsets + 1, mainStream));
                CUDACHECK(cudaFreeAsync(d_temp_ptr, mainStream));
            }

            DEBUGSYNC
    
            CUDACHECK(cudaMemcpyAsync(h_numAlignmentsPerPartition.data(), d_numAlignmentsPerPartition.data().get(), sizeof(int) * numLengthPartitions, cudaMemcpyDeviceToHost, mainStream));
            //wait for above D2H transfer. This also synchronizes with the H2D transfer so we can use different streams for computations without cudaStreamWaitEvent
            CUDACHECK(cudaStreamSynchronize(mainStream)); 
    
            // std::cout << "h_numAlignmentsPerPartition: ";
            // for(int i = 0; i< numSingleTileTileSizes; i++){
            //     std::cout << h_numAlignmentsPerPartition[i] << ", ";
            // }
            // std::cout << "\n";
    
            int numProcessedAlignmentsByCurrentChunk = 0;

            int currentStreamIndex = 0;
            for(int partitionId = 0; partitionId < numLengthPartitions; partitionId++){
                const int numAlignmentsInPartition = h_numAlignmentsPerPartition[partitionId];
                if(numAlignmentsInPartition > 0){
                    currentStreamIndex = (currentStreamIndex + 1) % computeStreams.size();
                    const cudaStream_t stream = computeStreams[currentStreamIndex];

                    #ifdef ENABLE_NVTX3
                    nvtx3::scoped_range sr3("partition");
                    #endif
                    const int* d_numIndicesPerBatch = d_numIndicesPerPartitionPerBatch.data().get() + partitionId*numSequenceGroupsInChunk;
                    thrust::transform(
                        thrust::cuda::par_nosync.on(stream),
                        d_numIndicesPerBatch,
                        d_numIndicesPerBatch + numSequenceGroupsInChunk,
                        d_numHapsPerSequenceGroup + numProcessedSequenceGroupsByChunks,
                        d_numAlignmentsPerBatch.begin() + partitionId * numSequenceGroupsInChunk,
                        thrust::multiplies<int>{}
                    );
                    thrust::inclusive_scan(
                        thrust::cuda::par_nosync(ThrustCudaMallocAsyncAllocator<int>(stream)).on(stream),
                        d_numAlignmentsPerBatch.begin() + partitionId * numSequenceGroupsInChunk,
                        d_numAlignmentsPerBatch.begin() + partitionId * numSequenceGroupsInChunk+ numSequenceGroupsInChunk,
                        d_numAlignmentsPerBatchInclPrefixSum.begin() + partitionId * numSequenceGroupsInChunk
                    ); 

                    if(partitionId < numSingleTileTileSizes){
                        multiConfigGpuPairHMM.launchPairHMMSingleTileKernel(
                            singleTileTilesizes[partitionId],
                            devAlignmentScoresFloat + numProcessedAlignmentsByChunks, 
                            d_readData, 
                            d_hapData, 
                            d_base_qual, 
                            d_ins_qual, 
                            d_del_qual, 
                            d_readBeginOffsets + firstReadInChunk, 
                            d_hapBeginOffsets + firstHapInChunk, 
                            d_readLengths + firstReadInChunk, 
                            d_hapLengths + firstHapInChunk,
                            d_numHapsPerSequenceGroup + firstBatchId, 
                            d_numHapsPerSequenceGroupPrefixSum + firstBatchId, 
                            d_indicesPerPartitionPerBatch.data().get() + partitionId * numReadsInChunk, 
                            d_numReadsPerSequenceGroupPrefixSum + firstBatchId,  
                            numSequenceGroupsInChunk, 
                            d_resultOffsetsPerBatch.data().get(), 
                            d_numAlignmentsPerBatch.data().get() + partitionId * numSequenceGroupsInChunk, 
                            d_numAlignmentsPerBatchInclPrefixSum.data().get() + partitionId * numSequenceGroupsInChunk, 
                            numAlignmentsInPartition,
                            stream
                        );
                        DEBUGSYNC
                    }else{
                        size_t tempBytes = 0;
                        thrust::device_vector<char, ThrustCudaMallocAsyncAllocator<char>> d_temp(0, ThrustCudaMallocAsyncAllocator<char>(stream));
                        try{
                            tempBytes = multiConfigGpuPairHMM.getSuggestedTempBytesForMultiTile(maximumHaplotypeLength);
                            d_temp.resize(tempBytes);
                        }catch(...){
                            tempBytes = multiConfigGpuPairHMM.getMinimumSuggestedTempBytesForMultiTile(maximumHaplotypeLength);
                            d_temp.resize(tempBytes);
                        }

                        multiConfigGpuPairHMM.launchPairHMMMultiTileKernel(
                            d_temp.data().get(),
                            d_temp.size(),
                            maximumHaplotypeLength,
                            devAlignmentScoresFloat + numProcessedAlignmentsByChunks, 
                            d_readData, 
                            d_hapData, 
                            d_base_qual, 
                            d_ins_qual, 
                            d_del_qual, 
                            d_readBeginOffsets + firstReadInChunk, 
                            d_hapBeginOffsets + firstHapInChunk, 
                            d_readLengths + firstReadInChunk, 
                            d_hapLengths + firstHapInChunk,
                            d_numHapsPerSequenceGroup + firstBatchId, 
                            d_numHapsPerSequenceGroupPrefixSum + firstBatchId, 
                            d_indicesPerPartitionPerBatch.data().get() + partitionId * numReadsInChunk, 
                            d_numReadsPerSequenceGroupPrefixSum + firstBatchId,  
                            numSequenceGroupsInChunk, 
                            d_resultOffsetsPerBatch.data().get(), 
                            d_numAlignmentsPerBatch.data().get() + partitionId * numSequenceGroupsInChunk, 
                            d_numAlignmentsPerBatchInclPrefixSum.data().get() + partitionId * numSequenceGroupsInChunk, 
                            numAlignmentsInPartition,
                            stream
                        );
                    }
                    numProcessedAlignmentsByCurrentChunk += numAlignmentsInPartition;
                }
            }

            numProcessedAlignmentsByChunks += numProcessedAlignmentsByCurrentChunk;
            numProcessedSequenceGroupsByChunks += numSequenceGroupsInChunk;                
        }
    
        pipelineDataPtr->h_scores.resize(totalNumberOfAlignments);

        CUDACHECK(cudaDeviceSynchronize()); //wait for all streams to complete before transfer
        CUDACHECK(cudaMemcpyAsync(pipelineDataPtr->h_scores.data(), devAlignmentScoresFloat, totalNumberOfAlignments*sizeof(float), cudaMemcpyDeviceToHost, transferStreams[0]));
        CUDACHECK(cudaStreamSynchronize(transferStreams[0]));
    
        totalTimer.stop();
        totalSecondsAllBatches += totalTimer.elapsed();
        auto cells = computeNumberOfDPCells(fullBatch);
        totalDPCellsAllBatches += cells;
        // if(options.verbose){
        //     totalTimer.printGCUPS(cells);
        // }    
    }

    std::vector<float> processBatchCPU(const BatchOfSequenceGroups& batch_, const std::vector<float>& ph2pr){
        helpers::CpuTimer totalTimer("processBatchCPU");
        //const uint64_t dp_cells = computeNumberOfDPCells(batch_);
    
        const int numSequenceGroups = batch_.numHapsPerSequenceGroup.size();
    
        const int totalNumberOfAlignments = batch_.getTotalNumberOfAlignments();
        const auto& numAlignmentsPerBatchInclusivePrefixSum = batch_.getNumberOfAlignmentsPerBatchInclusivePrefixSum();
    
        std::vector<float> results(totalNumberOfAlignments);
    
        #pragma omp parallel for schedule(dynamic, 16)
        for(int alignmentId = 0; alignmentId < totalNumberOfAlignments; alignmentId++){
            const int batchIdByGroupId = std::distance(
                numAlignmentsPerBatchInclusivePrefixSum.begin(),
                std::upper_bound(
                    numAlignmentsPerBatchInclusivePrefixSum.begin(),
                    numAlignmentsPerBatchInclusivePrefixSum.begin() + numSequenceGroups,
                    alignmentId
                )
            );
            const int batchId = min(batchIdByGroupId, numSequenceGroups-1);
            const int numHapsInSequenceGroup = batch_.numHapsPerSequenceGroup[batchId];
            const int alignmentOffset = (batchId == 0 ? 0 : numAlignmentsPerBatchInclusivePrefixSum[batchId-1]);
            const int alignmentIdInSequenceGroup = alignmentId - alignmentOffset;
            const int hapToProcessInSequenceGroup = alignmentIdInSequenceGroup % numHapsInSequenceGroup;
            const int readToProcessInSequenceGroup = alignmentIdInSequenceGroup / numHapsInSequenceGroup;
    
            int read = batch_.numReadsPerSequenceGroupPrefixSum[batchId]+readToProcessInSequenceGroup;
            int readLengths = batch_.readLengths[read];
            int hap = batch_.numHapsPerSequenceGroupPrefixSum[batchId]+hapToProcessInSequenceGroup;
            int hapLengths = batch_.hapLengths[hap];
            int h_off = batch_.hapBeginOffsets[hap];
            int r_off = batch_.readBeginOffsets[read];
            double score = align_host(&batch_.hapData[h_off],hapLengths,&batch_.readData[r_off],readLengths,&batch_.base_quals[r_off],&batch_.ins_quals[r_off],&batch_.del_quals[r_off],batch_.gcp_quals[r_off],&ph2pr[0]);
            
            const int outputIndex = alignmentOffset + readToProcessInSequenceGroup * numHapsInSequenceGroup + hapToProcessInSequenceGroup;
            // const int outputIndex = alignmentOffset + hapToProcessInSequenceGroup * numReadsInSequenceGroup + readToProcessInSequenceGroup;
            results[outputIndex] = score;
        }
    
        totalTimer.stop();
        //totalTimer.printGCUPS(dp_cells);
    
        return results;
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
};

std::future<void> launchGpuPairHMMWorker(
    const Options* options,
    PipelineDataQueue* inputQueue,
    PipelineDataQueue* outputQueue,
    int deviceId
);
