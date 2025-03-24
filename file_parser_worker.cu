#include "execution_pipeline.cuh"
#include "common.cuh"
#include "cuda_helpers.cuh"
#include "timers.cuh"
#include "cuda_errorcheck.cuh"

#include <algorithm>
#include <fstream>
#include <future>
#include <numeric>
#include <string>
#include <vector>
#include <functional>

#ifdef ENABLE_NVTX3
#include <nvtx3/nvtx3.hpp>
#endif


struct FileParserWorker{
    const Options* optionsPtr;
    PipelineDataQueue* pipelineDataQueueIn;
    PipelineDataQueue* pipelineDataQueueOut;

    FileParserWorker(const Options* optionsPtr_, PipelineDataQueue* pipelineDataQueueIn_, PipelineDataQueue* pipelineDataQueueOut_)
        : optionsPtr(optionsPtr_), pipelineDataQueueIn(pipelineDataQueueIn_), pipelineDataQueueOut(pipelineDataQueueOut_)
    {

    }

    void run(){
        #ifdef ENABLE_NVTX3
        nvtx3::scoped_range sr("FileParserWorker::run");
        #endif
        helpers::CpuTimer cputimer("FileParserWorker::run");
        const auto& options = *optionsPtr;

        std::ifstream file(options.inputfile);
        if(!file){
            throw std::runtime_error("Could not open file" + options.inputfile);
        }


        std::string linebuffer;
        std::vector<PairHmmSequenceGroup> sequenceGroups;
        size_t numAlignmentsInGroups = 0;

        PipelineData* pipelineDataPtr = pipelineDataQueueIn->pop();

        while(std::getline(file, linebuffer)){
            
            const size_t split = linebuffer.find(" ");
            const int numReads = std::stoi(linebuffer.substr(0,split));
            const int numHaps = std::stoi(linebuffer.substr(split));
            sequenceGroups.emplace_back(parseNextSequenceGroup(file, numReads, numHaps));
            numAlignmentsInGroups += sequenceGroups.back().numReads * sequenceGroups.back().numHaps;
            if(numAlignmentsInGroups >= size_t(options.fileBatchsize)){
                //pipelineDataPtr->batch = convertGroupsToBatch(sequenceGroups);
                convertGroupsToBatchInplace(pipelineDataPtr->batch, sequenceGroups);
                pipelineDataQueueOut->push(pipelineDataPtr);

                sequenceGroups.clear();
                numAlignmentsInGroups = 0;
                pipelineDataPtr = pipelineDataQueueIn->pop();
            }
        }
        if(numAlignmentsInGroups > 0){
            //pipelineDataPtr->batch = convertGroupsToBatch(sequenceGroups);
            convertGroupsToBatchInplace(pipelineDataPtr->batch, sequenceGroups);
            pipelineDataQueueOut->push(pipelineDataPtr);
        }
        pipelineDataQueueOut->push(nullptr); //notify consumer that parsing is complete

        cputimer.stop();
    
        if(options.verbose){
            cputimer.print();
        }
    }

    int getPaddingLength(int length){
        return 4-(length % 4);
    }
    int getPaddedLength(int length){
        return length + getPaddingLength(length);
    }

    PairHmmSequenceGroup parseNextSequenceGroup(std::ifstream& file, int numReads, int numHaps){

        PairHmmSequenceGroup sequenceGroup;
        sequenceGroup.numHaps = numHaps;
        sequenceGroup.numReads = numReads;
    
        constexpr char paddingchar = 'N';    
        constexpr int qualityoffset = -33;
        
        std::string linebuffer;
        for (int i=0; i < numReads; i++){
            std::getline(file, linebuffer);
    
            const int readLength = linebuffer.find(" ");
            const int paddingLength = getPaddingLength(readLength);
            const int paddedReadLength = getPaddedLength(readLength);
    
            sequenceGroup.readData.reserve(sequenceGroup.readData.size() + paddedReadLength);
            sequenceGroup.readData.insert(sequenceGroup.readData.end(), linebuffer.begin(), linebuffer.begin() + readLength);
            sequenceGroup.readData.insert(sequenceGroup.readData.end(), paddingLength, paddingchar);
            sequenceGroup.readBeginOffsets.push_back(sequenceGroup.readData.size());
            sequenceGroup.readLengths.push_back(readLength);
    
            sequenceGroup.base_quals.reserve(sequenceGroup.base_quals.size() + paddedReadLength);
            auto it0 = sequenceGroup.base_quals.end();
            sequenceGroup.base_quals.insert(sequenceGroup.base_quals.end(), linebuffer.begin() + 1*readLength+1, linebuffer.begin() + 1*readLength+1 + readLength);
            sequenceGroup.base_quals.insert(sequenceGroup.base_quals.end(), paddingLength, paddingchar);
            std::for_each(it0, sequenceGroup.base_quals.end(), [&](auto& x){x += qualityoffset;});
    
            sequenceGroup.ins_quals.reserve(sequenceGroup.ins_quals.size() + paddedReadLength);
            auto it1 = sequenceGroup.ins_quals.end();
            sequenceGroup.ins_quals.insert(sequenceGroup.ins_quals.end(), linebuffer.begin() + 2*readLength+2, linebuffer.begin() + 2*readLength+2 + readLength);
            sequenceGroup.ins_quals.insert(sequenceGroup.ins_quals.end(), paddingLength, paddingchar);
            std::for_each(it1, sequenceGroup.ins_quals.end(), [&](auto& x){x += qualityoffset;});
    
            sequenceGroup.del_quals.reserve(sequenceGroup.del_quals.size() + paddedReadLength);
            auto it2 = sequenceGroup.del_quals.end();
            sequenceGroup.del_quals.insert(sequenceGroup.del_quals.end(), linebuffer.begin() + 3*readLength+3, linebuffer.begin() + 3*readLength+3 + readLength);
            sequenceGroup.del_quals.insert(sequenceGroup.del_quals.end(), paddingLength, paddingchar);
            std::for_each(it2, sequenceGroup.del_quals.end(), [&](auto& x){x += qualityoffset;});
    
            sequenceGroup.gcp_quals.reserve(sequenceGroup.gcp_quals.size() + paddedReadLength);
            auto it3 = sequenceGroup.gcp_quals.end();
            sequenceGroup.gcp_quals.insert(sequenceGroup.gcp_quals.end(), linebuffer.begin() + 4*readLength+4, linebuffer.begin() + 4*readLength+4 + readLength);
            sequenceGroup.gcp_quals.insert(sequenceGroup.gcp_quals.end(), paddingLength, paddingchar);
            std::for_each(it3, sequenceGroup.gcp_quals.end(), [&](auto& x){x += qualityoffset;});
        }
    
        for (int i=0; i < numHaps; i++){
            std::getline(file, linebuffer);
    
            const int hapLength = linebuffer.length();
            const int paddingLength = getPaddingLength(hapLength);
            const int paddedHapLength = getPaddedLength(hapLength);
    
            sequenceGroup.hapLengths.push_back(hapLength);
    
            sequenceGroup.hapData.reserve(sequenceGroup.hapData.size() + paddedHapLength);
            sequenceGroup.hapData.insert(sequenceGroup.hapData.end(), linebuffer.begin(), linebuffer.begin() + hapLength);
            sequenceGroup.hapData.insert(sequenceGroup.hapData.end(), paddingLength, paddingchar);
            sequenceGroup.hapBeginOffsets.push_back(sequenceGroup.hapData.size());
        }
    
        return sequenceGroup;
    
    }

    void convertGroupsToBatchInplace(BatchOfSequenceGroups& result, const std::vector<PairHmmSequenceGroup>& groups){
        #ifdef ENABLE_NVTX3
        nvtx3::scoped_range sr("convertGroupsToBatchInplace");
        #endif

        #define CONCATVEC(v){ \
            size_t totalElements = std::accumulate(groups.begin(), groups.end(), 0ull, [](auto r, const auto& b){return r+b.v.size();}); \
            result.v.resize(totalElements); \
            auto it = result.v.begin(); \
            std::for_each(groups.begin(), groups.end(), [&](const auto& b){ it = std::copy(b.v.begin(), b.v.end(), it); }); \
        }
    
        CONCATVEC(readData)
        CONCATVEC(hapData)
        CONCATVEC(base_quals)
        CONCATVEC(ins_quals)
        CONCATVEC(del_quals)
        CONCATVEC(gcp_quals)
        CONCATVEC(readLengths)
        CONCATVEC(hapLengths)

        result.numReadsPerSequenceGroup.resize(groups.size());
        std::transform(groups.begin(), groups.end(), result.numReadsPerSequenceGroup.begin(), [](const auto& b){ return b.numReads; });
        result.numHapsPerSequenceGroup.resize(groups.size());
        std::transform(groups.begin(), groups.end(), result.numHapsPerSequenceGroup.begin(), [](const auto& b){ return b.numHaps; });

        result.readBeginOffsets.resize(result.readLengths.size() + 1);
        result.readBeginOffsets[0] = 0;
        std::transform_inclusive_scan(result.readLengths.begin(), result.readLengths.end(), result.readBeginOffsets.begin() + 1, std::plus<int>{}, 
            [&](const auto& l){ return getPaddedLength(l); }
        );

        result.hapBeginOffsets.resize(result.hapLengths.size() + 1);
        result.hapBeginOffsets[0] = 0;
        std::transform_inclusive_scan(result.hapLengths.begin(), result.hapLengths.end(), result.hapBeginOffsets.begin() + 1, std::plus<int>{}, 
            [&](const auto& l){ return getPaddedLength(l); }
        );

        result.numReadsPerSequenceGroupPrefixSum.resize(result.numReadsPerSequenceGroup.size()+1);
        result.numReadsPerSequenceGroupPrefixSum[0] = 0;
        std::inclusive_scan(result.numReadsPerSequenceGroup.begin(), result.numReadsPerSequenceGroup.end(), result.numReadsPerSequenceGroupPrefixSum.begin()+1);
        result.numHapsPerSequenceGroupPrefixSum.resize(result.numHapsPerSequenceGroup.size()+1);
        result.numHapsPerSequenceGroupPrefixSum[0] = 0;
        std::inclusive_scan(result.numHapsPerSequenceGroup.begin(), result.numHapsPerSequenceGroup.end(), result.numHapsPerSequenceGroupPrefixSum.begin()+1);
  
        result.numAlignmentsPerSequenceGroup.resize(groups.size());
        result.numAlignmentsPerSequenceGroupPrefixSum.resize(groups.size()+1);
        result.numAlignmentsPerSequenceGroupPrefixSum[0] = 0;
        std::transform(
            result.numReadsPerSequenceGroup.begin(), 
            result.numReadsPerSequenceGroup.end(),
            result.numHapsPerSequenceGroup.begin(),
            result.numAlignmentsPerSequenceGroup.begin(),
            std::multiplies<int>{}
        );
        std::inclusive_scan(
            result.numAlignmentsPerSequenceGroup.begin(),
            result.numAlignmentsPerSequenceGroup.end(),
            result.numAlignmentsPerSequenceGroupPrefixSum.begin()+1
        );
        result.totalNumberOfAlignments = result.numAlignmentsPerSequenceGroupPrefixSum.back();
    
        #undef CONCATVEC
    }

    BatchOfSequenceGroups convertGroupsToBatch(const std::vector<PairHmmSequenceGroup>& groups){
        #ifdef ENABLE_NVTX3
        nvtx3::scoped_range sr("convertGroupsToBatch");
        #endif
        BatchOfSequenceGroups result;
        convertGroupsToBatchInplace(result, groups);    
        return result;
    }
};

std::future<void> launchFileParser(
    const Options* options,
    PipelineDataQueue* inputQueue,
    PipelineDataQueue* outputQueue,
    int deviceId
){
    return std::async(std::launch::async,
        [=](){
            try{
                CUDACHECK(cudaSetDevice(deviceId));
                FileParserWorker worker(options, inputQueue, outputQueue);                   
                worker.run();
            }catch (const std::exception& e){
                std::cerr << e.what() << "\n";
                std::exit(EXIT_FAILURE);
            }catch(...){
                std::cerr << "Caught exception in file parser\n"; 
                std::exit(EXIT_FAILURE);
            }
        }
    );
}