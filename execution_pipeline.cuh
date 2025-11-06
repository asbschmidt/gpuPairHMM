#ifndef EXECUTION_PIPELINE_CUH
#define EXECUTION_PIPELINE_CUH

#include "common.cuh"

#include <future>

struct PipelineData{
    //std::vector<float, PinnedAllocator<float>> h_scores;
    std::vector<float> h_scores;
    BatchOfSequenceGroups batch;

    void reset(){
        batch.reset();
        h_scores.clear();
    }
};



using PipelineDataQueue = SimpleConcurrentQueue<PipelineData*>;


std::future<void> launchFileParser(
    const Options* options,
    PipelineDataQueue* inputQueue,
    PipelineDataQueue* outputQueue,
    int deviceId
);

std::future<void> launchOutputWriterWorker(
    const Options* options,
    PipelineDataQueue* inputQueue,
    PipelineDataQueue* outputQueue,
    int deviceId
);

std::future<void> launchGpuPairHMMWorker(
    const Options* options,
    PipelineDataQueue* inputQueue,
    PipelineDataQueue* outputQueue,
    int deviceId
);



#endif