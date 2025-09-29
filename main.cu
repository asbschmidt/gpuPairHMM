#include "execution_pipeline.cuh"
#include "common.cuh"
#include "cuda_errorcheck.cuh"

#include <limits>

void run(const Options& options){
    PipelineDataQueue inputQueueForParser;
    PipelineDataQueue inputQueueForHMM;
    PipelineDataQueue inputQueueForOutputWriter;

    std::vector<std::unique_ptr<PipelineData>> pipelineDataVector;

    for(int i = 0; i < options.queue_depth; i++){
        auto data = std::make_unique<PipelineData>();
        pipelineDataVector.push_back(std::move(data));
        inputQueueForParser.push(pipelineDataVector.back().get());
    }

    auto parserFuture = launchFileParser(&options, &inputQueueForParser, &inputQueueForHMM, 0);
    auto hmmFuture = launchGpuPairHMMWorker(&options, &inputQueueForHMM, &inputQueueForOutputWriter, 0);
    auto outputwriterFuture = launchOutputWriterWorker(&options, &inputQueueForOutputWriter, &inputQueueForParser, 0);

    parserFuture.wait();
    hmmFuture.wait();
    outputwriterFuture.wait();
}


int main(int argc, char** argv){

    const int deviceId = 0;
    size_t releaseThreshold = UINT64_MAX;
    cudaMemPool_t memPool;
    CUDACHECK(cudaDeviceGetDefaultMemPool(&memPool, deviceId));
    CUDACHECK(cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &releaseThreshold));
    //init memory pool with 512 MB
    {
        void* ptr;
        CUDACHECK(cudaMallocFromPoolAsync(&ptr, 512 * 1024 * 1024, memPool, (cudaStream_t)0));
        CUDACHECK(cudaFreeAsync(ptr, (cudaStream_t)0));
        CUDACHECK(cudaDeviceSynchronize());
    }

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
        if(argstring == "--transferBatchsize"){
            options.transferBatchsize = std::atoi(argv[x+1]);
            x++;
        }
        if(argstring == "--checkResults"){
            options.checkResults = true;
        }
        if(argstring == "--verbose"){
            options.verbose = true;
        }
        if(argstring == "--fileBatchsize"){
            options.fileBatchsize = std::atoi(argv[x+1]);
            x++;
        }
        if(argstring == "--queue_depth"){
            options.queue_depth = std::atoi(argv[x+1]);
            x++;
        }
    }

    if(options.transferBatchsize <= 0){
        options.transferBatchsize = std::numeric_limits<int>::max();
    }

    if(options.fileBatchsize <= 0){
        options.fileBatchsize = std::numeric_limits<int>::max();
    }

    std::cout << "options.inputfile = " << options.inputfile << "\n";
    std::cout << "options.outputfile = " << options.outputfile << "\n";
    std::cout << "options.transferBatchsize = " << options.transferBatchsize << "\n";
    std::cout << "options.verbose = " << options.verbose << "\n";
    std::cout << "options.fileBatchsize = " << options.fileBatchsize << "\n";

    run(options);
    
    return 0;
}