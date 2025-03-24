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

#ifdef ENABLE_NVTX3
#include <nvtx3/nvtx3.hpp>
#endif

struct OutputWriterWorker{
    const Options* optionsPtr;
    PipelineDataQueue* pipelineDataQueueIn;
    PipelineDataQueue* pipelineDataQueueOut;

    OutputWriterWorker(const Options* optionsPtr_, PipelineDataQueue* pipelineDataQueueIn_, PipelineDataQueue* pipelineDataQueueOut_)
        : optionsPtr(optionsPtr_), pipelineDataQueueIn(pipelineDataQueueIn_), pipelineDataQueueOut(pipelineDataQueueOut_)
    {

    }

    void run(){
        #ifdef ENABLE_NVTX3
        nvtx3::scoped_range sr("OutputWriterWorker::run");
        #endif
        helpers::CpuTimer cputimer("OutputWriterWorker::run");
        const auto& options = *optionsPtr;

        std::ofstream file(options.outputfile);
        if(!file){
            throw std::runtime_error("Could not open file" + options.outputfile);
        }

        std::ofstream outputfilestream;
        bool outputToFile = false;
        if(options.outputfile != ""){
            outputfilestream.open(options.outputfile);
            if(!(outputfilestream)){
                throw std::runtime_error("Could not open file" + options.outputfile);
            }
            outputToFile = true;
        }

        std::ostream* osPtr;
        if(outputToFile){
            osPtr = &outputfilestream;
        }else{
            osPtr = &std::cout;
        }

        bool noOutput = false;
        if(options.outputfile == "/dev/null"){
            noOutput = true;
        }


        PipelineData* pipelineDataPtr = pipelineDataQueueIn->pop();

        // std::stringstream sstream;
        // int numcached = 0;
        // auto reset_stringstream = [&](){
        //     sstream.str("");
        //     sstream.clear();
        //     numcached = 0;
        // };

        std::vector<char> charbuffer(1024*1024);
        int usedbuffer = 0;
        int numprinted = 0;
        auto reset_charbuffer = [&](){
            usedbuffer = 0;
        };
        auto flush_charbuffer = [&](){
            (*osPtr) << charbuffer.data();
        };

        auto writebuffered = [&](const char* format, auto... args){
            numprinted = snprintf(charbuffer.data() + usedbuffer, charbuffer.size() - usedbuffer, format, args...);
            if(numprinted < 0){
                throw std::runtime_error("error snprintf");
            }
            if(numprinted + usedbuffer > int(charbuffer.size())-1){
                flush_charbuffer();
                reset_charbuffer();
                //try again
                numprinted = snprintf(charbuffer.data() + usedbuffer, charbuffer.size() - usedbuffer, format, args...);
                if(numprinted < 0){
                    throw std::runtime_error("error snprintf");
                }
                assert(numprinted + usedbuffer <= int(charbuffer.size())-1);
            }
            usedbuffer += numprinted;
        };

        while(pipelineDataPtr != nullptr){
            const auto& pipelineData = *pipelineDataPtr;
            const auto& batch = pipelineDataPtr->batch;

            if(!noOutput){
                const int numSequenceGroupsInBatch = batch.numReadsPerSequenceGroup.size();

                size_t resultindex = 0;
                for(int b = 0; b < numSequenceGroupsInBatch; b++){
                    const int numReadsInGroup = batch.numReadsPerSequenceGroup[b];
                    const int numHapsInGroup = batch.numHapsPerSequenceGroup[b];
                    //sstream << numReadsInGroup << ' ' << numHapsInGroup << '\n';
                    writebuffered("%d %d\n", numReadsInGroup, numHapsInGroup);

                    for(int r = 0; r < numReadsInGroup; r++){
                        for(int h = 0; h < numHapsInGroup; h++){
                            if(h > 0){
                                //sstream << ' ';
                                writebuffered(" ");
                            }
                            //sstream << pipelineData.h_scores[resultindex++];
                            writebuffered("%.5f", pipelineData.h_scores[resultindex++]);
                        }
                        //sstream << '\n';
                        writebuffered("\n");
                    }
                    // numcached += numReadsInGroup * numHapsInGroup;
                    // if(numcached >= 10000){
                    //     if(sstream.rdbuf()->in_avail() > 0){
                    //         (*osPtr) << sstream.rdbuf();
                    //         reset_stringstream();
                    //     }
                    // }
                }
            }
            pipelineDataPtr->reset();
            pipelineDataQueueOut->push(pipelineDataPtr);
            pipelineDataPtr = pipelineDataQueueIn->pop();
        }

        if(usedbuffer > 0){
            flush_charbuffer();
        }

        // if(numcached > 0){
        //     if(sstream.rdbuf()->in_avail() > 0){
        //         (*osPtr) << sstream.rdbuf();
        //         reset_stringstream();
        //     }
        // }


        cputimer.stop();
    
        if(options.verbose){
            cputimer.print();
        }
    }
};

std::future<void> launchOutputWriterWorker(
    const Options* options,
    PipelineDataQueue* inputQueue,
    PipelineDataQueue* outputQueue,
    int deviceId
){
    return std::async(std::launch::async,
        [=](){
            try{
                CUDACHECK(cudaSetDevice(deviceId));
                OutputWriterWorker worker(options, inputQueue, outputQueue);                   
                worker.run();
            }catch (const std::exception& e){
                std::cerr << e.what() << "\n";
                std::exit(EXIT_FAILURE);
            }catch(...){
                std::cerr << "Caught exception in output writer\n"; 
                std::exit(EXIT_FAILURE);
            }
        }
    );
}