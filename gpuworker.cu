#include "gpuworker.cuh"
#include "tuningconfigs.hpp"


//#define TUNING_ARCH_MACRO 80


std::future<void> launchGpuPairHMMWorker(
    const Options* options,
    PipelineDataQueue* inputQueue,
    PipelineDataQueue* outputQueue,
    int deviceId
){
    #if defined(TUNING_ARCH_MACRO)
    constexpr int cudaArch = TUNING_ARCH_MACRO * 10;
    #else
    constexpr int cudaArch = 0;
    #endif

    using Selector = SelectRealConfigOrCompatibilityConfig<cudaArch, PairHMMKernelConfigsSingleTile>;
    constexpr int selectedArch = Selector::arch;
    using KernelConfigs = Selector::type;
    if(options->verbose){
        std::cout << "Using tuning config " << selectedArch / 10 << "\n";
    }

    return std::async(std::launch::async,
        [=](){
            try{
                CUDACHECK(cudaSetDevice(deviceId));
                GpuPairHMMWorker<MultiConfigGpuPairHMM<KernelConfigs>> worker(options, inputQueue, outputQueue);                   
                worker.run();
            }catch (const std::exception& e){
                std::cerr << e.what() << "\n";
                std::exit(EXIT_FAILURE);
            }catch(...){
                std::cerr << "Caught exception in gpu worker\n"; 
                std::exit(EXIT_FAILURE);
            }
        }
    );
}