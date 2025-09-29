# settings
DIALECT      = -std=c++17
OPTIMIZATION = -O3 -g
WARNINGS     = -Xcompiler="-Wall -Wextra -Wno-format-security"

NVCC_FLAGS   = -lineinfo --expt-relaxed-constexpr --extended-lambda -Xcompiler="-fopenmp"

GPUARCH ?= native
NVCC_GPU_ARCH = -arch=native

ifeq ($(GPUARCH), native)
	NVCC_GPU_ARCH = -arch=native
else
	NVCC_GPU_ARCH := $(foreach N, $(GPUARCH),-gencode=arch=compute_$(N),code=sm_$(N) )
endif

GPUARCH_NUM_COMPILE_THREADS ?= 1
NVCC_FLAGS += $(NVCC_GPU_ARCH) --threads $(GPUARCH_NUM_COMPILE_THREADS)

TUNINGARCH ?= 0


#INCLUDE_FLAGS = -INVTX/c/include
#NVCC_FLAGS += -DENABLE_NVTX3
LDFLAGS      = -Xcompiler="-pthread "  $(NVCC_FLAGS)
COMPILER     = nvcc
ARTIFACT     = gpuPairHMM


# make targets
.PHONY: clean

release: $(ARTIFACT)

clean :
	rm -f *.o
	rm -f $(ARTIFACT)
	rm -f $(benchmark_peakperformance)


# compiler call
COMPILE = $(COMPILER) $(INCLUDE_FLAGS) $(NVCC_FLAGS) $(DIALECT) $(OPTIMIZATION) $(WARNINGS) -c $< -o $@


# link object files into executable
$(ARTIFACT): main.o output_writer_worker.o file_parser_worker.o gpuworker.o
	$(COMPILER) $^ -o $(ARTIFACT) $(LDFLAGS)

benchmark_peakperformance: benchmark_peakperformance.o
	$(COMPILER) $^ -o benchmark_peakperformance $(LDFLAGS)


benchmark_peakperformance.o: benchmark_peakperformance.cu cuda_helpers.cuh pairhmm_kernels.cuh utility_kernels.cuh
	$(COMPILE)

main.o: main.cu
	$(COMPILE)

output_writer_worker.o: output_writer_worker.cu
	$(COMPILE)

file_parser_worker.o: file_parser_worker.cu
	$(COMPILE)

gpuworker.o: gpuworker.cu pairhmm_kernels.cuh
	$(COMPILE) -DTUNING_ARCH_MACRO=$(TUNINGARCH)
