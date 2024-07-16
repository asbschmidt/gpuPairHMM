# settings
DIALECT      = -std=c++14
OPTIMIZATION = -O3 -g
WARNINGS     = -Xcompiler="-Wall -Wextra"
# NVCC_FLAGS   = -arch=sm_89 -lineinfo --expt-relaxed-constexpr -rdc=true
# NVCC_FLAGS   = -arch=native -lineinfo --expt-relaxed-constexpr -rdc=true --extended-lambda -Xcompiler="-fopenmp"
NVCC_FLAGS   = -arch=native -lineinfo --expt-relaxed-constexpr -rdc=true --extended-lambda -Xcompiler="-fopenmp" #-res-usage 


INCLUDE_FLAGS = -INVTX/c/include
LDFLAGS      = -Xcompiler="-pthread "  $(NVCC_FLAGS)
COMPILER     = nvcc
ARTIFACT     = align


# make targets
.PHONY: clean

release: $(ARTIFACT)

clean :
	rm -f *.o
	rm -f $(ARTIFACT)


# compiler call
COMPILE = $(COMPILER) $(INCLUDE_FLAGS) $(NVCC_FLAGS) $(DIALECT) $(OPTIMIZATION) $(WARNINGS) -c $< -o $@


# link object files into executable
$(ARTIFACT): main.o 
	$(COMPILER) $^ -o $(ARTIFACT) $(LDFLAGS)

# compile CUDA files
main.o : main.cu cuda_helpers.cuh
	$(COMPILE)

# compile pure C++ files
#sequence_io.o : sequence_io.cpp sequence_io.h
#	$(COMPILE)

# compile pure C++ files
#dbdata.o : dbdata.cpp dbdata.hpp mapped_file.hpp sequence_io.cpp sequence_io.h
#	$(COMPILE)
