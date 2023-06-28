# settings
DIALECT      = -std=c++14
OPTIMIZATION = -O3
WARNINGS     = -Xcompiler="-Wall -Wextra"
# NVCC_FLAGS   = -arch=sm_61 -lineinfo --expt-relaxed-constexpr -rdc=true
NVCC_FLAGS   = -arch=sm_80 -lineinfo --expt-relaxed-constexpr -rdc=true --extended-lambda -res-usage -Xcompiler="-fopenmp"
LDFLAGS      = -Xcompiler="-pthread -s"  $(NVCC_FLAGS)
COMPILER     = nvcc
ARTIFACT     = align


# make targets
.PHONY: clean

release: $(ARTIFACT)

clean :
	rm -f *.o
	rm -f $(ARTIFACT)


# compiler call
COMPILE = $(COMPILER) $(NVCC_FLAGS) $(DIALECT) $(OPTIMIZATION) $(WARNINGS) -c $< -o $@


# link object files into executable
$(ARTIFACT): main.o sequence_io.o dbdata.o
	$(COMPILER) $^ -o $(ARTIFACT) $(LDFLAGS)

# compile CUDA files
main.o : main.cu sequence_io.h cuda_helpers.cuh
	$(COMPILE)

# compile pure C++ files
sequence_io.o : sequence_io.cpp sequence_io.h
	$(COMPILE)

# compile pure C++ files
dbdata.o : dbdata.cpp dbdata.hpp mapped_file.hpp sequence_io.cpp sequence_io.h
	$(COMPILE)
