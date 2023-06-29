# settings
DIALECT      = -std=c++17
#OPTIMIZATION = -O0 -g
OPTIMIZATION = -O3
WARNINGS     = -Xcompiler="-Wall -Wextra"
# NVCC_FLAGS   = -arch=sm_61 -lineinfo --expt-relaxed-constexpr -rdc=true
NVCC_FLAGS   = -arch=sm_80 -lineinfo --expt-relaxed-constexpr -rdc=true --extended-lambda -Xcompiler="-fopenmp" -Xptxas "-v"
LDFLAGS      = -Xcompiler="-pthread -s"  $(NVCC_FLAGS)
#LDFLAGS      = -Xcompiler="-pthread"  $(NVCC_FLAGS)
COMPILER     = nvcc
ARTIFACT     = align

MAKEDB = makedb

# make targets
.PHONY: clean

release: $(ARTIFACT) $(MAKEDB)

clean :
	rm -f *.o
	rm -f $(ARTIFACT)
	rm -f $(MAKEDB)


# compiler call
COMPILE = $(COMPILER) $(NVCC_FLAGS) $(DIALECT) $(OPTIMIZATION) $(WARNINGS) -c $< -o $@


# link object files into executable
$(ARTIFACT): main.o sequence_io.o dbdata.o
	$(COMPILER) $^ -o $(ARTIFACT) $(LDFLAGS)

$(MAKEDB): makedb.o sequence_io.o dbdata.o
	$(COMPILER) $^ -o $(MAKEDB) $(LDFLAGS)

# compile CUDA files
main.o : main.cu sequence_io.h cuda_helpers.cuh length_partitions.hpp
	$(COMPILE)

# compile pure C++ files
sequence_io.o : sequence_io.cpp sequence_io.h
	$(COMPILE)

# compile pure C++ files
dbdata.o : dbdata.cpp dbdata.hpp mapped_file.hpp sequence_io.h length_partitions.hpp
	$(COMPILE)

# compile pure C++ files
makedb.o : makedb.cpp dbdata.hpp sequence_io.h
	$(COMPILE)


