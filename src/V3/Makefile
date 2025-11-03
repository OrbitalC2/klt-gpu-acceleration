# Simple Makefile for KLT with CUDA
CC = gcc
NVCC = nvcc
CFLAGS = -O2 -Wall -g -I.
NVCCFLAGS = -O2 -arch=sm_86 -I.
LIBS = -lm -L/usr/local/cuda/lib64 -lcudart

# Common C source files (used by both CPU and GPU builds)
C_SOURCES = example3.c klt.c pnmio.c pyramid.c \
            storeFeatures.c trackFeatures.c klt_util.c \
            writeFeatures.c error.c convolve.c

# CUDA source files (GPU build only)
CU_SOURCES = convolve.cu selectGoodFeatures.cu klt_util_gpu.cu trackFeatures_gpu.cu

# Object files
C_OBJECTS_GPU = $(C_SOURCES:.c=.o)
C_OBJECTS_CPU = $(C_SOURCES:.c=.o_cpu)
CU_OBJECTS = convolve_cuda.o selectGoodFeatures_cuda.o klt_util_gpu_cuda.o trackFeatures_gpu_cuda.o

OBJECTS_GPU = $(C_OBJECTS_GPU) $(CU_OBJECTS)
OBJECTS_CPU = $(C_OBJECTS_CPU) selectGoodFeatures_cpu.o

TARGET_GPU = example3_gpu
TARGET_CPU = example3_cpu

# Default target: build both versions
all: $(TARGET_GPU) $(TARGET_CPU)

# Simple individual build targets
gpu: $(TARGET_GPU)
cpu: $(TARGET_CPU)

# Compile CUDA files with NVCC
convolve_cuda.o: convolve.cu
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA -c convolve.cu -o convolve_cuda.o

selectGoodFeatures_cuda.o: selectGoodFeatures.cu
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA -c selectGoodFeatures.cu -o selectGoodFeatures_cuda.o

klt_util_gpu_cuda.o: klt_util_gpu.cu
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA -c klt_util_gpu.cu -o klt_util_gpu_cuda.o

trackFeatures_gpu_cuda.o: trackFeatures_gpu.cu
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA -c trackFeatures_gpu.cu -o trackFeatures_gpu_cuda.o

# Compile selectGoodFeatures for CPU build (no CUDA)
selectGoodFeatures_cpu.o: selectGoodFeatures.cu
	$(CC) $(CFLAGS) -x c++ -c selectGoodFeatures.cu -o selectGoodFeatures_cpu.o

# Compile C files for GPU build (with USE_CUDA defined)
%.o: %.c
	$(CC) $(CFLAGS) -DUSE_CUDA -c $< -o $@

# Compile C files for CPU build (no USE_CUDA)
%.o_cpu: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Link GPU version
$(TARGET_GPU): $(OBJECTS_GPU)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LIBS)

# Link CPU version
$(TARGET_CPU): $(OBJECTS_CPU)
	$(CC) $(CFLAGS) -o $@ $^ -lm

# Run targets
run_gpu: $(TARGET_GPU)
	./$(TARGET_GPU)

run_cpu: $(TARGET_CPU)
	./$(TARGET_CPU)

run: run_gpu

# Profile GPU version
profile: $(TARGET_GPU)
	nsys profile --stats=true --force-overwrite=true \
	    --cuda-memory-usage=true \
	    --trace=cuda,nvtx \
	    -o klt_profile ./$(TARGET_GPU)
	@echo ""
	@echo "Profile saved to klt_profile.nsys-rep"

clean:
	rm -f *.o *.o_cpu $(TARGET_GPU) $(TARGET_CPU) *.ppm features.* *.nsys-rep *.sqlite gmon.out gprof.txt

.PHONY: all clean run run_gpu run_cpu profile gpu cpu
