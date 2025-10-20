# Simple Makefile for KLT with CUDA
CC = gcc
NVCC = nvcc
CFLAGS = -O2 -Wall -g -I.
NVCCFLAGS = -O2 -arch=sm_86 -I. # adjust arch for your GPU
LIBS = -lm -L/usr/local/cuda/lib64 -lcudart

# Source files - NOTE: convolve.c is REMOVED for GPU version
C_SOURCES_GPU = example3.c klt.c pnmio.c pyramid.c selectGoodFeatures.c \
 storeFeatures.c trackFeatures.c klt_util.c writeFeatures.c error.c

# All sources for CPU version
C_SOURCES_CPU = example3.c klt.c pnmio.c pyramid.c selectGoodFeatures.c \
 storeFeatures.c trackFeatures.c klt_util.c writeFeatures.c error.c convolve.c

CU_SOURCES = convolve.cu
C_OBJECTS_GPU = $(C_SOURCES_GPU:.c=.o)
CU_OBJECTS = convolve_cuda.o
OBJECTS_GPU = $(C_OBJECTS_GPU) $(CU_OBJECTS)

TARGET = example3_gpu

# Compile CUDA files
convolve_cuda.o: convolve.cu
	$(NVCC) $(NVCCFLAGS) -c convolve.cu -o convolve_cuda.o

# Compile C files
%.o: %.c
	$(CC) $(CFLAGS) -DUSE_GPU -c $< -o $@

# Link everything into GPU version (excluding convolve.o)
$(TARGET): $(OBJECTS_GPU)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LIBS)

# CPU-only version for comparison
cpu: $(C_SOURCES_CPU)
	$(CC) $(CFLAGS) -o example3_cpu $(C_SOURCES_CPU) -lm

clean:
	rm -f *.o $(TARGET) example3_cpu *.ppm features.* *.nsys-rep *.sqlite

run: $(TARGET)
	time ./$(TARGET)

run_cpu: cpu
	time ./example3_cpu

# Timing rules
time_gpu: $(TARGET)
	@echo "=== Running GPU version with timing ==="
	./$(TARGET)
	@echo ""

time_cpu: cpu
	@echo "=== Running CPU version with timing ==="
	./example3_cpu
	@echo ""

time_both: $(TARGET) cpu
	@echo "========================================="
	@echo "=== Running GPU version with timing ==="
	@echo "========================================="
	./$(TARGET)
	@echo ""
	@echo "========================================="
	@echo "=== Running CPU version with timing ==="
	@echo "========================================="
	./example3_cpu
	@echo ""
	@echo "========================================="
	@echo "=== Comparison Complete ==="
	@echo "========================================="

profile: $(TARGET)
	nsys profile --trace=cuda --stats=true ./$(TARGET)

profile_cpu:
	$(MAKE) clean
	$(MAKE) CFLAGS="$(CFLAGS) -pg -no-pie" cpu
	./example3_cpu
	gprof ./example3_cpu gmon.out > gprof.txt
	@echo "gprof report written to gprof.txt"

.PHONY: clean run run_cpu profile time_gpu time_cpu time_both