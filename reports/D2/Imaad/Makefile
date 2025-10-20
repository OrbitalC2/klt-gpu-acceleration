# V2 Makefile with CUDA support
CC = gcc
NVCC = nvcc
CFLAGS = -O2 -Wall -g
NVCCFLAGS = -O2 -arch=sm_86  # RTX 3080 architecture
LIBS = -lm -L/usr/local/cuda/lib64 -lcudart

# Source files
C_SOURCES = example3.c klt.c pnmio.c pyramid.c selectGoodFeatures.c \
            trackFeatures.c klt_util.c storeFeatures.c writeFeatures.c error.c
CU_SOURCES = convolve_gpu.cu
OBJECTS = $(C_SOURCES:.c=.o) $(CU_SOURCES:.cu=.o)

TARGET = example3_gpu

# Compile CUDA files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile C files
%.o: %.c
	$(CC) $(CFLAGS) -DUSE_GPU -I/usr/local/cuda/include -c $< -o $@

# Link everything
$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LIBS)

# CPU version for comparison
cpu: $(C_SOURCES)
	$(CC) $(CFLAGS) -o example3_cpu $(C_SOURCES) -lm

clean:
	rm -f $(OBJECTS) $(TARGET) example3_cpu *.ppm features.*

run: $(TARGET)
	time ./$(TARGET)

run_cpu: cpu
	time ./example3_cpu

.PHONY: clean run run_cpu
