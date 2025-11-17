######################################################################
# Choose your favorite C compiler
CC = nvc        

######################################################################
# Base Flags (Common to both)
BASE_FLAGS = -DNDEBUG -O3

# GPU Flags (OpenACC enabled)
GPU_FLAGS = -acc -Minfo=acc

# CPU Flags (OpenACC disabled explicitely)
CPU_FLAGS = -noacc

######################################################################
# Source files
EXAMPLES = example1.c example2.c example3.c example4.c example5.c
ARCH = convolve.c error.c pnmio.c pyramid.c selectGoodFeatures.c \
       storeFeatures.c trackFeatures.c klt.c klt_util.c writeFeatures.c
LIB = -L/usr/local/lib -L/usr/lib

.SUFFIXES: .c .o

# Default target
CFLAGS ?= $(BASE_FLAGS) $(GPU_FLAGS)

all: libklt.a $(EXAMPLES:.c=)

.c.o:
	$(CC) -c $(CFLAGS) $<

libklt.a: $(ARCH:.c=.o)
	rm -f libklt.a
	ar ruv libklt.a $(ARCH:.c=.o)
	rm -f *.o

example1: libklt.a
	$(CC) $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm

example2: libklt.a
	$(CC) $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm

example3: libklt.a
	$(CC) $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm

example4: libklt.a
	$(CC) $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm

example5: libklt.a
	$(CC) $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm

depend:
	makedepend $(ARCH) $(EXAMPLES)

clean:
	rm -f *.o *.a $(EXAMPLES:.c=) *.tar *.tar.gz libklt.a \
	      feat*.ppm features.ft features.txt cpu.time gpu.time

######################################################################
# INDIVIDUAL RUN COMMANDS
######################################################################

run_cpu:
	@echo ""
	@echo "   [CPU MODE]"
	@echo "   -> Cleaning previous builds..."
	@$(MAKE) clean > /dev/null 2>&1
	@echo "   -> Compiling with CPU flags (-noacc)..."
	@$(MAKE) example3 CFLAGS="$(BASE_FLAGS) $(CPU_FLAGS)" > /dev/null 2>&1
	@echo "   -> Running..."
	@echo "----------------------------------------"
	@time ./example3
	@echo "----------------------------------------"
	@echo "   -> Done."

run_gpu:
	@echo ""
	@echo "   [GPU MODE]"
	@echo "   -> Cleaning previous builds..."
	@$(MAKE) clean
	@echo "   -> Compiling library with GPU flags (-acc)..."
	@$(MAKE) libklt.a CFLAGS="$(BASE_FLAGS) $(GPU_FLAGS)"
	@echo "   -> Compiling example3 with GPU flags..."
	@$(MAKE) example3 CFLAGS="$(BASE_FLAGS) $(GPU_FLAGS)"
	@echo "   -> Running..."
	@echo "----------------------------------------"
	@time ./example3
	@echo "----------------------------------------"
	@echo "   -> Done."
