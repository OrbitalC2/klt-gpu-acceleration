#ifndef CUDA_ERRORS_H
#define CUDA_ERRORS_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

inline void cudaCheck(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA ERROR: %s:%d: %s\n", file, line, cudaGetErrorString(err));
        abort();
    }
}
#define CUDA_CHECK(call) cudaCheck((call), __FILE__, __LINE__)

#endif 
