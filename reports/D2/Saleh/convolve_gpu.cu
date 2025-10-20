// convolve_gpu.cu
// Naive GPU Implementation (V2) for Separable Convolution.

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>
#include "klt_util.h"
#include "convolve.h"
#include "convolve_gpu.h"

// Macro for error checking
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

inline void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s in %s at line %d: %s\n", func, file, line, cudaGetErrorString(err));

    }
}

// ====================================================================
// KERNEL 1: Horizontal Convolution
// Assigns one thread to compute the output value of one pixel (i, j).
// This pass is memory-friendly due to coalesced access along rows.
// ====================================================================

__global__ void gpu_convolveImageHoriz(
    const float *d_imgin,
    float *d_imgout,
    int ncols,
    int nrows,
    const float *d_kernel,
    int kernel_width)
{
    // Calculate 2D index (i=column, j=row) from 1D thread index
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check for bounds
    if (idx >= ncols * nrows) return;
    
    const int j = idx / ncols; // row index
    const int i = idx % ncols; // column index

    const int radius = kernel_width / 2;
    float sum = 0.0f;

    // Boundary Handling (Zero Padding)
    if (i < radius || i >= ncols - radius) {
        d_imgout[idx] = 0.0f;
        return;
    }

    // Actual Convolution
    // Replicates the CPU's pattern: image pixels accessed forward (i - radius + k),
    // kernel weights accessed backward (kernel_width - 1 - k).
    sum = 0.0f;
    for (int k = 0; k < kernel_width; k++) {
        // Input pixel index (j * ncols) + (i - radius + k)
        int input_pixel_index = j * ncols + (i - radius + k);
        
        // Kernel index for reverse access
        int kernel_index = kernel_width - 1 - k;
        
        sum += d_imgin[input_pixel_index] * d_kernel[kernel_index];
    }
    
    d_imgout[idx] = sum;
}


// ====================================================================
// KERNEL 2: Vertical Convolution
// Assigns one thread to compute the output value of one pixel (i, j).
// This pass is non-coalesced due to strided access along columns (V2 weakness).
// ====================================================================

__global__ void gpu_convolveImageVert(
    const float *d_imgin,
    float *d_imgout,
    int ncols,
    int nrows,
    const float *d_kernel,
    int kernel_width)
{
    // Calculate 2D index (i=column, j=row) from 1D thread index
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check for bounds
    if (idx >= ncols * nrows) return;
    
    const int j = idx / ncols; // row index
    const int i = idx % ncols; // column index

    const int radius = kernel_width / 2;
    float sum = 0.0f;

    // Boundary Handling (Zero Padding)
    if (j < radius || j >= nrows - radius) {
        d_imgout[idx] = 0.0f;
        return;
    }

    // Actual Convolution (Vertical access)
    for (int k = 0; k < kernel_width; k++) {
        // Pixel index in the input image: (j - radius + k) * ncols + i
        int input_pixel_index = (j - radius + k) * ncols + i;
        
        // d_kernel[kernel_width - 1 - k] matches the k-th element accessed in the CPU's reversed kernel loop
        sum += d_imgin[input_pixel_index] * d_kernel[kernel_width - 1 - k];
    }
    
    d_imgout[idx] = sum;
}


// ====================================================================
// HOST WRAPPER: KLTConvolveSeparate_GPU
// Handles memory management and kernel launching.
// ====================================================================

void KLTConvolveSeparate_GPU(
    _KLT_FloatImage imgin,
    ConvolutionKernel horiz_kernel,
    ConvolutionKernel vert_kernel,
    _KLT_FloatImage imgout)
{
    // Dimensions
    const int N = imgin->ncols * imgin->nrows;
    const int ncols = imgin->ncols;
    const int nrows = imgin->nrows;
    const int kernel_width_h = horiz_kernel.width;
    const int kernel_width_v = vert_kernel.width;

    // --- Device Pointers ---
    float *d_imgin, *d_tmpimg, *d_imgout;
    float *d_kernel_h, *d_kernel_v;
    
    // --- Launch Configuration (Naive V2) ---
    // One thread per pixel. 256 threads per block is a safe and typical choice.
    const int THREADS_PER_BLOCK = 256;
    const int NUM_BLOCKS = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // 1. Allocate Device Memory (Images)
    CHECK_CUDA_ERROR( cudaMalloc((void**)&d_imgin,  N * sizeof(float)) );
    CHECK_CUDA_ERROR( cudaMalloc((void**)&d_tmpimg, N * sizeof(float)) ); // Intermediate result
    CHECK_CUDA_ERROR( cudaMalloc((void**)&d_imgout, N * sizeof(float)) );

    // 2. Allocate Device Memory (Kernels)
    CHECK_CUDA_ERROR( cudaMalloc((void**)&d_kernel_h, kernel_width_h * sizeof(float)) );
    CHECK_CUDA_ERROR( cudaMalloc((void**)&d_kernel_v, kernel_width_v * sizeof(float)) );
    
    // 3. Transfer Data H2D (Input Image and Kernels)
    CHECK_CUDA_ERROR( cudaMemcpy(d_imgin, imgin->data, N * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUDA_ERROR( cudaMemcpy(d_kernel_h, horiz_kernel.data, kernel_width_h * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUDA_ERROR( cudaMemcpy(d_kernel_v, vert_kernel.data, kernel_width_v * sizeof(float), cudaMemcpyHostToDevice) );

    // --- PASS 1: Horizontal Convolution (Input: d_imgin, Output: d_tmpimg) ---
    gpu_convolveImageHoriz<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
        d_imgin, d_tmpimg, ncols, nrows, d_kernel_h, kernel_width_h
    );
    CHECK_CUDA_ERROR( cudaGetLastError() );
    // Ensure horizontal kernel completes before vertical starts (implicit sync after kernel launch)
    CHECK_CUDA_ERROR( cudaDeviceSynchronize() );


    // --- PASS 2: Vertical Convolution (Input: d_tmpimg, Output: d_imgout) ---
    // Note: The intermediate data d_tmpimg stays on the device. This is a crucial
    // communication efficiency step compared to a naive CPU-to-CPU implementation.
    gpu_convolveImageVert<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
        d_tmpimg, d_imgout, ncols, nrows, d_kernel_v, kernel_width_v
    );
    CHECK_CUDA_ERROR( cudaGetLastError() );
    CHECK_CUDA_ERROR( cudaDeviceSynchronize() );


    // 4. Transfer Data D2H (Output Image)
    CHECK_CUDA_ERROR( cudaMemcpy(imgout->data, d_imgout, N * sizeof(float), cudaMemcpyDeviceToHost) );

    // 5. Cleanup Device Memory
    CHECK_CUDA_ERROR( cudaFree(d_imgin) );
    CHECK_CUDA_ERROR( cudaFree(d_tmpimg) );
    CHECK_CUDA_ERROR( cudaFree(d_imgout) );
    CHECK_CUDA_ERROR( cudaFree(d_kernel_h) );
    CHECK_CUDA_ERROR( cudaFree(d_kernel_v) );
}
