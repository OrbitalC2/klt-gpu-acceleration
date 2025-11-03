#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include "cudaErrors.h"

extern "C"
{
#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt_util.h"
#include "convolve_gpu.h"
}

// Tile size for shared memory
#define TILE_WIDTH 32
#define TILE_HEIGHT 8
#define SMEM_PADDING 8

// Constant memory for convolution coefficients
__constant__ float cWindow[MAX_KERNEL_WIDTH];

// Kernel cache to avoid redundant cudaMemcpyToSymbol calls
static float cached_kernel[MAX_KERNEL_WIDTH];
static int cached_kernel_width = -1;

// Defined in convolve.c
extern "C" {
extern ConvolutionKernel gauss_kernel;
extern ConvolutionKernel gaussderiv_kernel;
extern float sigma_last;
}

// ============================================================================
// HELPER FUNCTION - Copy kernel to constant memory only if changed
// ============================================================================
static void _copyKernelToConstantMemory(const float* kernel_data, int width)
{
    if (width == cached_kernel_width && 
        memcmp(kernel_data, cached_kernel, width * sizeof(float)) == 0) {
        return;
    }
    
    CUDA_CHECK(cudaMemcpyToSymbol(cWindow, kernel_data, 
                                  width * sizeof(float), 
                                  0, cudaMemcpyHostToDevice));
    
    memcpy(cached_kernel, kernel_data, width * sizeof(float));
    cached_kernel_width = width;
}

// ============================================================================
// GPU KERNELS
// ============================================================================

__global__ void ConvolveHorizKernel(const float *__restrict__ in,
                                     float *__restrict__ out,
                                     int ncols, int nrows, int stride, int kWidth)
{
    extern __shared__ float tile[];  // Dynamic shared memory
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int radius = kWidth >> 1;
    
    // Calculate row stride for 2D indexing in 1D array
    int rowStride = blockDim.x + 2 * radius + SMEM_PADDING;
    
    // Load tile into shared memory with halo regions
    int haloLeft = blockIdx.x * blockDim.x - radius;
    int haloWidth = blockDim.x + 2 * radius;
    
    // Each thread loads multiple elements if needed
    for (int i = tx; i < haloWidth; i += blockDim.x) {
        int loadX = haloLeft + i;
        int smemX = i;
        
        if (y < nrows) {
            if (loadX >= 0 && loadX < ncols) {
                tile[ty * rowStride + smemX] = in[y * stride + loadX];
            } else {
                tile[ty * rowStride + smemX] = 0.0f;
            }
        }
    }
    
    __syncthreads();
    
    if (x >= ncols || y >= nrows)
        return;
    
    float v = 0.0f;
    
    if (x < radius || x >= (ncols - radius)) {
        v = 0.0f;
    }
    else {
        int smemStart = tx + radius;
        int spatialIdx = 0;
        
        // Apply kernel in REVERSE order to match CPU
        for (int k = kWidth - 1; k >= 0; --k) {
            v += tile[ty * rowStride + smemStart + spatialIdx - radius] * cWindow[k];
            spatialIdx++;
        }
    }
    
    out[y * stride + x] = v;
}

__global__ void ConvolveVertKernel(const float *__restrict__ in,
                                    float *__restrict__ out,
                                    int ncols, int nrows, int stride, int kWidth)
{
    extern __shared__ float tile[];  // Dynamic shared memory
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int radius = kWidth >> 1;
    
    // Calculate row stride for 2D indexing in 1D array
    int rowStride = blockDim.x + SMEM_PADDING;
    
    // Load tile into shared memory with halo regions
    int haloTop = blockIdx.y * blockDim.y - radius;
    int haloHeight = blockDim.y + 2 * radius;
    
    // Each thread loads multiple elements if needed
    for (int i = ty; i < haloHeight; i += blockDim.y) {
        int loadY = haloTop + i;
        int smemY = i;
        
        if (x < ncols) {
            if (loadY >= 0 && loadY < nrows) {
                tile[smemY * rowStride + tx] = in[loadY * stride + x];
            } else {
                tile[smemY * rowStride + tx] = 0.0f;
            }
        }
    }
    
    __syncthreads();
    
    if (x >= ncols || y >= nrows)
        return;
    
    float v = 0.0f;
    
    if (y < radius || y >= (nrows - radius)) {
        v = 0.0f;
    }
    else {
        int smemStart = ty + radius;
        int spatialIdx = 0;
        
        // Apply kernel in REVERSE order to match CPU
        for (int k = kWidth - 1; k >= 0; --k) {
            v += tile[(smemStart + spatialIdx - radius) * rowStride + tx] * cWindow[k];
            spatialIdx++;
        }
    }
    
    out[y * stride + x] = v;
}

// ============================================================================
// MAIN GPU CONVOLUTION FUNCTION
// ============================================================================

extern "C"
void _convolveSeparateGPU_DeviceToDevice(
  float *d_in, int ncols, int nrows, int in_stride,
  ConvolutionKernel horizontalWindow,
  ConvolutionKernel verticalWindow,
  float *d_out, int out_stride)
{
  assert(horizontalWindow.width % 2 == 1);
  assert(verticalWindow.width % 2 == 1);
  assert(horizontalWindow.width <= MAX_KERNEL_WIDTH);
  assert(verticalWindow.width <= MAX_KERNEL_WIDTH);

  // Both kernels take a single stride, so d_tmp uses the input stride
  float *d_tmp = nullptr;
  CUDA_CHECK(cudaMalloc(&d_tmp, in_stride * nrows * sizeof(float)));

  // Launch configuration
  dim3 block(TILE_WIDTH, TILE_HEIGHT);
  dim3 grid((ncols + block.x - 1) / block.x, 
            (nrows + block.y - 1) / block.y);

  int hRadius = horizontalWindow.width >> 1;
  int horizSharedMemSize = TILE_HEIGHT * (TILE_WIDTH + 2 * hRadius + SMEM_PADDING) * sizeof(float);
  
  _copyKernelToConstantMemory(horizontalWindow.data, horizontalWindow.width);
  
  ConvolveHorizKernel<<<grid, block, horizSharedMemSize>>>(d_in, d_tmp, ncols, nrows, in_stride, 
                                       horizontalWindow.width);
  CUDA_CHECK(cudaGetLastError());

  assert(in_stride == out_stride && "Stride mismatch not supported by current kernel design");
  int vRadius = verticalWindow.width >> 1;
  int vertSharedMemSize = (TILE_HEIGHT + 2 * vRadius) * (TILE_WIDTH + SMEM_PADDING) * sizeof(float);
  
  _copyKernelToConstantMemory(verticalWindow.data, verticalWindow.width);
  
  ConvolveVertKernel<<<grid, block, vertSharedMemSize>>>(d_tmp, d_out, ncols, nrows, in_stride, 
                                      verticalWindow.width);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaFree(d_tmp));
}

// GPU wrapper for pyramid operations (with H2D/D2H transfers)
static void _convolveSeparateGPU(
  _KLT_FloatImage imgin,
  ConvolutionKernel horizontalWindow,
  ConvolutionKernel verticalWindow,
  _KLT_FloatImage imgout)
{
  assert(horizontalWindow.width % 2 == 1);
  assert(verticalWindow.width % 2 == 1);
  assert(imgin != imgout);
  assert(imgout->ncols >= imgin->ncols);
  assert(imgout->nrows >= imgin->nrows);

  const int ncols = imgin->ncols;
  const int nrows = imgin->nrows;

  _KLTSyncToDevice(imgin);

  if (imgout->d_data == NULL) {
    _KLTAllocFloatImageGPU(imgout);
  }

  _convolveSeparateGPU_DeviceToDevice(
    imgin->d_data, ncols, nrows, imgin->d_stride,
    horizontalWindow, verticalWindow,
    imgout->d_data, imgout->d_stride
  );

  _KLTMarkDeviceValid(imgout);
}
// ============================================================================
// GPU versions of gradient/smoothing functions for pyramid operations
// ============================================================================

extern "C" {

void _KLTComputeGradients_GPU(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage gradx,
  _KLT_FloatImage grady)
{
  assert(gradx->ncols >= img->ncols);
  assert(gradx->nrows >= img->nrows);
  assert(grady->ncols >= img->ncols);
  assert(grady->nrows >= img->nrows);

  // Compute kernels, if necessary
  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

  // Use GPU convolution
  _convolveSeparateGPU(img, gaussderiv_kernel, gauss_kernel, gradx);
  _convolveSeparateGPU(img, gauss_kernel, gaussderiv_kernel, grady);
}

void _KLTComputeSmoothedImage_GPU(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage smooth)
{
  assert(smooth->ncols >= img->ncols);
  assert(smooth->nrows >= img->nrows);

  // Compute kernel, if necessary
  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

  // Use GPU convolution
  _convolveSeparateGPU(img, gauss_kernel, gauss_kernel, smooth);
}

} // extern "C"