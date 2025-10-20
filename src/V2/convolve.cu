#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include "cudaErrors.h"

extern "C"
{
#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt_util.h"
}

// GPU stride multiple for warp-friendly memory access
#ifndef KLT_GPU_STRIDE_MULT
#define KLT_GPU_STRIDE_MULT 32
#endif

// Constant memory for convolution coefficients
__constant__ float cWindow[MAX_KERNEL_WIDTH];

// Static kernels (same as CPU version)
static ConvolutionKernel gauss_kernel;
static ConvolutionKernel gaussderiv_kernel;
static float sigma_last = -10.0;

// Global timing accumulators
static float total_kernel_time = 0.0f;
static float total_gradients_time = 0.0f;
static float total_smoothed_time = 0.0f;
static int gradients_call_count = 0;
static int smoothed_call_count = 0;

// ============================================================================
// GPU KERNELS
// ============================================================================

__global__ void ConvolveHorizKernel(const float *__restrict__ in,
                                     float *__restrict__ out,
                                     int ncols, int nrows, int stride, int kWidth)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= ncols || y >= nrows)
        return;
    
    int radius = kWidth >> 1;
    float v = 0.0f;
    
    if (x < radius || x >= (ncols - radius)) {
        v = 0.0f;
    }
    else {
        const int base = y * stride;
        int spatialIdx = x - radius;        
        // Apply kernel in REVERSE order to match CPU
        for (int k = kWidth - 1; k >= 0; --k) {
            v += in[base + spatialIdx] * cWindow[k];
            spatialIdx++;        }
    }
    
    out[y * stride + x] = v;
}

__global__ void ConvolveVertKernel(const float *__restrict__ in,
                                    float *__restrict__ out,
                                    int ncols, int nrows, int stride, int kWidth)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= ncols || y >= nrows)
        return;
    
    int radius = kWidth >> 1;
    float v = 0.0f;
    
    if (y < radius || y >= (nrows - radius)) {
        v = 0.0f;
    }
    else {
        int spatialIdx = y - radius;        
        // Apply kernel in REVERSE order to match CPU
        for (int k = kWidth - 1; k >= 0; --k) {
            v += in[spatialIdx * stride + x] * cWindow[k];
            spatialIdx++;        }
    }
    
    out[y * stride + x] = v;
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

static inline int round_up(int x, int multiple)
{
  return ((x + multiple - 1) / multiple) * multiple;
}

// ============================================================================
// MAIN GPU CONVOLUTION FUNCTION
// ============================================================================

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
  assert(horizontalWindow.width <= MAX_KERNEL_WIDTH);
  assert(verticalWindow.width <= MAX_KERNEL_WIDTH);

  const int ncols = imgin->ncols;
  const int nrows = imgin->nrows;

  // Round up the row length to a warp-friendly multiple
  const int stride = round_up(ncols, KLT_GPU_STRIDE_MULT);
  const int pitchBytes = stride * sizeof(float);

  // Allocate device memory
  float *d_in = nullptr;
  float *d_tmp = nullptr;
  float *d_out = nullptr;

  CUDA_CHECK(cudaMalloc(&d_in, stride * nrows * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_tmp, stride * nrows * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, stride * nrows * sizeof(float)));

  // Copy input to device
  CUDA_CHECK(cudaMemcpy2D(d_in, pitchBytes, 
                          imgin->data, ncols * sizeof(float), 
                          ncols * sizeof(float), nrows, 
                          cudaMemcpyHostToDevice));

  // Launch configuration
  dim3 block(32, 8);
  dim3 grid((ncols + block.x - 1) / block.x, 
            (nrows + block.y - 1) / block.y);

  // ===== START KERNEL TIMER =====
  cudaEvent_t kernelStart, kernelStop;
  cudaEventCreate(&kernelStart);
  cudaEventCreate(&kernelStop);
  cudaEventRecord(kernelStart);

  // PASS 1: Horizontal convolution
  CUDA_CHECK(cudaMemcpyToSymbol(cWindow, horizontalWindow.data, 
                                horizontalWindow.width * sizeof(float), 
                                0, cudaMemcpyHostToDevice));
  ConvolveHorizKernel<<<grid, block>>>(d_in, d_tmp, ncols, nrows, stride, 
                                       horizontalWindow.width);
  CUDA_CHECK(cudaGetLastError());

  // PASS 2: Vertical convolution
  CUDA_CHECK(cudaMemcpyToSymbol(cWindow, verticalWindow.data, 
                                verticalWindow.width * sizeof(float), 
                                0, cudaMemcpyHostToDevice));
  ConvolveVertKernel<<<grid, block>>>(d_tmp, d_out, ncols, nrows, stride, 
                                      verticalWindow.width);
  CUDA_CHECK(cudaGetLastError());

  // ===== END KERNEL TIMER =====
  cudaEventRecord(kernelStop);
  cudaEventSynchronize(kernelStop);
  float kernelTime = 0;
  cudaEventElapsedTime(&kernelTime, kernelStart, kernelStop);
  total_kernel_time += kernelTime;
  
  cudaEventDestroy(kernelStart);
  cudaEventDestroy(kernelStop);

  // Copy result back to host
  CUDA_CHECK(cudaMemcpy2D(imgout->data, ncols * sizeof(float), 
                          d_out, pitchBytes, 
                          ncols * sizeof(float), nrows, 
                          cudaMemcpyDeviceToHost));

  // Cleanup
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_tmp));
  CUDA_CHECK(cudaFree(d_out));
}

// ============================================================================
// CPU UTILITY FUNCTIONS (from convolve.c) - Wrapped in extern "C"
// ============================================================================

extern "C" {

void _KLTToFloatImage(
  KLT_PixelType *img,
  int ncols, int nrows,
  _KLT_FloatImage floatimg)
{
  KLT_PixelType *ptrend = img + ncols*nrows;
  float *ptrout = floatimg->data;

  assert(floatimg->ncols >= ncols);
  assert(floatimg->nrows >= nrows);

  floatimg->ncols = ncols;
  floatimg->nrows = nrows;

  while (img < ptrend)
    *ptrout++ = (float) *img++;
}

static void _computeKernels(
  float sigma,
  ConvolutionKernel *gauss,
  ConvolutionKernel *gaussderiv)
{
  const float factor = 0.01f;
  int i;

  assert(MAX_KERNEL_WIDTH % 2 == 1);
  assert(sigma >= 0.0);

  // Compute kernels, and automatically determine widths
  {
    const int hw = MAX_KERNEL_WIDTH / 2;
    float max_gauss = 1.0f;
    float max_gaussderiv = (float)(sigma * exp(-0.5f));

    // Compute gauss and deriv
    for (i = -hw; i <= hw; i++) {
      gauss->data[i+hw] = (float)exp(-i*i / (2*sigma*sigma));
      gaussderiv->data[i+hw] = -i * gauss->data[i+hw];
    }

    // Compute widths
    gauss->width = MAX_KERNEL_WIDTH;
    for (i = -hw; fabs(gauss->data[i+hw] / max_gauss) < factor; 
         i++, gauss->width -= 2);
    
    gaussderiv->width = MAX_KERNEL_WIDTH;
    for (i = -hw; fabs(gaussderiv->data[i+hw] / max_gaussderiv) < factor; 
         i++, gaussderiv->width -= 2);
    
    if (gauss->width == MAX_KERNEL_WIDTH || 
        gaussderiv->width == MAX_KERNEL_WIDTH)
      KLTError("(_computeKernels) MAX_KERNEL_WIDTH %d is too small for "
               "a sigma of %f", MAX_KERNEL_WIDTH, sigma);
  }

  // Shift if width less than MAX_KERNEL_WIDTH
  for (i = 0; i < gauss->width; i++)
    gauss->data[i] = gauss->data[i+(MAX_KERNEL_WIDTH-gauss->width)/2];
  
  for (i = 0; i < gaussderiv->width; i++)
    gaussderiv->data[i] = gaussderiv->data[i+(MAX_KERNEL_WIDTH-gaussderiv->width)/2];
  
  // Normalize gauss and deriv
  {
    const int hw = gaussderiv->width / 2;
    float den;

    den = 0.0;
    for (i = 0; i < gauss->width; i++)
      den += gauss->data[i];
    for (i = 0; i < gauss->width; i++)
      gauss->data[i] /= den;
    
    den = 0.0;
    for (i = -hw; i <= hw; i++)
      den -= i * gaussderiv->data[i+hw];
    for (i = -hw; i <= hw; i++)
      gaussderiv->data[i+hw] /= den;
  }

  sigma_last = sigma;
}

void _KLTGetKernelWidths(
  float sigma,
  int *gauss_width,
  int *gaussderiv_width)
{
  _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
  *gauss_width = gauss_kernel.width;
  *gaussderiv_width = gaussderiv_kernel.width;
}

void _KLTComputeGradients(
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

  // ===== START TOTAL TIMER =====
  cudaEvent_t totalStart, totalStop;
  cudaEventCreate(&totalStart);
  cudaEventCreate(&totalStop);
  cudaEventRecord(totalStart);

  // Use GPU convolution
  _convolveSeparateGPU(img, gaussderiv_kernel, gauss_kernel, gradx);
  _convolveSeparateGPU(img, gauss_kernel, gaussderiv_kernel, grady);

  // ===== END TOTAL TIMER =====
  cudaEventRecord(totalStop);
  cudaEventSynchronize(totalStop);
  float totalTime = 0;
  cudaEventElapsedTime(&totalTime, totalStart, totalStop);
  total_gradients_time += totalTime;
  gradients_call_count++;
  
  cudaEventDestroy(totalStart);
  cudaEventDestroy(totalStop);
}

void _KLTComputeSmoothedImage(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage smooth)
{
  assert(smooth->ncols >= img->ncols);
  assert(smooth->nrows >= img->nrows);

  // Compute kernel, if necessary
  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

  // ===== START TOTAL TIMER =====
  cudaEvent_t totalStart, totalStop;
  cudaEventCreate(&totalStart);
  cudaEventCreate(&totalStop);
  cudaEventRecord(totalStart);

  // Use GPU convolution
  _convolveSeparateGPU(img, gauss_kernel, gauss_kernel, smooth);

  // ===== END TOTAL TIMER =====
  cudaEventRecord(totalStop);
  cudaEventSynchronize(totalStop);
  float totalTime = 0;
  cudaEventElapsedTime(&totalTime, totalStart, totalStop);
  total_smoothed_time += totalTime;
  smoothed_call_count++;
  
  cudaEventDestroy(totalStart);
  cudaEventDestroy(totalStop);
}

// Function to print accumulated timing statistics
void _KLTPrintGPUTimingStats()
{
  printf("\n========================================\n");
  printf("GPU Timing Statistics (Accumulated)\n");
  printf("========================================\n");
  printf("Total kernel time: %.3f ms\n", total_kernel_time);
  printf("Total _KLTComputeGradients time: %.3f ms (%d calls)\n", 
         total_gradients_time, gradients_call_count);
  printf("Total _KLTComputeSmoothedImage time: %.3f ms (%d calls)\n", 
         total_smoothed_time, smoothed_call_count);
  printf("Total GPU time: %.3f ms\n", 
         total_gradients_time + total_smoothed_time);
  printf("Memory/overhead time: %.3f ms\n", 
         (total_gradients_time + total_smoothed_time) - total_kernel_time);
  printf("========================================\n");
}

// Function to reset timing statistics
void _KLTResetGPUTimingStats()
{
  total_kernel_time = 0.0f;
  total_gradients_time = 0.0f;
  total_smoothed_time = 0.0f;
  gradients_call_count = 0;
  smoothed_call_count = 0;
}

} // extern "C"