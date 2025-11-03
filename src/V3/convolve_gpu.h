#ifndef _CONVOLVE_GPU_H_
#define _CONVOLVE_GPU_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "base.h"

// GPU stride multiple for warp-friendly memory access
#ifndef KLT_GPU_STRIDE_MULT
#define KLT_GPU_STRIDE_MULT 32
#endif

// Helper function to round up to nearest multiple
static inline int round_up(int x, int multiple)
{
  return ((x + multiple - 1) / multiple) * multiple;
}

// Shared kernel state variables (extern declarations)
extern ConvolutionKernel gauss_kernel;
extern ConvolutionKernel gaussderiv_kernel;
extern float sigma_last;

// Shared function declarations
void _computeKernels(
  float sigma,
  ConvolutionKernel *gauss,
  ConvolutionKernel *gaussderiv);

void _convolveSeparateGPU_DeviceToDevice(
  float *d_in,           // Device pointer input
  int ncols, int nrows,
  int stride,            // Stride for d_in
  ConvolutionKernel horizontalWindow,
  ConvolutionKernel verticalWindow,
  float *d_out,          // Device pointer output
  int out_stride);       // Stride for d_out

#ifdef __cplusplus
}
#endif

#endif /* _CONVOLVE_GPU_H_ */

