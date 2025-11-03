/*********************************************************************
 * convolve.h
 *********************************************************************/

#ifndef _CONVOLVE_H_
#define _CONVOLVE_H_

#include "klt.h"
#include "klt_util.h"

#define MAX_KERNEL_WIDTH 71

typedef struct {
  int   width;
  float data[MAX_KERNEL_WIDTH];
} ConvolutionKernel;

#ifdef __cplusplus
extern "C" {
#endif

void _KLTToFloatImage(
  KLT_PixelType *img,
  int ncols, int nrows,
  _KLT_FloatImage floatimg);

void _KLTComputeGradients(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage gradx,
  _KLT_FloatImage grady);

void _computeKernels(
  float sigma,
  ConvolutionKernel *gauss,
  ConvolutionKernel *gaussderiv);

void _KLTGetKernelWidths(
  float sigma,
  int *gauss_width,
  int *gaussderiv_width);

void _KLTComputeSmoothedImage(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage smooth);

void KLT_ConvolveSeparateCUDA(const _KLT_FloatImage imgin, 
  const ConvolutionKernel horizontalWindow, 
  const ConvolutionKernel verticalWindow, 
  _KLT_FloatImage imgout);

#ifdef __cplusplus
}
#endif

#endif
