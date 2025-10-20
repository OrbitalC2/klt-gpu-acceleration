/*********************************************************************
 * convolve.h
 *********************************************************************/

#ifndef _CONVOLVE_H_
#define _CONVOLVE_H_

#include "klt.h"
#include "klt_util.h"

#ifndef MAX_KERNEL_WIDTH
#define MAX_KERNEL_WIDTH 71
#endif

/* Make this struct visible to both CPU and GPU builds */
typedef struct {
    int width;
    float data[MAX_KERNEL_WIDTH];
} ConvolutionKernel;

/* Shared function prototypes */
void _KLTToFloatImage(
  KLT_PixelType *img,
  int ncols, int nrows,
  _KLT_FloatImage floatimg);

void _KLTComputeGradients(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage gradx,
  _KLT_FloatImage grady);

void _KLTGetKernelWidths(
  float sigma,
  int *gauss_width,
  int *gaussderiv_width);

void _KLTComputeSmoothedImage(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage smooth);

#endif /* _CONVOLVE_H_ */


#ifndef CONVOLVE_GPU_H
#define CONVOLVE_GPU_H

#include "klt_util.h"

/* GPU convolution interface */
void KLTConvolveSeparate_GPU(
    _KLT_FloatImage imgin,
    ConvolutionKernel horiz_kernel,
    ConvolutionKernel vert_kernel,
    _KLT_FloatImage imgout);

#endif /* CONVOLVE_GPU_H */
