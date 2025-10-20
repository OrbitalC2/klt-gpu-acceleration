#ifndef CONVOLVE_GPU_H
#define CONVOLVE_GPU_H

#include "klt_util.h"
#include "convolve.h"

#ifndef CONVOLUTION_KERNEL_DEFINED
#define CONVOLUTION_KERNEL_DEFINED

#ifndef MAX_KERNEL_WIDTH
#define MAX_KERNEL_WIDTH 71
#endif

typedef struct {
    int width;
    float data[MAX_KERNEL_WIDTH];
} ConvolutionKernel;

#endif /* CONVOLUTION_KERNEL_DEFINED */

/* GPU convolution interface */
void KLTConvolveSeparate_GPU(
    _KLT_FloatImage imgin,
    ConvolutionKernel horiz_kernel,
    ConvolutionKernel vert_kernel,
    _KLT_FloatImage imgout);

#endif /* CONVOLVE_GPU_H */
