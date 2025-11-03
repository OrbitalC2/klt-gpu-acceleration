/*********************************************************************
 * convolve.c
 *********************************************************************/

/* Standard includes */
#include <assert.h>
#include <math.h>
#include <stdlib.h>   /* malloc(), realloc() */
#include <time.h>     /* clock() for timing */
#include <stdio.h>    /* printf() */

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt_util.h"   /* printing */


// #define MAX_KERNEL_WIDTH 	71 done in headewr

/* Kernels - non-static so they can be accessed from convolve.cu */
ConvolutionKernel gauss_kernel;
ConvolutionKernel gaussderiv_kernel;
float sigma_last = -10.0;

/* Global timing accumulators */
static double total_convolution_time = 0.0;
static double total_gradients_time = 0.0;
static double total_smoothed_time = 0.0;
static int gradients_call_count = 0;
static int smoothed_call_count = 0;


/*********************************************************************
 * _KLTToFloatImage
 *
 * Given a pointer to image data (probably unsigned chars), copy
 * data to a float image.
 */

void _KLTToFloatImage(
  KLT_PixelType *img,
  int ncols, int nrows,
  _KLT_FloatImage floatimg)
{
  KLT_PixelType *ptrend = img + ncols*nrows;
  float *ptrout = floatimg->data;

  /* Output image must be large enough to hold result */
  assert(floatimg->ncols >= ncols);
  assert(floatimg->nrows >= nrows);

  floatimg->ncols = ncols;
  floatimg->nrows = nrows;

  while (img < ptrend)  *ptrout++ = (float) *img++;
  
#ifdef USE_CUDA
  /* Host data was just updated, so mark host as valid and device as invalid */
  floatimg->h_valid = TRUE;
  floatimg->d_valid = FALSE;
#endif
}


/*********************************************************************
 * _computeKernels
 * Made non-static so it can be called from selectGoodFeatures.cu
 */

void _computeKernels(
  float sigma,
  ConvolutionKernel *gauss,
  ConvolutionKernel *gaussderiv)
{
  const float factor = 0.01f;   /* for truncating tail */
  int i;

  assert(MAX_KERNEL_WIDTH % 2 == 1);
  assert(sigma >= 0.0);

  /* Compute kernels, and automatically determine widths */
  {
    const int hw = MAX_KERNEL_WIDTH / 2;
    float max_gauss = 1.0f, max_gaussderiv = (float) (sigma*exp(-0.5f));
	
    /* Compute gauss and deriv */
    for (i = -hw ; i <= hw ; i++)  {
      gauss->data[i+hw]      = (float) exp(-i*i / (2*sigma*sigma));
      gaussderiv->data[i+hw] = -i * gauss->data[i+hw];
    }

    /* Compute widths */
    gauss->width = MAX_KERNEL_WIDTH;
    for (i = -hw ; fabs(gauss->data[i+hw] / max_gauss) < factor ; 
         i++, gauss->width -= 2);
    gaussderiv->width = MAX_KERNEL_WIDTH;
    for (i = -hw ; fabs(gaussderiv->data[i+hw] / max_gaussderiv) < factor ; 
         i++, gaussderiv->width -= 2);
    if (gauss->width == MAX_KERNEL_WIDTH || 
        gaussderiv->width == MAX_KERNEL_WIDTH)
      KLTError("(_computeKernels) MAX_KERNEL_WIDTH %d is too small for "
               "a sigma of %f", MAX_KERNEL_WIDTH, sigma);
  }

  /* Shift if width less than MAX_KERNEL_WIDTH */
  for (i = 0 ; i < gauss->width ; i++)
    gauss->data[i] = gauss->data[i+(MAX_KERNEL_WIDTH-gauss->width)/2];
  for (i = 0 ; i < gaussderiv->width ; i++)
    gaussderiv->data[i] = gaussderiv->data[i+(MAX_KERNEL_WIDTH-gaussderiv->width)/2];
  /* Normalize gauss and deriv */
  {
    const int hw = gaussderiv->width / 2;
    float den;
			
    den = 0.0;
    for (i = 0 ; i < gauss->width ; i++)  den += gauss->data[i];
    for (i = 0 ; i < gauss->width ; i++)  gauss->data[i] /= den;
    den = 0.0;
    for (i = -hw ; i <= hw ; i++)  den -= i*gaussderiv->data[i+hw];
    for (i = -hw ; i <= hw ; i++)  gaussderiv->data[i+hw] /= den;
  }

  sigma_last = sigma;
}
	

/*********************************************************************
 * _KLTGetKernelWidths
 *
 */

void _KLTGetKernelWidths(
  float sigma,
  int *gauss_width,
  int *gaussderiv_width)
{
  _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
  *gauss_width = gauss_kernel.width;
  *gaussderiv_width = gaussderiv_kernel.width;
}


/*********************************************************************
 * _convolveImageHoriz
 */

static void _convolveImageHoriz(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout)
{
  float *ptrrow = imgin->data;           /* Points to row's first pixel */
  register float *ptrout = imgout->data, /* Points to next output pixel */
    *ppp;
  register float sum;
  register int radius = kernel.width / 2;
  register int ncols = imgin->ncols, nrows = imgin->nrows;
  register int i, j, k;

  /* Kernel width must be odd */
  assert(kernel.width % 2 == 1);

  /* Must read from and write to different images */
  assert(imgin != imgout);

  /* Output image must be large enough to hold result */
  assert(imgout->ncols >= imgin->ncols);
  assert(imgout->nrows >= imgin->nrows);

  /* For each row, do ... */
  for (j = 0 ; j < nrows ; j++)  {

    /* Zero leftmost columns */
    for (i = 0 ; i < radius ; i++)
      *ptrout++ = 0.0;

    /* Convolve middle columns with kernel */
    for ( ; i < ncols - radius ; i++)  {
      ppp = ptrrow + i - radius;
      sum = 0.0;
      for (k = kernel.width-1 ; k >= 0 ; k--)
        sum += *ppp++ * kernel.data[k];
      *ptrout++ = sum;
    }

    /* Zero rightmost columns */
    for ( ; i < ncols ; i++)
      *ptrout++ = 0.0;

    ptrrow += ncols;
  }
}


/*********************************************************************
 * _convolveImageVert
 */

static void _convolveImageVert(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout)
{
  float *ptrcol = imgin->data;            /* Points to row's first pixel */
  register float *ptrout = imgout->data,  /* Points to next output pixel */
    *ppp;
  register float sum;
  register int radius = kernel.width / 2;
  register int ncols = imgin->ncols, nrows = imgin->nrows;
  register int i, j, k;

  /* Kernel width must be odd */
  assert(kernel.width % 2 == 1);

  /* Must read from and write to different images */
  assert(imgin != imgout);

  /* Output image must be large enough to hold result */
  assert(imgout->ncols >= imgin->ncols);
  assert(imgout->nrows >= imgin->nrows);

  /* For each column, do ... */
  for (i = 0 ; i < ncols ; i++)  {

    /* Zero topmost rows */
    for (j = 0 ; j < radius ; j++)  {
      *ptrout = 0.0;
      ptrout += ncols;
    }

    /* Convolve middle rows with kernel */
    for ( ; j < nrows - radius ; j++)  {
      ppp = ptrcol + ncols * (j - radius);
      sum = 0.0;
      for (k = kernel.width-1 ; k >= 0 ; k--)  {
        sum += *ppp * kernel.data[k];
        ppp += ncols;
      }
      *ptrout = sum;
      ptrout += ncols;
    }

    /* Zero bottommost rows */
    for ( ; j < nrows ; j++)  {
      *ptrout = 0.0;
      ptrout += ncols;
    }

    ptrcol++;
    ptrout -= nrows * ncols - 1;
  }
}


/*********************************************************************
 * _convolveSeparate
 */

static void _convolveSeparate(
  _KLT_FloatImage imgin,
  ConvolutionKernel horiz_kernel,
  ConvolutionKernel vert_kernel,
  _KLT_FloatImage imgout)
{
  /* Create temporary image */
  _KLT_FloatImage tmpimg;
  tmpimg = _KLTCreateFloatImage(imgin->ncols, imgin->nrows);
  
  /* ===== START CONVOLUTION TIMER ===== */
  clock_t start = clock();
  
  /* Do convolution */
  _convolveImageHoriz(imgin, horiz_kernel, tmpimg);
  _convolveImageVert(tmpimg, vert_kernel, imgout);
  
  /* ===== END CONVOLUTION TIMER ===== */
  clock_t end = clock();
  double convTime = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
  total_convolution_time += convTime;

  /* Free memory */
  _KLTFreeFloatImage(tmpimg);

  #ifdef USE_CUDA
  // Mark that host data is now valid, device is stale
    _KLTMarkHostValid(imgout);
  #endif
}

	
/*********************************************************************
 * _KLTComputeGradients
 */

// Forward declaration of GPU version
#ifdef USE_CUDA
extern void _KLTComputeGradients_GPU(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage gradx,
  _KLT_FloatImage grady);
#endif

void _KLTComputeGradients(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage gradx,
  _KLT_FloatImage grady)
{
#ifdef USE_CUDA
  /* Use GPU version for better performance */
  _KLTComputeGradients_GPU(img, sigma, gradx, grady);
#else			
  /* Output images must be large enough to hold result */
  assert(gradx->ncols >= img->ncols);
  assert(gradx->nrows >= img->nrows);
  assert(grady->ncols >= img->ncols);
  assert(grady->nrows >= img->nrows);

  /* Compute kernels, if necessary */
  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
  
  /* ===== START TOTAL TIMER ===== */
  clock_t totalStart = clock();
  
  _convolveSeparate(img, gaussderiv_kernel, gauss_kernel, gradx);
  _convolveSeparate(img, gauss_kernel, gaussderiv_kernel, grady);
  
  /* ===== END TOTAL TIMER ===== */
  clock_t totalEnd = clock();
  double totalTime = ((double)(totalEnd - totalStart)) / CLOCKS_PER_SEC * 1000.0;
  total_gradients_time += totalTime;
  gradients_call_count++;
#endif
}
	

/*********************************************************************
 * _KLTComputeSmoothedImage
 */

// Forward declaration of GPU version
#ifdef USE_CUDA
extern void _KLTComputeSmoothedImage_GPU(_KLT_FloatImage img,float sigma,_KLT_FloatImage smooth);
#endif

void _KLTComputeSmoothedImage(_KLT_FloatImage img,float sigma,_KLT_FloatImage smooth)
{
#ifdef USE_CUDA
  /* Use GPU version for better performance */
  _KLTComputeSmoothedImage_GPU(img, sigma, smooth);
#else
  /* Output image must be large enough to hold result */
  assert(smooth->ncols >= img->ncols);
  assert(smooth->nrows >= img->nrows);

  /* Compute kernel, if necessary; gauss_deriv is not used */
  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

  /* ===== START TOTAL TIMER ===== */
  clock_t totalStart = clock();
  
  _convolveSeparate(img, gauss_kernel, gauss_kernel, smooth);
  
  /* ===== END TOTAL TIMER ===== */
  clock_t totalEnd = clock();
  double totalTime = ((double)(totalEnd - totalStart)) / CLOCKS_PER_SEC * 1000.0;
  total_smoothed_time += totalTime;
  smoothed_call_count++;
#endif
}


/*********************************************************************
 * _KLTPrintCPUTimingStats
 *
 * Print accumulated timing statistics
 */

void _KLTPrintCPUTimingStats(void)
{
  printf("\n========================================\n");
  printf("CPU Timing Statistics (Accumulated)\n");
  printf("========================================\n");
  printf("Total convolution time: %.3f ms\n", total_convolution_time);
  printf("Total _KLTComputeGradients time: %.3f ms (%d calls)\n", total_gradients_time, gradients_call_count);
  printf("Total _KLTComputeSmoothedImage time: %.3f ms (%d calls)\n", total_smoothed_time, smoothed_call_count);
  printf("Total CPU time: %.3f ms\n", total_gradients_time + total_smoothed_time);
  printf("Memory/overhead time: %.3f ms\n", (total_gradients_time + total_smoothed_time) - total_convolution_time);
  printf("========================================\n");
}


/*********************************************************************
 * _KLTResetCPUTimingStats
 *
 * Reset timing statistics
 */

void _KLTResetCPUTimingStats(void)
{
  total_convolution_time = 0.0;
  total_gradients_time = 0.0;
  total_smoothed_time = 0.0;
  gradients_call_count = 0;
  smoothed_call_count = 0;
}