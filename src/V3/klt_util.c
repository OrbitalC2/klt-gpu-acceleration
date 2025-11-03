/*********************************************************************
 * klt_util.c
 *********************************************************************/

/* Standard includes */
#include <assert.h>
#include <stdlib.h>  /* malloc() */
#include <math.h>		/* fabs() */

/* Our includes */
#include "base.h"
#include "error.h"
#include "pnmio.h"
#include "klt.h"
#include "klt_util.h"


/*********************************************************************/

float _KLTComputeSmoothSigma(
  KLT_TrackingContext tc)
{
  return (tc->smooth_sigma_fact * max(tc->window_width, tc->window_height));
}


/*********************************************************************
 * _KLTCreateFloatImage
 */

_KLT_FloatImage _KLTCreateFloatImage(
  int ncols,
  int nrows)
{
  _KLT_FloatImage floatimg;
  int nbytes = sizeof(_KLT_FloatImageRec) +
    ncols * nrows * sizeof(float);

  floatimg = (_KLT_FloatImage)  malloc(nbytes);
  if (floatimg == NULL)
    KLTError("(_KLTCreateFloatImage) Out of memory");
  floatimg->ncols = ncols;
  floatimg->nrows = nrows;
  floatimg->data = (float *) (floatimg + 1);

#ifdef USE_CUDA
  floatimg->d_data = NULL;
  floatimg->d_stride = 0;
  floatimg->d_valid = FALSE;
  floatimg->h_valid = TRUE;  // Host data is valid initially
#endif

  return(floatimg);
}

/*********************************************************************
 * _KLTFreeFloatImage
 */

void _KLTFreeFloatImage(
  _KLT_FloatImage img)
{
  if (img == NULL) return;
  
#ifdef USE_CUDA
  if (img->d_data != NULL) {
    cudaFree(img->d_data);
    img->d_data = NULL;
  }
#endif
  free(img);
}


/*********************************************************************
 * _KLTPrintSubFloatImage
 */

void _KLTPrintSubFloatImage(
  _KLT_FloatImage floatimg,
  int x0, int y0,
  int width, int height)
{
  int ncols = floatimg->ncols;
  int offset;
  int i, j;

  assert(x0 >= 0);
  assert(y0 >= 0);
  assert(x0 + width <= ncols);
  assert(y0 + height <= floatimg->nrows);

  fprintf(stderr, "\n");
  for (j = 0 ; j < height ; j++)  {
    for (i = 0 ; i < width ; i++)  {
      offset = (j+y0)*ncols + (i+x0);
      fprintf(stderr, "%6.2f ", *(floatimg->data + offset));
    }
    fprintf(stderr, "\n");
  }
  fprintf(stderr, "\n");
}
	

/*********************************************************************
 * _KLTWriteFloatImageToPGM
 */

void _KLTWriteFloatImageToPGM(
  _KLT_FloatImage img,
  char *filename)
{
  int npixs = img->ncols * img->nrows;
  float mmax = -999999.9f, mmin = 999999.9f;
  float fact;
  float *ptr;
  uchar *byteimg, *ptrout;
  int i;

#ifdef USE_CUDA
  /* Ensure data is available on host for writing */
  _KLTSyncToHost(img);
#endif

  /* Calculate minimum and maximum values of float image */
  ptr = img->data;
  for (i = 0 ; i < npixs ; i++)  {
    mmax = max(mmax, *ptr);
    mmin = min(mmin, *ptr);
    ptr++;
  }
	
  /* Allocate memory to hold converted image */
  byteimg = (uchar *) malloc(npixs * sizeof(uchar));

  /* Convert image from float to uchar */
  fact = 255.0f / (mmax-mmin);
  ptr = img->data;
  ptrout = byteimg;
  for (i = 0 ; i < npixs ; i++)  {
    *ptrout++ = (uchar) ((*ptr++ - mmin) * fact);
  }

  /* Write uchar image to PGM */
  pgmWriteFile(filename, byteimg, img->ncols, img->nrows);

  /* Free memory */
  free(byteimg);
}

/*********************************************************************
 * _KLTWriteFloatImageToPGM
 */

void _KLTWriteAbsFloatImageToPGM(
  _KLT_FloatImage img,
  char *filename,float scale)
{
  int npixs = img->ncols * img->nrows;
  float fact;
  float *ptr;
  uchar *byteimg, *ptrout;
  int i;
  float tmp;
	
  /* Allocate memory to hold converted image */
  byteimg = (uchar *) malloc(npixs * sizeof(uchar));

  /* Convert image from float to uchar */
  fact = 255.0f / scale;
  ptr = img->data;
  ptrout = byteimg;
  for (i = 0 ; i < npixs ; i++)  {
    tmp = (float) (fabs(*ptr++) * fact);
    if(tmp > 255.0) tmp = 255.0;
    *ptrout++ =  (uchar) tmp;
  }

  /* Write uchar image to PGM */
  pgmWriteFile(filename, byteimg, img->ncols, img->nrows);

  /* Free memory */
  free(byteimg);
}
