
/* Standard includes */
#include <assert.h>
#include <stdlib.h> 
#include <stdio.h>  
#include <string.h> 
#include <math.h>  
#define fsqrt(X) sqrt(X)

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt.h"
#include "klt_util.h"
#include "pyramid.h"

int KLT_verbose = 1;

typedef enum {SELECTING_ALL, REPLACING_SOME} selectionMode;

/*********************************************************************
 * _quicksort
 * Replacement for qsort().  Computing time is decreased by taking
 * advantage of specific knowledge of our array (that there are 
 * three ints associated with each point).
 *
 * This routine generously provided by 
 *      Manolis Lourakis <lourakis@csi.forth.gr>
 *
 * NOTE: The results of this function may be slightly different from
 * those of qsort().  This is due to the fact that different sort 
 * algorithms have different behaviours when sorting numbers with the 
 * same value: Some leave them in the same relative positions in the 
 * array, while others change their relative positions. For example, 
 * if you have the array [c d b1 a b2] with b1=b2, it may be sorted as 
 * [a b1 b2 c d] or [a b2 b1 c d].
 */

#define SWAP3(list, i, j)               \
{register int *pi, *pj, tmp;            \
     pi=list+3*(i); pj=list+3*(j);      \
                                        \
     tmp=*pi;    \
     *pi++=*pj;  \
     *pj++=tmp;  \
                 \
     tmp=*pi;    \
     *pi++=*pj;  \
     *pj++=tmp;  \
                 \
     tmp=*pi;    \
     *pi=*pj;    \
     *pj=tmp;    \
}

static void _quicksort(int *pointlist, int n)
{
  unsigned int i, j, ln, rn;

  while (n > 1)
  {
    SWAP3(pointlist, 0, n/2);
    for (i = 0, j = n; ; )
    {
      do --j; while (pointlist[3*j+2] < pointlist[2]);
      do ++i; while (i < j && pointlist[3*i+2] > pointlist[2]);
      if (i >= j) break;
      SWAP3(pointlist, i, j);
    }
    SWAP3(pointlist, j, 0);
    ln = j;
    rn = n - ++j;
    if (ln < rn)
    {
      _quicksort(pointlist, ln);
      pointlist += 3*j;
      n = rn;
    }
    else
    {
      _quicksort(pointlist + 3*j, rn);
      n = ln;
    }
  }
}
#undef SWAP3

static void _fillFeaturemap(
  int x, int y,
  uchar *featuremap,
  int mindist,
  int ncols,
  int nrows)
{
  int ix, iy;
  for (iy = y - mindist ; iy <= y + mindist ; iy++)
    for (ix = x - mindist ; ix <= x + mindist ; ix++)
      if (ix >= 0 && ix < ncols && iy >= 0 && iy < nrows)
        featuremap[iy*ncols+ix] = 1;
}

static void _enforceMinimumDistance(
  int *pointlist,              /* featurepoints */
  int npoints,                 /* number of featurepoints */
  KLT_FeatureList featurelist, /* features */
  int ncols, int nrows,        /* size of images */
  int mindist,                 /* min. dist b/w features */
  int min_eigenvalue,          /* min. eigenvalue */
  KLT_BOOL overwriteAllFeatures)
{
  int indx;          /* Index into features */
  int x, y, val;     /* Location and trackability of pixel under consideration */
  uchar *featuremap; /* Boolean array recording proximity of features */
  int *ptr;

  if (min_eigenvalue < 1)  min_eigenvalue = 1;

  featuremap = (uchar *) malloc(ncols * nrows * sizeof(uchar));
  memset(featuremap, 0, ncols*nrows);

  mindist--;

  if (!overwriteAllFeatures)
    for (indx = 0 ; indx < featurelist->nFeatures ; indx++)
      if (featurelist->feature[indx]->val >= 0)  {
        x   = (int) featurelist->feature[indx]->x;
        y   = (int) featurelist->feature[indx]->y;
        _fillFeaturemap(x, y, featuremap, mindist, ncols, nrows);
      }

  ptr = pointlist;
  indx = 0;
  while (1)  {

    if (ptr >= pointlist + 3*npoints)  {
      while (indx < featurelist->nFeatures)  {
        if (overwriteAllFeatures ||
            featurelist->feature[indx]->val < 0) {
          featurelist->feature[indx]->x   = -1;
          featurelist->feature[indx]->y   = -1;
          featurelist->feature[indx]->val = KLT_NOT_FOUND;
          featurelist->feature[indx]->aff_img = NULL;
          featurelist->feature[indx]->aff_img_gradx = NULL;
          featurelist->feature[indx]->aff_img_grady = NULL;
          featurelist->feature[indx]->aff_x = -1.0;
          featurelist->feature[indx]->aff_y = -1.0;
          featurelist->feature[indx]->aff_Axx = 1.0;
          featurelist->feature[indx]->aff_Ayx = 0.0;
          featurelist->feature[indx]->aff_Axy = 0.0;
          featurelist->feature[indx]->aff_Ayy = 1.0;
        }
        indx++;
      }
      break;
    }

    x   = *ptr++;
    y   = *ptr++;
    val = *ptr++;

    assert(x >= 0 && x < ncols);
    assert(y >= 0 && y < nrows);

    while (!overwriteAllFeatures &&
           indx < featurelist->nFeatures &&
           featurelist->feature[indx]->val >= 0)
      indx++;

    if (indx >= featurelist->nFeatures)  break;

    if (!featuremap[y*ncols+x] && val >= min_eigenvalue)  {
      featurelist->feature[indx]->x   = (KLT_locType) x;
      featurelist->feature[indx]->y   = (KLT_locType) y;
      featurelist->feature[indx]->val = (int) val;
      featurelist->feature[indx]->aff_img = NULL;
      featurelist->feature[indx]->aff_img_gradx = NULL;
      featurelist->feature[indx]->aff_img_grady = NULL;
      featurelist->feature[indx]->aff_x = -1.0;
      featurelist->feature[indx]->aff_y = -1.0;
      featurelist->feature[indx]->aff_Axx = 1.0;
      featurelist->feature[indx]->aff_Ayx = 0.0;
      featurelist->feature[indx]->aff_Axy = 0.0;
      featurelist->feature[indx]->aff_Ayy = 1.0;
      indx++;

      _fillFeaturemap(x, y, featuremap, mindist, ncols, nrows);
    }
  }

  free(featuremap);
}

#ifdef KLT_USE_QSORT
static int _comparePoints(const void *a, const void *b)
{
  int v1 = *(((int *) a) + 2);
  int v2 = *(((int *) b) + 2);
  if (v1 > v2)  return(-1);
  else if (v1 < v2)  return(1);
  else return(0);
}
#endif

static void _sortPointList(int *pointlist, int npoints)
{
#ifdef KLT_USE_QSORT
  qsort(pointlist, npoints, 3*sizeof(int), _comparePoints);
#else
  _quicksort(pointlist, npoints);
#endif
}

static float _minEigenvalue(float gxx, float gxy, float gyy)
{
  return (float) ((gxx + gyy - sqrt((gxx - gyy)*(gxx - gyy) + 4*gxy*gxy))/2.0f);
}

/* ======================= GPU SECTION ======================= */
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "cudaErrors.h"
#include "convolve_gpu.h"

__device__ static float _minEigenvalueDevice(float gxx, float gxy, float gyy)
{
    return (gxx + gyy - sqrtf((gxx - gyy)*(gxx - gyy) + 4.0f*gxy*gxy)) * 0.5f;
}

__global__ void ComputeMinEigenvaluesKernel(
    const float* __restrict__ gradx,
    const float* __restrict__ grady,
    int* __restrict__ pointlist,
    int ncols, int nrows,
    int stride_gradx,
    int stride_grady,
    int borderx, int bordery,
    int window_hw, int window_hh,
    int skip_stride,
    unsigned int limit)
{
    // Shared memory for tile of gradx and grady
    extern __shared__ float shared_mem[];
    
    // Calculate tile dimensions
    int tile_width = blockDim.x * skip_stride + 2 * window_hw;
    int tile_height = blockDim.y * skip_stride + 2 * window_hh;
    int tile_size = tile_width * tile_height;
    
    // Split shared memory into two arrays
    float* s_gradx = shared_mem;
    float* s_grady = shared_mem + tile_size;
    
    // Global coordinates for this thread's pixel
    int x = borderx + (blockIdx.x * blockDim.x + threadIdx.x) * skip_stride;
    int y = bordery + (blockIdx.y * blockDim.y + threadIdx.y) * skip_stride;
    
    // Top-left corner of the tile in global memory
    int tile_start_x = borderx + blockIdx.x * blockDim.x * skip_stride - window_hw;
    int tile_start_y = bordery + blockIdx.y * blockDim.y * skip_stride - window_hh;
    
    // Cooperatively load tile into shared memory
    int total_threads = blockDim.x * blockDim.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    for (int idx = tid; idx < tile_size; idx += total_threads) {
        int tile_y = idx / tile_width;
        int tile_x = idx % tile_width;
        int global_x = tile_start_x + tile_x;
        int global_y = tile_start_y + tile_y;
        
        // Clamp to valid image boundaries
        global_x = max(0, min(global_x, ncols - 1));
        global_y = max(0, min(global_y, nrows - 1));
        
        int global_idx_x = global_y * stride_gradx + global_x;
        int global_idx_y = global_y * stride_grady + global_x;
        s_gradx[idx] = gradx[global_idx_x];
        s_grady[idx] = grady[global_idx_y];
    }
    
    __syncthreads();
    
    // Check if this thread is within valid bounds
    if (x >= ncols - borderx || y >= nrows - bordery) return;
    
    // Compute structure tensor elements using shared memory
    float gxx = 0.0f, gxy = 0.0f, gyy = 0.0f;
    
    // Base position of this thread's window in the tile
    int local_base_x = threadIdx.x * skip_stride + window_hw;
    int local_base_y = threadIdx.y * skip_stride + window_hh;
    
    for (int dy = -window_hh; dy <= window_hh; dy++) {
        for (int dx = -window_hw; dx <= window_hw; dx++) {
            int tile_idx = (local_base_y + dy) * tile_width + (local_base_x + dx);
            float gx = s_gradx[tile_idx];
            float gy = s_grady[tile_idx];
            gxx += gx * gx;
            gxy += gx * gy;
            gyy += gy * gy;
        }
    }
    
    // Compute minimum eigenvalue
    float val = _minEigenvalueDevice(gxx, gxy, gyy);
    if (val > (float)limit) val = (float)limit;
    
    // Calculate output index
    int x_idx = (x - borderx) / skip_stride;
    int y_idx = (y - bordery) / skip_stride;
    int valid_cols = (ncols - 2 * borderx + skip_stride - 1) / skip_stride;
    int idx = y_idx * valid_cols + x_idx;
    
    // Write result
    pointlist[3 * idx + 0] = x;
    pointlist[3 * idx + 1] = y;
    pointlist[3 * idx + 2] = (int)val;
}

// Helper function to ensure GPU buffers are allocated
static void _ensureGPUBuffers(KLT_TrackingContext tc, int ncols, int nrows)
{
    const int stride = round_up(ncols, KLT_GPU_STRIDE_MULT);
    const size_t buffer_size = stride * nrows * sizeof(float);
    
    // Check if buffers need to be allocated or reallocated
    if (tc->d_floatimg == nullptr || 
        tc->gpu_ncols != ncols || 
        tc->gpu_nrows != nrows) {
        
        // Free old buffers if they exist
        if (tc->d_floatimg != nullptr) {
            CUDA_CHECK(cudaFree(tc->d_floatimg));
            CUDA_CHECK(cudaFree(tc->d_gradx));
            CUDA_CHECK(cudaFree(tc->d_grady));
        }
        
        // Allocate new buffers
        CUDA_CHECK(cudaMalloc(&tc->d_floatimg, buffer_size));
        CUDA_CHECK(cudaMalloc(&tc->d_gradx, buffer_size));
        CUDA_CHECK(cudaMalloc(&tc->d_grady, buffer_size));
        
        // Store dimensions
        tc->gpu_ncols = ncols;
        tc->gpu_nrows = nrows;
        tc->gpu_stride = stride;
    }
}

static void _KLTSelectGoodFeatures_GPU(
  KLT_TrackingContext tc,
  KLT_PixelType *img, 
  int ncols, 
  int nrows,
  KLT_FeatureList featurelist,
  selectionMode mode)
{
  _KLT_FloatImage floatimg_host = nullptr;
  int window_hw, window_hh;
  int *pointlist;
  int npoints = 0;
  KLT_BOOL overwriteAllFeatures = (mode == SELECTING_ALL) ? TRUE : FALSE;
  KLT_BOOL need_gradient_computation = TRUE;

  if (tc->window_width % 2 != 1) {
    tc->window_width = tc->window_width+1;
    KLTWarning("Tracking context's window width must be odd.  "
               "Changing to %d.\n", tc->window_width);
  }
  if (tc->window_height % 2 != 1) {
    tc->window_height = tc->window_height+1;
    KLTWarning("Tracking context's window height must be odd.  "
               "Changing to %d.\n", tc->window_height);
  }
  if (tc->window_width < 3) {
    tc->window_width = 3;
    KLTWarning("Tracking context's window width must be at least three.  \n"
               "Changing to %d.\n", tc->window_width);
  }
  if (tc->window_height < 3) {
    tc->window_height = 3;
    KLTWarning("Tracking context's window height must be at least three.  \n"
               "Changing to %d.\n", tc->window_height);
  }
  window_hw = tc->window_width/2;
  window_hh = tc->window_height/2;

  pointlist = (int *) malloc(ncols * nrows * 3 * sizeof(int));

  _ensureGPUBuffers(tc, ncols, nrows);
  
  const int stride = tc->gpu_stride;
  const int pitchBytes = stride * sizeof(float);

  if (mode == REPLACING_SOME && tc->sequentialMode && tc->pyramid_last != NULL && tc->gradients_valid) {
    need_gradient_computation = FALSE;
  }
  else  {
    floatimg_host = _KLTCreateFloatImage(ncols, nrows);
    
    // smoothBeforeSelecting is not applied on this path; the gradient
    // filters already include Gaussian smoothing
    _KLTToFloatImage(img, ncols, nrows, floatimg_host);

    CUDA_CHECK(cudaMemcpy2D(tc->d_floatimg, pitchBytes, 
                            floatimg_host->data, ncols * sizeof(float), 
                            ncols * sizeof(float), nrows, 
                            cudaMemcpyHostToDevice));

    if (fabs(tc->grad_sigma - sigma_last) > 0.05)
      _computeKernels(tc->grad_sigma, &gauss_kernel, &gaussderiv_kernel);

    _convolveSeparateGPU_DeviceToDevice(tc->d_floatimg, ncols, nrows, stride,
                                        gaussderiv_kernel, gauss_kernel,
                                        tc->d_gradx, stride);
    _convolveSeparateGPU_DeviceToDevice(tc->d_floatimg, ncols, nrows, stride,
                                        gauss_kernel, gaussderiv_kernel,
                                        tc->d_grady, stride);

    tc->gradients_valid = TRUE;
  }

  if (tc->writeInternalImages)  {
    _KLT_FloatImage temp_gradx = _KLTCreateFloatImage(ncols, nrows);
    _KLT_FloatImage temp_grady = _KLTCreateFloatImage(ncols, nrows);
    
    CUDA_CHECK(cudaMemcpy2D(temp_gradx->data, ncols * sizeof(float),
                            tc->d_gradx, pitchBytes,
                            ncols * sizeof(float), nrows,
                            cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy2D(temp_grady->data, ncols * sizeof(float),
                            tc->d_grady, pitchBytes,
                            ncols * sizeof(float), nrows,
                            cudaMemcpyDeviceToHost));
    
    if (floatimg_host) {
      _KLTWriteFloatImageToPGM(floatimg_host, "kltimg_sgfrlf.pgm");
    }
    _KLTWriteFloatImageToPGM(temp_gradx, "kltimg_sgfrlf_gx.pgm");
    _KLTWriteFloatImageToPGM(temp_grady, "kltimg_sgfrlf_gy.pgm");
    
    _KLTFreeFloatImage(temp_gradx);
    _KLTFreeFloatImage(temp_grady);
  }

  {
      unsigned int limit = 1;
      int borderx = tc->borderx;
      int bordery = tc->bordery;
      int i;

      if (borderx < window_hw)  borderx = window_hw;
      if (bordery < window_hh)  bordery = window_hh;

      for (i = 0 ; i < sizeof(int) ; i++)  limit *= 256;
      limit = limit/2 - 1;

      int skip_stride = tc->nSkippedPixels + 1;

      int valid_cols = (ncols - 2 * borderx + skip_stride - 1) / skip_stride;
      int valid_rows = (nrows - 2 * bordery + skip_stride - 1) / skip_stride;
      int max_points = valid_cols * valid_rows;

      int *d_pointlist = nullptr;

      CUDA_CHECK(cudaMalloc(&d_pointlist, max_points * 3 * sizeof(int)));

      dim3 block(16, 16);
      dim3 grid((valid_cols + block.x - 1) / block.x,
                (valid_rows + block.y - 1) / block.y);

      int tile_width = block.x * skip_stride + 2 * window_hw;
      int tile_height = block.y * skip_stride + 2 * window_hh;
      int shared_mem_size = 2 * tile_width * tile_height * sizeof(float);

      ComputeMinEigenvaluesKernel<<<grid, block, shared_mem_size>>>(
          tc->d_gradx, tc->d_grady, d_pointlist,
          ncols, nrows, stride, stride,
          borderx, bordery,
          window_hw, window_hh, skip_stride, limit);

      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      CUDA_CHECK(cudaMemcpy(pointlist, d_pointlist, max_points * 3 * sizeof(int),
                            cudaMemcpyDeviceToHost));

      CUDA_CHECK(cudaFree(d_pointlist));

      npoints = max_points;
  }

  _sortPointList(pointlist, npoints);

  if (tc->mindist < 0)  {
    KLTWarning("(_KLTSelectGoodFeatures) Tracking context field tc->mindist "
               "is negative (%d); setting to zero", tc->mindist);
    tc->mindist = 0;
  }

  _enforceMinimumDistance(
    pointlist,
    npoints,
    featurelist,
    ncols, nrows,
    tc->mindist,
    tc->min_eigenvalue,
    overwriteAllFeatures);

  free(pointlist);
  if (floatimg_host)  {
    _KLTFreeFloatImage(floatimg_host);
  }
}

void KLTSelectGoodFeatures_GPU(
  KLT_TrackingContext tc,
  KLT_PixelType *img,
  int ncols,
  int nrows,
  KLT_FeatureList fl)
{
  if (KLT_verbose >= 1)  {
    fprintf(stderr,  "(KLT) Selecting the %d best features "
            "from a %d by %d image...  ", fl->nFeatures, ncols, nrows);
    fflush(stderr);
  }

  _KLTSelectGoodFeatures_GPU(tc, img, ncols, nrows,
                             fl, SELECTING_ALL);

  if (KLT_verbose >= 1)  {
    fprintf(stderr,  "\n\t%d features found.\n",
            KLTCountRemainingFeatures(fl));
    if (tc->writeInternalImages)
      fprintf(stderr,  "\tWrote images to 'kltimg_sgfrlf*.pgm'.\n");
    fflush(stderr);
  }
}

void KLTReplaceLostFeatures_GPU(
  KLT_TrackingContext tc,
  KLT_PixelType *img,
  int ncols,
  int nrows,
  KLT_FeatureList fl)
{
  int nLostFeatures = fl->nFeatures - KLTCountRemainingFeatures(fl);

  if (KLT_verbose >= 1)  {
    fprintf(stderr,  "(KLT) Attempting to replace %d features "
            "in a %d by %d image...  ", nLostFeatures, ncols, nrows);
    fflush(stderr);
  }

  if (nLostFeatures > 0)
    _KLTSelectGoodFeatures_GPU(tc, img, ncols, nrows,
                               fl, REPLACING_SOME);

  if (KLT_verbose >= 1)  {
    fprintf(stderr,  "\n\t%d features replaced.\n",
            nLostFeatures - fl->nFeatures + KLTCountRemainingFeatures(fl));
    if (tc->writeInternalImages)
      fprintf(stderr,  "\tWrote images to 'kltimg_sgfrlf*.pgm'.\n");
    fflush(stderr);
  }
}

// Cleanup function - call this when done with tracking context
void KLTFreeGPUResources(KLT_TrackingContext tc)
{
  if (tc->d_floatimg != nullptr) {
    CUDA_CHECK(cudaFree(tc->d_floatimg));
    tc->d_floatimg = nullptr;
  }
  if (tc->d_gradx != nullptr) {
    CUDA_CHECK(cudaFree(tc->d_gradx));
    tc->d_gradx = nullptr;
  }
  if (tc->d_grady != nullptr) {
    CUDA_CHECK(cudaFree(tc->d_grady));
    tc->d_grady = nullptr;
  }
  tc->gradients_valid = FALSE;
  tc->gpu_ncols = 0;
  tc->gpu_nrows = 0;
  tc->gpu_stride = 0;
}

#endif /* USE_CUDA */

/* ======================= CPU SECTION ======================= */

static void _KLTSelectGoodFeatures_CPU(KLT_TrackingContext tc,KLT_PixelType *img, int ncols, int nrows,KLT_FeatureList featurelist,selectionMode mode){
  _KLT_FloatImage floatimg, gradx, grady;
  int window_hw, window_hh;
  int *pointlist;
  int npoints = 0;
  KLT_BOOL overwriteAllFeatures = (mode == SELECTING_ALL) ? TRUE : FALSE;
  KLT_BOOL floatimages_created = FALSE;

  if (tc->window_width % 2 != 1) {
    tc->window_width = tc->window_width+1;
    KLTWarning("Tracking context's window width must be odd.  "
               "Changing to %d.\n", tc->window_width);
  }
  if (tc->window_height % 2 != 1) {
    tc->window_height = tc->window_height+1;
    KLTWarning("Tracking context's window height must be odd.  "
               "Changing to %d.\n", tc->window_height);
  }
  if (tc->window_width < 3) {
    tc->window_width = 3;
    KLTWarning("Tracking context's window width must be at least three.  \n"
               "Changing to %d.\n", tc->window_width);
  }
  if (tc->window_height < 3) {
    tc->window_height = 3;
    KLTWarning("Tracking context's window height must be at least three.  \n"
               "Changing to %d.\n", tc->window_height);
  }
  window_hw = tc->window_width/2;
  window_hh = tc->window_height/2;

  pointlist = (int *) malloc(ncols * nrows * 3 * sizeof(int));

  if (mode == REPLACING_SOME && tc->sequentialMode && tc->pyramid_last != NULL)  {
    floatimg = ((_KLT_Pyramid) tc->pyramid_last)->img[0];
    gradx = ((_KLT_Pyramid) tc->pyramid_last_gradx)->img[0];
    grady = ((_KLT_Pyramid) tc->pyramid_last_grady)->img[0];
    assert(gradx != NULL);
    assert(grady != NULL);
  }
  else  {
    floatimages_created = TRUE;
    floatimg = _KLTCreateFloatImage(ncols, nrows);
    gradx = _KLTCreateFloatImage(ncols, nrows);
    grady = _KLTCreateFloatImage(ncols, nrows);
    if (tc->smoothBeforeSelecting)  {
      _KLT_FloatImage tmpimg;
      tmpimg = _KLTCreateFloatImage(ncols, nrows);
      _KLTToFloatImage(img, ncols, nrows, tmpimg);
      _KLTComputeSmoothedImage(tmpimg, _KLTComputeSmoothSigma(tc), floatimg);
      _KLTFreeFloatImage(tmpimg);
    }
     else
      _KLTToFloatImage(img, ncols, nrows, floatimg);

    _KLTComputeGradients(floatimg, tc->grad_sigma, gradx, grady);
  }

  if (tc->writeInternalImages)  {
    _KLTWriteFloatImageToPGM(floatimg, "kltimg_sgfrlf.pgm");
    _KLTWriteFloatImageToPGM(gradx, "kltimg_sgfrlf_gx.pgm");
    _KLTWriteFloatImageToPGM(grady, "kltimg_sgfrlf_gy.pgm");
  }

  {
    register float gx, gy;
    register float gxx, gxy, gyy;
    register int xx, yy;
    register int *ptr;
    float val;
    unsigned int limit = 1;
    int borderx = tc->borderx; /* Must not touch cols */
    int bordery = tc->bordery; /* lost by convolution */
    int x, y;
    int i;

    if (borderx < window_hw)  borderx = window_hw;
    if (bordery < window_hh)  bordery = window_hh;

    for (i = 0 ; i < sizeof(int) ; i++)  limit *= 256;
    limit = limit/2 - 1;

    ptr = pointlist;
    for (y = bordery ; y < nrows - bordery ; y += tc->nSkippedPixels + 1)
      for (x = borderx ; x < ncols - borderx ; x += tc->nSkippedPixels + 1)  {

        gxx = 0;  gxy = 0;  gyy = 0;
        for (yy = y-window_hh ; yy <= y+window_hh ; yy++)
          for (xx = x-window_hw ; xx <= x+window_hw ; xx++)  {
            gx = *(gradx->data + ncols*yy+xx);
            gy = *(grady->data + ncols*yy+xx);
            gxx += gx * gx;
            gxy += gx * gy;
            gyy += gy * gy;
          }

        *ptr++ = x;
        *ptr++ = y;
        val = _minEigenvalue(gxx, gxy, gyy);
        if (val > limit)  {
          KLTWarning("(_KLTSelectGoodFeatures) minimum eigenvalue %f is "
                     "greater than the capacity of an int; setting "
                     "to maximum value", val);
          val = (float) limit;
        }
        *ptr++ = (int) val;
        npoints++;
      }
  }

  _sortPointList(pointlist, npoints);

  if (tc->mindist < 0)  {
    KLTWarning("(_KLTSelectGoodFeatures) Tracking context field tc->mindist "
               "is negative (%d); setting to zero", tc->mindist);
    tc->mindist = 0;
  }

  _enforceMinimumDistance(
    pointlist,
    npoints,
    featurelist,
    ncols, nrows,
    tc->mindist,
    tc->min_eigenvalue,
    overwriteAllFeatures);

  free(pointlist);
  if (floatimages_created)  {
    _KLTFreeFloatImage(floatimg);
    _KLTFreeFloatImage(gradx);
    _KLTFreeFloatImage(grady);
  }
}

void KLTSelectGoodFeatures_CPU(
  KLT_TrackingContext tc,
  KLT_PixelType *img,
  int ncols,
  int nrows,
  KLT_FeatureList fl)
{
  if (KLT_verbose >= 1)  {
    fprintf(stderr,  "(KLT) Selecting the %d best features "
            "from a %d by %d image...  ", fl->nFeatures, ncols, nrows);
    fflush(stderr);
  }

  _KLTSelectGoodFeatures_CPU(tc, img, ncols, nrows,
                             fl, SELECTING_ALL);

  if (KLT_verbose >= 1)  {
    fprintf(stderr,  "\n\t%d features found.\n",
            KLTCountRemainingFeatures(fl));
    if (tc->writeInternalImages)
      fprintf(stderr,  "\tWrote images to 'kltimg_sgfrlf*.pgm'.\n");
    fflush(stderr);
  }
}

void KLTReplaceLostFeatures_CPU(
  KLT_TrackingContext tc,
  KLT_PixelType *img,
  int ncols,
  int nrows,
  KLT_FeatureList fl)
{
  int nLostFeatures = fl->nFeatures - KLTCountRemainingFeatures(fl);

  if (KLT_verbose >= 1)  {
    fprintf(stderr,  "(KLT) Attempting to replace %d features "
            "in a %d by %d image...  ", nLostFeatures, ncols, nrows);
    fflush(stderr);
  }

  if (nLostFeatures > 0)
    _KLTSelectGoodFeatures_CPU(tc, img, ncols, nrows,
                               fl, REPLACING_SOME);

  if (KLT_verbose >= 1)  {
    fprintf(stderr,  "\n\t%d features replaced.\n",
            nLostFeatures - fl->nFeatures + KLTCountRemainingFeatures(fl));
    if (tc->writeInternalImages)
      fprintf(stderr,  "\tWrote images to 'kltimg_sgfrlf*.pgm'.\n");
    fflush(stderr);
  }
}

/* ======================= Public wrapper ======================= */
/* Keeps the original API name; selects GPU when available (if built with USE_CUDA). */
void KLTSelectGoodFeatures(
  KLT_TrackingContext tc,
  KLT_PixelType *img,
  int ncols,
  int nrows,
  KLT_FeatureList fl)
{
#ifdef USE_CUDA
  int ndev = 0;
  if (cudaGetDeviceCount(&ndev) == cudaSuccess && ndev > 0) {
    KLTSelectGoodFeatures_GPU(tc, img, ncols, nrows, fl);
    return;
  }
#endif
  KLTSelectGoodFeatures_CPU(tc, img, ncols, nrows, fl);
}
