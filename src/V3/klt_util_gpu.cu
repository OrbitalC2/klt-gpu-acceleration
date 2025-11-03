#include "klt.h"
#include "klt_util.h"
#include "cuda_runtime.h"
#include "cudaErrors.h"
#include <assert.h>

#define KLT_GPU_STRIDE_MULT 32  //warp friendly

static int round_up(int value, int multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

// Allocate GPU memory for a FloatImage
void _KLTAllocFloatImageGPU(_KLT_FloatImage img)
{
  if (img->d_data != NULL) {
    return;  // Already allocated
  }

  const int ncols = img->ncols;
  const int nrows = img->nrows;
  const int stride = round_up(ncols, KLT_GPU_STRIDE_MULT);

  CUDA_CHECK(cudaMalloc(&img->d_data, stride * nrows * sizeof(float)));
  img->d_stride = stride;
  img->d_valid = FALSE;
}

// Free GPU memory (called automatically by _KLTFreeFloatImage)
void _KLTFreeFloatImageGPU(_KLT_FloatImage img)
{
  if (img->d_data != NULL) {
    cudaFree(img->d_data);
    img->d_data = NULL;
    img->d_stride = 0;
    img->d_valid = FALSE;
  }
}

//H2D memcpy and management only if needed
void _KLTSyncToDevice(_KLT_FloatImage img)
{
  if (img->d_data == NULL) {
    _KLTAllocFloatImageGPU(img);
  }

  if (!img->d_valid && img->h_valid) {
    const int ncols = img->ncols;
    const int nrows = img->nrows;
    const int pitchBytes = img->d_stride * sizeof(float);

    CUDA_CHECK(cudaMemcpy2D(img->d_data, pitchBytes,img->data, ncols * sizeof(float),ncols * sizeof(float), nrows,cudaMemcpyHostToDevice));
    img->d_valid = TRUE;
    // h_valid remains TRUE - both copies are now valid
  }
}

//D2H memcpy and management only if needed
void _KLTSyncToHost(_KLT_FloatImage img)
{
  if (img->d_data == NULL || !img->d_valid) {
    return;  // No device data to sync
  }

  if (!img->h_valid) {
    const int ncols = img->ncols;
    const int nrows = img->nrows;
    const int pitchBytes = img->d_stride * sizeof(float);

    CUDA_CHECK(cudaMemcpy2D(
      img->data, ncols * sizeof(float),
      img->d_data, pitchBytes,
      ncols * sizeof(float), nrows,
      cudaMemcpyDeviceToHost
    ));

    img->h_valid = TRUE;
    // d_valid remains TRUE - both copies are now valid
  }
}

// Mark device data as valid, host as invalid (after GPU computation)
void _KLTMarkDeviceValid(_KLT_FloatImage img)
{
  img->d_valid = TRUE;
  img->h_valid = FALSE;
}

// Mark host data as valid, device as invalid (after CPU computation)
void _KLTMarkHostValid(_KLT_FloatImage img)
{
  img->h_valid = TRUE;
  img->d_valid = FALSE;
}