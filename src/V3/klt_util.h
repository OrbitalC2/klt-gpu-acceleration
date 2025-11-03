/*********************************************************************
 * klt_util.h
 *********************************************************************/

#ifndef _KLT_UTIL_H_
#define _KLT_UTIL_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct  {
  int ncols, nrows;
  float *data;

#ifdef USE_CUDA
  float *d_data;           // Device pointer
  int d_stride;            // Device memory stride (for alignment)
  KLT_BOOL d_valid;        // TRUE if device data is up-to-date
  KLT_BOOL h_valid;        // TRUE if host data is up-to-date
#endif

} _KLT_FloatImageRec, *_KLT_FloatImage;

_KLT_FloatImage _KLTCreateFloatImage(
  int ncols, 
  int nrows);

void _KLTFreeFloatImage(
  _KLT_FloatImage);
	
void _KLTPrintSubFloatImage(
  _KLT_FloatImage floatimg,
  int x0, int y0,
  int width, int height);

void _KLTWriteFloatImageToPGM(
  _KLT_FloatImage img,
  char *filename);

/* for affine mapping */
void _KLTWriteAbsFloatImageToPGM(
  _KLT_FloatImage img,
  char *filename,float scale);

#ifdef USE_CUDA
void _KLTAllocFloatImageGPU(_KLT_FloatImage img);
void _KLTFreeFloatImageGPU(_KLT_FloatImage img);
void _KLTSyncToDevice(_KLT_FloatImage img);
void _KLTSyncToHost(_KLT_FloatImage img);
void _KLTMarkDeviceValid(_KLT_FloatImage img);
void _KLTMarkHostValid(_KLT_FloatImage img);
#endif

#ifdef __cplusplus
}
#endif

#endif


