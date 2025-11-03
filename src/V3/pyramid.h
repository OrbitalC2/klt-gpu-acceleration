/*********************************************************************
 * pyramid.h
 *********************************************************************/

#ifndef _PYRAMID_H_
#define _PYRAMID_H_

#include "klt_util.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct  {
  int subsampling;
  int nLevels;
  _KLT_FloatImage *img;
  int *ncols, *nrows;
}  _KLT_PyramidRec, *_KLT_Pyramid;


_KLT_Pyramid _KLTCreatePyramid(
  int ncols,
  int nrows,
  int subsampling,
  int nlevels);

void _KLTComputePyramid(
  _KLT_FloatImage floatimg, 
  _KLT_Pyramid pyramid,
  float sigma_fact);

void _KLTFreePyramid(
  _KLT_Pyramid pyramid);

#ifdef USE_CUDA
// GPU tracking function
void KLTTrackFeatures_GPU(
  KLT_TrackingContext tc,
  KLT_FeatureList featurelist,
  _KLT_Pyramid pyramid1,
  _KLT_Pyramid pyramid1_gradx,
  _KLT_Pyramid pyramid1_grady,
  _KLT_Pyramid pyramid2,
  _KLT_Pyramid pyramid2_gradx,
  _KLT_Pyramid pyramid2_grady);
#endif

#ifdef __cplusplus
}
#endif

#endif
