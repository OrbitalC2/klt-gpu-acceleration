#include "base.h"
#include "klt.h"
#include "klt_util.h"
#include "pyramid.h"
#include <cuda_runtime.h>
#include <stdio.h>

// Device-side bilinear interpolation
__device__ inline float interpolate_device(
    float x, float y,
    const float* __restrict__ data,
    int ncols, int nrows, int stride)
{
    int xt = (int)x;
    int yt = (int)y;
    
    // Clamp to valid range
    if (xt < 0 || yt < 0 || xt >= ncols-1 || yt >= nrows-1)
        return 0.0f;
    
    float ax = x - xt;
    float ay = y - yt;
    
    const float* ptr = data + yt * stride + xt;
    
    return (1.0f - ax) * (1.0f - ay) * ptr[0] +
           ax * (1.0f - ay) * ptr[1] +
           (1.0f - ax) * ay * ptr[stride] +
           ax * ay * ptr[stride + 1];
}

// Device-side compute intensity difference
__device__ void computeIntensityDifference_device(
    const float* __restrict__ img1_data, int img1_stride,
    const float* __restrict__ img2_data, int img2_stride,
    int ncols, int nrows,
    float x1, float y1, float x2, float y2,
    int width, int height,
    float* imgdiff)
{
    int hw = width / 2;
    int hh = height / 2;
    int idx = 0;
    
    for (int j = -hh; j <= hh; j++) {
        for (int i = -hw; i <= hw; i++) {
            float g1 = interpolate_device(x1 + i, y1 + j, img1_data, ncols, nrows, img1_stride);
            float g2 = interpolate_device(x2 + i, y2 + j, img2_data, ncols, nrows, img2_stride);
            imgdiff[idx++] = g1 - g2;
        }
    }
}

// Device-side compute gradient sum
__device__ void computeGradientSum_device(
    const float* __restrict__ gradx1_data, int gradx1_stride,
    const float* __restrict__ grady1_data, int grady1_stride,
    const float* __restrict__ gradx2_data, int gradx2_stride,
    const float* __restrict__ grady2_data, int grady2_stride,
    int ncols, int nrows,
    float x1, float y1, float x2, float y2,
    int width, int height,
    float* gradx, float* grady)
{
    int hw = width / 2;
    int hh = height / 2;
    int idx = 0;
    
    for (int j = -hh; j <= hh; j++) {
        for (int i = -hw; i <= hw; i++) {
            float gx1 = interpolate_device(x1 + i, y1 + j, gradx1_data, ncols, nrows, gradx1_stride);
            float gx2 = interpolate_device(x2 + i, y2 + j, gradx2_data, ncols, nrows, gradx2_stride);
            float gy1 = interpolate_device(x1 + i, y1 + j, grady1_data, ncols, nrows, grady1_stride);
            float gy2 = interpolate_device(x2 + i, y2 + j, grady2_data, ncols, nrows, grady2_stride);
            gradx[idx] = gx1 + gx2;
            grady[idx] = gy1 + gy2;
            idx++;
        }
    }
}

// Device-side compute 2x2 gradient matrix
__device__ void compute2by2GradientMatrix_device(
    const float* gradx, const float* grady,
    int width, int height,
    float* gxx, float* gxy, float* gyy)
{
    float sum_gxx = 0.0f, sum_gxy = 0.0f, sum_gyy = 0.0f;
    int n = width * height;
    
    for (int i = 0; i < n; i++) {
        float gx = gradx[i];
        float gy = grady[i];
        sum_gxx += gx * gx;
        sum_gxy += gx * gy;
        sum_gyy += gy * gy;
    }
    
    *gxx = sum_gxx;
    *gxy = sum_gxy;
    *gyy = sum_gyy;
}

// Device-side compute 2x1 error vector
__device__ void compute2by1ErrorVector_device(
    const float* imgdiff, const float* gradx, const float* grady,
    int width, int height, float step_factor,
    float* ex, float* ey)
{
    float sum_ex = 0.0f, sum_ey = 0.0f;
    int n = width * height;
    
    for (int i = 0; i < n; i++) {
        float diff = imgdiff[i];
        sum_ex += diff * gradx[i];
        sum_ey += diff * grady[i];
    }
    
    *ex = sum_ex * step_factor;
    *ey = sum_ey * step_factor;
}

// Device-side solve equation
__device__ int solveEquation_device(
    float gxx, float gxy, float gyy,
    float ex, float ey, float small,
    float* dx, float* dy)
{
    float det = gxx * gyy - gxy * gxy;
    
    if (det < small) return KLT_SMALL_DET;
    
    *dx = (gyy * ex - gxy * ey) / det;
    *dy = (gxx * ey - gxy * ex) / det;
    return KLT_TRACKED;
}

// Device-side sum absolute window
__device__ float sumAbsFloatWindow_device(const float* fw, int width, int height)
{
    float sum = 0.0f;
    int n = width * height;
    for (int i = 0; i < n; i++) {
        sum += fabsf(fw[i]);
    }
    return sum;
}

// Main tracking kernel - one thread per feature
__global__ void TrackFeaturesKernel(
    // Input feature positions
    const float* __restrict__ x1_in,
    const float* __restrict__ y1_in,
    float* x2_inout,
    float* y2_inout,
    int* status_out,
    // Pyramid images (array of device pointers)
    const float** img1_ptrs,
    const float** gradx1_ptrs,
    const float** grady1_ptrs,
    const float** img2_ptrs,
    const float** gradx2_ptrs,
    const float** grady2_ptrs,
    const int* ncols_arr,
    const int* nrows_arr,
    const int* stride_arr,
    // Tracking parameters
    int nFeatures,
    int nLevels,
    int subsampling,
    int window_width,
    int window_height,
    float step_factor,
    int max_iterations,
    float small_det,
    float threshold,
    float max_residue,
    int borderx,
    int bordery)
{
    int feat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (feat_idx >= nFeatures) return;
    
    // Skip lost features
    if (x1_in[feat_idx] < 0.0f) {
        status_out[feat_idx] = -1;
        return;
    }
    
    float xloc = x1_in[feat_idx];
    float yloc = y1_in[feat_idx];
    float xlocout = x2_inout[feat_idx];
    float ylocout = y2_inout[feat_idx];
    
    int hw = window_width / 2;
    int hh = window_height / 2;
    float one_plus_eps = 1.001f;
    
    int final_status = KLT_TRACKED;
    
    // Allocate workspace in registers/local memory
    float imgdiff[256];  // max 16x16 window
    float gradx[256];
    float grady[256];
    
    // Coarse-to-fine pyramid tracking
    for (int level = nLevels - 1; level >= 0; level--) {
        const float* img1_data = img1_ptrs[level];
        const float* gradx1_data = gradx1_ptrs[level];
        const float* grady1_data = grady1_ptrs[level];
        const float* img2_data = img2_ptrs[level];
        const float* gradx2_data = gradx2_ptrs[level];
        const float* grady2_data = grady2_ptrs[level];
        
        int ncols = ncols_arr[level];
        int nrows = nrows_arr[level];
        int stride = stride_arr[level];
        
        // Scale coordinates for this level
        // Level 0 is finest (scale=1), level n-1 is coarsest (scale=subsamp^(n-1))
        float scale = powf((float)subsampling, (float)level);
        float x1 = xloc / scale;
        float y1 = yloc / scale;
        float x2 = xlocout / scale;
        float y2 = ylocout / scale;
        
        // KLT iteration loop
        int iteration = 0;
        int status = KLT_TRACKED;
        float dx = 0.0f, dy = 0.0f;
        
        do {
            // Bounds check
            if (x1 - hw < 0.0f || ncols - (x1 + hw) < one_plus_eps ||
                x2 - hw < 0.0f || ncols - (x2 + hw) < one_plus_eps ||
                y1 - hh < 0.0f || nrows - (y1 + hh) < one_plus_eps ||
                y2 - hh < 0.0f || nrows - (y2 + hh) < one_plus_eps) {
                status = KLT_OOB;
                break;
            }
            
            // Compute windows
            computeIntensityDifference_device(img1_data, stride, img2_data, stride,
                                            ncols, nrows, x1, y1, x2, y2,
                                            window_width, window_height, imgdiff);
            
            computeGradientSum_device(gradx1_data, stride, grady1_data, stride,
                                     gradx2_data, stride, grady2_data, stride,
                                     ncols, nrows, x1, y1, x2, y2,
                                     window_width, window_height, gradx, grady);
            
            // Compute matrices
            float gxx, gxy, gyy, ex, ey;
            compute2by2GradientMatrix_device(gradx, grady, window_width, window_height,
                                            &gxx, &gxy, &gyy);
            compute2by1ErrorVector_device(imgdiff, gradx, grady, window_width, window_height,
                                         step_factor, &ex, &ey);
            
            // Solve for displacement
            status = solveEquation_device(gxx, gxy, gyy, ex, ey, small_det, &dx, &dy);
            if (status == KLT_SMALL_DET) break;
            
            x2 += dx;
            y2 += dy;
            iteration++;
            
        } while ((fabsf(dx) >= threshold || fabsf(dy) >= threshold) && iteration < max_iterations);
        
        // Check final bounds
        if (x2 - hw < 0.0f || ncols - (x2 + hw) < one_plus_eps ||
            y2 - hh < 0.0f || nrows - (y2 + hh) < one_plus_eps) {
            status = KLT_OOB;
        }
        
        // Check residue at finest level
        if (level == 0 && status == KLT_TRACKED) {
            computeIntensityDifference_device(img1_data, stride, img2_data, stride,
                                            ncols, nrows, x1, y1, x2, y2,
                                            window_width, window_height, imgdiff);
            float residue = sumAbsFloatWindow_device(imgdiff, window_width, window_height);
            if (residue / (window_width * window_height) > max_residue) {
                status = KLT_LARGE_RESIDUE;
            }
        }
        
        // Handle iteration timeout
        if (status == KLT_TRACKED && iteration >= max_iterations) {
            status = KLT_MAX_ITERATIONS;
        }
        
        if (status != KLT_TRACKED) {
            final_status = status;
            break;
        }
        
        // Scale back to current level's coordinate for next finer level
        xlocout = x2 * scale;
        ylocout = y2 * scale;
    }
    
    // Final border check
    int ncols_full = ncols_arr[0];
    int nrows_full = nrows_arr[0];
    if (xlocout < borderx || xlocout > ncols_full - 1 - borderx ||
        ylocout < bordery || ylocout > nrows_full - 1 - bordery) {
        final_status = KLT_OOB;
    }
    
    // Write output
    if (final_status == KLT_TRACKED) {
        x2_inout[feat_idx] = xlocout;
        y2_inout[feat_idx] = ylocout;
        status_out[feat_idx] = KLT_TRACKED;
    } else {
        x2_inout[feat_idx] = -1.0f;
        y2_inout[feat_idx] = -1.0f;
        status_out[feat_idx] = final_status;
    }
}

// Host wrapper function
extern "C" {

void KLTTrackFeatures_GPU(
    KLT_TrackingContext tc,
    KLT_FeatureList featurelist,
    _KLT_Pyramid pyramid1,
    _KLT_Pyramid pyramid1_gradx,
    _KLT_Pyramid pyramid1_grady,
    _KLT_Pyramid pyramid2,
    _KLT_Pyramid pyramid2_gradx,
    _KLT_Pyramid pyramid2_grady)
{
    int nFeatures = featurelist->nFeatures;
    int nLevels = tc->nPyramidLevels;
    
    // Allocate device memory for feature arrays
    float *d_x1, *d_y1, *d_x2, *d_y2;
    int *d_status;
    
    cudaMalloc(&d_x1, nFeatures * sizeof(float));
    cudaMalloc(&d_y1, nFeatures * sizeof(float));
    cudaMalloc(&d_x2, nFeatures * sizeof(float));
    cudaMalloc(&d_y2, nFeatures * sizeof(float));
    cudaMalloc(&d_status, nFeatures * sizeof(int));
    
    // Copy feature positions to device
    float *h_x1 = (float*)malloc(nFeatures * sizeof(float));
    float *h_y1 = (float*)malloc(nFeatures * sizeof(float));
    float *h_x2 = (float*)malloc(nFeatures * sizeof(float));
    float *h_y2 = (float*)malloc(nFeatures * sizeof(float));
    
    for (int i = 0; i < nFeatures; i++) {
        h_x1[i] = featurelist->feature[i]->x;
        h_y1[i] = featurelist->feature[i]->y;
        h_x2[i] = featurelist->feature[i]->x;  // Initial guess
        h_y2[i] = featurelist->feature[i]->y;
    }
    
    cudaMemcpy(d_x1, h_x1, nFeatures * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y1, h_y1, nFeatures * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x2, h_x2, nFeatures * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y2, h_y2, nFeatures * sizeof(float), cudaMemcpyHostToDevice);
    
    // Build pyramid pointer arrays
    const float **h_img1_ptrs = (const float**)malloc(nLevels * sizeof(float*));
    const float **h_gradx1_ptrs = (const float**)malloc(nLevels * sizeof(float*));
    const float **h_grady1_ptrs = (const float**)malloc(nLevels * sizeof(float*));
    const float **h_img2_ptrs = (const float**)malloc(nLevels * sizeof(float*));
    const float **h_gradx2_ptrs = (const float**)malloc(nLevels * sizeof(float*));
    const float **h_grady2_ptrs = (const float**)malloc(nLevels * sizeof(float*));
    int *h_ncols = (int*)malloc(nLevels * sizeof(int));
    int *h_nrows = (int*)malloc(nLevels * sizeof(int));
    int *h_stride = (int*)malloc(nLevels * sizeof(int));
    
    for (int i = 0; i < nLevels; i++) {
        // Ensure device data is allocated and valid
        if (pyramid1->img[i]->d_data == NULL) _KLTAllocFloatImageGPU(pyramid1->img[i]);
        if (pyramid1_gradx->img[i]->d_data == NULL) _KLTAllocFloatImageGPU(pyramid1_gradx->img[i]);
        if (pyramid1_grady->img[i]->d_data == NULL) _KLTAllocFloatImageGPU(pyramid1_grady->img[i]);
        if (pyramid2->img[i]->d_data == NULL) _KLTAllocFloatImageGPU(pyramid2->img[i]);
        if (pyramid2_gradx->img[i]->d_data == NULL) _KLTAllocFloatImageGPU(pyramid2_gradx->img[i]);
        if (pyramid2_grady->img[i]->d_data == NULL) _KLTAllocFloatImageGPU(pyramid2_grady->img[i]);
        
        _KLTSyncToDevice(pyramid1->img[i]);
        _KLTSyncToDevice(pyramid1_gradx->img[i]);
        _KLTSyncToDevice(pyramid1_grady->img[i]);
        _KLTSyncToDevice(pyramid2->img[i]);
        _KLTSyncToDevice(pyramid2_gradx->img[i]);
        _KLTSyncToDevice(pyramid2_grady->img[i]);
        
        h_img1_ptrs[i] = pyramid1->img[i]->d_data;
        h_gradx1_ptrs[i] = pyramid1_gradx->img[i]->d_data;
        h_grady1_ptrs[i] = pyramid1_grady->img[i]->d_data;
        h_img2_ptrs[i] = pyramid2->img[i]->d_data;
        h_gradx2_ptrs[i] = pyramid2_gradx->img[i]->d_data;
        h_grady2_ptrs[i] = pyramid2_grady->img[i]->d_data;
        h_ncols[i] = pyramid1->img[i]->ncols;
        h_nrows[i] = pyramid1->img[i]->nrows;
        h_stride[i] = pyramid1->img[i]->d_stride;
    }
    
    // Copy pyramid metadata to device
    const float **d_img1_ptrs, **d_gradx1_ptrs, **d_grady1_ptrs;
    const float **d_img2_ptrs, **d_gradx2_ptrs, **d_grady2_ptrs;
    int *d_ncols, *d_nrows, *d_stride;
    
    cudaMalloc(&d_img1_ptrs, nLevels * sizeof(float*));
    cudaMalloc(&d_gradx1_ptrs, nLevels * sizeof(float*));
    cudaMalloc(&d_grady1_ptrs, nLevels * sizeof(float*));
    cudaMalloc(&d_img2_ptrs, nLevels * sizeof(float*));
    cudaMalloc(&d_gradx2_ptrs, nLevels * sizeof(float*));
    cudaMalloc(&d_grady2_ptrs, nLevels * sizeof(float*));
    cudaMalloc(&d_ncols, nLevels * sizeof(int));
    cudaMalloc(&d_nrows, nLevels * sizeof(int));
    cudaMalloc(&d_stride, nLevels * sizeof(int));
    
    cudaMemcpy(d_img1_ptrs, h_img1_ptrs, nLevels * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gradx1_ptrs, h_gradx1_ptrs, nLevels * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grady1_ptrs, h_grady1_ptrs, nLevels * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_img2_ptrs, h_img2_ptrs, nLevels * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gradx2_ptrs, h_gradx2_ptrs, nLevels * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grady2_ptrs, h_grady2_ptrs, nLevels * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ncols, h_ncols, nLevels * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nrows, h_nrows, nLevels * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stride, h_stride, nLevels * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (nFeatures + threadsPerBlock - 1) / threadsPerBlock;
    
    TrackFeaturesKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_x1, d_y1, d_x2, d_y2, d_status,
        d_img1_ptrs, d_gradx1_ptrs, d_grady1_ptrs,
        d_img2_ptrs, d_gradx2_ptrs, d_grady2_ptrs,
        d_ncols, d_nrows, d_stride,
        nFeatures, nLevels, pyramid1->subsampling,
        tc->window_width, tc->window_height,
        tc->step_factor, tc->max_iterations,
        tc->min_determinant, tc->min_displacement,
        tc->max_residue, tc->borderx, tc->bordery);
    
    cudaDeviceSynchronize();
    
    // Copy results back
    int *h_status = (int*)malloc(nFeatures * sizeof(int));
    cudaMemcpy(h_x2, d_x2, nFeatures * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y2, d_y2, nFeatures * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_status, d_status, nFeatures * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Update feature list
    for (int i = 0; i < nFeatures; i++) {
        featurelist->feature[i]->x = h_x2[i];
        featurelist->feature[i]->y = h_y2[i];
        featurelist->feature[i]->val = h_status[i];
    }
    
    // Cleanup
    cudaFree(d_x1);
    cudaFree(d_y1);
    cudaFree(d_x2);
    cudaFree(d_y2);
    cudaFree(d_status);
    cudaFree(d_img1_ptrs);
    cudaFree(d_gradx1_ptrs);
    cudaFree(d_grady1_ptrs);
    cudaFree(d_img2_ptrs);
    cudaFree(d_gradx2_ptrs);
    cudaFree(d_grady2_ptrs);
    cudaFree(d_ncols);
    cudaFree(d_nrows);
    cudaFree(d_stride);
    
    free(h_x1);
    free(h_y1);
    free(h_x2);
    free(h_y2);
    free(h_status);
    free(h_img1_ptrs);
    free(h_gradx1_ptrs);
    free(h_grady1_ptrs);
    free(h_img2_ptrs);
    free(h_gradx2_ptrs);
    free(h_grady2_ptrs);
    free(h_ncols);
    free(h_nrows);
    free(h_stride);
}

} // extern "C"

