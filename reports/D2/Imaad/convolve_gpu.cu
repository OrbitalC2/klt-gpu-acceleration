// convolve_gpu.cu
#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 32

__global__ void convolveHorizontalKernel(float *input, float *output, float *kernel, int width, int height, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int half_kernel = kernel_size / 2;
    float sum = 0.0f;

    for (int k = -half_kernel; k <= half_kernel; k++) {
        int nx = x + k;
        if (nx < 0) nx = 0;
        if (nx >= width) nx = width - 1;
        sum += input[y * width + nx] * kernel[k + half_kernel];
    }
    output[y * width + x] = sum;
}

__global__ void convolveVerticalKernel(float *input, float *output, float *kernel, int width, int height, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int half_kernel = kernel_size / 2;
    float sum = 0.0f;

    for (int k = -half_kernel; k <= half_kernel; k++) {
        int ny = y + k;
        if (ny < 0) ny = 0;
        if (ny >= height) ny = height - 1;
        sum += input[ny * width + x] * kernel[k + half_kernel];
    }
    output[y * width + x] = sum;
}

extern "C" void convolveSeparate_GPU(float *input, float *output, float *kernel, int width, int height, int kernel_size) {
    float *d_input, *d_temp, *d_output, *d_kernel;
    size_t image_size = width * height * sizeof(float);
    size_t kernel_mem = kernel_size * sizeof(float);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate device memory
    cudaMalloc(&d_input, image_size);
    cudaMalloc(&d_temp, image_size);
    cudaMalloc(&d_output, image_size);
    cudaMalloc(&d_kernel, kernel_mem);

    // Time memory transfer Host → Device
    cudaEventRecord(start);
    cudaMemcpy(d_input, input, image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_mem, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float h2d_time = 0;
    cudaEventElapsedTime(&h2d_time, start, stop);

    // Setup kernel launch
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Time kernel execution (both passes)
    cudaEventRecord(start);
    convolveHorizontalKernel<<<gridSize, blockSize>>>(d_input, d_temp, d_kernel, width, height, kernel_size);
    cudaDeviceSynchronize();
    convolveVerticalKernel<<<gridSize, blockSize>>>(d_temp, d_output, d_kernel, width, height, kernel_size);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float kernel_time = 0;
    cudaEventElapsedTime(&kernel_time, start, stop);

    // Time memory transfer Device → Host
    cudaEventRecord(start);
    cudaMemcpy(output, d_output, image_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float d2h_time = 0;
    cudaEventElapsedTime(&d2h_time, start, stop);

    // Print timing (only once)
    static int first_call = 1;
    if (first_call) {
        printf("[GPU Timing] H2D: %.3f ms, Kernel: %.3f ms, D2H: %.3f ms, Total: %.3f ms\n",
               h2d_time, kernel_time, d2h_time, h2d_time + kernel_time + d2h_time);
        first_call = 0;
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_temp);
    cudaFree(d_output);
    cudaFree(d_kernel);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
