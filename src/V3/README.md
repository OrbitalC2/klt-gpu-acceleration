# V3 — Optimized CUDA pipeline (Deliverable 3)

V2 proved the convolution kernel was cheap and the copies were not: 96.9% of GPU time went to allocating, copying and freeing on every call. V3 attacks that from two sides. The kernels were rewritten around shared memory, and then most of the pipeline was moved onto the GPU so images stay there between stages instead of crossing PCIe on every call.

The team tried several optimizations over this deliverable and kept the ones that measured faster. The intermediate attempts were not committed, so what follows is what the final code does, with each item pointed at the code that implements it.

## What changed from V2

### Convolution kernels: shared-memory tiling ([`convolve.cu`](convolve.cu))

- `ConvolveHorizKernel` and `ConvolveVertKernel` each load a 32×8 tile plus a halo of `kernel_width/2` pixels into dynamic shared memory (`extern __shared__ float tile[]`), then convolve out of shared memory instead of re-reading global memory for every tap. Rows are padded by `SMEM_PADDING` (8 floats).
- Coefficients stay in `__constant__ float cWindow[MAX_KERNEL_WIDTH]`, as in V2, but `_copyKernelToConstantMemory` now `memcmp`s against the last upload and skips `cudaMemcpyToSymbol` when the kernel hasn't changed.
- New entry point `_convolveSeparateGPU_DeviceToDevice` takes device pointers in and out, so callers that already hold data on the GPU never touch the host.

### Images live on the GPU ([`klt_util.h`](klt_util.h), [`klt_util_gpu.cu`](klt_util_gpu.cu))

- `_KLT_FloatImageRec` gains `d_data`, `d_stride`, `d_valid` and `h_valid`. Each image tracks which copy is current.
- `_KLTSyncToDevice` / `_KLTSyncToHost` only call `cudaMemcpy2D` when the other side is stale; `_KLTMarkDeviceValid` / `_KLTMarkHostValid` are set after GPU and CPU writes.
- `_KLTAllocFloatImageGPU` allocates once per image with the row stride rounded up to 32 floats; `_KLTFreeFloatImage` releases it.
- `_KLTComputeGradients` and `_KLTComputeSmoothedImage` in [`convolve.c`](convolve.c) forward to `_KLTComputeGradients_GPU` / `_KLTComputeSmoothedImage_GPU` under `USE_CUDA`, which read and write the device copies.

### Feature selection on the GPU ([`selectGoodFeatures.cu`](selectGoodFeatures.cu), replaces `selectGoodFeatures.c`)

- `ComputeMinEigenvaluesKernel`: 16×16 blocks, each loading a shared tile of `gradx` and `grady` (block plus window halo, accounting for `nSkippedPixels`) and computing the 2×2 structure tensor and its minimum eigenvalue per candidate pixel.
- The float image, `gradx` and `grady` are persistent device buffers on the tracking context (`tc->d_floatimg`, `d_gradx`, `d_grady`, sized by `_ensureGPUBuffers`). The image is converted to float on the host and copied up once. Gradients are then computed device-to-device, and only the candidate list comes back to the host.
- `_quicksort` and `_enforceMinimumDistance` remain on the CPU.
- The same file still contains the CPU path, built into `example3_cpu` by compiling it as C++ without `USE_CUDA`.

### Feature tracking on the GPU ([`trackFeatures_gpu.cu`](trackFeatures_gpu.cu))

- `TrackFeaturesKernel` runs one thread per feature through the whole coarse-to-fine pyramid: bilinear interpolation, window differences and gradient sums, the 2×2 solve, and the residue and bounds checks are all `__device__` ports of the functions in `trackFeatures.c`. Window scratch is held in three per-thread arrays of 256 floats. Nothing checks the window size against that, so anything larger than 16×16 overflows them. The default window is 7×7.
- [`trackFeatures.c`](trackFeatures.c) builds both pyramids, allocates device storage for every level, and calls `KLTTrackFeatures_GPU` instead of the per-feature CPU loop.

## What still runs on the CPU or costs a copy

- **Pyramid subsampling** ([`pyramid.c`](pyramid.c)): smoothing runs on the GPU, but `_KLTComputePyramid` syncs each smoothed level back with `_KLTSyncToHost` and subsamples on the host.
- **Sequential mode:** the reused previous-frame pyramid is synced back to the host at the top of `KLTTrackFeatures` before the new frame is processed.
- **Per-call allocations that remain:** `d_tmp` in `_convolveSeparateGPU_DeviceToDevice`, `d_pointlist` in selection, and the feature and pyramid-pointer arrays in `KLTTrackFeatures_GPU` are `cudaMalloc`ed and freed on every call.
- **Affine consistency checking:** if `affineConsistencyCheck >= 0`, tracking falls back to the CPU loop.
- **`smoothBeforeSelecting`:** not applied on the GPU selection path; the gradient filters already smooth.
- **Gradient reuse:** `tc->gradients_valid` stays set after the first selection, so `KLTReplaceLostFeatures` in sequential mode would reuse the first frame's gradients. `example3` doesn't exercise this, since `REPLACE` is off in every version.

## Everything that was tried

| Optimization | Source | In final V3? | Where |
|---|---|---|---|
| Shared-memory tiling (convolution) | D2 plan | Kept | `ConvolveHorizKernel`, `ConvolveVertKernel` |
| Shared-memory tiling (eigenvalues) | — | Kept | `ComputeMinEigenvaluesKernel` |
| Constant-memory coefficients, upload only on change | V2 + new cache | Kept | `_copyKernelToConstantMemory` |
| Persistent device memory, data resident across stages | D2 plan | Kept | `klt_util_gpu.cu`, `_ensureGPUBuffers` |
| GPU feature selection | D1 #2 hotspot | Kept | `selectGoodFeatures.cu` |
| GPU feature tracking | — | Kept | `trackFeatures_gpu.cu` |
| Pitched, 32-float-aligned rows | V2 | Kept | `_KLTAllocFloatImageGPU` |
| CUDA streams / async copies | D2 plan | Tried, dropped | No `cudaStream*` or `cudaMemcpyAsync` in the code; the nsys trace runs on one stream |
| Pinned host memory | — | Tried, dropped | No `cudaMallocHost` / `cudaHostAlloc` |
| Kernel fusion (horizontal + vertical) | D2 plan | Not in final code | Still two launches per convolution |
| Texture memory | D2 plan | Not in final code | No texture objects |

Streams and pinned memory are listed as tried on the team's account; there is no surviving code or measurement for them. For kernel fusion and texture memory there's no record either way beyond the D2 plan.

## Profiling evidence

[`reports/D3/nsys_trace.json`](../../reports/D3/nsys_trace.json) is an Nsight Systems export (`make profile`) captured partway through V3, after the shared-memory kernels went in but before the data was kept on the device. Only the two convolution kernels appear in it:

| | Count | Total time |
|---|---|---|
| `ConvolveHorizKernel` + `ConvolveVertKernel` | 2,997 each | 130 ms |
| Host→device copies (image + two `cudaMemcpyToSymbol`) | 8,991 | 1,765 ms |
| Device→host copies | 2,997 | 2,025 ms |

The images are 1920×1070 (8,217,600 bytes per copy). That works out to about 43 µs of kernel time per convolution, against about 94 µs for V2's kernel at the same size (103 ms / 1,098 calls). The timing methods differ: nsys kernel durations here, CUDA events in V2. The copies still cost 29× the kernel time, which is what the rest of V3 removes.

[`reports/D3/gprof_report.txt`](../../reports/D3/gprof_report.txt) is a gprof profile of the CPU path from this folder on a short run (33 convolution calls).

## Timing

[`reports/D3/time.txt`](../../reports/D3/time.txt), from [`time.sh`](../../reports/D3/time.sh) on 3 Nov 2025: one wall-clock run of each binary, 275 frames, 1,000 features, I/O included.

| `example3_cpu` | `example3_gpu` |
|---|---|
| 31.038 s | 3.711 s (8.4×) |

## Build

```bash
make                 # example3_gpu (nvcc, -arch=sm_86) and example3_cpu (gcc)
make profile         # nsys profile --stats=true
```

`example3` reads `klt_dataset/pgm/img1.pgm` … `img274.pgm`; that dataset is not in the repo. The CPU target builds without CUDA (`make example3_cpu`); the GPU target needs `nvcc`.

Demo recordings of both builds are in [`reports/D3/media/`](../../reports/D3/media/).
