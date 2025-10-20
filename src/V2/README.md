# V2 — Naive CUDA convolution (Deliverable 2)

V1 profiling put 86% of runtime in `_convolveSeparate`, so V2 moves only that function to the GPU. Changes from V1:
- New: [`convolve.cu`](convolve.cu), [`cudaErrors.h`](cudaErrors.h) and the [`Makefile`](Makefile).
- [`convolve.c`](convolve.c) is V1's with `clock()` accumulators around each convolution and public call. It is used only for the CPU comparison build, so that the same calls are timed on both sides.
- [`convolve.h`](convolve.h) now defines `ConvolutionKernel` so the `.cu` file can see it.
- Every other `.c` and `.h` file is byte-identical to V1.

## Three attempts at the same kernel

Each team member wrote a GPU convolution independently. All three do a horizontal pass into a temporary image followed by a vertical pass, one thread per output pixel, and all three allocate, copy in, copy out and free device memory on every call. They differ in how closely they follow `convolve.c` and in whether they were ever connected to the tracker:

| File | Author | Launch | Coefficients | Borders | Called by KLT? |
|---|---|---|---|---|---|
| [`reports/D2/Imaad/convolve_gpu.cu`](../../reports/D2/Imaad/convolve_gpu.cu) | Imaad | 32×32 blocks | global memory, copied per call | clamp to edge, kernel read forwards (differs from `convolve.c`) | No. Exports `convolveSeparate_GPU`, which nothing calls |
| [`reports/D2/Saleh/convolve_gpu.cu`](../../reports/D2/Saleh/convolve_gpu.cu) | Saleh | 1D grid, 256 threads | global memory, copied per call | zeroed, kernel read backwards (matches `convolve.c`) | No. `KLTConvolveSeparate_GPU` is declared in `convolve.h` but never called |
| [`convolve.cu`](convolve.cu) | Rayyan | 32×8 blocks | `__constant__ cWindow[]` via `cudaMemcpyToSymbol` | zeroed, kernel read backwards (matches `convolve.c`) | Yes. Replaces `convolve.c` outright: defines `_KLTComputeGradients`, `_KLTComputeSmoothedImage`, `_KLTGetKernelWidths`, `_KLTToFloatImage` |

`convolve.cu` also uses a pitched layout (row length rounded up to a multiple of 32, copied with `cudaMemcpy2D`), `__restrict__` pointers, and CUDA-event timers around the kernels and around each public call. Those timers are where the kernel-vs-transfer split below comes from. It is the only one of the three that the D2 numbers describe and the one V3 grew out of. The other two attempts are kept with their authors' logs in [`reports/D2/`](../../reports/D2/), along with the Makefile that targeted Imaad's kernel; that Makefile does not link, since nothing in KLT calls `convolveSeparate_GPU`.

## Building

```bash
make            # example3_gpu: KLT + convolve.cu (convolve.c excluded)
make cpu        # example3_cpu: KLT + timed convolve.c
```

When the project was first uploaded, this folder held Imaad's Makefile, and the timed `convolve.c`, `cudaErrors.h` and Makefile behind the D2 numbers were in Rayyan's report folder. The `ConvolutionKernel` header those files need is Saleh's D2 `convolve.h`. They were moved here unchanged so the folder builds as it did for the report. The CPU build was checked to produce the same `features.txt` and `feat9.ppm` as V1 on the sample frames; the GPU build was not re-run for this repo.

Needs `nvcc`; `-arch=sm_86` targets the RTX 3080 it was run on. `example3.c` in this folder is V1's (10 frames, `img0.pgm`…). The 275-frame driver used for the results below was not uploaded with V2; V3's `example3.c` is the 275-frame version.

## Results

275 frames at 1920×1070, RTX 3080 server. Times are accumulated over all 1,098 convolution calls (549 gradient, 549 smoothing), not whole-program wall time ([`D2stats.txt`](../../reports/D2/Rayyan/D2stats.txt)):

| | CPU `convolve.c` | GPU `convolve.cu` |
|---|---|---|
| Total convolution time | 29,281 ms | 3,276 ms |
| of which kernel execution | — | 103 ms |
| of which allocation and transfers | — | 3,173 ms (96.9%) |

That is 8.94× on convolution time, and 284× if only kernel time is counted. Tracking output matched the CPU for all 275 frames. Table 1 of [`D2_performance_report.pdf`](../../reports/D2/D2_performance_report.pdf) prints the CPU time as 1.777 s; that is the 10-frame figure from the run below, and the 8.94× in the same table comes from 29.281 s.

Two earlier whole-program timings on the 10-frame 1920×1080 set are also recorded: CPU 1.777 s vs. GPU 1.836 s, 0.97× ([`CCP-D2-Performance-Results.txt`](../../reports/D2/Imaad/CCP-D2-Performance-Results.txt)), and 1.73 s for both builds with a size-mismatch exit on the last frame ([`D2_Performance_Results.txt`](../../reports/D2/Saleh/D2_Performance_Results.txt)). Neither Imaad's nor Saleh's kernel is called from the tracker in the uploaded code, so these timings can't be tied to a GPU path that exists in the repo.

## What D2 said to do next

The report listed why the speedup stopped at 8.94× and proposed fixes for V3:

| D2 diagnosis | Proposed for V3 |
|---|---|
| Host↔device copy on every call, no persistent device memory | Keep images on the GPU between calls |
| No shared-memory tiling | Shared-memory tiles with halos |
| Horizontal and vertical passes are separate launches | Kernel fusion |
| — | Texture memory for spatial locality |
| No streams or async copies | CUDA streams to overlap copies and compute |

What actually landed is in [`../V3/README.md`](../V3/README.md).
