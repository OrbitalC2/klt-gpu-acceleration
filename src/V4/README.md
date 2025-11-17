# V4 — OpenACC comparison (Deliverable 4)

The same V1 code, offloaded with compiler directives instead of hand-written CUDA. The question for this deliverable was how much of V3's speedup a directive-based port gets for a fraction of the effort.

V4 starts from V1, not V3. `klt.c`, `pyramid.c`, `klt_util.c` and the I/O files are unchanged from V1; the work is in three files plus the Makefile.

## What was offloaded

### Convolution ([`convolve.c`](convolve.c))

- `_convolveImageHoriz` and `_convolveImageVert`: the outer row (or column) loop is `#pragma acc parallel loop gang vector` with `present(data_in, data_out)` and `copyin(kernel)`.
- The inner loop indices and the accumulator are `private`. Without that they're shared across threads and race, and feature selection then finds 0 features. The source comments record this as a bug fixed during the port.
- `_convolveSeparate` opens an `acc data` region around both passes: input image and both kernels `copyin`, the temporary image `create`d on the device, output `copyout`. Data moves in and out on every call, the same pattern as V2.

### Feature selection ([`selectGoodFeatures.c`](selectGoodFeatures.c))

- The min-eigenvalue loop over candidate pixels is `#pragma acc parallel loop collapse(2)`. Each thread sums its own window serially and writes into a full-size `eigen_map`, so there are no shared writes.
- `gradx` and `grady` are `copyin`, and `eigen_map` is initialised to -1 on the device and `copyout`.
- The candidate list is then filled serially on the host from `eigen_map`, keeping V1's ordering for the sort. `_quicksort` and `_enforceMinimumDistance` stay on the CPU.

### Feature tracking ([`trackFeatures.c`](trackFeatures.c))

- All 12 directives here are data movement. `#pragma acc enter data copyin` puts every level of the six pyramids (image, gradx and grady for both frames) on the device before the feature loop, and `exit data delete` removes them after.
- There is no `parallel` or `kernels` region in the file, so the per-feature tracking loop and `_trackFeature` still run on the CPU, and the pyramid copies are not used by any device code.

### Build ([`Makefile`](Makefile))

- `nvc -O3 -acc -Minfo=acc` for the GPU build, and the same with `-noacc` for the CPU baseline. Both come from the same source, so the comparison isolates the directives.

## Differences from V1 beyond the directives

- `trackFeatures.c` was restructured during the port. Three pieces of V1 behaviour did not come back:
  - the `writeInternalImages` debug dump
  - freeing `aff_img` / `aff_img_gradx` / `aff_img_grady` when a feature is lost
  - the whole affine-consistency branch
- With `example3`'s settings (`affineConsistencyCheck = -1`, `writeInternalImages = FALSE`) none of these paths runs, so tracking output is unaffected, but V4 does not support affine checking.
- `example3.c` has the per-frame PPM output and feature-table writes commented out, so timing excludes output I/O.

## Results

Reported in the team's original V4 README, RTX 3080 server, whole-program time:

| Dataset | V1 (`-noacc`) | V4 (`-acc`) | Speedup |
|---|---|---|---|
| Large tracking dataset | 32.99 s | 8.9 s | 3.7× |
| 10 frames upscaled to 4024 px wide | 7.49 s | 1.91 s | 3.92× |

The raw logs for these runs are not in the repo. Given what was offloaded, the gain comes from convolution and eigenvalue scoring; tracking itself is still serial.

## Build and run

```bash
make run_cpu      # rebuilds with -noacc and times ./example3
make run_gpu      # rebuilds with -acc and times ./example3
```

Needs the NVIDIA HPC SDK (`nvc`). `example3.c` here reads `img0.pgm` … `img9.pgm`, the 320×240 KLT sample frames included in this folder; the benchmark runs above used larger inputs. With `gcc` the pragmas are ignored and it builds as plain V1 code.
