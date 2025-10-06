# V1 — CPU baseline and profiling (Deliverable 1)

Unmodified KLT 1.3.4 (Stan Birchfield's C implementation). Every later version is measured against this code, so nothing here was changed; the work in this deliverable was finding out where the time goes.

## Build and run

```bash
make lib example3          # plain build, uses the original KLT Makefile
./example3                 # expects img0.pgm ... img9.pgm in the working directory
```

The 320×240 sample frames that ship with KLT are in `../V4/img*.pgm`. The profiling run below used 1920×1080 frames, which are not in the repo.

## Profiling

```bash
gcc -O2 -pg -g -o example3_prof *.c -lm
./example3_prof
gprof example3_prof gmon.out > profile_report.txt
```

Each team member profiled independently; their flat profiles and call graphs are in [`reports/D1/`](../../reports/D1/). The consolidated report is [`D1_profiling_report.pdf`](../../reports/D1/D1_profiling_report.pdf).

## What the profile showed

RTX 3080 HPC server (CPU only for this run), 1920×1080 grayscale input, 150 features, 10 frames, 1.28 s total:

| Function | Share of runtime | Self time | Calls |
|---|---|---|---|
| `_convolveSeparate` (`convolve.c`) | 85.94% | 1.10 s | 63 |
| `_KLTSelectGoodFeatures` (`selectGoodFeatures.c`) | 7.03% | 0.09 s | 1 |
| `_quicksort` (`selectGoodFeatures.c`) | 3.12% | 0.04 s | 1 (+1,147,784 recursive) |

`_convolveSeparate` sits under both `_KLTComputeGradients` (called twice per image, once per gradient direction) and `_KLTComputeSmoothedImage`, which runs once per input frame and again for every pyramid level in `_KLTComputePyramid`. That made it the first GPU target in V2; min-eigenvalue scoring in feature selection was second, and got ported in V3.
