# KLT feature tracker: CPU to GPU

Taking Birchfield's KLT 1.3.4 feature tracker from a single-threaded C baseline to a CUDA pipeline, one deliverable at a time. Built for CS 4110 (High Performance Computing with GPUs) by Rayyan Imran, Imaad Fazal and Saleh Mubashar, and run on an RTX 3080 server.

Each version sits in its own folder so they can be diffed against each other, and each was added in its own commit.

| Version | What it is |
|---|---|
| [V1](src/V1/) | Original CPU code, gprof profiling and call graphs |
| [V2](src/V2/) | Naive CUDA port of `_convolveSeparate`: 8.94x on convolution time, 97% of it spent on transfers |
