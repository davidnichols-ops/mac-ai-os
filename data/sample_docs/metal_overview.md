# Metal Framework Overview

Metal is Apple's low-level, low-overhead GPU programming API.
It provides near-direct access to the GPU for compute and graphics workloads.

## Unified Memory

On Apple Silicon, the CPU and GPU share a unified memory architecture. This
means buffers do not need to be copied between CPU and GPU address spaces —
both can read and write the same memory. This dramatically reduces the cost
of passing data to the GPU compared to discrete-GPU systems.

## Compute Kernels

Metal compute kernels are written in the Metal Shading Language, a C++14-based
language. Kernels are dispatched over a grid of threadgroups using
`MTLComputeCommandEncoder`.

## Performance

Metal is the foundation for Core ML inference, MPSGraph, and the MLX framework.
MLX uses Metal compute kernels to run large language models on the Apple
Silicon GPU and unified memory.
