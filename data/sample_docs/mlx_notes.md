# MLX Notes

MLX is an array framework for machine learning on Apple Silicon, built by
Apple's machine learning research team.

## Lazy Computation

MLX uses lazy evaluation: operations build a computation graph that is only
materialized when explicitly requested. This allows fused kernels and
optimizations.

## Unified Memory

Because MLX targets Apple Silicon's unified memory, arrays live in a single
memory space accessible from both CPU and GPU. There is no explicit host-to-device
copy.

## mlx-lm

`mlx-lm` is a package on top of MLX for running large language models locally.
It supports loading quantized 4-bit and 8-bit models from the Hugging Face
`mlx-community` organization. As of version 0.31, mlx-lm focuses on generation
and does not ship an embedding API.
