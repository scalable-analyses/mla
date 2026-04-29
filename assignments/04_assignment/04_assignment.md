# Assignment 04: Tensor Contractions on GPUs

In this assignment, you will implement and optimize GPU tensor-contraction
kernels using [cuTile](https://github.com/nvidia/cutile-python). You will
explore the effects of parallelization strategies, primitive-size merging, and
kernel fusion on performance.

All code should be written in `src/`, one file per task.

We assume the following import conventions:

```python
import cuda.tile as ct
import cupy as cp
import torch
import triton
```

**Use FP16 data type for tensor inputs and outputs, accumulate in FP32.** We assume row-major order for all tensors.

**Be careful with choosing dimension sizes for the following tasks. Assert that all tensors will fit in memory (less than 32 GiB) first.**
If the machine runs out of memory, it crashes. This will impact you and other students. If this happens, please notify us via the Matrix channel!

---

## Task 1: Tiled Contraction Kernel Variants

a) Classify all dimensions in the einsum string `eabklxy, ecklyz -> eabcxz`.

b) Implement a cuTile kernel that computes the contraction `eabklxy, ecklyz -> eabcxz`. Use dimensions `xyz` as your GEMM dimensions. Sequentialize all other K-dimensions, parallelize the remaining dimensions. The kernel should work with arbitrary dimension sizes. You can hand them to your kernel as function arguments.

c) Implement a cuTile kernel that computes the contraction `eabklxy, ecklyz -> eabcxz`. Use dimensions `xyz` as your GEMM dimensions. Sequentialize all other K-dimensions, **as well as the `b` dimension**. Parallelize the remaining dimensions. The kernel should work with arbitrary dimension sizes. You can hand them to your kernel as function arguments.

Find one configuration (dimension sizes) where your kernel from b) performs better and one configuration where your new kernel from c) performs better.

d) Implement a cuTile kernel that computes the contraction `eabklxy, ecklyz -> eabcxz`. **Use dimensions `xyzl` as your GEMM dimensions** by permuting the input tiles of the `ct.mma` instruction, as well as reshaping so that `y` and `l` are merged.

Find one configuration (dimension sizes) where your kernel from b) performs better and one configuration where your new kernel from d) performs better.

e) Implement a cuTile kernel that computes the contraction `eabklxy, ecklyz -> eabcxz`. **Use dimensions `exyz` as your GEMM dimensions**, meaning that you perform a 3D `ct.mma` inside the kernel. Sequentialize all other K-dimensions, parallelize the remaining dimensions. The kernel should work with arbitrary dimension sizes.

**Verify** every kernel variant against `torch.einsum()`.

Use `triton.testing.do_bench` (or a similar benchmark function provided by torch/cupy) for all benchmarks.


## Task 2: Kernel Fusion

a) Implement a cuTile kernel for the contraction `eabklxy, ecklyz -> eabcxz` where you fuse an elementwise tensor multiplication of a tensor `D` of shape `eabcxz` with the output tensor. The output tensor can be overwritten by the multiplication.

b) Implement a kernel that computes the elementwise multiplication only. Compare runtime results of your fused kernel with sequentially calling the cuTile contraction kernel, then the elementwise multiplication. Choose tensor sizes such that the FLOP count of the contraction is similar to a 2048x2048x2048 matrix multiplication.

## Task 3: GEMM Dimension Size Sweep

a) Implement a contraction kernel that computes the contraction `ackm, bcnk -> abnm`. Assume fixed dimension sizes `|a| = 16`, `|b| = 16`, and `|c| = 32`. The kernel should be able to handle arbitrary sizes for dimensions `mnk`.

b) Perform the following benchmarks, visualize your results and *explain* your findings:
1. Assume dimension sizes `|k| = 64`, `|m| = 64`. Benchmark a dimension size sweep for dimension `n`, ranging **from 17 to 129** (non-power-of-two sizes included).
2. Assume dimension sizes `|m| = 64`, `|n| = 64`. Benchmark a dimension size sweep for dimension `k`, ranging **from 17 to 129** (non-power-of-two sizes included).
