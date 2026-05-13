# Assignment 06: Multi-Input Einsum Contraction

In this assignment you will contract two intermediate tensors of a light-field tensor-ring decomposition loaded from disk, first by using PyTorch's `torch.einsum` as a reference, and then by building a cuTile kernel driven by the `Config`/`Optimizer` interface you implemented in Assignment 05.

All code should be written inside `src/` (`src/main.py` already contains some boilerplate code).

**Store** the [tensor data](https://cloud.uni-jena.de/s/4aeP53cgxoiXQEp) inside your assignment directory (next to the `src` directory). **Do not add the `data` directory to your git repository!**

We assume the following import conventions:

```python
import cuda.tile as ct
import cupy as cp
import numpy as np
import torch
import opt_einsum
import triton
```

---

## Task 1: PyTorch Reference Contraction

Two intermediate tensors of a light-field tensor-ring decomposition are stored in `data/lf_tr_64_intermediate.npz`:

| Name           | Shape             |
|----------------|-------------------|
| `tensor_acspx` | `(a, c, s, p, x)` |
| `tensor_bspy`  | `(b, s, p, y)`    |

The skeleton in `src/main.py` already loads both tensors as CPU numpy tensors.

a) **Classify** every index that appears in the two tensors. State which indices are of type M, N, K, or C (use the definitions from the lecture).

b) **Write** the einsum string for the contraction and compute the result `tensor_abcyx` using `torch.einsum`. Convert all tensors to torch tensors and move them to the GPU before calling `torch.einsum`. Run the contraction **twice**: once with `torch.float32` inputs and once with `torch.float16` inputs (cast the tensors before contracting).

c) **Visualize** both results side-by-side by calling the `plot_tensor()` helper provided in `src/main.py`. Save the fp32 result to `results/torch_32.png` and the fp16 result to `results/torch_16.png`. **Report** if you see any visible differences between the two images.

---

## Task 2: Generating a Basic Config

Use the `generate_config` function you implemented in Assignment 05.

a) Call `generate_config` with the einsum string from Task 1 and the shapes of `tensor_acspx` and `tensor_bspy` to produce an initial `Config`. You may choose either fp32 or fp16 as the data types for the config.

b) **Report** the resulting config (all fields).

---

## Task 3: Optimized Config

a) **Apply** optimizations to the configuration of Task 2 and ensure the config is valid and launchable. Optimize for performance.

b) **Report** the final optimized config (all fields).

---

## Task 4: cuTile Kernel

a) **Implement** a cuTile kernel that computes the contraction following your configuration from Task 3.

b) **Verify** correctness by comparing the kernel output against the `torch.einsum` result from Task 1 using `torch.allclose` with a suitable tolerance.

c) Use `triton.testing.do_bench` to measure the average kernel runtime. **Compute** and **report** the achieved performance in TFLOPS.

---

## Optional Task

Optimize your kernel so that its performance is higher than `torch.einsum()`.
