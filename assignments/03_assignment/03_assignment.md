# Assignment 03: Matrix Multiplication with cuTile

In this assignment you will implement and optimize GPU matrix multiplication kernels using [cuTile](https://github.com/nvidia/cutile-python). You will explore the impact of numeric precision, tiling strategies, and memory access patterns on performance.

All code should be written in `src/`, one file per task.

We assume the following import conventions:

```python
import cuda.tile as ct
import cupy as cp
import torch
import triton
```

---

## Task 1: FP32 vs FP16 Performance

a) **Your task** is to implement two cuTile kernels that each compute a matrix multiplication $A \times B = C$ with `shape(A) = (64, 4096)`, `shape(B) = (4096, 64)` and `shape(C) = (64, 64)`. 

1. **`kernel_fp16`**: Inputs `A` and `B` are `FP16`, accumulator/output `C` is `FP32`
2. **`kernel_fp32`**: Inputs `A` and `B` are `FP32`, accumulator/output `C` is `FP32`

Both kernels should use `ct.mma` to perform the tile-level multiply-accumulate. Use a single CTA (1 block in the grid) per kernel launch and use the fixed tile shape of `(m_tile=64, n_tile=64, k_tile=64)`.

**Verify** that both kernels compute correct results via `torch.matmul` using `torch.allclose`.

b) Use `triton.testing.do_bench` (or an equivalent benchmark function from `torch` / `cupy`) to measure the average kernel runtime for both variants. **Report** the measured runtimes and the resulting **speedup** of `kernel_fp16` over `kernel_fp32`.

---

## Task 2: Simple Matrix Multiplication Kernel

Write a cuTile kernel that computes `C = A @ B` for input matrices:

- `A` of shape `(M, K)` 
- `B` of shape `(K, N)`
- `C` of shape `(M, N)`

**Requirements:**

- Each kernel program is responsible for producing one output tile of shape `(m_tile, n_tile)`
- The kernel should work with tile sizes that are specified by the calling function
- Map block IDs (BIDs) in row-major order: `BID 0` covers the top-left output tile, `BID 1` the tile to its right, and so on, wrapping to the next row when the current row of tiles is exhausted
- **The kernel should support matrix shapes that are not powers of 2**
- Use `ct.mma` for the inner accumulation step

**Verify** correctness by comparing `C` against `torch.matmul(A, B)` using `torch.allclose`.

---

## Task 3: Benchmarking the Matrix Multiplication Kernel

Use the kernel from Task 2 for all benchmarks in this task. Report all performance numbers in **TFLOPS**, computed as:

$$
\text{TFLOPS} = \frac{2 \cdot M \cdot N \cdot K}{t_s \cdot 10^{12}}
$$

where $t_s$ is the measured kernel runtime in seconds.

a) Benchmark your kernel with tile shapes `(64, 64, 64)` for square matrix multiplications of sizes:

$$M = N = K \in \{256,\ 512,\ 1024,\ 2048,\ 4096,\ 8192\}$$

**Plot** the achieved TFLOPS and **report** your observations.

b) Fix the matrix size at `2048 × 2048 × 2048`, as well as `512 × 512 × 512`, and benchmark all tile shape combinations (27 total):

$$m\_{tile},\ n\_{tile},\ k\_{tile}\ \in \{32,\ 64,\ 128\}$$

**Visualize** your results as a **heatmap** with `m_tile` on one axis and `n_tile` on the other, fixing `k_tile = 64`.

**Report** the best-performing tile shape combination.

---

## Task 4: L2 Cache Optimization via Block Swizzling

a) **Your task** is to implement a swizzled matrix multiplication kernel. The requirements are the same as in _Task 2_, except block IDs should not be mapped in row-major order. Swizzle them for L2 cache reuse. You can assume a contraction dimension size of `4096`.

**Report** how you choose to map the BIDs and why. **Verify** correctness of the swizzled kernel against `torch.matmul`.

b) Repeat the tile shape sweep from _Task 3b_ for your swizzled kernel and **report** the best performing tile shape combination. **Compare** the performance of your swizzled kernel to the performance of your kernel from _Task 2_ for a matrix multiplication of shape `8192 × 8192 × 4096` (`m × n × k`).

