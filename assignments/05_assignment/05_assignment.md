# Assignment 05: Contraction Interface and L2 Optimization

In this assignment you will build a high-level configuration interface for tensor contractions, implement an optimizer that manipulates those configurations, and use it to derive and benchmark an L2-optimized cuTile kernel.

All code should be written in `src/`.

We assume the following import conventions:

```python
import cuda.tile as ct
import cupy as cp
import torch
import triton
from dataclasses import dataclass, field
```

**Use FP16 data type for tensor inputs and outputs, accumulate in FP32.**
We assume row-major order for all tensors.

---

## Task 1: Config Class

Define the following Python types that together represent a tensor contraction configuration as introduced in the lecture.

a) Define enumeration types (e.g. using Python's `enum.Enum` or simple class constants) for:

- **`DimType`**: `M`, `N`, `K`, `C`
- **`ExecType`**: `SEQ`, `PAR`, `PRIM`
- **`PrimType`**: `GEMM`, `BGEMM`
- **`LastType`**: `NONE`, `ELWISE_MUL`
- **`FirstType`**: `ZERO`
- **`DataType`**: `FLOAT16`, `FLOAT32`

b) Define a `Config` dataclass with the following fields, matching the interface
shown in the lecture:

| Field | Type | Description |
|---|---|---|
| `data_type` | `DataType` | Numeric precision of the operands |
| `prim_main` | `PrimType` | Main (B)GEMM primitive used inside the kernel |
| `prim_last` | `LastType` | Optional elementwise operation applied after the accumulation |
| `prim_first` | `FirstType` | Initialization of the accumulator |
| `dim_types` | `list[DimType]` | Per-dimension index type |
| `exec_types` | `list[ExecType]` | Per-dimension execution strategy |
| `dim_sizes` | `list[int]` | Per-dimension size |
| `strides` | `list[list[int]]` | Per-tensor, per-dimension stride (one inner list per tensor) |

---

## Task 2: Generating a Basic Config

Write a function `generate_config` that takes an einsum string and a list of shapes for the input tensors (the output shape is implied by the einsum) and returns a basic `Config`.

**Requirements:**

- Classify each dimension index automatically by inspecting in which tensors it appears.
- Compute strides for every tensor assuming **row-major layout**. A stride of `0` indicates that the dimension does not appear in that tensor.
- Set **all** `exec_types` to `SEQ`.
- Set `data_type = DataType.FLOAT16`, `prim_main = PrimType.GEMM`, `prim_last = LastType.NONE`, `prim_first = FirstType.ZERO`.

---

## Task 3: Optimizer Class

Implement a class `Optimizer` that wraps a `Config` and exposes methods to transform it.

a) **Implement** the function `split_dim(dim_id: int, outer_size: int, inner_size: int)`.

It splits one dimension into two. `outer_size * inner_size` must equal the original size; raise a `ValueError` otherwise.

After splitting:
- Insert two new dimensions at the position of the original dimension.
- The outer dimension (left) gets `size = outer_size`.
- The inner dimension (right) gets `size = inner_size`.
- Strides have to be updated accordingly.
- Both new dimensions inherit `dim_type` and `exec_type` from the original.

b) **Implement** the function `fuse_dims(dim_id_a: int, dim_id_b: int)`.

Fuse two dimensions into a single one. Two dimensions can only be fused if they are **adjacent** in every tensor they both appear in, i.e., the two dimensions are contiguous in memory (`stride[a] == stride[b] * size[b]` or `stride[a] * size[a] == stride[b]`) in every tensor.

Check this condition for all tensors before performing the fusion. Raise a descriptive `ValueError` if the check fails.

After a valid fusion:
- The new size is `size[a] * size[b]`.
- Update the strides lists accordingly.
- The fused dimension inherits the `dim_type` and `exec_type` of `dim_id_a`.
- Remove one dimension from all lists.

c) **Implement** the function `permute_dims(permutation: list[int])`.

Reorder all per-dimension lists (`dim_types`, `exec_types`, `dim_sizes`, and each tensor's strides list) according to `permutation`, following the syntax of `torch.permute`.

d) **Implement** the function `make_executable()`.

Set exec types and permute the config's dimensions so that the config becomes executable via cuTile. Use the parallel execution type where possible. Test the resulting configuration with your `verify()` function from e).

e) **Implement** the function `verify()`.

Check that the current configuration is executable. Raise a descriptive `ValueError` for each violated condition:

1. No `K`-dimension may have `exec_type = PAR`.
2. All dimensions with `exec_type = SEQ` must appear to the **left** of all dimensions with `exec_type = PRIM` in the config.
3. All dimensions with `exec_type = PAR` must appear to the **left** of all dimensions with `exec_type = SEQ` in the config.
4. The rightmost dimensions must be `PRIM` and the `PRIM` dimensions must include at least one dimension of each type `M`, `N`, and `K`.

---

## Task 4: L2-Optimized Batched Contraction

Consider the batched matrix multiplication expressed as `cmk, ckn -> cmn` with dimension sizes $|c| = 4$, $|m| = |n| = |k| = 4096$.

a) Use your `generate_config` function from Task 2 to produce the initial `Config` for this contraction. **Report** the resulting config.

b) Use your `Optimizer` and the implemented functions from Task 3 to transform the basic config into an L2-optimized one, following the general L2-reuse pattern from the lecture.
```
config.dim_sizes = [ [...], |m_l2|, |n_l2|, |m_prim|, |n_prim|, |k_prim|]
```

**Choose** the sizes for `m_l2`, `m_prim`, `n_l2`, `n_prim` and **justify** your choice with respect to L2 cache reuse.
**Report** the final config.

c) Implement the kernel

Implement a cuTile kernel that computes `cmk, ckn -> cmn` following your optimized config from b). **Verify** correctness of your kernel.

d) Use `triton.testing.do_bench` (or a similar benchmark function provided by cuTile/Torch) to measure the average kernel runtime. **Report** the achieved performance in TFLOPS.
**Compare** the performance of your L2-optimized kernel to a baseline kernel that maps BIDs in plain row-major order over `(c, m, n)` without any splitting or permuting. **Report** your findings.
