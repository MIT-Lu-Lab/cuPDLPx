---
description: Sparse matrix storage and CUDA and ROCm sparse backends in cuPDLPx.
---

# SpMV

The two sparse matrix–vector products (SpMV) in the PDHG update are
highlighted below:

$$
\begin{aligned}
\widehat x^{k+1}
&=\operatorname{proj}_{\mathcal X}
\left(
x^k-\tau\left(c-
\htmlClass{spmv-highlight}{\boldsymbol{A^\top y^k}}
\right)
\right),\\[0.4em]
\widehat y^{k+1}
&=y^k
-\sigma\htmlClass{spmv-highlight}{\boldsymbol{A(2\widehat x^{k+1}-x^k)}}
-\sigma\operatorname{proj}_{-\mathcal S}
\left(
\sigma^{-1}y^k
-\htmlClass{spmv-highlight}{\boldsymbol{A(2\widehat x^{k+1}-x^k)}}
\right).
\end{aligned}
$$

cuPDLPx computes $A(2\widehat x^{k+1}-x^k)$ once and reuses the result in both
terms of the dual update.

cuPDLPx stores both $A$ and $A^\top$ in compressed sparse row (CSR) form.
This increases matrix storage, but lets both products use optimized
non-transposed SpMV kernels throughout the solve.

## CUDA sparse backends

For CUDA versions before 13.3, cuPDLPx uses `cusparseSpMV` with
`CUSPARSE_SPMV_CSR_ALG2`. This path is deterministic across runs. The solver
also calls `cusparseSpMV_preprocess` so the one-time analysis cost is amortized
over the repeated products.

Starting with CUDA 13.3, cuPDLPx uses `cusparseSpMVOp`. On the LP workloads
reported in the cuPDLPx paper, this backend is often faster, but it may request
a larger temporary buffer.

The sparse descriptors, preprocessing state, and temporary buffers are created
during initialization and reused. No sparse workspace allocation occurs in the
main iteration loop.

See the [cuSPARSE documentation for CUDA
13.3](https://docs.nvidia.com/cuda/archive/13.3.0/cusparse/contents.html)
for `cusparseSpMV`, `cusparseSpMV_preprocess`, and `cusparseSpMVOp`.

## ROCm/HIP

The HIP build uses `hipsparseSpMV` for the same two products. `cusparseSpMVOp`
is CUDA-specific, so it is not selected on the HIP path. The higher-level
iteration and solver interfaces are shared between both backends.

Build requirements and architecture selection are listed under
[GPU backends](gpu-backends.md).

The AMD [hipSPARSE generic API
reference](https://rocm.docs.amd.com/projects/hipSPARSE/en/latest/reference/generic.html)
documents `hipsparseSpMV`, preprocessing, and workspace requirements.
