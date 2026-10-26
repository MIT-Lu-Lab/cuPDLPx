---
description: Overview of the cuPDLPx solve pipeline and its GPU-oriented implementation.
---

# Implementation

cuPDLPx preprocesses the LP on the CPU and runs the main iteration on the GPU.
Sparse matrices and iterate vectors remain in GPU memory until solution
recovery.

## The solve pipeline

| Stage | Main job | Where it runs |
| --- | --- | --- |
| Preprocess | Remove small matrix entries; map sufficiently negative lower bounds to $-\infty$ and sufficiently positive upper bounds to $+\infty$ ([thresholds](../reference/parameters.md#scaling-and-preprocessing)) | CPU |
| [Presolve](../algorithm/presolve.md) | Reduce the numbers of variables, constraints, and matrix nonzeros with PSLP | CPU |
| GPU setup | Transfer data, [preconditioning](../algorithm/preconditioning.md), allocate workspaces, estimate $\lVert A\rVert_2$ | CPU + GPU |
| Main iteration | Apply reflected Halpern PDHG between termination checks | GPU |
| Termination and restart | Form residuals, check termination and restart criteria, update primal weight | GPU + CPU |
| Optional polishing | Solve primal and dual feasibility problems | GPU |
| Recovery | Undo scaling and apply postsolve, then assemble the result | CPU + GPU |

## Implementation details

- [GPU backends](gpu-backends.md) covers CUDA, ROCm/HIP, and build targets.
- [SpMV](sparse-matrix-vector-products.md) explains why
  both $A$ and $A^\top$ are stored and how the sparse backend is selected.
- [Kernel fusion](kernel-fusion.md) explains the fused primal and dual updates.
- [CUDA Graphs](cuda-graphs.md) explains capture, replay, and synchronization.
- [Results and status](../getting-started/results.md) documents the values
  returned after recovery.
- [Log interpretation](../getting-started/log-interpretation.md) explains the
  progress table and final solve summary.

See [Algorithm](../algorithm/index.md) for the mathematical formulation.
