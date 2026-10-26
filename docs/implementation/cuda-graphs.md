---
description: How cuPDLPx uses CUDA Graphs to reduce launch overhead and reuse GPU work across restarts.
---

# CUDA Graphs

cuPDLPx uses [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
to reduce kernel launch overhead. A graph records GPU operations and their
dependencies, allowing the sequence to be replayed with a single CPU launch.
[Kernel fusion](kernel-fusion.md) also reduces memory traffic within each
iteration.

<figure markdown="span">
  ![Individual kernel launches compared with a single CUDA Graph launch](../assets/cuda-graphs-launch-overhead.svg#only-light)
  ![Individual kernel launches compared with a single CUDA Graph launch](../assets/cuda-graphs-launch-overhead-dark.svg#only-dark)
  <figcaption>
    Launch overhead can leave gaps between short GPU kernels;
    a graph launch submits the recorded sequence together. Adapted from
    NVIDIA, <a href="https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s32082/">Effortless CUDA Graphs</a>, GTC 2021.
  </figcaption>
</figure>

## Graph capture and replay

cuPDLPx evaluates termination and restart criteria every
[`termination_evaluation_frequency`](../reference/parameters.md#limits-and-logging) iterations
(`200` by default). The iterations between consecutive checks form a
window, executed as follows:

1. Execute the first iteration of each window outside the graph. After a
   restart, its PDHG iterate gives the initial fixed-point error of the new
   epoch, the baseline for the [restart criteria](../algorithm/restart.md)
   and the [active-set step size boost](../algorithm/step-size.md#divergence-protection).
2. **Launch a graph** covering iterations 2 through
   `termination_evaluation_frequency` within the window. Before its first
   launch, the graph is captured and instantiated; subsequent windows reuse
   it. The final iteration also saves the intermediate values needed for
   the checks.
3. Compute residual vectors and perform reductions on the GPU. The CPU uses
   the resulting scalar measures to evaluate termination and restart criteria.

## Graph reuse across restarts

The iterate arrays and work buffers keep the same device addresses across
restarts. cuPDLPx updates the anchor, iteration counter, and step sizes in
GPU memory, so the same graph can be reused without recapture.

!!! note "Check frequency"

    Increasing `termination_evaluation_frequency` reduces graph launches,
    reductions, and host synchronization, but delays convergence checks,
    restarts, and detection of time or iteration limits.

ROCm builds use the corresponding HIP Graph APIs through the
[GPU backend compatibility layer](gpu-backends.md).
