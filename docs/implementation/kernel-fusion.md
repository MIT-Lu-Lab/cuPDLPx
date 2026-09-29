---
description: How cuPDLPx fuses memory-bound PDHG vector operations into primal and dual GPU kernels.
---

# Kernel fusion

cuPDLPx fuses the affine updates, projections, reflection, and Halpern
updates into two vector kernels. These operations have low arithmetic
intensity; fusion reduces global memory traffic and kernel launches.
See the [Roofline model](https://modal.com/gpu-glossary/perf/roofline-model)
for the relationship between arithmetic intensity and memory bandwidth.

## Fused operations

The fused updates combine PDHG, reflection, and Halpern anchoring, with
vector operations highlighted below. Within each epoch, $(x^0,y^0)$ is the
fixed anchor and $k$ is the local iteration index.

$$
\begin{aligned}
x^{k+1}
&=\frac{k+1}{k+2}
\left[
\gamma\,\underbrace{\left(
2\htmlClass{vector-operation-highlight}{\operatorname{proj}_{\mathcal X}}
\left(
x^k\htmlClass{vector-operation-highlight}{\boldsymbol{-}}
\tau\left(c\htmlClass{vector-operation-highlight}{\boldsymbol{-}}
A^\top y^k
\right)
\right)
\htmlClass{vector-operation-highlight}{\boldsymbol{-}}x^k
\right)}_{\bar x^{k+1}}
\htmlClass{vector-operation-highlight}{\boldsymbol{+}}
(1-\gamma)x^k
\right]
\htmlClass{vector-operation-highlight}{\boldsymbol{+}}
\frac{1}{k+2}x^0,\\[0.6em]
y^{k+1}
&=\frac{k+1}{k+2}
\left[
y^k
\htmlClass{vector-operation-highlight}{\boldsymbol{-}}
2\gamma\sigma
\left(
A\bar x^{k+1}
\htmlClass{vector-operation-highlight}{\boldsymbol{+}}
\htmlClass{vector-operation-highlight}{\operatorname{proj}_{-\mathcal S}}
\left(
\sigma^{-1}y^k
\htmlClass{vector-operation-highlight}{\boldsymbol{-}}
A\bar x^{k+1}
\right)
\right)
\right]
\htmlClass{vector-operation-highlight}{\boldsymbol{+}}
\frac{1}{k+2}y^0.
\end{aligned}
$$

The primal kernel computes $\bar x^{k+1}$ once and reuses it for both the
primal reflection and the PDHG extrapolation in the dual update.

## Kernel execution

Each iteration uses two SpMV calls and two fused vector kernels:

1. **SpMV:** compute $A^\top y^k$ for the primal update.
2. **Fused primal kernel:** perform the primal PDHG, reflection, and Halpern updates.
3. **SpMV:** compute $A\bar x^{k+1}$ for the dual update.
4. **Fused dual kernel:** perform the dual PDHG, reflection, and Halpern updates.

Each GPU thread computes all three updates for one coordinate.
The primal kernel writes $x^{k+1}$ and
$\bar x^{k+1}$ to global memory, and the dual kernel writes $y^{k+1}$.
The second SpMV uses the stored vector $\bar x^{k+1}$.

!!! note "Major iterations"

    cuPDLPx evaluates termination and restart criteria at regular intervals;
    these iterations are called **major iterations**. For these checks, the
    fused kernels also save the PDHG iterates and reflected dual values to global
    memory. These values otherwise remain in registers. The primal kernel also performs
    [dual-slack recovery](../algorithm/termination.md#dual-slack-recovery).
