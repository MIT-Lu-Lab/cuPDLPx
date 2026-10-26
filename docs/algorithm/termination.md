---
description: Primal feasibility, dual feasibility, and objective-gap termination checks used by cuPDLPx.
---

# Termination criteria

cuPDLPx checks optimality using relative primal and dual residuals and the
primal–dual gap. These quantities are evaluated on the unscaled LP, after
presolve if enabled. The formulas below use the default $\ell_2$ norm.
Selecting $\ell_\infty$ replaces the norms in both residual numerators
and denominators.

## Optimality conditions

The optimality test requires all three quantities to satisfy the following
bounds<sup>[\[1\]](#ref-cupdlpx)</sup>. Here $x$ denotes the primal variables,
$y$ the dual multipliers, $r$ the dual slacks, and $p$ the
[support function](index.md#dual-problem). The vector $b$ collects the finite
constraint bounds, counting each equality bound once.

- **Primal feasibility**

    $$
    \frac{
    \left\lVert Ax-\operatorname{proj}_{[\ell_c,u_c]}(Ax)\right\rVert_2
    }{
    1+\lVert b\rVert_2
    }
    <\varepsilon_{\mathrm{feas}}.
    $$

- **Dual feasibility**

    $$
    \frac{\left\lVert c-A^\top y-r\right\rVert_2}
    {1+\lVert c\rVert_2}
    <\varepsilon_{\mathrm{feas}}.
    $$

- **Relative objective gap**

    $$
    \frac{
    \left|c^\top x+p(-y;\ell_c,u_c)+p(-r;\ell_v,u_v)\right|
    }{
    1+\left|c^\top x+c_0\right|
    +\left|-p(-y;\ell_c,u_c)-p(-r;\ell_v,u_v)+c_0\right|
    }
    <\varepsilon_{\mathrm{opt}}.
    $$

Variable-bound feasibility and the dual-slack sign conditions are satisfied
by the primal projection and Moreau recovery, respectively. Both tolerances
default to `1e-4`.

## Dual-slack recovery

At each termination check, cuPDLPx recovers the dual-slack vector $r$
using the Moreau identity:

$$
\widehat r^{\,k+1}
=\tau^{-1}\left(\widehat x^{k+1}-x^{k}\right)+(c-A^\top y^k).
$$

The recovered vector is used to evaluate dual feasibility and the
primal–dual gap at the PDHG iterate $\widehat x^{k+1}$. The returned dual
slacks are computed separately from $c-A^\top y$, with each component
restricted to the sign permitted by its variable bounds.

## Unscaling

The iterations and dual-slack recovery use the
[scaled LP](preconditioning.md), with $\widetilde A=D_1AD_2$ and
$\widetilde c=\theta_cD_2c$. The variables map to the unscaled LP as

$$
x=\theta_b^{-1}D_2\widetilde x,
\qquad
y=\theta_c^{-1}D_1\widetilde y,
\qquad
r=\theta_c^{-1}D_2^{-1}\widetilde r.
$$

Residuals are unscaled before taking norms:

$$
\begin{aligned}
Ax-\operatorname{proj}_{[\ell_c,u_c]}(Ax)
&=\theta_b^{-1}D_1^{-1}
\Bigl(\widetilde A\widetilde x
-\operatorname{proj}_{[\widetilde\ell_c,\widetilde u_c]}
(\widetilde A\widetilde x)\Bigr),\\[0.4em]
c-A^\top y-r
&=\theta_c^{-1}D_2^{-1}
\bigl(\widetilde c-\widetilde A^\top\widetilde y-\widetilde r\bigr).
\end{aligned}
$$

The objective values are recovered from
$c^\top x=(\theta_b\theta_c)^{-1}\,\widetilde c^{\,\top}\widetilde x$ and its
dual counterpart. The denominators $1+\lVert b\rVert_2$ and
$1+\lVert c\rVert_2$ use the unscaled data. When presolve is enabled,
[postsolve](presolve.md#postsolve) then recovers the original model's
solution vectors.

## Check frequency

cuPDLPx checks optimality, infeasibility, and restart criteria every `200`
iterations by default. Time and iteration limits are checked at the same
interval, so the iteration count can exceed its limit by less than one
check interval.

An iterate returned after a time or iteration limit may not satisfy the
optimality conditions. See [Results and status](../getting-started/results.md)
for interpretation and [CUDA Graphs](../implementation/cuda-graphs.md)
for execution between checks.

## Parameters

See [Parameters](../reference/parameters.md#accuracy-and-termination) for
the tolerances and norm, and
[Limits and logging](../reference/parameters.md#limits-and-logging) for the
check interval.

## References

<span id="ref-cupdlpx">\[1\]</span> Haihao Lu, Zedong Peng, and Jinwen Yang.
[*cuPDLPx: A Further Enhanced GPU-Based First-Order Solver for Linear
Programming*](https://arxiv.org/abs/2507.14051), 2025.
