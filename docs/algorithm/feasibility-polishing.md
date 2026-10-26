---
description: Primal and dual feasibility polishing after the main solve in cuPDLPx.
---

# Feasibility polishing

Feasibility polishing solves separate primal and dual feasibility problems
after the main solve. It can reduce feasibility residuals, but may increase
the primal–dual gap. Polishing is disabled by default.

!!! note "Scheduling"

    cuPDLPx uses the feasibility problems from PDLP
    (Algorithm 4)<sup>[\[1\]](#ref-pdlp)</sup>, but runs polishing only after the
    main solve. PDLP interleaves polishing with the main iterations.

## Activation

When enabled, polishing runs unless

- the main solve ended with primal or dual infeasibility, or
- both relative feasibility residuals already meet the polishing tolerance,
  `1e-6` by default.

The primal phase runs first, followed by the dual phase.

## Primal polishing

$$
\begin{aligned}
\operatorname*{minimize}_{x\in\mathbb R^n}\quad
  & 0 \\
\text{subject to}\quad
  & \ell_c \le Ax \le u_c, \\
  & \ell_v \le x \le u_v.
\end{aligned}
$$

The primal phase initializes $x$ from the main solve and sets $y=0$.
It applies restarted reflected Halpern PDHG, using the relative primal
residual as the convergence criterion.

## Dual polishing

$$
\begin{aligned}
\operatorname*{maximize}_{y\in\mathbb R^m,\,r\in\mathbb R^n}\quad
  & 0 \\
\text{subject to}\quad
  & c-A^\top y=r, \\
  & y\in\mathcal Y, \\
  & r\in\mathcal R.
\end{aligned}
$$

Here $r$ is the dual-slack vector, and $\mathcal Y$ and $\mathcal R$ impose
the [dual sign constraints](index.md#dual-problem). The dual phase
initializes $y$ from the main solve and sets $x=0$. It uses the relative
dual residual as the convergence criterion.

## Result

Each phase replaces its corresponding primal or dual values only if it
reaches the polishing tolerance. Otherwise, the values from the main solve
are retained. The objective values and primal–dual gap reflect the accepted
updates, but the main solve's termination status is unchanged. The final
gap can therefore exceed the optimality tolerance even when the status is
`OPTIMAL`. Check the final residuals and gap before using the result; see
[Results and status](../getting-started/results.md).

## Parameters

See [Parameters](../reference/parameters.md#feasibility-polishing) to enable
polishing and set its tolerance.

## References

<span id="ref-pdlp">\[1\]</span> David Applegate et al.
[*PDLP: A Practical First-Order Method for Large-Scale Linear
Programming*](https://arxiv.org/abs/2501.07018), 2025.
