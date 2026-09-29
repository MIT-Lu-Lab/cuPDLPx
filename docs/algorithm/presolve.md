---
description: LP reductions with PSLP and recovery of the original model's solution.
---

# Presolve and postsolve

cuPDLPx uses [PSLP](https://github.com/dance858/PSLP)<sup>[\[1\]](#ref-pslp)</sup>
to reduce the LP before applying r<sup>2</sup>HPDHG. Postsolve recovers the
primal and dual variables for the original LP.

## Presolve

Presolve is enabled by default and applies reductions such as:

| Reduction | What it does |
| --- | --- |
| Fixed-variable elimination | Substitute variables with equal lower and upper bounds into the constraints and objective. |
| Redundant-constraint removal | Remove constraints already implied by the remaining model. |
| Bound tightening | Use constraints and existing bounds to derive tighter variable bounds. |

Reducing the number of rows, columns, and nonzeros can lower the cost of
the sparse matrix–vector products in each PDHG iteration. PSLP records
these transformations for postsolve.

cuPDLPx skips the main iteration if presolve solves the LP or returns
an infeasible or infeasible-or-unbounded status.

!!! note

    When presolve is enabled, [termination checks](termination.md) are performed
    on the presolved model, not the original model.

## Postsolve

After the reduced LP is solved and any feasibility polishing is complete,
PSLP recovers the primal variables, dual variables, and dual slacks for the
original LP. cuPDLPx returns these recovered vectors.

!!! warning "Presolve and warm starts"

    User-provided warm starts require presolve to be disabled.

## References

<span id="ref-pslp">\[1\]</span> Daniel Cederberg and Stephen Boyd.
[*Presolving for GPU-Accelerated First-Order LP
Solvers*](https://arxiv.org/abs/2604.23951), 2026.
