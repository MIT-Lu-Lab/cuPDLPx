---
description: Row, column, objective, and bound scaling in cuPDLPx.
---

# Preconditioning

cuPDLPx rescales the constraint matrix, objective, and bounds to improve
numerical conditioning and accelerate convergence.

Positive diagonal matrices $D_1$ and $D_2$ scale the rows and columns of $A$,
and two positive scalars $\theta_b$ and $\theta_c$ scale the bounds and the
objective. The scaled data are

$$
\begin{aligned}
\widetilde A&=D_1AD_2,\\
\widetilde c&=\theta_c\,D_2c,\\
(\widetilde\ell_c,\widetilde u_c)&=\theta_b\,D_1(\ell_c,u_c),\\
(\widetilde\ell_v,\widetilde u_v)&=\theta_b\,D_2^{-1}(\ell_v,u_v).
\end{aligned}
$$

To compute $D_1$ and $D_2$, cuPDLPx applies geometric mean scaling as used
in Gurobi<sup>[\[1\]](#ref-gurobi-scaleflag)</sup>, followed by Ruiz
equilibration and Pock–Chambolle scaling as used in
PDLP<sup>[\[2\]](#ref-pdlp)</sup>. Each stage rescales $A$ and accumulates
its row and column factors in $D_1$ and $D_2$. Empty rows and columns have
scaling factor $1$.

The final [objective and bound scaling](#objective-and-bound-scaling) stage
follows HPR-LP<sup>[\[3\]](#ref-hpr-lp)</sup> and determines $\theta_b$ and
$\theta_c$. Both are $1$ when this stage is disabled.

## Geometric mean scaling

One geometric mean scaling pass<sup>[\[1\]](#ref-gurobi-scaleflag)</sup> first scales the rows,
then the columns. For each row, the factor is the reciprocal geometric mean
of its smallest and largest nonzero entry magnitudes:

$$
d_{1,i}=
\left(\min_{j:\,a_{ij}\ne 0}|a_{ij}|
\;\max_{j:\,a_{ij}\ne 0}|a_{ij}|\right)^{-1/2}.
$$

After applying the row factors, compute and apply the column factors from the
row-scaled matrix:

$$
d_{2,j}=
\left(\min_{i:\,a_{ij}\ne 0}|a_{ij}|
\;\max_{i:\,a_{ij}\ne 0}|a_{ij}|\right)^{-1/2}.
$$

The number of row–column passes defaults to `12`; setting it to `0`
disables this stage.

## Ruiz equilibration scaling

One Ruiz iteration<sup>[\[2\]](#ref-pdlp)</sup> computes the row and column factors
from the same matrix and applies them together:

$$
d_{1,i}=\left(\max_{j}|a_{ij}|\right)^{-1/2},
\qquad
d_{2,j}=\left(\max_{i}|a_{ij}|\right)^{-1/2}.
$$

Repeating the iteration balances the row and column $\ell_\infty$ norms.
The number of iterations defaults to `10`.

## Pock–Chambolle scaling

The Pock–Chambolle factors<sup>[\[2\]](#ref-pdlp)</sup> are computed from the
same matrix and applied together:

$$
d_{1,i}=\left(\sum_{j}|a_{ij}|^\alpha\right)^{-1/2},
\qquad
d_{2,j}=\left(\sum_{i}|a_{ij}|^{2-\alpha}\right)^{-1/2}.
$$

At the default $\alpha=1$, they are the reciprocal square roots of the
corresponding row and column $\ell_1$ norms. This stage is enabled by
default.

## Objective and bound scaling

The final stage runs after the diagonal scaling and multiplies the scaled
data by two scalars<sup>[\[3\]](#ref-hpr-lp)</sup>:

$$
\theta_b=\frac{1}{1+\|b\|_2},
\qquad
\theta_c=\frac{1}{1+\|c\|_2}.
$$

Here $b$ collects the finite entries of the scaled $\ell_c$ and $u_c$,
counting each equality bound once, and $c$ is the scaled
objective vector. The constraint and variable bounds are multiplied by
$\theta_b$, and the objective vector by $\theta_c$.
This stage is enabled by default. Whether it runs also decides the
[initial primal weight](primal-weight.md#initial-weight).

## Parameters

See [Parameters](../reference/parameters.md#scaling-and-preprocessing) for
scaling settings.

## References

<span id="ref-gurobi-scaleflag">\[1\]</span> Gurobi Optimization.
[*ScaleFlag*](https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html#scaleflag).
Gurobi Optimizer Reference Manual.

<span id="ref-pdlp">\[2\]</span> David Applegate, Mateo Díaz, Oliver Hinder, Haihao Lu,
Miles Lubin, Brendan O'Donoghue, and Warren Schudy.
[*Practical Large-Scale Linear Programming Using Primal-Dual Hybrid
Gradient*](https://proceedings.neurips.cc/paper/2021/hash/a8fbbd3b11424ce032ba813493d95ad7-Abstract.html).
*NeurIPS*, 2021.

<span id="ref-hpr-lp">\[3\]</span> Kaihuang Chen, Defeng Sun, Yancheng Yuan, Guojun Zhang,
and Xinyuan Zhao.
[*HPR-LP: An Implementation of an HPR Method for Solving Linear
Programming*](https://arxiv.org/abs/2408.12179), 2024.
