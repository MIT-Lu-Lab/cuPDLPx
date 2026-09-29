---
description: PDHG, Halpern anchoring, and reflection in the base iteration of cuPDLPx.
---

# Base algorithm

cuPDLPx combines PDHG, Halpern anchoring, and reflection. Within each restart
epoch, let $(x^k,y^k)$ be the current primal–dual iterate and
$(x^0,y^0)$ the anchor. The index $k$ counts iterations within the epoch;
we omit the epoch index used in
[Algorithm 1](index.md#restarted-reflected-halpern-pdhg).

## PDHG

For the [bound-form LP](index.md#lp-formulation), let
$\mathcal X=[\ell_v,u_v]$ and $\mathcal S=[\ell_c,u_c]$.
With primal and dual [step sizes](step-size.md) $\tau$ and $\sigma$, a
primal–dual hybrid gradient (PDHG) step<sup>[\[1\]](#ref-chambolle-pock)</sup> is

$$
\begin{aligned}
\widehat x^{k+1}
&=\operatorname{proj}_{\mathcal X}
\left(x^k-\tau(c-A^\top y^k)\right),\\[0.4em]
\widehat y^{k+1}
&=y^k-\sigma A(2\widehat x^{k+1}-x^k)
-\sigma\operatorname{proj}_{-\mathcal S}
\left(\sigma^{-1}y^k-A(2\widehat x^{k+1}-x^k)\right).
\end{aligned}
$$

The primal step projects a gradient update onto the variable bounds. The
dual step uses the extrapolated primal iterate $2\widehat x^{k+1}-x^k$ and
a projection onto $-\mathcal S=[-u_c,-\ell_c]$. Both projections clip each
coordinate to its allowed interval.

Standard PDHG sets
$(x^{k+1},y^{k+1})=(\widehat x^{k+1},\widehat y^{k+1})$.
The fixed points of this update are primal–dual solutions of the LP.

## Halpern iteration

The Halpern iteration<sup>[\[2\]](#ref-halpern)</sup> takes a convex combination
of the PDHG update and a fixed anchor:

$$
\begin{aligned}
x^{k+1}
&=\frac{k+1}{k+2}\widehat x^{k+1}
+\frac{1}{k+2}x^0,\\[0.4em]
y^{k+1}
&=\frac{k+1}{k+2}\widehat y^{k+1}
+\frac{1}{k+2}y^0.
\end{aligned}
$$

The anchor remains fixed within each epoch, while its weight $1/(k+2)$
decreases with each iteration.

## Restart

Restart is an important technique for accelerating the convergence of
Halpern PDHG. The anchor draws the iterates toward the epoch's starting
point. When this point is far from a solution, anchoring can slow further
progress.

cuPDLPx therefore uses an [adaptive restart scheme](restart.md) based on the
fixed-point error and the number of iterations since the last restart.
At each restart, the anchor is replaced with the latest PDHG iterate and
$k$ is reset to zero.

## Reflection

Reflection (a special case of
**over-relaxation**<sup>[\[4\]](#ref-eckstein-bertsekas)</sup>) doubles the displacement
from the current iterate to its PDHG update:

$$
\begin{aligned}
\bar x^{k+1}&=2\widehat x^{k+1}-x^k,\\[0.4em]
\bar y^{k+1}&=2\widehat y^{k+1}-y^k.
\end{aligned}
$$

The reflected operator has the same fixed points as PDHG and, under the
step-size condition discussed in [Theoretical insights](#theoretical-insights),
is nonexpansive in the
[PDHG metric](restart.md#fixed-point-error): applying the operator to two
points does not increase the distance between them in that metric.
The Halpern iteration can therefore be applied to this
operator<sup>[\[3\]](#ref-lu-yang),[\[5\]](#ref-hpr-lp)</sup>.

With reflection coefficient $\gamma$, the combined update is

$$
\begin{aligned}
x^{k+1}
&=\frac{k+1}{k+2}
\left[\gamma\bar x^{k+1}+(1-\gamma)x^k\right]
+\frac{1}{k+2}x^0,\\[0.4em]
y^{k+1}
&=\frac{k+1}{k+2}
\left[\gamma\bar y^{k+1}+(1-\gamma)y^k\right]
+\frac{1}{k+2}y^0.
\end{aligned}
$$

Here $\gamma$ is
[`reflection_coefficient`](../reference/parameters.md#step-size-and-reflection).
The default $\gamma=1$ uses full reflection; $\gamma=\tfrac12$ recovers
the unreflected Halpern PDHG update. The primal reflection
$2\widehat x^{k+1}-x^k$ is also the extrapolated point used in the PDHG dual
step.

## Theoretical insights

For fixed step sizes satisfying $\tau\sigma\lVert A\rVert_2^2<1$, the PDHG
operator $\mathcal T$ is **firmly nonexpansive** in the
[PDHG metric](restart.md#fixed-point-error). Its reflection $2\mathcal T-I$
is therefore **nonexpansive**, which is sufficient for applying Halpern
iteration.

**Halpern iteration** uses anchoring to accelerate fixed-point iterations.
When the LP has a primal–dual solution, both Halpern PDHG and its reflected
variant have an $O(1/k)$ bound on the
[fixed-point error](restart.md#fixed-point-error)
(Lemmas 3 and 11)<sup>[\[3\]](#ref-lu-yang)</sup>.
Reflection thus allows a larger step while retaining this convergence
guarantee.

**Restart** improves this sublinear rate by exploiting the **sharpness of
LP**. For feasible and bounded LPs, sharpness bounds the distance to the
solution set by a constant multiple of the fixed-point error on bounded
regions. Combining this relationship with Halpern's $O(1/k)$ bound, suitable
restart rules yield accelerated linear convergence by refreshing the
anchor after sufficient progress.

For the precise assumptions, convergence rates, and proofs, see Sections 3
and 6 of the theoretical paper,
[*Restarted Halpern PDHG for Linear Programming*](https://arxiv.org/abs/2407.16144)<sup>[\[3\]](#ref-lu-yang)</sup>.

## References

<span id="ref-chambolle-pock">\[1\]</span> Antonin Chambolle and Thomas Pock.
[*A First-Order Primal-Dual Algorithm for Convex Problems with Applications
to Imaging*](https://doi.org/10.1007/s10851-010-0251-1).
*Journal of Mathematical Imaging and Vision*, 40:120–145, 2011.

<span id="ref-halpern">\[2\]</span> Benjamin Halpern.
[*Fixed Points of Nonexpanding Maps*](https://doi.org/10.1090/S0002-9904-1967-11864-0).
*Bulletin of the American Mathematical Society*, 73:957–961, 1967.

<span id="ref-lu-yang">\[3\]</span> Haihao Lu and Jinwen Yang.
[*Restarted Halpern PDHG for Linear Programming*](https://arxiv.org/abs/2407.16144),
2024.

<span id="ref-eckstein-bertsekas">\[4\]</span> Jonathan Eckstein and Dimitri P. Bertsekas.
[*On the Douglas–Rachford Splitting Method and the Proximal Point Algorithm
for Maximal Monotone Operators*](https://doi.org/10.1007/BF01581204).
*Mathematical Programming*, 55:293–318, 1992.

<span id="ref-hpr-lp">\[5\]</span> Kaihuang Chen, Defeng Sun, Yancheng Yuan, Guojun Zhang,
and Xinyuan Zhao.
[*HPR-LP: An Implementation of an HPR Method for Solving Linear
Programming*](https://arxiv.org/abs/2408.12179), 2024.
