---
description: Fixed-point-error restart criteria and epoch transitions in cuPDLPx.
---

# Adaptive restart

cuPDLPx restarts the Halpern iteration based on the fixed-point error and
the number of iterations since the last restart<sup>[\[1\]](#ref-lu-yang),[\[2\]](#ref-cupdlpx)</sup>.
The iterations between restarts form an epoch.

## Fixed-point error

For the PDHG operator $\mathcal T$, define the fixed-point error of a point
$(x,y)$ as

$$
r(x,y)=\left\lVert (x,y)-\mathcal T(x,y)\right\rVert_P,
$$

where $P$ is the canonical PDHG metric:

$$
P=
\begin{bmatrix}
\dfrac{\omega}{\eta}I & A^\top \\
A & \dfrac{1}{\eta\omega}I
\end{bmatrix}\succ0.
$$

PDLP uses the normalized duality gap for restarting, while the earlier
cuPDLP uses the KKT error<sup>[\[2\]](#ref-cupdlpx)</sup>.

Let $(x^{n,0},y^{n,0})$ be the anchor of epoch $n$,
$(x^{n,k},y^{n,k})$ its current iterate, and $N$ the total iteration count. A
restart occurs when any of the following conditions is satisfied.

## Sufficient reduction

$$
r(x^{n,k},y^{n,k})\le
\beta_{\mathrm{sufficient}}r(x^{n,0},y^{n,0}).
$$

This condition requires the fixed-point error to fall to a fraction
$\beta_{\mathrm{sufficient}}$ of its value at the anchor. The default is `0.2`.

## Necessary reduction and local increase

$$
\begin{aligned}
r(x^{n,k},y^{n,k})&\le
\beta_{\mathrm{necessary}}r(x^{n,0},y^{n,0}),\\
r(x^{n,k},y^{n,k})&>r(x^{n,k'},y^{n,k'}).
\end{aligned}
$$

Here $(x^{n,k'},y^{n,k'})$ is the iterate at the previous termination check,
[`termination_evaluation_frequency`](../reference/parameters.md#limits-and-logging)
iterations earlier. The error must be sufficiently below its value at the
anchor but larger than at the previous check. The default
$\beta_{\mathrm{necessary}}$ is `0.5`.

## Artificial epoch limit

$$
k\ge\beta_{\mathrm{artificial}}N.
$$

This condition limits the epoch length relative to the total iteration count.
The default $\beta_{\mathrm{artificial}}$ is `0.36`.

## Restart operation

At a restart, cuPDLPx:

1. replaces the anchor with the latest PDHG iterate;
2. resets the local Halpern iteration count;
3. updates the [primal weight](primal-weight.md).

The restart criteria are evaluated at every termination check.

## Parameters

See [Parameters](../reference/parameters.md#adaptive-restart) for restart settings.

## References

<span id="ref-lu-yang">\[1\]</span> Haihao Lu and Jinwen Yang.
[*Restarted Halpern PDHG for Linear Programming*](https://arxiv.org/abs/2407.16144),
2024.

<span id="ref-cupdlpx">\[2\]</span> Haihao Lu, Zedong Peng, and Jinwen Yang.
[*cuPDLPx: A Further Enhanced GPU-Based First-Order Solver for Linear
Programming*](https://arxiv.org/abs/2507.14051), 2025.
