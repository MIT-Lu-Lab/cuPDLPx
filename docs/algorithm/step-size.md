---
description: Spectral-norm-based step size and active-set step size boost in cuPDLPx.
---

# Step size

cuPDLPx initializes the step size from the spectral norm of the scaled
constraint matrix. The [active-set step size boost](#active-set-step-size-boost)
can increase it near a solution.

## Initial constant step size

For the scaled constraint matrix $A$, the initial step size
is<sup>[\[1\]](#ref-cupdlpx)</sup>

$$
\eta=\frac{0.998}{\lVert A\rVert_2}.
$$

The spectral norm is estimated by power iteration on $AA^\top$ after
[preconditioning](preconditioning.md).
The primal and dual step sizes are

$$
\tau=\frac{\eta}{\omega},\qquad
\sigma=\eta\omega.
$$

The factor `0.998` provides a safety margin relative to the estimated norm.
The canonical PDHG metric is positive definite when
$\eta<1/\lVert A\rVert_2$. At each restart, the
[primal-weight controller](primal-weight.md) can change $\tau$ and $\sigma$
while preserving $\tau\sigma=\eta^2$ for a fixed $\eta$.

## Active-set step size boost

The initial step size $\eta^0=0.998/\lVert A\rVert_2$ uses the full constraint
matrix. Near a solution, inactive constraints and variables that remain at
their bounds can permit a larger step size. The active-set step size boost
(ASB) retains potentially binding constraints in $B$ and variables not
identified as persistently clamped at a bound in $F$. The tests below
define these sets. The resulting submatrix $A_{B,F}$ satisfies

$$
\lVert A_{B,F}\rVert_2\le\lVert A\rVert_2.
$$

ASB estimates $\lVert A_{B,F}\rVert_2$ and uses it to propose a larger step
size at an adaptive restart. A divergence check can trigger rollback and
start a new epoch. ASB is enabled by default.

### Active set

The sets $B$ and $F$ retain constraints and variables that do not meet the
exclusion tests over the [trailing window](#trailing-window).
These tests use the PDHG update from $(x,y)$ to $(\widehat x,\widehat y)$:

$$
\begin{aligned}
\widehat x
&=\operatorname{proj}_{[\ell_v,u_v]}
\left(x-\tau(c-A^\top y)\right),\\[0.4em]
\widehat y
&=y-\sigma A(2\widehat x-x)
-\sigma\operatorname{proj}_{[-u_c,-\ell_c]}
\left(\sigma^{-1}y-A(2\widehat x-x)\right).
\end{aligned}
$$

**Variables.** The primal projection gives the dual-slack estimate

$$
\tilde r=c-A^\top y+\tau^{-1}(\widehat x-x).
$$

A positive $\tilde r_i$ indicates projection onto the lower bound; a
negative value indicates projection onto the upper bound. Variable $i$
meets the exclusion test when $\ell_{v,i}=u_{v,i}$ or when

$$
\begin{aligned}
\tilde r_i&>\varepsilon_r\max\{1,|c_i|\}
&&\text{if } \ell_{v,i}>-\infty,
\qquad\text{or}\\
\tilde r_i&<-\varepsilon_r\max\{1,|c_i|\}
&&\text{if } u_{v,i}<\infty.
\end{aligned}
$$

**Constraints.** The dual update projects

$$
\tilde s=\sigma^{-1}y-A(2\widehat x-x)
$$

onto $[-u_c,-\ell_c]$. It gives $\widehat y_j=0$ when $\tilde s_j$ lies
in this interval. Constraint $j$ meets the exclusion test when
$\ell_{c,j}<u_{c,j}$ and $\tilde s_j$ lies inside $[-u_{c,j},-\ell_{c,j}]$
with a margin at each finite endpoint:

$$
\begin{aligned}
\tilde s_j+u_{c,j}&>\varepsilon_c\max\{1,|\tilde s_j|,|u_{c,j}|\}
&&\text{if } u_{c,j}<\infty,\\
-\ell_{c,j}-\tilde s_j&>\varepsilon_c\max\{1,|\tilde s_j|,|\ell_{c,j}|\}
&&\text{if } \ell_{c,j}>-\infty.
\end{aligned}
$$

Both tolerances default to `1e-8`. The submatrix $A_{B,F}$ contains the rows in
$B$ and the columns in $F$.

### Activation

#### Trailing window

The exclusion tests run at every termination check. An index is removed
from $B$ or $F$ only if it meets the exclusion test at every check over the
last `10000` iterations by default. It is restored as soon as it fails the test.

#### Activation tolerance

ASB activates when the relative primal residual, relative dual residual,
and relative primal–dual gap are all below the activation tolerance
($10^{-4}$ by default), and at least one index has been excluded.
See [Termination criteria](termination.md#optimality-conditions)
for the residual definitions.

### Step-size increase

After activation, ASB uses adaptive restarts to estimate

$$
\hat\sigma\approx\lVert A_{B,F}\rVert_2
$$

by power iteration, setting entries outside $B$ and $F$ to zero after each
product and warm-starting from the previous eigenvector.

ASB re-estimates the norm when the accumulated additions and removals
reach `1%` of the current $|B|+|F|$ by default. It skips the estimate if the
step-size limit rules out an increase, and stops power iteration early if
the running estimate does so. These decisions are retained until the same
change threshold is reached.

For a positive norm estimate, the proposed step size and acceptance condition are

$$
\eta^{+}
=\max\left\{\eta^0,\
\min\left\{\frac{\alpha}{\hat\sigma},\ \eta_{\mathrm{ceil}}\right\}\right\},
\qquad
\eta\leftarrow\eta^{+}
\ \text{ if }\ \eta^{+}>\eta
\ \text{ and }\ \eta^{+}\ge\rho\,\eta.
$$

Here $\alpha$ is a safety factor (default `0.9`), $\rho$ is the minimum
increase ratio (default `1.1`), and the upper limit
$\eta_{\mathrm{ceil}}$ is initially $\infty$.
When an increase is accepted, ASB saves the restart iterate, primal weight,
and controller state for rollback.

### Divergence protection

While $\eta>\eta^0$, ASB monitors the
[fixed-point error](restart.md#fixed-point-error) $r$, with the metric $P$
evaluated at the initial step size. The divergence check triggers when the
error or residuals are nonfinite, or when

$$
r(z^k)>(1+\delta)\,r(z^0).
$$

Here $z^0$ is the first iterate of the current epoch, and $\delta$ defaults
to `0.05`. A rollback

- restores the saved iterate, primal weight, and controller state;
- resets the step size to $\eta^0$ and discards $\hat\sigma$;
- resets the local iteration count;
- sets $\eta_{\mathrm{ceil}}$ to `0.7` times the rejected step size by default.

By default, ASB is disabled for the rest of the solve after two rollbacks.
The C result and command-line output report step-size increases, rollbacks,
and power iteration counts.

## Parameters

See [Step size and reflection](../reference/parameters.md#step-size-and-reflection)
for power iteration settings and
[Active-set step size boost](../reference/parameters.md#active-set-step-size-boost) for
ASB settings.

## References

<span id="ref-cupdlpx">\[1\]</span> Haihao Lu, Zedong Peng, and Jinwen Yang.
[*cuPDLPx: A Further Enhanced GPU-Based First-Order Solver for Linear
Programming*](https://arxiv.org/abs/2507.14051), 2025.
