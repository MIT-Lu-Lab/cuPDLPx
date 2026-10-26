---
description: LP formulation and the restarted reflected Halpern PDHG algorithm implemented by cuPDLPx.
---

# Algorithm

cuPDLPx solves linear programs using **restarted reflected Halpern
primal–dual hybrid gradient (r<sup>2</sup>HPDHG)**<sup>[\[1\]](#ref-cupdlpx)</sup>.
Each iteration updates the primal and dual iterates using sparse
matrix–vector products and projections onto the bound constraints.

## LP formulation

### Primal problem

cuPDLPx solves linear programs in bound form:

$$
\begin{aligned}
\operatorname*{minimize}_{x\in\mathbb R^n}\quad
  & c^\top x + c_0 \\
\text{subject to}\quad
  & \ell_c \le Ax \le u_c, \\
  & \ell_v \le x \le u_v.
\end{aligned}
$$

Here $A\in\mathbb R^{m\times n}$ is the constraint matrix,
$c\in\mathbb R^n$ is the objective vector, and $c_0$ is the objective
constant. The vectors $\ell_c,u_c$ and $\ell_v,u_v$ are the constraint
and variable bounds, respectively. Bounds may be infinite.

Define the variable and constraint sets

$$
\mathcal X=\{x:\ell_v\le x\le u_v\},\qquad
\mathcal S=\{s:\ell_c\le s\le u_c\}.
$$

### Dual problem

$$
\begin{aligned}
\operatorname*{maximize}_{y\in\mathcal Y,\,r\in\mathcal R}\quad
  & -p(-y;\ell_c,u_c)-p(-r;\ell_v,u_v)+c_0 \\
\text{subject to}\quad
  & c-A^\top y=r.
\end{aligned}
$$

The support function of the interval $[\ell,u]$ is

$$
p(z;\ell,u)=u^\top z^+-\ell^\top z^-,
\qquad
z^+=\max(z,0),\quad z^-=\max(-z,0).
$$

Let $\mathbb R_+=[0,+\infty)$ and $\mathbb R_-=(-\infty,0]$. The dual and
dual-slack domains are the Cartesian products
$\mathcal Y=\prod_{i=1}^m\mathcal Y_i$ and
$\mathcal R=\prod_{j=1}^n\mathcal R_j$, where

$$
\mathcal Y_i=
\begin{cases}
\{0\}, & \ell_{c,i}=-\infty,\ u_{c,i}=+\infty,\\
\mathbb R_-, & \ell_{c,i}=-\infty,\ u_{c,i}\in\mathbb R,\\
\mathbb R_+, & \ell_{c,i}\in\mathbb R,\ u_{c,i}=+\infty,\\
\mathbb R, & \text{otherwise},
\end{cases}
\qquad
\mathcal R_j=
\begin{cases}
\{0\}, & \ell_{v,j}=-\infty,\ u_{v,j}=+\infty,\\
\mathbb R_-, & \ell_{v,j}=-\infty,\ u_{v,j}\in\mathbb R,\\
\mathbb R_+, & \ell_{v,j}\in\mathbb R,\ u_{v,j}=+\infty,\\
\mathbb R, & \text{otherwise}.
\end{cases}
$$

Here $r$ is the dual-slack variable.

### Saddle-point formulation

The LP is equivalent to

$$
\min_{x\in\mathcal X}\max_{y\in\mathcal Y}
L(x,y)
=c^\top x+c_0-y^\top Ax-p(y;-u_c,-\ell_c).
$$

The objective constant $c_0$ shifts both objective values without changing
the feasible set or the PDHG iterates. The
paper<sup>[\[1\]](#ref-cupdlpx)</sup> uses $c_0=0$.

## Restarted reflected Halpern PDHG

The algorithm applies reflection and Halpern anchoring to the PDHG update,
with adaptive restarts<sup>[\[1\]](#ref-cupdlpx)</sup>.

<div class="latex-algorithm" role="group" aria-labelledby="algorithm-1-caption">
  <div class="latex-algorithm__caption" id="algorithm-1-caption">
    <strong>Algorithm 1.</strong> Restarted reflected Halpern PDHG
  </div>
  <div class="latex-algorithm__input">
    <strong>Input:</strong> An LP, an initial point
    $(x^{0,0},y^{0,0})$, and a termination tolerance $\varepsilon$.
  </div>
  <ol class="latex-algorithm__lines">
    <li>
      <strong>Precondition.</strong> Apply geometric mean scaling, Ruiz equilibration,
      Pock–Chambolle scaling, and objective and bound scaling.
    </li>
    <li>
      <strong>Initialize.</strong> Set
      $$
      \eta=\frac{0.998}{\lVert A\rVert_2},\qquad
      \omega^0=1,\qquad
      \tau=\frac{\eta}{\omega^0},\qquad
      \sigma=\eta\omega^0.
      $$
    </li>
    <li>
      <span class="latex-algorithm__keyword">for</span>
      $n=0,1,\ldots$ <span class="latex-algorithm__keyword">do</span>
    </li>
    <li class="latex-algorithm__indent-1">
      Set $k\leftarrow 0$.
    </li>
    <li class="latex-algorithm__indent-1">
      <span class="latex-algorithm__keyword">repeat</span>
    </li>
    <li class="latex-algorithm__indent-2">
      <div class="latex-algorithm__operation">
        <strong>PDHG update.</strong>
        $$
        \begin{aligned}
        \widehat x^{n,k+1}
        &=\operatorname{proj}_{\mathcal X}
        \left(x^{n,k}-\tau(c-A^\top y^{n,k})\right),\\[0.4em]
        \widehat y^{n,k+1}
        &=y^{n,k}
        -\sigma A(2\widehat x^{n,k+1}-x^{n,k})
        -\sigma\operatorname{proj}_{[-u_c,-\ell_c]}
        \left(\sigma^{-1}y^{n,k}
        -A(2\widehat x^{n,k+1}-x^{n,k})\right).
        \end{aligned}
        $$
      </div>
    </li>
    <li class="latex-algorithm__indent-2">
      <div class="latex-algorithm__operation latex-algorithm__operation--check">
        <strong>Termination check.</strong>
        <div class="latex-algorithm__statement">
          <span class="latex-algorithm__keyword">if</span>
          $\operatorname{KKT}(\widehat x^{n,k+1},\widehat y^{n,k+1})
          \le\varepsilon$
          <span class="latex-algorithm__keyword">then</span>
        </div>
        <div class="latex-algorithm__statement latex-algorithm__statement--indent">
          <span class="latex-algorithm__keyword">return</span>
          $(\widehat x^{n,k+1},\widehat y^{n,k+1})$.
        </div>
        <div class="latex-algorithm__statement">
          <span class="latex-algorithm__keyword">end if</span>
        </div>
      </div>
    </li>
    <li class="latex-algorithm__indent-2">
      <div class="latex-algorithm__operation">
        <strong>Reflection and Halpern update.</strong>
        $$
        \begin{aligned}
        x^{n,k+1}
        &=\frac{k+1}{k+2}
          \left(\gamma\bigl(2\widehat x^{n,k+1}-x^{n,k}\bigr)
          +(1-\gamma)x^{n,k}\right)
          +\frac{1}{k+2}x^{n,0},\\
        y^{n,k+1}
        &=\frac{k+1}{k+2}
          \left(\gamma\bigl(2\widehat y^{n,k+1}-y^{n,k}\bigr)
          +(1-\gamma)y^{n,k}\right)
          +\frac{1}{k+2}y^{n,0}.
        \end{aligned}
        $$
      </div>
    </li>
    <li class="latex-algorithm__indent-2">
      Set $k\leftarrow k+1$.
    </li>
    <li class="latex-algorithm__indent-1">
      <span class="latex-algorithm__keyword">until</span> a restart condition
      holds.
    </li>
    <li class="latex-algorithm__indent-1">
      <div class="latex-algorithm__operation latex-algorithm__operation--restart">
        <strong>Restart.</strong>
        <div class="latex-algorithm__statement">Set the next anchors:</div>
        $$
        (x^{n+1,0},y^{n+1,0})
        =(\widehat x^{n,k},\widehat y^{n,k}).
        $$
        <div class="latex-algorithm__statement">
          Update $\omega^{n+1}$ with the PID controller, then update $\tau$
          and $\sigma$.
        </div>
      </div>
    </li>
    <li>
      <span class="latex-algorithm__keyword">end for</span>
    </li>
  </ol>
</div>

The algorithm uses the following notation:

- $\operatorname{proj}_{\mathcal X}$: projection onto the variable bounds,
  computed by clipping each coordinate to its interval;
- $\eta$: the step size, based on the scaled matrix's spectral norm;
  the [active-set step size boost](step-size.md#active-set-step-size-boost) may increase it at a restart;
- $\omega^n$: the primal weight for epoch $n$; the initial value
  $\omega^0=1$ assumes the default objective and bound scaling described
  under [Preconditioning](preconditioning.md);
- $\tau$ and $\sigma$: the primal and dual step sizes, respectively;
- $\gamma$: the reflection coefficient
  ([`reflection_coefficient`](../reference/parameters.md#step-size-and-reflection), default `1`),
  which weights the reflected point $2\widehat x^{n,k+1}-x^{n,k}$ against
  the current iterate $x^{n,k}$;
- $n$ and $k$: the restart epoch and the iteration within that epoch
  (here $n$ is an epoch index, rather than the variable count in the LP);
- $(x^{n,0},y^{n,0})$: the primal and dual anchors for epoch $n$;
- $(\widehat x^{n,k+1},\widehat y^{n,k+1})$: the PDHG iterate before the
  reflection and Halpern update, at which the KKT residual is evaluated.

The **KKT** (Karush–Kuhn–Tucker) check tests primal feasibility, dual
feasibility, and the primal–dual gap. See [Termination criteria](termination.md)
for the residuals and tolerances.

The following pages describe the components:
[Base algorithm](base-algorithm.md),
[Presolve and postsolve](presolve.md), [Preconditioning](preconditioning.md),
[Step size](step-size.md),
[Adaptive restart](restart.md), [Primal weight](primal-weight.md), and
[Termination criteria](termination.md). Optional
[feasibility polishing](feasibility-polishing.md) can reduce primal and dual
feasibility residuals after the main solve.

## References

<span id="ref-cupdlpx">\[1\]</span> Haihao Lu, Zedong Peng, and Jinwen Yang.
[*cuPDLPx: A Further Enhanced GPU-Based First-Order Solver for Linear
Programming*](https://arxiv.org/abs/2507.14051), 2025.
