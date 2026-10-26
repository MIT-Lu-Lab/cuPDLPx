---
description: Primal and dual step sizes and the adaptive primal-weight update in cuPDLPx.
---

# Primal weight

The primal weight $\omega>0$ determines the relative sizes of the primal
and dual steps<sup>[\[1\]](#ref-pdlp)</sup>:

$$
\tau=\frac{\eta}{\omega},\qquad
\sigma=\eta\omega.
$$

Increasing $\omega$ decreases the primal step size and increases the dual
step size. cuPDLPx updates $\omega$ at each restart to balance the scaled
primal and dual displacements.

## Initial weight

With
[objective and bound scaling](preconditioning.md#objective-and-bound-scaling)
enabled, the initial weight is $\omega^0=1$. Otherwise,

$$
\omega^0=\frac{1+\lVert c\rVert}{1+\lVert b\rVert},
$$

where $c$ is the objective vector and $b$ collects the finite constraint
bounds before preconditioning, counting each equality bound once. The norm
is selected by
[`optimality_norm`](../reference/parameters.md#accuracy-and-termination).

## PID controller

cuPDLPx uses a PID controller with a discounted integral
term<sup>[\[2\]](#ref-cupdlpx)</sup>.

For epoch $n$, let $(x^{n,0},y^{n,0})$ be the anchor and
$(\widehat x^{n,k},\widehat y^{n,k})$ the PDHG iterate at the restart.
The imbalance between the scaled displacements is

$$
e^n=\log\left(
\frac{\sqrt{\omega^n}\lVert \widehat x^{n,k}-x^{n,0}\rVert_2}
{(1/\sqrt{\omega^n})\lVert \widehat y^{n,k}-y^{n,0}\rVert_2}
\right).
$$

Positive $e^n$ indicates a larger scaled primal displacement; negative
$e^n$ indicates a larger scaled dual displacement. The update is

$$
\log\omega^{n+1}=\log\omega^n-
\left[
K_Pe^n
+K_I\sum_{i=1}^{n}\rho^{n-i}e^i
+K_D(e^n-e^{n-1})
\right].
$$

The proportional, integral, and derivative terms use the current imbalance,
its discounted history, and its change since the previous epoch.
Updating $\log\omega$ preserves $\omega>0$.

cuPDLPx uses

$$
K_P=0.99,\qquad K_I=0.01,\qquad K_D=0,\qquad \rho=0.3
$$

by default. The weight and step sizes remain constant within each epoch.

## Safeguard

cuPDLPx resets $\omega$ and skips the PID update if either of the following
conditions holds<sup>[\[3\]](#ref-hpr-lp)</sup>:

- either displacement norm, $\lVert\widehat x^{n,k}-x^{n,0}\rVert_2$ or
  $\lVert\widehat y^{n,k}-y^{n,0}\rVert_2$, lies outside
  $[10^{-16},10^{12}]$;
- the ratio of the relative dual residual to the relative primal residual
  lies outside $[10^{-8},10^{8}]$.

The reset restores $\omega_{\mathrm{best}}$ and clears the controller's
integral and derivative state. This weight was produced at the restart
where the relative primal and dual residuals, $\delta_p^{\,n}$ and
$\delta_d^{\,n}$, were closest on a logarithmic scale:

$$
\omega_{\mathrm{best}}=\omega^{m+1},
\qquad
m=\operatorname*{arg\,min}_{n}
\left|\log_{10}\frac{\delta_d^{\,n}}{\delta_p^{\,n}}\right|.
$$

The stored weight is initialized to $\omega^0$ and updated when a restart
improves this measure.

## Parameters

See [Parameters](../reference/parameters.md#adaptive-restart) for the PID gains.

## References

<span id="ref-pdlp">\[1\]</span> David Applegate, Mateo Díaz, Oliver Hinder, Haihao Lu,
Miles Lubin, Brendan O'Donoghue, and Warren Schudy.
[*Practical Large-Scale Linear Programming Using Primal-Dual Hybrid
Gradient*](https://proceedings.neurips.cc/paper/2021/hash/a8fbbd3b11424ce032ba813493d95ad7-Abstract.html).
*NeurIPS*, 2021.

<span id="ref-cupdlpx">\[2\]</span> Haihao Lu, Zedong Peng, and Jinwen Yang.
[*cuPDLPx: A Further Enhanced GPU-Based First-Order Solver for Linear
Programming*](https://arxiv.org/abs/2507.14051), 2025.

<span id="ref-hpr-lp">\[3\]</span> Kaihuang Chen, Defeng Sun, Yancheng Yuan, Guojun Zhang,
and Xinyuan Zhao.
[*HPR-LP: An Implementation of an HPR Method for Solving Linear
Programming*](https://arxiv.org/abs/2408.12179), 2024.
