---
description: cuPDLPx parameter names, defaults, ranges, and usage across Python, Julia, C, and the command line.
hide:
  - navigation
---

# Parameters

Solver settings are grouped by function. Each table lists the parameter
name, default, type, and allowed values for the selected interface.

## Usage

=== "Python"

    Set parameters with `setParams`, `setParam`, or `Params`:

    ```python
    model.setParams(
        TimeLimit=300,
        OptimalityTol=1e-6,
        FeasibilityTol=1e-6,
        OutputFlag=True,
    )
    ```

    Python also accepts the leaf C field names, such as `time_sec_limit` and
    `eps_optimal_relative`.

    Read and reset values with:

    ```python
    time_limit = model.getParam("TimeLimit")
    same_value = model.Params.TimeLimit

    for key, value in model.Params.items():
        print(key, value)

    model.resetParams()
    ```

    `setParams(...)` validates all updates before applying them. If any name
    or value is invalid, no parameters are changed.

=== "Julia"

    CuPDLPx.jl exposes supported solver settings as JuMP attributes using the
    C leaf field names:

    ```julia
    using JuMP, CuPDLPx

    model = Model(CuPDLPx.Optimizer)
    set_attribute(model, "time_sec_limit", 300.0)
    set_attribute(model, "eps_optimal_relative", 1e-6)
    set_attribute(model, "eps_feasible_relative", 1e-6)
    set_attribute(model, "verbose", true)
    ```

    See the [CuPDLPx.jl documentation](https://jump.dev/JuMP.jl/stable/packages/CuPDLPx/#Supported-parameters)
    for the attributes exposed by the current Julia wrapper.

=== "C"

    Initialize `pdhg_parameters_t` before overriding individual fields:

    ```c
    pdhg_parameters_t params;
    set_default_parameters(&params);

    params.termination_criteria.time_sec_limit = 300.0;
    params.termination_criteria.eps_optimal_relative = 1e-6;
    params.termination_criteria.eps_feasible_relative = 1e-6;
    params.verbose = false;
    ```

    Use values in the documented ranges; assigning C fields does not perform
    the Python interface's runtime parameter validation.

=== "Command line"

    Pass CLI settings before the input and output paths:

    ```bash
    cupdlpx \
      --time_limit 300 \
      --eps_opt 1e-6 \
      --eps_feas 1e-6 \
      --quiet \
      model.mps.gz output
    ```

    Run `cupdlpx --help` for the full list of flags.

<div class="parameter-reference" markdown>

## Limits and logging

=== "Python"

    | Parameter | Default | Type and range | Description |
    | --- | --- | --- | --- |
    | `TimeLimit` | `3600.0` | float<br>$[0,\infty)$ | Wall-clock limit in seconds. |
    | `IterationLimit` | $2^{31}-1$ | integer<br>$[0,2^{31}-1]$ | Maximum number of iterations. |
    | `OutputFlag`, `LogToConsole` | `true` | boolean<br>`false`, `true` | Print progress and the solve summary. |
    | `TermCheckFreq` | `200` | integer<br>$[3,2^{31}-1]$ | Iterations between termination and restart checks. |
    | `Debug` | `false` | boolean<br>`false`, `true` | Print developer diagnostics: restart reasons, active-set step size boost events, and a primal-weight log column. |

=== "C and Julia"

    | Parameter | Default | Type and range | Description |
    | --- | --- | --- | --- |
    | `time_sec_limit` | `3600.0` | float<br>$[0,\infty)$ | Wall-clock limit in seconds. |
    | `iteration_limit` | $2^{31}-1$ | integer<br>$[0,2^{31}-1]$ | Maximum number of iterations. |
    | `verbose` | `true` | boolean<br>`false`, `true` | Print progress and the solve summary. |
    | `termination_evaluation_frequency` | `200` | integer<br>$[3,2^{31}-1]$ | Iterations between termination and restart checks. |
    | `debug` | `false` | boolean<br>`false`, `true` | Print developer diagnostics: restart reasons, active-set step size boost events, and a primal-weight log column. |

=== "Command line"

    | Parameter | Default | Type and range | Description |
    | --- | --- | --- | --- |
    | `--time_limit` | `3600.0` | float<br>$[0,\infty)$ | Wall-clock limit in seconds. |
    | `--iter_limit` | $2^{31}-1$ | integer<br>$[0,2^{31}-1]$ | Maximum number of iterations. |
    | `--verbose`, `--quiet` | `true` | boolean<br>`false`, `true` | Print progress and the solve summary. |
    | `--eval_freq` | `200` | integer<br>$[3,2^{31}-1]$ | Iterations between termination and restart checks. |
    | `--debug` | `false` | boolean<br>`false`, `true` | Print developer diagnostics: restart reasons, active-set step size boost events, and a primal-weight log column. |

## Accuracy and termination

=== "Python"

    | Parameter | Default | Type and range | Description |
    | --- | --- | --- | --- |
    | `OptimalityTol` | `1e-4` | float<br>$(0,\infty)$ | Relative objective-gap tolerance. |
    | `FeasibilityTol` | `1e-4` | float<br>$(0,\infty)$ | Relative primal and dual feasibility tolerance. |
    | `OptimalityNorm` | `l2` | string<br>`l2`, `linf` | Norm used by the termination criteria. |

=== "C and Julia"

    | Parameter | Default | Type and range | Description |
    | --- | --- | --- | --- |
    | `eps_optimal_relative` | `1e-4` | float<br>$(0,\infty)$ | Relative objective-gap tolerance. |
    | `eps_feasible_relative` | `1e-4` | float<br>$(0,\infty)$ | Relative primal and dual feasibility tolerance. |
    | `optimality_norm` | `NORM_TYPE_L2` | `norm_type_t`<br>`NORM_TYPE_L2`, `NORM_TYPE_L_INF` | Norm used by the termination criteria. |

=== "Command line"

    | Parameter | Default | Type and range | Description |
    | --- | --- | --- | --- |
    | `--eps_opt` | `1e-4` | float<br>$(0,\infty)$ | Relative objective-gap tolerance. |
    | `--eps_feas` | `1e-4` | float<br>$(0,\infty)$ | Relative primal and dual feasibility tolerance. |
    | `--opt_norm` | `l2` | string<br>`l2`, `linf` | Norm used by the termination criteria. |

Residuals are evaluated in the unscaled model; when
[presolve](../algorithm/presolve.md) is enabled, this is the presolved model.

## Scaling and preprocessing

=== "Python"

    | Parameter | Default | Type and range | Description |
    | --- | --- | --- | --- |
    | `GeoMeanIters` | `12` | integer<br>$[0,2^{31}-1]$ | Number of [geometric mean scaling](../algorithm/preconditioning.md#geometric-mean-scaling) passes; `0` disables this stage. |
    | `RuizIters` | `10` | integer<br>$[0,2^{31}-1]$ | Number of $\ell_\infty$ [Ruiz equilibration scaling](../algorithm/preconditioning.md#ruiz-equilibration-scaling) passes. |
    | `UsePCAlpha` | `true` | boolean<br>`false`, `true` | Enable Pock–Chambolle scaling. |
    | `PCAlpha` | `1.0` | float<br>$\mathbb R$ | Exponent used by Pock–Chambolle scaling. |
    | `BoundObjRescaling` | `true` | boolean<br>`false`, `true` | Enable objective and bound scaling. |
    | `Presolve` | `true` | boolean<br>`false`, `true` | Enable presolve. |
    | `MatrixZeroTol` | `1e-9` | float<br>$[0,\infty)$ | Remove entries of $A$ at or below this absolute magnitude. |
    | `InfiniteBound` | `1e20` | float<br>$(0,\infty)$ | Replace lower bounds $\le -t$ and upper bounds $\ge t$ with infinities, where $t$ is this threshold. |

=== "C and Julia"

    | Parameter | Default | Type and range | Description |
    | --- | --- | --- | --- |
    | `geometric_mean_iterations` | `12` | integer<br>$[0,2^{31}-1]$ | Number of [geometric mean scaling](../algorithm/preconditioning.md#geometric-mean-scaling) passes; `0` disables this stage. |
    | `l_inf_ruiz_iterations` | `10` | integer<br>$[0,2^{31}-1]$ | Number of $\ell_\infty$ [Ruiz equilibration scaling](../algorithm/preconditioning.md#ruiz-equilibration-scaling) passes. |
    | `has_pock_chambolle_alpha` | `true` | boolean<br>`false`, `true` | Enable Pock–Chambolle scaling. |
    | `pock_chambolle_alpha` | `1.0` | float<br>$\mathbb R$ | Exponent used by Pock–Chambolle scaling. |
    | `bound_objective_rescaling` | `true` | boolean<br>`false`, `true` | Enable objective and bound scaling. |
    | `presolve` | `true` | boolean<br>`false`, `true` | Enable presolve. |
    | `matrix_zero_tol` | `1e-9` | float<br>$[0,\infty)$ | Remove entries of $A$ at or below this absolute magnitude. |
    | `infinite_bound` | `1e20` | float<br>$(0,\infty)$ | Replace lower bounds $\le -t$ and upper bounds $\ge t$ with infinities, where $t$ is this threshold. |

=== "Command line"

    | Parameter | Default | Type and range | Description |
    | --- | --- | --- | --- |
    | `--geo_mean_iter` | `12` | integer<br>$[0,2^{31}-1]$ | Number of [geometric mean scaling](../algorithm/preconditioning.md#geometric-mean-scaling) passes; `0` disables this stage. |
    | `--l_inf_ruiz_iter` | `10` | integer<br>$[0,2^{31}-1]$ | Number of $\ell_\infty$ [Ruiz equilibration scaling](../algorithm/preconditioning.md#ruiz-equilibration-scaling) passes. |
    | `--no_pock_chambolle` | Not passed | flag | Disable Pock–Chambolle scaling (enabled by default). |
    | `--pock_chambolle_alpha` | `1.0` | float<br>$\mathbb R$ | Exponent used by Pock–Chambolle scaling. |
    | `--no_bound_obj_rescaling` | Not passed | flag | Disable objective and bound scaling (enabled by default). |
    | `--no_presolve` | Not passed | flag | Disable presolve (enabled by default). |
    | `--matrix_zero_tol` | `1e-9` | float<br>$[0,\infty)$ | Remove entries of $A$ at or below this absolute magnitude. |
    | `--infinite_bound` | `1e20` | float<br>$(0,\infty)$ | Replace lower bounds $\le -t$ and upper bounds $\ge t$ with infinities, where $t$ is this threshold. |

## Step size and reflection

=== "Python"

    | Parameter | Default | Type and range | Description |
    | --- | --- | --- | --- |
    | `SVMaxIter` | `5000` | integer<br>$[1,2^{31}-1]$ | Maximum iterations for estimating $\lVert A\rVert_2$. |
    | `SVTol` | `1e-4` | float<br>$(0,\infty)$ | Stopping tolerance for the norm estimate. |
    | `ReflectionCoeff` | `1.0` | float<br>$\mathbb R$ | Weight $\gamma$ on the reflected point in the Halpern update; `1.0` uses the fully reflected point $2\widehat x-x$. |

=== "C and Julia"

    | Parameter | Default | Type and range | Description |
    | --- | --- | --- | --- |
    | `sv_max_iter` | `5000` | integer<br>$[1,2^{31}-1]$ | Maximum iterations for estimating $\lVert A\rVert_2$. |
    | `sv_tol` | `1e-4` | float<br>$(0,\infty)$ | Stopping tolerance for the norm estimate. |
    | `reflection_coefficient` | `1.0` | float<br>$\mathbb R$ | Weight $\gamma$ on the reflected point in the Halpern update; `1.0` uses the fully reflected point $2\widehat x-x$. |

=== "Command line"

    | Parameter | Default | Type and range | Description |
    | --- | --- | --- | --- |
    | `--sv_max_iter` | `5000` | integer<br>$[1,2^{31}-1]$ | Maximum iterations for estimating $\lVert A\rVert_2$. |
    | `--sv_tol` | `1e-4` | float<br>$(0,\infty)$ | Stopping tolerance for the norm estimate. |

    The command-line interface does not expose the reflection coefficient.

## Active-set step size boost

=== "Python"

    | Parameter | Default | Type and range | Description |
    | --- | --- | --- | --- |
    | `ActiveSetBoost` | `true` | boolean<br>`false`, `true` | Enable the [active-set step size boost](../algorithm/step-size.md#active-set-step-size-boost). |
    | `ASBActivationTol` | `1e-4` | float<br>$[0,\infty)$ | Threshold for the maximum relative primal residual, dual residual, and primal–dual gap; see [activation](../algorithm/step-size.md#activation). |
    | `ASBWindowIter` | `10000` | integer<br>$[1,2^{31}-1]$ | Length of the [trailing window](../algorithm/step-size.md#trailing-window) in iterations. |
    | `ASBVariableTol` | `1e-8` | float<br>$[0,\infty)$ | Dual-slack tolerance for identifying a variable at a bound. |
    | `ASBConstraintTol` | `1e-8` | float<br>$[0,\infty)$ | Margin for identifying an inactive constraint. |
    | `ASBReestimateChangeRatio` | `0.01` | float<br>$[0,\infty)$ | Threshold for accumulated additions and removals, as a fraction of the current active-set size, before norm re-estimation. |
    | `ASBSafetyFactor` | `0.9` | float<br>$(0,\infty)$ | Safety factor $\alpha$ in the target step $\alpha/\hat\sigma$. |
    | `ASBMinRaiseRatio` | `1.1` | float<br>$[1,\infty)$ | Minimum ratio of the target step to the current step for a step-size increase. |
    | `ASBDivergenceMargin` | `0.05` | float<br>$[0,\infty)$ | Allowed relative increase in fixed-point error within an epoch before rollback. |
    | `ASBDivergenceCeilingRatio` | `0.7` | float<br>$(0,1]$ | Limit on subsequent step-size increases, as a fraction of the rejected step size; the step never falls below its initial value. |
    | `ASBMaxReverts` | `2` | integer<br>$[0,2^{31}-1]$ | Number of rollbacks before ASB is disabled. |

=== "C and Julia"

    | Parameter | Default | Type and range | Description |
    | --- | --- | --- | --- |
    | `active_set_boost` | `true` | boolean<br>`false`, `true` | Enable the [active-set step size boost](../algorithm/step-size.md#active-set-step-size-boost). |
    | `asb_activation_tol` | `1e-4` | float<br>$[0,\infty)$ | Threshold for the maximum relative primal residual, dual residual, and primal–dual gap; see [activation](../algorithm/step-size.md#activation). |
    | `asb_window_iter` | `10000` | integer<br>$[1,2^{31}-1]$ | Length of the [trailing window](../algorithm/step-size.md#trailing-window) in iterations. |
    | `asb_variable_tol` | `1e-8` | float<br>$[0,\infty)$ | Dual-slack tolerance for identifying a variable at a bound. |
    | `asb_constraint_tol` | `1e-8` | float<br>$[0,\infty)$ | Margin for identifying an inactive constraint. |
    | `asb_reestimate_change_ratio` | `0.01` | float<br>$[0,\infty)$ | Threshold for accumulated additions and removals, as a fraction of the current active-set size, before norm re-estimation. |
    | `asb_safety_factor` | `0.9` | float<br>$(0,\infty)$ | Safety factor $\alpha$ in the target step $\alpha/\hat\sigma$. |
    | `asb_min_raise_ratio` | `1.1` | float<br>$[1,\infty)$ | Minimum ratio of the target step to the current step for a step-size increase. |
    | `asb_divergence_margin` | `0.05` | float<br>$[0,\infty)$ | Allowed relative increase in fixed-point error within an epoch before rollback. |
    | `asb_divergence_ceiling_ratio` | `0.7` | float<br>$(0,1]$ | Limit on subsequent step-size increases, as a fraction of the rejected step size; the step never falls below its initial value. |
    | `asb_max_reverts` | `2` | integer<br>$[0,2^{31}-1]$ | Number of rollbacks before ASB is disabled. |

=== "Command line"

    | Parameter | Default | Type and range | Description |
    | --- | --- | --- | --- |
    | `--no_active_set_boost` | Not passed | flag | Disable the [active-set step size boost](../algorithm/step-size.md#active-set-step-size-boost) (enabled by default). |
    | `--asb_activation_tol` | `1e-4` | float<br>$[0,\infty)$ | Threshold for the maximum relative primal residual, dual residual, and primal–dual gap; see [activation](../algorithm/step-size.md#activation). |
    | `--asb_window_iter` | `10000` | integer<br>$[1,2^{31}-1]$ | Length of the [trailing window](../algorithm/step-size.md#trailing-window) in iterations. |
    | `--asb_variable_tol` | `1e-8` | float<br>$[0,\infty)$ | Dual-slack tolerance for identifying a variable at a bound. |
    | `--asb_constraint_tol` | `1e-8` | float<br>$[0,\infty)$ | Margin for identifying an inactive constraint. |
    | `--asb_reestimate_change_ratio` | `0.01` | float<br>$[0,\infty)$ | Threshold for accumulated additions and removals, as a fraction of the current active-set size, before norm re-estimation. |
    | `--asb_safety_factor` | `0.9` | float<br>$(0,\infty)$ | Safety factor $\alpha$ in the target step $\alpha/\hat\sigma$. |
    | `--asb_min_raise_ratio` | `1.1` | float<br>$[1,\infty)$ | Minimum ratio of the target step to the current step for a step-size increase. |
    | `--asb_divergence_margin` | `0.05` | float<br>$[0,\infty)$ | Allowed relative increase in fixed-point error within an epoch before rollback. |
    | `--asb_divergence_ceiling_ratio` | `0.7` | float<br>$(0,1]$ | Limit on subsequent step-size increases, as a fraction of the rejected step size; the step never falls below its initial value. |
    | `--asb_max_reverts` | `2` | integer<br>$[0,2^{31}-1]$ | Number of rollbacks before ASB is disabled. |

## Adaptive restart

=== "Python"

    | Parameter | Default | Type and range | Description |
    | --- | --- | --- | --- |
    | `RestartArtificialThresh` | `0.36` | float<br>$\mathbb R$ | Maximum epoch length relative to the total iteration count. |
    | `RestartSufficientReduction` | `0.2` | float<br>$\mathbb R$ | Fixed-point-error ratio for a restart triggered by sufficient reduction. |
    | `RestartNecessaryReduction` | `0.5` | float<br>$\mathbb R$ | Required reduction before a local-increase restart. |
    | `RestartKp` | `0.99` | float<br>$\mathbb R$ | Proportional gain in the primal-weight controller. |

    The Python interface does not expose the integral gain, derivative gain,
    or integral decay. Their defaults are listed in the C and Julia tab.

=== "C and Julia"

    | Parameter | Default | Type and range | Description |
    | --- | --- | --- | --- |
    | `artificial_restart_threshold` | `0.36` | float<br>$\mathbb R$ | Maximum epoch length relative to the total iteration count. |
    | `sufficient_reduction_for_restart` | `0.2` | float<br>$\mathbb R$ | Fixed-point-error ratio for a restart triggered by sufficient reduction. |
    | `necessary_reduction_for_restart` | `0.5` | float<br>$\mathbb R$ | Required reduction before a local-increase restart. |
    | `k_p` | `0.99` | float<br>$\mathbb R$ | Proportional gain in the primal-weight controller. |
    | `k_i` | `0.01` | float<br>$\mathbb R$ | Integral gain in the primal-weight controller. |
    | `k_d` | `0.0` | float<br>$\mathbb R$ | Derivative gain in the primal-weight controller. |
    | `i_smooth` | `0.3` | float<br>$\mathbb R$ | Decay factor for the accumulated integral error. |

=== "Command line"

    These settings are not exposed by the command-line interface.

See [Adaptive restart](../algorithm/restart.md) for the restart conditions and
[Primal weight](../algorithm/primal-weight.md) for the controller update.

## Feasibility polishing

=== "Python"

    | Parameter | Default | Type and range | Description |
    | --- | --- | --- | --- |
    | `FeasibilityPolishing` | `false` | boolean<br>`false`, `true` | Run the feasibility-polishing phases after the main solve. |
    | `FeasibilityPolishingTol` | `1e-6` | float<br>$(0,\infty)$ | Target relative feasibility residual. |

=== "C and Julia"

    | Parameter | Default | Type and range | Description |
    | --- | --- | --- | --- |
    | `feasibility_polishing` | `false` | boolean<br>`false`, `true` | Run the feasibility-polishing phases after the main solve. |
    | `eps_feas_polish_relative` | `1e-6` | float<br>$(0,\infty)$ | Target relative feasibility residual. |

=== "Command line"

    | Parameter | Default | Type and range | Description |
    | --- | --- | --- | --- |
    | `-f`, `--feasibility_polishing` | `false` | boolean<br>`false`, `true` | Run the feasibility-polishing phases after the main solve. |
    | `--eps_feas_polish` | `1e-6` | float<br>$(0,\infty)$ | Target relative feasibility residual. |

</div>
