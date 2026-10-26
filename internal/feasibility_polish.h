#pragma once

#include "internal_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    void feasibility_polish(const pdhg_parameters_t *params, pdhg_solver_state_t *state);

    void compute_polish_fixed_point_error_device(pdhg_solver_state_t *state, bool is_primal_polish);

#ifdef __cplusplus
}
#endif
