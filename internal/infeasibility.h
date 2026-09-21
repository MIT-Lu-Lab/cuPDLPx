/*
Copyright 2025 Haihao Lu

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#pragma once

#include "utils.h"

#ifdef __cplusplus
extern "C"
{
#endif

    // Same two halves as termination.h. |A||d| and |A^T||y| cost two more SpMMs
    // and matter only when some candidate passes its objective margin, so the
    // host half computes them itself, with one more synchronization, only then.

    // Owns every buffer of a check; created before the main loop so that a check
    // never allocates.
    infeas_ctx_t *infeasibility_context_create(pdhg_solver_state_t *state);

    void infeasibility_context_destroy(infeas_ctx_t *ctx);

    void compute_infeasibility_device(pdhg_solver_state_t *state);

    void compute_infeasibility_host(pdhg_solver_state_t *state, double objective_margin);

#ifdef __cplusplus
}
#endif
