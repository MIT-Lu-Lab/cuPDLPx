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

    // Detection of primal and dual infeasibility from candidate rays; see the
    // comment above compute_infeasibility_device in infeasibility.cu for the
    // method. Same protocol as termination.h: compute_infeasibility_device
    // only queues kernels (the candidates, A d, A^T y and their reductions) and
    // never waits; the caller fetches the scalars; compute_infeasibility_host
    // turns them into the certificate errors of the state.
    //
    // A check has a second round, because |A||d| and |A^T||y| cost two more
    // SpMMs and are only needed when some candidate passes its objective
    // margin. compute_infeasibility_host runs it when needed, with one more
    // copy and synchronization; otherwise it is host arithmetic only.

    // The context owns every buffer of a check. It is created once before the
    // main loop, so that a check never allocates, and destroyed with the solver
    // state.
    infeas_ctx_t *infeasibility_context_create(pdhg_solver_state_t *state);

    void infeasibility_context_destroy(infeas_ctx_t *ctx);

    void compute_infeasibility_device(pdhg_solver_state_t *state);

    void compute_infeasibility_host(pdhg_solver_state_t *state);

#ifdef __cplusplus
}
#endif
