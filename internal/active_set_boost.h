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

#include "internal_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    typedef enum
    {
        ASB_PHASE_WAITING = 0,
        ASB_PHASE_ACTIVE = 1,
        ASB_PHASE_OFF = 2
    } asb_phase_t;

    typedef enum
    {
        ASB_ACTION_NONE = 0,
        ASB_ACTION_REVERT = 1
    } asb_action_t;

    void active_set_boost_init(pdhg_solver_state_t *state);

    void active_set_boost_free(pdhg_solver_state_t *state);

    void active_set_boost_update_window(pdhg_solver_state_t *state, const pdhg_parameters_t *params);

    asb_action_t active_set_boost_check(pdhg_solver_state_t *state, const pdhg_parameters_t *params);

    void active_set_boost_on_restart(pdhg_solver_state_t *state, const pdhg_parameters_t *params);

#ifdef __cplusplus
}
#endif
