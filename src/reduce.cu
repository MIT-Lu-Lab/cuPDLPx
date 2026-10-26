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

// Temporary storage of the reductions; see reduce.h.

#include "reduce.h"

void reduce_temp_storage_create(pdhg_solver_state_t *state)
{
    int n_max = state->num_variables > state->num_constraints ? state->num_variables : state->num_constraints;
    if (n_max < 1)
        n_max = 1;
    const double *in = NULL;
    double *out = NULL;
    size_t sum_bytes = 0, max_bytes = 0;
    CUDA_CHECK(cub::DeviceReduce::Sum(NULL, sum_bytes, in, out, n_max, state->stream));
    CUDA_CHECK(cub::DeviceReduce::Max(NULL, max_bytes, in, out, n_max, state->stream));
    state->reduce_temp_bytes = sum_bytes > max_bytes ? sum_bytes : max_bytes;
    CUDA_CHECK(cudaMalloc(&state->reduce_temp_d, state->reduce_temp_bytes));
}
