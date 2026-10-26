// Copyright 2025 Haihao Lu
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Reductions of a termination check (C++ only). Each scalar is one
// cub::DeviceReduce over a thrust::transform_iterator whose lambda computes the
// quantity from the index; the result stays in device memory. The lambdas are
// __host__ __device__ so the iterator can deduce their type, which needs nvcc
// --extended-lambda.

#ifndef CUPDLPX_REDUCE_H
#define CUPDLPX_REDUCE_H

#include "utils.h"

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

// The element function of a reduction: DEVICE_FN(i) { return ...; }
#define DEVICE_FN(i) [=] __host__ __device__(int i)

// Temporary storage of cub::DeviceReduce, sized once for the longest vector and
// shared by every call. cub rejects a call that needs more.
void reduce_temp_storage_create(pdhg_solver_state_t *state);

// out = sum over i < n of f(i).
template <typename F> void reduce_sum(pdhg_solver_state_t *state, int n, F f, double *out)
{
    if (n <= 0)
    {
        CUDA_CHECK(cudaMemsetAsync(out, 0, sizeof(double), state->stream));
        return;
    }
    auto it = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), f);
    size_t bytes = state->reduce_temp_bytes;
    CUDA_CHECK(cub::DeviceReduce::Sum(state->reduce_temp_d, bytes, it, out, n, state->stream));
}

// out = max over i < n of f(i); 0 when n is 0, since every maximum taken here is
// nonnegative.
template <typename F> void reduce_max(pdhg_solver_state_t *state, int n, F f, double *out)
{
    if (n <= 0)
    {
        CUDA_CHECK(cudaMemsetAsync(out, 0, sizeof(double), state->stream));
        return;
    }
    auto it = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), f);
    size_t bytes = state->reduce_temp_bytes;
    CUDA_CHECK(cub::DeviceReduce::Max(state->reduce_temp_d, bytes, it, out, n, state->stream));
}

#endif // CUPDLPX_REDUCE_H
