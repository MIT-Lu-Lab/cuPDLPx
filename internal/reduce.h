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

// Temporary storage of cub::DeviceReduce, shared by every call. cub rejects a
// call that needs more, and captured graphs keep its address, so it is sized
// once by planning every reduction.
void reduce_temp_storage_create(pdhg_solver_state_t *state, size_t bytes);

// With required_bytes set, a reduction only raises it to the storage it needs.
// Planning calls the same functions as execution, so cub sees the same types.
// On Windows, nvcc rejects these lambdas in functions with internal linkage, so
// functions that define reductions are not static.

// out = sum over i < n of f(i).
template <typename F> void reduce_sum(const pdhg_solver_state_t *state, size_t *required_bytes, int n, F f, double *out)
{
    if (n <= 0)
    {
        if (!required_bytes)
            CUDA_CHECK(cudaMemsetAsync(out, 0, sizeof(double), state->stream));
        return;
    }
    auto it = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), f);
    void *temp = required_bytes ? NULL : state->reduce_temp_d;
    size_t bytes = state->reduce_temp_bytes;
    CUDA_CHECK(cub::DeviceReduce::Sum(temp, bytes, it, out, n, state->stream));
    if (required_bytes && bytes > *required_bytes)
        *required_bytes = bytes;
}

// out = max over i < n of f(i); 0 when n is 0, since every maximum taken here is
// nonnegative.
template <typename F> void reduce_max(const pdhg_solver_state_t *state, size_t *required_bytes, int n, F f, double *out)
{
    if (n <= 0)
    {
        if (!required_bytes)
            CUDA_CHECK(cudaMemsetAsync(out, 0, sizeof(double), state->stream));
        return;
    }
    auto it = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), f);
    void *temp = required_bytes ? NULL : state->reduce_temp_d;
    size_t bytes = state->reduce_temp_bytes;
    CUDA_CHECK(cub::DeviceReduce::Max(temp, bytes, it, out, n, state->stream));
    if (required_bytes && bytes > *required_bytes)
        *required_bytes = bytes;
}

#endif // CUPDLPX_REDUCE_H
