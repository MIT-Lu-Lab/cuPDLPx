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

#include "reduce.h"

#include <math.h>
#include <stdio.h>

void launch_test_reductions(
    const pdhg_solver_state_t *state, size_t *required_bytes, int n, const double *input, double *out);
static int expect_results(const double *out, double sum, double max_abs, double sum_sq);

int main()
{
    const int n = 12000;
    pdhg_solver_state_t state = {};
    CUDA_CHECK(cudaStreamCreate(&state.stream));
    double *input_h = (double *)safe_malloc(n * sizeof(double));
    for (int i = 0; i < n; ++i)
        input_h[i] = (double)(i % 3) - 1.0;
    double *input, *out;
    CUDA_CHECK(cudaMalloc(&input, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&out, 3 * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(input, input_h, n * sizeof(double), cudaMemcpyHostToDevice));
    free(input_h);

    size_t bytes = 0;
    launch_test_reductions(&state, &bytes, n, input, out);
    reduce_temp_storage_create(&state, bytes);

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    CUDA_CHECK(cudaStreamBeginCapture(state.stream, cudaStreamCaptureModeGlobal));
    launch_test_reductions(&state, NULL, n, input, out);
    CUDA_CHECK(cudaStreamEndCapture(state.stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    int failed = 0;
    for (int replay = 0; replay < 2; ++replay)
    {
        CUDA_CHECK(cudaGraphLaunch(graphExec, state.stream));
        CUDA_CHECK(cudaStreamSynchronize(state.stream));
        failed |= expect_results(out, 0.0, 1.0, 8000.0);
    }
    launch_test_reductions(&state, NULL, 0, input, out);
    CUDA_CHECK(cudaStreamSynchronize(state.stream));
    failed |= expect_results(out, 0.0, 0.0, 0.0);

    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaFree(state.reduce_temp_d));
    CUDA_CHECK(cudaFree(input));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaStreamDestroy(state.stream));
    if (failed)
        return 1;
    printf("Planned reduction workspace checks passed.\n");
    return 0;
}

void launch_test_reductions(
    const pdhg_solver_state_t *state, size_t *required_bytes, int n, const double *input, double *out)
{
    reduce_sum(state, required_bytes, n, DEVICE_FN(i) { return input[i]; }, out);
    reduce_max(state, required_bytes, n, DEVICE_FN(i) { return fabs(input[i]); }, out + 1);
    reduce_sum(state, required_bytes, n, DEVICE_FN(i) { return input[i] * input[i]; }, out + 2);
}

static int expect_results(const double *out, double sum, double max_abs, double sum_sq)
{
    double expected[3] = {sum, max_abs, sum_sq};
    double actual[3];
    CUDA_CHECK(cudaMemcpy(actual, out, sizeof(actual), cudaMemcpyDeviceToHost));
    for (int k = 0; k < 3; ++k)
    {
        if (actual[k] != expected[k])
        {
            fprintf(stderr, "Reduction %d: expected %.17g, got %.17g\n", k, expected[k], actual[k]);
            return 1;
        }
    }
    return 0;
}
