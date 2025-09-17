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

#include "cupdlpx/interface.h"
#include <stdio.h>
#include <math.h> 

int main() {
    // Example: min c^T x
    // s.t. l <= A x <= u, x >= 0

    int m = 3; // number of constraints
    int n = 2; // number of variables

    // A as a dense matrix
    double A[3][2] = {
        {1.0, 2.0},
        {0.0, 1.0},
        {3.0, 2.0}
    };

    // c: objective coefficients
    double c[2] = {1.0, 1.0};

    // l&u: constraintbounds
    double l[3] = {5.0, -INFINITY, -1e20};  // lower bounds
    double u[3] = {5.0, 2.0, 8.0};          // upper bounds

    printf("Objective c = [");
    for (int j = 0; j < n; j++) {
        printf(" %g", c[j]);
    }
    printf(" ]\n");

    printf("Matrix A:\n");
    for (int i = 0; i < m; i++) {
        printf("  [");
        for (int j = 0; j < n; j++) {
            printf(" %g", A[i][j]);
        }
        printf(" ]\n");
    }

    printf("Constraint bounds (l <= A x <= b):\n");
    for (int i = 0; i < m; i++) {
        printf("  row %d: l = %g, u = %g\n", i, l[i], u[i]);
    }

    return 0;
}
