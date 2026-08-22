# Copyright 2025 Haihao Lu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gzip

import numpy as np
import pytest
import scipy.sparse as sp

import cupdlpx
from cupdlpx import Model, PDLP

# Same LP as base_lp_data in conftest.py:
# Minimize  x1 + x2
# Subject to
#     x1 + 2*x2 == 5
#            x2 <= 2
#   3*x1 + 2*x2 <= 8
#        x1, x2 >= 0
MPS_MIN = """NAME          TESTLP
ROWS
 N  COST
 E  R1
 L  R2
 L  R3
COLUMNS
    X1        COST      1.0   R1        1.0
    X1        R3        3.0
    X2        COST      1.0   R1        2.0
    X2        R2        1.0   R3        2.0
RHS
    RHS       R1        5.0   R2        2.0
    RHS       R3        8.0
ENDATA
"""

# Same LP but maximization via OBJSENSE
MPS_MAX = MPS_MIN.replace(
    "ROWS\n",
    "OBJSENSE\n    MAXIMIZE\nROWS\n",
)


@pytest.fixture
def mps_min_file(tmp_path):
    path = tmp_path / "test_min.mps"
    path.write_text(MPS_MIN)
    return path


@pytest.fixture
def mps_max_file(tmp_path):
    path = tmp_path / "test_max.mps"
    path.write_text(MPS_MAX)
    return path


@pytest.fixture
def mps_gz_file(tmp_path):
    path = tmp_path / "test_min.mps.gz"
    with gzip.open(path, "wt") as f:
        f.write(MPS_MIN)
    return path


def _check_min_model_data(model):
    """
    Verify the parsed model matches the LP encoded in MPS_MIN.
    """
    assert isinstance(model, Model)
    assert model.num_vars == 2
    assert model.num_constrs == 3
    # objective
    assert np.allclose(model.c, [1.0, 1.0])
    assert model.c0 == 0.0
    # constraint matrix
    A = model.A.toarray() if sp.issparse(model.A) else np.asarray(model.A)
    assert np.allclose(A, [[1.0, 2.0], [0.0, 1.0], [3.0, 2.0]])
    # constraint bounds
    assert np.allclose(model.constr_lb, [5.0, -np.inf, -np.inf])
    assert np.allclose(model.constr_ub, [5.0, 2.0, 8.0])
    # variable bounds (MPS default: 0 <= x < inf)
    assert np.allclose(model.lb, [0.0, 0.0])
    assert np.allclose(model.ub, [np.inf, np.inf])


def test_read_builds_model(mps_min_file):
    """
    cupdlpx.read should build a Model with the exact problem data.
    """
    model = cupdlpx.read(str(mps_min_file))
    _check_min_model_data(model)
    assert model.ModelSense == PDLP.MINIMIZE


def test_read_accepts_pathlike(mps_min_file):
    """
    cupdlpx.read should accept a pathlib.Path.
    """
    model = cupdlpx.read(mps_min_file)
    _check_min_model_data(model)


def test_read_gzip(mps_gz_file):
    """
    cupdlpx.read should read .mps.gz files.
    """
    model = cupdlpx.read(str(mps_gz_file))
    _check_min_model_data(model)


def test_read_objsense_maximize(mps_max_file):
    """
    OBJSENSE MAXIMIZE should set ModelSense to PDLP.MAXIMIZE without
    negating the objective vector.
    """
    model = cupdlpx.read(str(mps_max_file))
    _check_min_model_data(model)
    assert model.ModelSense == PDLP.MAXIMIZE


def test_read_missing_file_raises(tmp_path):
    """
    Reading a nonexistent file should raise FileNotFoundError.
    """
    with pytest.raises(FileNotFoundError):
        cupdlpx.read(str(tmp_path / "no_such_file.mps"))


def test_read_and_optimize_minimize(mps_min_file, atol):
    """
    Solve the model built from the MPS file and verify the solution.
    Optimal solution: x* = (1, 2), objective = 3
    """
    model = cupdlpx.read(str(mps_min_file))
    model.setParams(OutputFlag=False, Presolve=False)
    model.optimize()
    assert model.Status == PDLP.OPTIMAL, f"Unexpected termination status: {model.Status}"
    assert np.allclose(model.X, [1, 2], atol=atol), f"Unexpected primal solution: {model.X}"
    assert np.isclose(model.ObjVal, 3, atol=atol), f"Unexpected objective value: {model.ObjVal}"


def test_read_and_optimize_maximize(mps_max_file, atol):
    """
    Solve the maximization model built from the MPS file.
    Optimal solution: x* = (1.5, 1.75), objective = 3.25
    """
    model = cupdlpx.read(str(mps_max_file))
    model.setParams(OutputFlag=False, Presolve=False)
    model.optimize()
    assert model.Status == PDLP.OPTIMAL, f"Unexpected termination status: {model.Status}"
    assert np.allclose(model.X, [1.5, 1.75], atol=atol), f"Unexpected primal solution: {model.X}"
    assert np.isclose(model.ObjVal, 3.25, atol=atol), f"Unexpected objective value: {model.ObjVal}"


# ---------------------------------------------------------------------------
# BOUNDS-section semantics
# ---------------------------------------------------------------------------

MPS_BOUNDS = """NAME          BOUNDS
ROWS
 N  COST
 L  R1
COLUMNS
    C0        COST      1.0   R1        1.0
    MARKER                 'MARKER'                 'INTORG'
    I1        COST      1.0   R1        1.0
    I2        COST      1.0   R1        1.0
    I3        COST      1.0   R1        1.0
    I4        COST      1.0   R1        1.0
    I5        COST      1.0   R1        1.0
    MARKER                 'MARKER'                 'INTEND'
    C6        COST      1.0   R1        1.0
    C7        COST      1.0   R1        1.0
    C8        COST      1.0   R1        1.0
RHS
    RHS       R1        5.0
BOUNDS
 UP BND       I2        5.0
 LO BND       I3        2.0
 MI BND       I4
 UP BND       I5        -3.0
 UP BND       C6        -3.0
 LO BND       C7        -10.0
 UP BND       C7        -3.0
 LO BND       C8        -1e30
 UP BND       C8        1e20
ENDATA
"""


def _read_text(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return cupdlpx.read(path)


def test_read_bounds_conventions(tmp_path):
    """
    - integer (MARKER) columns with no BOUNDS entry default to [0, 1]
    - any BOUNDS entry on such a column cancels that default
    - negative UP with no explicit lower bound implies lb = -inf, but an
      explicit lower bound (C7) is kept

    Large finite bounds (C8) are read as written; turning them into infinities
    is the solver's infinite_bound parameter, not the reader's job.
    """
    model = _read_text(tmp_path, "bounds.mps", MPS_BOUNDS)
    inf = np.inf
    #              C0   I1   I2   I3   I4    I5    C6    C7     C8
    expected_lb = [0.0, 0.0, 0.0, 2.0, -inf, -inf, -inf, -10.0, -1e30]
    expected_ub = [inf, 1.0, 5.0, inf, inf, -3.0, -3.0, -3.0, 1e20]
    assert np.array_equal(model.lb, expected_lb), model.lb
    assert np.array_equal(model.ub, expected_ub), model.ub


MPS_NO_OBJ_ROW = """NAME          NOOBJ
ROWS
 E  R1
 L  R2
COLUMNS
    X1        R1        1.0   R2        1.0
    X2        R1        2.0
RHS
    RHS       R1        5.0   R2        2.0
ENDATA
"""


def test_read_without_objective_row(tmp_path):
    """
    A file with no N row has a zero objective; all rows stay constraints
    (previously the first row was silently taken as the objective).
    """
    model = _read_text(tmp_path, "noobj.mps", MPS_NO_OBJ_ROW)
    assert model.num_vars == 2
    assert model.num_constrs == 2
    assert np.allclose(model.c, [0.0, 0.0])
    assert np.allclose(model.constr_lb, [5.0, -np.inf])
    assert np.allclose(model.constr_ub, [5.0, 2.0])


MPS_SOS = MPS_MIN.replace(
    "ENDATA\n",
    "SOS\n S1 SOS       s1\n    X1        1\n    X2        2\nENDATA\n",
)

MPS_QUADOBJ = MPS_MIN.replace(
    "ENDATA\n",
    "QUADOBJ\n    X1        X1        2.0\nENDATA\n",
)


def test_read_sos_section_is_ignored(tmp_path):
    """
    SOS only restricts the integer feasible set, so the LP relaxation is
    unchanged and the section is skipped.
    """
    model = _read_text(tmp_path, "sos.mps", MPS_SOS)
    _check_min_model_data(model)


def test_read_unsupported_section_raises(tmp_path):
    """
    Sections that change the model beyond an LP (quadratic terms, cones,
    indicators, ...) must not be silently dropped.
    """
    with pytest.raises(Exception):
        _read_text(tmp_path, "quadobj.mps", MPS_QUADOBJ)
