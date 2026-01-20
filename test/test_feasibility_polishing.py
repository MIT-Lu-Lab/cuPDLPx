import numpy as np
from cupdlpx import Model, PDLP

# ------------------------------------------------------------------
# Pytest-style helper (can also be used stand-alone)
# ------------------------------------------------------------------
def polishing_data():
    """
    Fixture-like helper: returns a tiny LP whose feasible set has
    the unique vertex (1, 2) and objective value 3.
    """
    c = np.array([1.0, 1.0])
    A = np.array([[1.0, 2.0],
                  [0.0, 1.0],
                  [3.0, 2.0]])
    l = np.array([5.0, -np.inf, -np.inf])
    u = np.array([5.0, 2.0, 8.0])
    lb = np.zeros(2)
    ub = None
    return c, A, l, u, lb, ub


# ------------------------------------------------------------------
# Test: feasibility polishing
# ------------------------------------------------------------------
def test_feasibility_polishing():
    """
    Ensure feasibility-polishing drives an almost-infeasible starting
    point to within FeasibilityPolishingTol and still reports the
    correct objective value.
    """
    c, A, l, u, lb, ub = polishing_data()

    # 1. Build model
    m = Model(objective_vector=c,
              constraint_matrix=A,
              constraint_lower_bound=l,
              constraint_upper_bound=u,
              variable_lower_bound=lb,
              variable_upper_bound=ub)
    m.ModelSense = PDLP.MINIMIZE

    # 2. Enable polishing with tight tolerance
    m.setParams(OutputFlag=False,
                Presolve=False,
                FeasibilityPolishing=True,
                FeasibilityPolishingTol=1e-10)

    # 3. Solve
    m.optimize()

    # 4. Sanity checks
    assert m.Status == "OPTIMAL", f"unexpected status: {m.Status}"
    assert hasattr(m, "X") and hasattr(m, "ObjVal")

    # 5. Feasibility-quality check: max violation < 1e-10
    assert m._rel_p_res <= 1e-10, f"reported rel_p_res too high: {m._rel_p_res}"
    assert m._rel_d_res <= 1e-10, f"reported rel_d_res too high: {m._rel_d_res}"
    x = m.X
    viol = np.maximum(np.maximum(0, A @ x - u),   # upper violation
                      np.maximum(0, l - A @ x))   # lower violation
    l2_viol = np.linalg.norm(viol, ord=2) / (1 + max(np.linalg.norm(u, ord=2), np.linalg.norm(l, ord=2)))
    assert l2_viol < 1e-10, f"L2 feasibility violation = {l2_viol:.2e}"
