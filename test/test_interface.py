import numpy as np
import scipy.sparse as sp

from pycupdlpx.model import Model

# construct a simple LP for testing
m, n = 3, 2
A_dense = np.array([[1.0, 2.0],
                    [0.0, 1.0],
                    [3.0, 2.0]], dtype=np.float64)
c = np.array([1.0, 1.0], dtype=np.float64)
l = np.array([5.0, -np.inf, -np.inf], dtype=np.float64)
u = np.array([5.0, 2.0, 8.0], dtype=np.float64)

# different matrix formats
A_csr = sp.csr_matrix(A_dense)
A_csc = sp.csc_matrix(A_dense)
A_coo = sp.coo_matrix(A_dense)

# define a function to run a test
def run_once(title, A, c, l, u):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    # create model
    model = Model(
        objective_vector=c,
        constraint_matrix=A,
        constraint_lower_bound=l,
        constraint_upper_bound=u,
    )
    # set some parameters
    model.setParam("TimeLimit", 30.0)
    model.setParam("TermCheckFreq", 10)
    model.setParams(OutputFlag=True, IterationLimit=200000)
    model.Params.FeasibilityTol = 1e-8
    model.Params.OptimalityTol = 1e-8
    # solve
    model.optimize()
    print("Primal x   :", model.X)
    print("Dual y     :", model.Pi)

# run tests
run_once("Test 1: Dense Matrix", A_dense, c, l, u)
run_once("Test 2: CSR Matrix", A_csr, c, l, u)
run_once("Test 3: CSC Matrix", A_csc, c, l, u)
run_once("Test 4: COO Matrix", A_coo, c, l, u)