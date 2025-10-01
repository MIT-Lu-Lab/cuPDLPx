import numpy as np
import scipy.sparse as sp

from cupdlpx import PDLP, Model

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
    print("\n")
    print(title)
    # create model
    model = Model(
        objective_vector=c,
        constraint_matrix=A,
        constraint_lower_bound=l,
        constraint_upper_bound=u,
    )
    # set model sense
    model.ModelSense = PDLP.MAXIMIZE
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

# warm start tests
def test_warm_start(title, A, primal_start=None, dual_start=None):
    print(f"\n{title}")
    try:
        model = Model(
            objective_vector=c,
            constraint_matrix=A,
            constraint_lower_bound=l,
            constraint_upper_bound=u,
        )
        model.ModelSense = PDLP.MAXIMIZE
        model.setParam("TimeLimit", 30.0)
        model.setParam("TermCheckFreq", 10)
        model.setParams(OutputFlag=False, IterationLimit=200000)
        model.Params.FeasibilityTol = 1e-8
        model.Params.OptimalityTol = 1e-8
        
        # Set warm start
        model.setWarmStart(primal=primal_start, dual=dual_start)
        model.optimize()
            
    except Exception as e:
        print(f"Unexpected exception - {e}")

correct_primal = np.array([1.0, 1.0]) 
correct_dual = np.array([1.0, 1.0, 1.0]) 
wrong_primal = np.array([1.0])  # size 1 (too small)
wrong_dual = np.array([1.0, 1.0])  # size 2 (too small)

matrices = [
    ("Dense", A_dense),
    ("CSR", A_csr), 
    ("CSC", A_csc),
    ("COO", A_coo)
]

for name, A in matrices:
    # Test 1: Correct warm start
    test_warm_start(f"Test {name} - Correct warm start", A, correct_primal, correct_dual)
    
    # Test 2: Wrong primal size (should fallback to zeros)
    test_warm_start(f"Test {name} - Wrong primal size (fallback to zeros)", A, wrong_primal, correct_dual)

    # Test 3: Wrong dual size (should fallback to zeros)
    test_warm_start(f"Test {name} - Wrong dual size (fallback to zeros)", A, correct_primal, wrong_dual)

    # Test 4: Wrong primal & dual size (should fallback to zeros)
    test_warm_start(f"Test {name} - Wrong primal & dual size (fallback to zeros)", A, wrong_primal, wrong_dual)