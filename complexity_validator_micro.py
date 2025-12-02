import time
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.optimize import curve_fit
from typing import List, Any

# // Import Custom Modules
from graph_loader import GraphLoader, Graph
from pure_random import PureRandomSolver
from randomized_greedy import RandomizedGreedySolver
from simulated_annealing import SimulatedAnnealingSolver

# // ===========================================================================
# // FIX: PICKLE CLASS PATH ISSUE
# // ===========================================================================
# // Ensures pickle finds the Graph class correctly during loading
sys.modules['__main__'].Graph = Graph

# // ===========================================================================
# // CONFIGURATION
# // ===========================================================================
# // Micro-interval to validate theory in detail (Small N)
TEST_SIZES: List[int] = list(range(4, 31))
TARGET_DENSITY: str = "50,0"
DATA_PATH: str = "../data/processed/"

# // High number of repetitions to capture tiny execution times accurately
ITERATIONS_FOR_AVG: int = 2000


# // ===========================================================================
# // BENCHMARK ENGINE
# // ===========================================================================
def measure_iteration_cost(solver_class: Any, graph: Graph) -> float:
    # // Setup for SA (needs initial solution to perturb)
    if solver_class == SimulatedAnnealingSolver:
        solver_inst = solver_class(graph)
        current_sol = solver_inst._generate_initial_solution()

    start_time = time.perf_counter()

    for _ in range(ITERATIONS_FOR_AVG):
        if solver_class == RandomizedGreedySolver:
            s = solver_class(graph)
            s._construct_solution(k_best=3)
        elif solver_class == PureRandomSolver:
            s = solver_class(graph)
            s._construct_solution()
        elif solver_class == SimulatedAnnealingSolver:
            # // Measure atomic step: Neighbor generation + Validation
            solver_inst._get_neighbor(current_sol)

    end_time = time.perf_counter()
    return (end_time - start_time) / ITERATIONS_FOR_AVG


# // ===========================================================================
# // THEORETICAL MODELS
# // ===========================================================================
def model_poly(n, c): return c * (n ** 2)  # Greedy Model (Quadratic)


def model_lin(n, c): return c * n  # Random/SA Model (Linear)


# // ===========================================================================
# // MAIN EXECUTION
# // ===========================================================================
def main():
    print(f">> [Micro-Benchmark] Starting validation N=4..30 ({ITERATIONS_FOR_AVG} iterations)...")

    res_greedy, res_random, res_sa = [], [], []
    valid_n = []

    for n in TEST_SIZES:
        filename = f"graph_n{n}_p{TARGET_DENSITY}.gml.bin"
        try:
            g = GraphLoader.load_from_bin(DATA_PATH + filename)
            valid_n.append(n)

            res_greedy.append(measure_iteration_cost(RandomizedGreedySolver, g))
            res_random.append(measure_iteration_cost(PureRandomSolver, g))
            res_sa.append(measure_iteration_cost(SimulatedAnnealingSolver, g))

            if n % 5 == 0:
                print(f"   [Processed] N={n}")
        except:
            print(f"   [Skip] File not found: {filename}")

    # // Curve Fitting (Find constant 'c')
    x = np.array(valid_n)
    popt_g, _ = curve_fit(model_poly, x, res_greedy)
    popt_r, _ = curve_fit(model_lin, x, res_random)
    popt_s, _ = curve_fit(model_lin, x, res_sa)

    # // Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 5))

    # // Data Points (Experimental)
    plt.plot(x, res_greedy, 'o', color='firebrick', alpha=0.6, label='Greedy (Exp.)')
    plt.plot(x, res_random, 'o', color='green', alpha=0.6, label='Random (Exp.)')
    plt.plot(x, res_sa, 'o', color='orange', alpha=0.6, label='SA (Exp.)')

    # // Theoretical Curves (Fitted)
    x_smooth = np.linspace(min(x), max(x), 100)
    plt.plot(x_smooth, model_poly(x_smooth, popt_g[0]), 'r--', label=f'Theoretical $O(n^2)$')
    plt.plot(x_smooth, model_lin(x_smooth, popt_r[0]), 'g--', label=f'Theoretical $O(n)$')
    plt.plot(x_smooth, model_lin(x_smooth, popt_s[0]), color='orange', linestyle='--', label=f'Theoretical $O(n)$')

    # // Labels & Styling
    plt.xlabel('Number of Vertices ($N$)')
    plt.ylabel('Average Time per Iteration (s)')
    plt.title('Experimental Complexity Validation (Micro-Scale)')
    plt.legend()
    plt.tight_layout()

    # // Save and Show
    output_file = "../results/micro_complexity_validation.png"
    plt.savefig(output_file, dpi=300)
    print(f">> Plot generated: {output_file}")
    plt.show()


if __name__ == "__main__":
    main()