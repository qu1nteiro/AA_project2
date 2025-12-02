import time
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from typing import List, Tuple

# // Import Custom Modules
# // We explicitly import Graph to map it for pickle
from graph_loader import GraphLoader, Graph
from pure_random import PureRandomSolver
from randomized_greedy import RandomizedGreedySolver
from simulated_annealing import SimulatedAnnealingSolver

# // ===========================================================================
# // FIX: PICKLE CLASS PATH ISSUE
# // ===========================================================================
# // The .bin files were likely saved when 'Graph' was in the __main__ scope.
# // This line redirects pickle to find the class in the imported module.
sys.modules['__main__'].Graph = Graph

# // ===========================================================================
# // CONFIGURATION
# // ===========================================================================
DATA_PATH = "../data/processed/"
REAL_GRAPHS = [
    ("football.bin", "Football\n(N=115)", 115),
    ("Facebook_Ego.bin", "Facebook\n(N=4039)", 4039)
]


# // ===========================================================================
# // HELPER: ITERATION COUNT
# // ===========================================================================
def get_iterations(n_nodes: int) -> int:
    """
    Adjusts the number of test runs based on graph size to save time.
    For Facebook (large), we run fewer times because Greedy is slow.
    """
    return 50 if n_nodes < 500 else 5


# // ===========================================================================
# // BENCHMARK ENGINE
# // ===========================================================================
def measure_iteration_cost(solver_class, graph, n_nodes):
    iters = get_iterations(n_nodes)

    # // Setup SA (needs an initial solution to perturb)
    if solver_class == SimulatedAnnealingSolver:
        solver_inst = solver_class(graph)
        current_sol = solver_inst._generate_initial_solution()

    start_time = time.perf_counter()

    for _ in range(iters):
        if solver_class == RandomizedGreedySolver:
            s = solver_class(graph)
            s._construct_solution(k_best=3)
        elif solver_class == PureRandomSolver:
            s = solver_class(graph)
            s._construct_solution()
        elif solver_class == SimulatedAnnealingSolver:
            # // Measure one neighbor generation + validation
            solver_inst._get_neighbor(current_sol)

    end_time = time.perf_counter()
    return (end_time - start_time) / iters


# // ===========================================================================
# // MAIN EXECUTION
# // ===========================================================================
def main():
    print(">> [Macro-Benchmark] Starting comparison: Football vs Facebook...")

    # // Data Containers
    x_labels = []
    x_values = []

    y_greedy = []
    y_random = []
    y_sa = []

    # // 1. Data Collection Loop
    for filename, label, n in REAL_GRAPHS:
        full_path = DATA_PATH + filename

        # // Fallback: Check local directory if not found in processed
        if not os.path.exists(full_path):
            full_path = filename

        if os.path.exists(full_path):
            clean_name = label.replace('\n', ' ')
            print(f"   >> Loading and processing: {clean_name}...")

            # // Load Graph
            g = GraphLoader.load_from_bin(full_path)

            x_labels.append(label)
            x_values.append(n)

            # // Measure Execution Times
            t_grd = measure_iteration_cost(RandomizedGreedySolver, g, n)
            t_rnd = measure_iteration_cost(PureRandomSolver, g, n)
            t_sa = measure_iteration_cost(SimulatedAnnealingSolver, g, n)

            y_greedy.append(t_grd)
            y_random.append(t_rnd)
            y_sa.append(t_sa)

            print(f"      [Results] Greedy: {t_grd:.5f}s | Random: {t_rnd:.5f}s | SA: {t_sa:.5f}s")
        else:
            print(f"   [ERROR] Graph file not found: {filename}")

    # // Safety check
    if len(x_values) < 2:
        print(">> Not enough data points to plot comparison. Check file paths.")
        return

    # // 2. Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(9, 7))

    # // Plot Lines
    plt.plot(x_values, y_greedy, 'o-', color='firebrick', linewidth=3, markersize=10, label='Greedy ($O(n^2)$)')
    plt.plot(x_values, y_random, 'o-', color='green', linewidth=3, markersize=10, label='Random ($O(n)$)')
    plt.plot(x_values, y_sa, 'o-', color='orange', linewidth=3, markersize=10, label='SA ($O(n)$)')

    # // Calculate Fold Increase (How many times slower?)
    increase_greedy = y_greedy[1] / y_greedy[0]
    increase_random = y_random[1] / y_random[0]

    # // Add Text Annotations to the Graph
    plt.text(x_values[1], y_greedy[1], f"  x{increase_greedy:.0f} slower",
             va='bottom', color='firebrick', fontweight='bold', fontsize=11)

    plt.text(x_values[1], y_random[1], f"  x{increase_random:.0f} slower",
             va='top', color='green', fontsize=11)

    # // Axis Configuration (Log Scale is crucial here)
    plt.xscale('log')
    plt.yscale('log')

    # // Custom Ticks
    plt.xticks(x_values, x_labels, fontsize=12, fontweight='bold')
    plt.ylabel('Time per Iteration (seconds) - Log Scale', fontsize=12, fontweight='bold')
    plt.title('Real-World Scale Impact: Football vs Facebook', fontsize=14, pad=15)

    plt.legend(frameon=True, shadow=True, fontsize=11)
    plt.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()

    # // Save Output
    output_file = "../results/macro_complexity_comparison.png"
    plt.savefig(output_file, dpi=300)
    print(f">> Plot generated successfully: {output_file}")
    plt.show()


if __name__ == "__main__":
    main()