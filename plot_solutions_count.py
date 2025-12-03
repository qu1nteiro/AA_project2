import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import time
from typing import List, Tuple

# // ===========================================================================
# // IMPORTS
# // ===========================================================================
from graph_loader import GraphLoader, Graph
from pure_random import PureRandomSolver
from randomized_greedy import RandomizedGreedySolver
from simulated_annealing import SimulatedAnnealingSolver

# // FIX Pickle
sys.modules['__main__'].Graph = Graph

# // ===========================================================================
# // CONFIGURATION
# // ===========================================================================
DATA_DIR = "../data/processed/"
RESULTS_DIR = "../results/"

# // Dataset Configuration
SYNTHETIC_RANGE = list(range(4, 31))
SYNTHETIC_DENSITY = "50,0"

REAL_GRAPHS = [
    ("football.bin", "Football"),
    ("Facebook_Ego.bin", "Facebook")
]

# // Test Duration: 0.5s is enough to get large numbers
TEST_DURATION = 0.5


# // ===========================================================================
# // BENCHMARK ENGINE
# // ===========================================================================
def measure_throughput(graph: Graph) -> Tuple[int, int, int]:
    """
    Runs each algorithm for X seconds.
    Returns: (Random_Count, Greedy_Count, SA_Count) normalized per second.
    """

    # // 1. Pure Random
    s1 = PureRandomSolver(graph)
    s1.solve(max_time_seconds=TEST_DURATION)
    rnd_count = s1.solutions_tested

    # // 2. Randomized Greedy
    s2 = RandomizedGreedySolver(graph)
    s2.solve(max_time_seconds=TEST_DURATION, k_best=3)
    grd_count = s2.solutions_tested

    # // 3. Simulated Annealing
    s3 = SimulatedAnnealingSolver(graph)
    s3.solve(max_time_seconds=TEST_DURATION)
    sa_count = s3.solutions_tested

    # // Normalize to "Solutions per Second"
    factor = 1.0 / TEST_DURATION
    return (int(rnd_count * factor), int(grd_count * factor), int(sa_count * factor))


# // ===========================================================================
# // MAIN
# // ===========================================================================
def main():
    print(f">> [Throughput Counter] Measuring solutions per second (Duration={TEST_DURATION}s)...")

    data_n = []
    data_rnd = []
    data_grd = []
    data_sa = []

    # // 1. SYNTHETIC GRAPHS
    print("   [1/2] Processing Synthetic Graphs...")
    for n in SYNTHETIC_RANGE:
        filename = f"graph_n{n}_p{SYNTHETIC_DENSITY}.gml.bin"
        path = os.path.join(DATA_DIR, filename)

        if os.path.exists(path):
            try:
                g = GraphLoader.load_from_bin(path)
                counts = measure_throughput(g)

                data_n.append(n)
                data_rnd.append(counts[0])
                data_grd.append(counts[1])
                data_sa.append(counts[2])
            except:
                pass

    # // 2. REAL WORLD GRAPHS
    print("   [2/2] Processing Real World Graphs...")
    real_indices = []

    for filename, label in REAL_GRAPHS:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path): path = filename

        if os.path.exists(path):
            print(f"    >> Analyzing {label}...")
            try:
                g = GraphLoader.load_from_bin(path)
                counts = measure_throughput(g)

                idx = len(data_n)
                real_indices.append((idx, label))

                data_n.append(g.n)
                data_rnd.append(counts[0])
                data_grd.append(counts[1])
                data_sa.append(counts[2])
                print(f"       Nodes: {g.n} | Greedy: {counts[1]} | Random: {counts[0]} | SA: {counts[2]}")
            except Exception as e:
                print(e)

    # // 3. PLOTTING
    if not data_n: return

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(11, 7))  # Slightly wider to fit text

    # // Plot Lines
    plt.plot(data_n, data_rnd, 'o-', color='green', label='Pure Random (High Throughput)', linewidth=2)
    plt.plot(data_n, data_sa, 'o-', color='orange', label='Simulated Annealing (High Throughput)', linewidth=2)
    plt.plot(data_n, data_grd, 'o-', color='firebrick', label='Randomized Greedy (Low Throughput)', linewidth=2)

    # // ANNOTATIONS (Updated to include SA)
    for idx, label in real_indices:
        x_val = data_n[idx]
        y_val_rnd = data_rnd[idx]
        y_val_grd = data_grd[idx]
        y_val_sa = data_sa[idx]

        # // 1. Random (Top Green) - Text Above
        plt.text(x_val, y_val_rnd, f"{label}\n(~{y_val_rnd:.0e} sols/s)",
                 ha='right', va='bottom', color='green', fontsize=9, fontweight='bold')

        # // 2. SA (Middle Orange) - Text Below (offset slightly to avoid Green line)
        # // We define a small vertical offset factor
        plt.text(x_val, y_val_sa * 0.7, f"(~{y_val_sa:.0e} sols/s)",
                 ha='right', va='top', color='orange', fontsize=9, fontweight='bold')

        # // 3. Greedy (Bottom Red) - Text Below
        plt.text(x_val, y_val_grd, f"(~{y_val_grd} sols/s)",
                 ha='right', va='top', color='firebrick', fontsize=9, fontweight='bold')

    # // Log Scale
    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('Graph Size (N) - Log Scale', fontweight='bold')
    plt.ylabel('Solutions Tested per Second - Log Scale', fontweight='bold')
    plt.title('Exploration Power: Number of Configurations Tested per Second', fontsize=14, pad=15)

    plt.legend(frameon=True, shadow=True)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()

    if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)
    output_path = os.path.join(RESULTS_DIR, "solutions_throughput.png")
    plt.savefig(output_path, dpi=300)
    print(f">> Chart saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()