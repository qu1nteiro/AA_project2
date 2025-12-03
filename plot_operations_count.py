import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from typing import List, Tuple, Any

# // ===========================================================================
# // IMPORTS & SETUP
# // ===========================================================================
from graph_loader import GraphLoader, Graph
from pure_random import PureRandomSolver
from randomized_greedy import RandomizedGreedySolver
from simulated_annealing import SimulatedAnnealingSolver

# // FIX: Pickle serialization issue
sys.modules['__main__'].Graph = Graph

# // ===========================================================================
# // CONFIGURATION
# // ===========================================================================
DATA_DIR = "../data/processed/"
RESULTS_DIR = "../results/"

# // Synthetic Graphs Configuration (N=4 to N=30)
SYNTHETIC_RANGE = list(range(4, 31))
SYNTHETIC_DENSITY = "50,0"

# // Real World Graphs
REAL_GRAPHS = [
    ("football.bin", "Football"),
    ("Facebook_Ego.bin", "Facebook")
]

# // How many times to repeat counting to average out the randomness
# // Crucial for SA which has cheap moves (Add) and expensive moves (Remove)
OPS_AVERAGE_ITERS = 100


# // ===========================================================================
# // INSTRUMENTATION CLASSES (The "Spies")
# // ===========================================================================

class SpyRandom(PureRandomSolver):
    def __init__(self, graph):
        super().__init__(graph)
        self.ops = 0

    def _construct_solution(self):
        self.ops = 0
        self.ops += self.graph.n  # Shuffle cost

        current_nodes = set()
        covered_nodes = set()
        candidates = list(range(self.graph.n))

        for node in candidates:
            if len(covered_nodes) == self.graph.n:
                break

            self.ops += 1  # Visit Node
            current_nodes.add(node)
            covered_nodes.add(node)

            # Visit Neighbors
            degree = len(self.graph.adj[node])
            self.ops += (1 + degree)

            for neighbor in self.graph.adj[node]:
                covered_nodes.add(neighbor)
        return None


class SpyGreedy(RandomizedGreedySolver):
    def __init__(self, graph):
        super().__init__(graph)
        self.ops = 0

    def _calculate_gain(self, node_idx, nodes_to_cover):
        # 1 access to adj list + N accesses for neighbors
        self.ops += (1 + len(self.graph.adj[node_idx]))
        return super()._calculate_gain(node_idx, nodes_to_cover)

    def _construct_solution(self, k_best):
        self.ops = 0
        return super()._construct_solution(k_best)


class SpySA(SimulatedAnnealingSolver):
    def __init__(self, graph):
        super().__init__(graph)
        self.ops = 0

    def _is_dominating(self, nodes):
        cost = 0
        for u in nodes:
            # Visit node + Visit neighbors
            cost += (1 + len(self.graph.adj[u]))
        self.ops += cost
        return super()._is_dominating(nodes)


# // ===========================================================================
# // BENCHMARK ENGINE
# // ===========================================================================
def count_operations(graph: Graph) -> Tuple[float, float, float]:
    """
    Runs algorithms multiple times and returns AVERAGE ops count.
    """

    # // 1. Pure Random Spy
    spy_rnd = SpyRandom(graph)
    total_rnd = 0
    # Random is quite stable, but averaging doesn't hurt
    for _ in range(10):
        spy_rnd._construct_solution()
        total_rnd += spy_rnd.ops
    avg_rnd = total_rnd / 10

    # // 2. Randomized Greedy Spy
    spy_grd = SpyGreedy(graph)
    # Greedy is deterministic in ops count (mostly), 1 run is enough for speed
    spy_grd._construct_solution(k_best=3)
    avg_grd = spy_grd.ops

    # // 3. SA Spy (CRITICAL: MUST AVERAGE)
    spy_sa = SpySA(graph)
    sol = spy_sa._generate_initial_solution()

    total_sa = 0
    for _ in range(OPS_AVERAGE_ITERS):
        spy_sa.ops = 0  # Reset
        spy_sa._get_neighbor(sol)
        total_sa += spy_sa.ops

    avg_sa = total_sa / OPS_AVERAGE_ITERS

    return (avg_rnd, avg_grd, avg_sa)


# // ===========================================================================
# // MAIN
# // ===========================================================================
def main():
    print(">> [Ops Counter] Starting analysis (Averaging SA steps)...")

    data_n = []
    data_ops_rnd = []
    data_ops_grd = []
    data_ops_sa = []

    # // 1. SYNTHETIC
    print("   [1/2] Processing Synthetic Graphs...")
    for n in SYNTHETIC_RANGE:
        filename = f"graph_n{n}_p{SYNTHETIC_DENSITY}.gml.bin"
        path = os.path.join(DATA_DIR, filename)

        if os.path.exists(path):
            try:
                g = GraphLoader.load_from_bin(path)
                ops = count_operations(g)

                data_n.append(n)
                data_ops_rnd.append(ops[0])
                data_ops_grd.append(ops[1])
                data_ops_sa.append(ops[2])
            except:
                pass

    # // 2. REAL WORLD
    print("   [2/2] Processing Real World Graphs...")
    real_indices = []

    for filename, label in REAL_GRAPHS:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path): path = filename

        if os.path.exists(path):
            print(f"    >> Analyzing {label}...")
            try:
                g = GraphLoader.load_from_bin(path)
                ops = count_operations(g)

                idx = len(data_n)
                real_indices.append((idx, label))

                data_n.append(g.n)
                data_ops_rnd.append(ops[0])
                data_ops_grd.append(ops[1])
                data_ops_sa.append(ops[2])
                print(f"       Nodes: {g.n} | SA Avg Ops: {ops[2]:.1f}")
            except Exception as e:
                print(e)

    # // 3. PLOTTING
    if not data_n: return

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 7))

    # Plot Lines
    plt.plot(data_n, data_ops_grd, 'o-', color='firebrick', label='Randomized Greedy (High Cost)', linewidth=2, markersize=6)
    plt.plot(data_n, data_ops_rnd, 'o-', color='green', label='Pure Random (Linear)', linewidth=2, markersize=6)
    plt.plot(data_n, data_ops_sa, 'o-', color='orange', label='SA (Avg Linear)', linewidth=2, markersize=6)

    # Annotations
    for idx, label in real_indices:
        x_val = data_n[idx]
        y_val = data_ops_grd[idx]
        plt.text(x_val, y_val * 1.5, f"{label}\n(N={x_val})",
                 ha='center', va='bottom', fontweight='bold', color='black', fontsize=9)

    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('Graph Size (Number of Vertices) - Log Scale', fontweight='bold')
    plt.ylabel('Average Fundamental Operations - Log Scale', fontweight='bold')
    plt.title('Algorithmic Complexity: Operations Count vs Graph Size', fontsize=14, pad=15)

    plt.legend(frameon=True, shadow=True, fontsize=11)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()

    if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)
    output_path = os.path.join(RESULTS_DIR, "operations_count_analysis.png")
    plt.savefig(output_path, dpi=300)
    print(f">> Success! Chart saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()