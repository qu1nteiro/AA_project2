import csv
import os
import glob
import time
import networkx as nx

# Imports Project 2
from graph_loader import GraphLoader, Graph
from pure_random import PureRandomSolver
from randomized_greedy import RandomizedGreedySolver
from simulated_annealing import SimulatedAnnealingSolver

# Imports Project 1 (Legacy)
import project1_greedy
import project1_exhaustive

# ==============================================================================
# CONFIGURATION
# ==============================================================================
RESULTS_FILE = "../results/final_results.csv"
PROCESSED_DATA_PATH = "../data/processed"

# 2 Seconds for Random Algorithms
TIME_LIMIT_RANDOM = 2.0

# Limit for Exhaustive Search (Safety Guard)
# Do NOT run exhaustive if nodes > MAX_NODES_EXHAUSTIVE
MAX_NODES_EXHAUSTIVE = 22

TARGET_GRAPHS = []

# ==============================================================================
# HELPER: ADAPTER PATTERN
# ==============================================================================
def convert_custom_to_nx(custom_graph: Graph) -> nx.Graph:
    """
    Bridges the gap between Project 2 (Custom Graph) and Project 1 (NetworkX).
    """
    G = nx.Graph()
    # Add nodes with weights
    for i in range(custom_graph.n):
        G.add_node(i, weight=custom_graph.weights[i])

    # Add edges
    for u in range(custom_graph.n):
        for v in custom_graph.adj[u]:
            if u < v:  # Avoid duplicates
                G.add_edge(u, v)
    return G


def log_result(writer, graph_name, n, algo, limit, time_taken, weight, size, ops):
    """ Writes a standardized row to CSV. """
    row = [
        graph_name, n, algo,
        limit, f"{time_taken:.5f}",
        weight, size, ops
    ]
    writer.writerow(row)
    print(f"   > {algo:<22} | W: {str(weight):<6} | Time: {time_taken:.4f}s")


# ==============================================================================
# MAIN TEST SUITE
# ==============================================================================
def run_suite():
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

    # 1. DETETAR FICHEIROS AUTOMATICAMENTE
    # Procura tudo o que começa por "graph_" (Micro) e os grandes específicos
    all_bins = glob.glob(os.path.join(PROCESSED_DATA_PATH, "*.bin"))

    # Filtra e organiza
    micro_graphs = [os.path.basename(f) for f in all_bins if "graph_" in f]
    medium_large = ["football.bin", "Facebook_Ego.bin"]  # Garante que estes nomes estão exatos (case sensitive!)

    # Lista final ordenada (Micros primeiro para veres logo o Exaustivo a funcionar)
    target_list = sorted(micro_graphs) + medium_large

    print(f"--- STARTING FULL EXPERIMENT SUITE ---")
    print(f"--- Graphs to Process: {len(target_list)} ---")
    print(f"--- Saving to: {RESULTS_FILE} ---\n")

    with open(RESULTS_FILE, mode='w', newline='') as file:
        # ... (O resto do código mantém-se igual, usa 'target_list' no loop)
        writer = csv.writer(file)

        # Header
        header = [
            "Graph", "N", "Algorithm",
            "Time_Limit", "Actual_Time",
            "Best_Weight", "Sol_Size", "Operations"
        ]
        writer.writerow(header)

        for filename in target_list:
            full_path = os.path.join(PROCESSED_DATA_PATH, filename)

            if not os.path.exists(full_path):
                print(f"[SKIP] File not found: {filename}")
                continue

            # Load Data
            g = GraphLoader.load_from_bin(full_path)
            print(f"\n>> Processing: {g.name} (N={g.n}, M={g.m})")

            # ----------------------------------------------------
            # 1. EXHAUSTIVE SEARCH (Project 1)
            # ----------------------------------------------------
            if g.n <= MAX_NODES_EXHAUSTIVE:
                # Convert to NetworkX for legacy code
                G_nx = convert_custom_to_nx(g)

                t0 = time.time()
                _, w_ex, ops_ex = project1_exhaustive.solve_exhaustive(G_nx)
                elapsed = time.time() - t0

                log_result(writer, g.name, g.n, "Exhaustive (P1)", "N/A", elapsed, w_ex, "N/A", ops_ex)
            else:
                print(f"   > Exhaustive (P1)      | SKIPPED (N={g.n} > {MAX_NODES_EXHAUSTIVE})")
                # Log a placeholder so the table isn't empty in reports
                log_result(writer, g.name, g.n, "Exhaustive (P1)", "N/A", 0, "N/A", "N/A", 0)

            # ----------------------------------------------------
            # 2. GREEDY HEURISTIC (Project 1)
            # ----------------------------------------------------
            # Convert to NetworkX
            G_nx = convert_custom_to_nx(g)

            t0 = time.time()
            _, w_gr, ops_gr = project1_greedy.find_dominating_set_greedy(G_nx)
            elapsed = time.time() - t0

            log_result(writer, g.name, g.n, "Greedy (P1)", "1 Iter", elapsed, w_gr, "N/A", ops_gr)

            # ----------------------------------------------------
            # 3. PURE RANDOM (Project 2)
            # ----------------------------------------------------
            s1 = PureRandomSolver(g)
            t0 = time.time()
            res1 = s1.solve(TIME_LIMIT_RANDOM)
            elapsed = time.time() - t0

            log_result(writer, g.name, g.n, "Pure Random (P2)", TIME_LIMIT_RANDOM, elapsed,
                       res1.weight if res1 else "N/A", len(res1.nodes) if res1 else "N/A", s1.solutions_tested)

            # ----------------------------------------------------
            # 4. RANDOMIZED GREEDY (Project 2)
            # ----------------------------------------------------
            s2 = RandomizedGreedySolver(g)
            t0 = time.time()
            res2 = s2.solve(TIME_LIMIT_RANDOM, k_best=5)
            elapsed = time.time() - t0

            log_result(writer, g.name, g.n, "Rand Greedy (P2)", TIME_LIMIT_RANDOM, elapsed,
                       res2.weight if res2 else "N/A", len(res2.nodes) if res2 else "N/A", s2.solutions_tested)

            # ----------------------------------------------------
            # 5. SIMULATED ANNEALING (Project 2)
            # ----------------------------------------------------
            s3 = SimulatedAnnealingSolver(g)
            t0 = time.time()
            res3 = s3.solve(TIME_LIMIT_RANDOM)
            elapsed = time.time() - t0

            log_result(writer, g.name, g.n, "Sim Annealing (P2)", TIME_LIMIT_RANDOM, elapsed,
                       res3.weight if res3 else "N/A", len(res3.nodes) if res3 else "N/A", s3.solutions_tested)

            file.flush()


if __name__ == "__main__":
    run_suite()