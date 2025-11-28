import random
import time
import os
import sys
from typing import Set, List, Tuple, Optional

# Importar o nosso Loader (Project 2 Standard)
from graph_loader import GraphLoader, Graph


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================
class Solution:
    """
    Holds the result: Selected nodes and total cost.
    """

    def __init__(self, nodes: Set[int], weight: int):
        self.nodes: Set[int] = nodes
        self.weight: int = weight

    def __repr__(self):
        return f"Solution(Size={len(self.nodes)}, Weight={self.weight})"


# ==============================================================================
# STRATEGY 2: RANDOMIZED GREEDY (GRASP Construction)
# ==============================================================================
class RandomizedGreedySolver:
    """
    Adaptation of Project 1 Greedy Heuristic to be Randomized (GRASP).
    Logic: Instead of picking the best node, pick one from the Top-K best.
    """

    def __init__(self, graph: Graph):
        self.graph = graph
        self.best_solution: Optional[Solution] = None

        # History to avoid re-testing the same solution
        self.history_hashes: Set[int] = set()
        self.solutions_tested: int = 0

    def _calculate_gain(self, node_idx: int, nodes_to_cover: Set[int]) -> int:
        """
        Helper: Calculates how many nodes from 'nodes_to_cover'
        would be covered if we select 'node_idx'.
        """
        gain = 0

        # 1. Check if the node itself needs covering
        if node_idx in nodes_to_cover:
            gain += 1

        # 2. Check neighbors
        for neighbor in self.graph.adj[node_idx]:
            if neighbor in nodes_to_cover:
                gain += 1
        return gain

    def _construct_solution(self, k_best: int) -> Solution:
        """
        Generates ONE solution using the Randomized Greedy Strategy.
        Similar to 'find_dominating_set_greedy' from Project 1, but with randomness.
        """
        current_solution_nodes = set()
        current_weight = 0

        # Set of all nodes that still need to be dominated
        # (Initially all nodes 0 to n-1)
        nodes_to_cover = set(range(self.graph.n))

        # While there are nodes left to cover
        while nodes_to_cover:

            candidates: List[Tuple[float, int]] = []

            # --- EVALUATION PHASE ---
            # Evaluate all nodes NOT yet in the solution
            for node in range(self.graph.n):
                if node not in current_solution_nodes:

                    gain = self._calculate_gain(node, nodes_to_cover)

                    if gain > 0:
                        weight = self.graph.weights[node]
                        # Avoid division by zero (safety)
                        if weight <= 0: weight = 0.0001

                        ratio = gain / weight
                        candidates.append((ratio, node))

            # Fallback: If no node provides gain (should not happen in connected graph),
            # pick the cheapest remaining node to cover itself.
            if not candidates:
                remaining = list(nodes_to_cover)
                # Pick random one from remaining to ensure progress
                best_node = random.choice(remaining)
            else:
                # --- RANKING PHASE ---
                # Sort by Ratio (Descending: Best first)
                candidates.sort(key=lambda x: x[0], reverse=True)

                # --- SELECTION PHASE (The Random Twist) ---
                # Limit the list to the Top K candidates
                limit = min(len(candidates), k_best)

                # Pick one index randomly from 0 to limit-1
                choice_idx = random.randint(0, limit - 1)
                best_node = candidates[choice_idx][1]

            # --- UPDATE PHASE ---
            current_solution_nodes.add(best_node)
            current_weight += self.graph.weights[best_node]

            # Remove covered nodes from the 'To Do' list
            nodes_to_cover.discard(best_node)
            for neighbor in self.graph.adj[best_node]:
                nodes_to_cover.discard(neighbor)

        return Solution(current_solution_nodes, current_weight)

    def solve(self, max_time_seconds: float, k_best: int = 3, trace_callback=None) -> Solution:
        """
        Runs the randomized construction multiple times within the time limit.
        Keeps the best result found.

        Args:
            max_time_seconds: Tempo limite.
            k_best: Parâmetro de aleatoriedade (Top-K).
            trace_callback: Função (tempo, peso) chamada quando encontramos melhoria.
        """
        start_time = time.time()
        self.solutions_tested = 0
        self.history_hashes.clear()
        self.best_solution = None

        print(f"[Solver] Starting Randomized Greedy (Top-{k_best}, Time Limit: {max_time_seconds}s)...")

        while (time.time() - start_time) < max_time_seconds:

            # 1. Construct a Candidate
            candidate = self._construct_solution(k_best)
            self.solutions_tested += 1

            # 2. Duplicate Check
            sol_hash = hash(frozenset(candidate.nodes))
            if sol_hash in self.history_hashes:
                continue
            self.history_hashes.add(sol_hash)

            # 3. Update Best Solution
            if (self.best_solution is None) or (candidate.weight < self.best_solution.weight):
                self.best_solution = candidate

                # --- HOOK PARA O GRÁFICO DE CONVERGÊNCIA ---
                if trace_callback:
                    elapsed = time.time() - start_time
                    trace_callback(elapsed, candidate.weight)
                # -------------------------------------------

        return self.best_solution


# ==============================================================================
# MAIN EXECUTION (Test)
# ==============================================================================
if __name__ == "__main__":

    # 1. Define Path (Testing on Football)
    processed_path = "../data/processed/football.bin"

    if not os.path.exists(processed_path):
        # Fallback para execução local
        processed_path = "football.bin"

    if os.path.exists(processed_path):
        # 2. Load Graph
        print(f"--- Loading Graph from {processed_path} ---")
        my_graph = GraphLoader.load_from_bin(processed_path)

        # 3. Initialize Solver
        solver = RandomizedGreedySolver(my_graph)

        # 4. Run Algorithm
        k_value = 5
        # trace_callback=None por defeito
        final_sol = solver.solve(max_time_seconds=2.0, k_best=k_value)

        # 5. Output
        print("\n" + "=" * 40)
        print(f"FINAL RESULT FOR: {my_graph.name}")
        print("=" * 40)
        print(f"Strategy:        Randomized Greedy (k={k_value})")
        print(f"Time Limit:      2.0s")
        print(f"Solutions Tried: {solver.solutions_tested}")
        if final_sol:
            print(f"BEST WEIGHT:     {final_sol.weight}")
            print(f"SIZE (Nodes):    {len(final_sol.nodes)}")
        else:
            print("No solution found.")
        print("=" * 40)
    else:
        print(f"File not found: {processed_path}")