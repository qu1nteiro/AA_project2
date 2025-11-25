import random
import time
import math
import os
import sys
from typing import Set, List, Optional
from copy import deepcopy

# Importar o Loader (Project 2 Standard)
from graph_loader import GraphLoader, Graph


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================
class Solution:
    """
    Standard container for a solution state.
    """

    def __init__(self, nodes: Set[int], weight: int):
        self.nodes: Set[int] = nodes
        self.weight: int = weight

    def __repr__(self):
        return f"Solution(Size={len(self.nodes)}, Weight={self.weight})"


# ==============================================================================
# STRATEGY 3: SIMULATED ANNEALING
# ==============================================================================
class SimulatedAnnealingSolver:
    """
    Implements the Simulated Annealing metaheuristic.
    Physics Metaphor:
      - High Temperature: Molecules move wildly (High exploration, accepts bad moves).
      - Low Temperature: Molecules stabilize (Low exploration, acts like Greedy).
    """

    def __init__(self, graph: Graph):
        self.graph = graph
        self.best_solution_overall: Optional[Solution] = None
        self.solutions_tested: int = 0

        # SA Parameters
        self.alpha: float = 0.98  # Cooling rate (0.8 to 0.99)
        self.restart_tolerance: int = 500  # Max iterations without improvement

    # ------------------------------------------------------
    # Helper: Validity Check
    # ------------------------------------------------------
    def _is_dominating(self, nodes: Set[int]) -> bool:
        """
        Validation function: O(Nodes * AvgDegree)
        Checks if the set covers the entire graph.
        """
        if not nodes: return False

        covered_count = 0
        # Create a boolean array for fast lookup (0 = not covered, 1 = covered)
        # Using a list is faster than a set for dense checking in Python
        status = [False] * self.graph.n

        for u in nodes:
            if not status[u]:
                status[u] = True
                covered_count += 1

            for v in self.graph.adj[u]:
                if not status[v]:
                    status[v] = True
                    covered_count += 1

        return covered_count == self.graph.n

    # ------------------------------------------------------
    # Helper: Generate Initial Solution
    # ------------------------------------------------------
    def _generate_initial_solution(self) -> Solution:
        """
        Starts with a randomized constructive solution (fast and valid).
        Does not need to be optimized, SA will fix it.
        """
        nodes = set()
        covered_count = 0
        status = [False] * self.graph.n

        # Random permutation
        candidates = list(range(self.graph.n))
        random.shuffle(candidates)

        for u in candidates:
            if covered_count == self.graph.n:
                break

            nodes.add(u)
            # Update coverage
            if not status[u]:
                status[u] = True
                covered_count += 1
            for v in self.graph.adj[u]:
                if not status[v]:
                    status[v] = True
                    covered_count += 1

        # Calculate initial weight
        w = sum(self.graph.weights[u] for u in nodes)
        return Solution(nodes, w)

    # ------------------------------------------------------
    # Core: Neighbor Generation (Perturbation)
    # ------------------------------------------------------
    def _get_neighbor(self, current_sol: Solution) -> Optional[Solution]:
        """
        Tries to modify the solution slightly.
        Move Types:
        1. REMOVE: Remove a random node (Optimizes weight).
        2. ADD: Add a random node (Adds diversity/escapes local optima).

        Returns None if the move creates an invalid solution.
        """
        new_nodes = current_sol.nodes.copy()
        move_type = random.random()

        # ACTION 1: Try to Remove a node (Greedy direction) - 60% chance
        if move_type < 0.60 and len(new_nodes) > 1:
            node_to_remove = random.choice(list(new_nodes))
            new_nodes.remove(node_to_remove)

            # Constraint Check: Must remain dominating
            if self._is_dominating(new_nodes):
                new_weight = current_sol.weight - self.graph.weights[node_to_remove]
                return Solution(new_nodes, new_weight)
            else:
                return None  # Invalid move, rejected immediately

        # ACTION 2: Add a random node (Exploration direction) - 40% chance
        else:
            # Pick a node that is NOT in the set
            candidates = [x for x in range(self.graph.n) if x not in new_nodes]
            if not candidates:
                return None

            node_to_add = random.choice(candidates)
            new_nodes.add(node_to_add)

            # Adding is always valid for domination, but increases weight
            new_weight = current_sol.weight + self.graph.weights[node_to_add]
            return Solution(new_nodes, new_weight)

    # ------------------------------------------------------
    # Manager: Main Solver Loop
    # ------------------------------------------------------
    def solve(self, max_time_seconds: float) -> Solution:
        """
        Runs the Annealing process.
        Uses a 'Restart' strategy: if the system cools down too much
        but time remains, it restarts from a new random point.
        """
        start_time = time.time()
        self.solutions_tested = 0

        print(f"[Solver] Starting Simulated Annealing ({max_time_seconds}s limit)...")

        # Outer Loop: Restarts
        while (time.time() - start_time) < max_time_seconds:

            # 1. Initialization
            current_sol = self._generate_initial_solution()

            # Set global best
            if self.best_solution_overall is None or current_sol.weight < self.best_solution_overall.weight:
                self.best_solution_overall = current_sol

            # 2. Temperature Setup
            # Heuristic: Start T at 50% of initial weight to allow some bad moves initially
            temperature = current_sol.weight * 0.5
            min_temperature = 0.1

            # Inner Loop: The Annealing Cycle
            while temperature > min_temperature:

                # Check Time Limit inside inner loop too
                if (time.time() - start_time) >= max_time_seconds:
                    break

                # A. Generate Neighbor
                neighbor = self._get_neighbor(current_sol)
                self.solutions_tested += 1

                if neighbor is None:
                    continue  # Invalid move, try again

                # B. Calculate Delta (Change in Energy/Weight)
                # Negative delta is GOOD (weight went down)
                delta = neighbor.weight - current_sol.weight

                # C. Acceptance Probability
                accepted = False

                if delta < 0:
                    # Always accept improvements
                    accepted = True
                else:
                    # Probabilistically accept bad moves based on Temperature
                    # Boltzmann distribution: P = exp(-delta / T)
                    try:
                        probability = math.exp(-delta / temperature)
                    except OverflowError:
                        probability = 0

                    if random.random() < probability:
                        accepted = True

                # D. Update State
                if accepted:
                    current_sol = neighbor

                    # Update Global Best if found
                    if current_sol.weight < self.best_solution_overall.weight:
                        self.best_solution_overall = deepcopy(current_sol)
                        # print(f"  > New Best (SA): W={current_sol.weight}")

                # E. Cool Down
                temperature *= self.alpha

            # End of one cooling cycle.
            # If time remains, the loop restarts with a fresh random solution.

        return self.best_solution_overall


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":

    # Path Configuration
    processed_path = "../data/processed/football.bin"

    if not os.path.exists(processed_path):
        print(f"Error: File not found at {processed_path}")
        sys.exit(1)

    # Load
    print(f"--- Loading Graph from {processed_path} ---")
    my_graph = GraphLoader.load_from_bin(processed_path)

    # Initialize
    solver = SimulatedAnnealingSolver(my_graph)

    # Run (e.g., 2 seconds)
    final_sol = solver.solve(max_time_seconds=2.0)

    # Results
    print("\n" + "=" * 40)
    print(f"FINAL RESULT FOR: {my_graph.name}")
    print("=" * 40)
    print(f"Strategy:        Simulated Annealing")
    print(f"Solutions Tried: {solver.solutions_tested}")
    if final_sol:
        print(f"BEST WEIGHT:     {final_sol.weight}")
        print(f"SIZE (Nodes):    {len(final_sol.nodes)}")
    else:
        print("No solution found.")
    print("=" * 40)