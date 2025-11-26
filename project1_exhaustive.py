import networkx as nx
import itertools


# ==============================================================================
# PROJECT 1 LEGACY CODE: EXHAUSTIVE SEARCH
# ==============================================================================

def subset_verifier(graph, subset):
    """
    Helper: Checks if a subset is a Dominating Set.
    """
    if not subset:
        return graph.number_of_nodes() == 0

    dominated_nodes = set()
    dominated_nodes.update(subset)

    for node in subset:
        dominated_nodes.update(graph.neighbors(node))

    return len(dominated_nodes) == graph.number_of_nodes()


def weight_calculation(graph, subset):
    """
    Helper: Sums the weights of the nodes in the subset.
    """
    total_weight = 0
    for node in subset:
        total_weight += graph.nodes[node]['weight']
    return total_weight


def solve_exhaustive(graph):
    """
    Main Logic from Project 1.
    Iterates through ALL combinations to find the absolute minimum weight.
    """
    all_nodes = list(graph.nodes())
    n = graph.number_of_nodes()

    best_weight = float('inf')
    best_subset = None

    tested_configs = 0

    # Iteration through all possible sizes k (0 to N)
    for k in range(n + 1):
        # Iteration through all combinations of size k
        for subset in itertools.combinations(all_nodes, k):
            tested_configs += 1

            # 1. Verification
            if subset_verifier(graph, subset):

                # 2. Weight Calculation
                weight_at_moment = weight_calculation(graph, subset)

                # 3. Best Solution Update
                if weight_at_moment < best_weight:
                    best_weight = weight_at_moment
                    best_subset = set(subset)

                    # Optimization Note for Report:
                    # Even if we find a solution, we continue searching
                    # because a better weight might exist in other combinations.

    return best_subset, best_weight, tested_configs