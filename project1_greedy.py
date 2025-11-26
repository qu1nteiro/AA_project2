import networkx as nx

# ==============================================================================
# PROJECT 1 LEGACY CODE: GREEDY HEURISTIC
# ==============================================================================

def calculate_weight(graph, subset):
    """Calculates the sum of weights of the nodes in the subset."""
    total = 0
    for node in subset:
        total += graph.nodes[node]['weight']
    return total

def find_dominating_set_greedy(graph):
    """
    Original Greedy Algorithm from Project 1.
    Expects a NetworkX graph object.
    """
    greedy_set = set()
    nodes_to_cover = set(graph.nodes())
    ops_count = 0

    while nodes_to_cover:
        best_node = None
        best_ratio = -1.0

        # Only consider nodes that are NOT yet in our solution set
        candidates = set(graph.nodes()) - greedy_set

        for node in candidates:
            weight = graph.nodes[node]['weight']

            # Prevent division by zero
            if weight <= 0: weight = 0.0001

            # Calculate how many new nodes this candidate covers
            neighbors = set(graph.neighbors(node)) | {node}
            newly_covered = len(neighbors.intersection(nodes_to_cover))

            # Calculate Efficiency Ratio
            if newly_covered > 0:
                ratio = newly_covered / weight
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_node = node

        # Fallback
        if best_node is None:
            if not nodes_to_cover: break
            best_node = min(nodes_to_cover, key=lambda n: graph.nodes[n]['weight'])

        # Add to solution
        greedy_set.add(best_node)
        ops_count += 1

        # Remove covered nodes
        covered_now = set(graph.neighbors(best_node)) | {best_node}
        nodes_to_cover.difference_update(covered_now)

    final_weight = calculate_weight(graph, greedy_set)
    return greedy_set, final_weight, ops_count