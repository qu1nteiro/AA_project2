import random
import time
import os
import sys
from typing import Set, List, Optional

# Importamos o Loader apenas para carregar os dados .bin corretamente
from graph_loader import GraphLoader, Graph


# ==============================================================================
# STRUCTURES (C-Style Structs)
# ==============================================================================
class Solution:
    """
    Simples contentor para guardar uma solução candidata.
    Guarda os nós escolhidos e o peso total.
    """

    def __init__(self, nodes: Set[int], weight: int):
        self.nodes: Set[int] = nodes
        self.weight: int = weight

    def __repr__(self):
        return f"Solution(Nodes={len(self.nodes)}, Weight={self.weight})"


# ==============================================================================
# ALGORITHM CLASS (The Engine)
# ==============================================================================
class PureRandomSolver:
    """
    Implementação da Estratégia 1: Construção Puramente Aleatória.
    """

    def __init__(self, graph: Graph):
        self.graph = graph
        self.best_solution: Optional[Solution] = None

        # Histórico (Hash Set) para evitar testar soluções repetidas
        # Requisito: "Ensuring that no such solutions are tested more than once"
        self.history_hashes: Set[int] = set()
        self.solutions_tested: int = 0

    def _construct_solution(self) -> Solution:
        """
        Gera UMA solução válida usando aleatoriedade.
        Lógica:
        1. Baralha a lista de todos os vértices.
        2. Adiciona vértices um a um até o grafo estar coberto.
        """
        current_nodes = set()
        covered_nodes = set()  # Auxiliar para verificar cobertura rapidamente
        current_weight = 0

        # 1. Random Permutation (Baralhar os candidatos)
        candidates = list(range(self.graph.n))
        random.shuffle(candidates)

        # 2. Construção Gulosa-Aleatória (Adiciona até cobrir)
        total_nodes_needed = self.graph.n

        for node in candidates:
            # Critério de Paragem: Se já cobrimos tudo, para.
            if len(covered_nodes) == total_nodes_needed:
                break

            # Adiciona nó à solução
            current_nodes.add(node)
            current_weight += self.graph.weights[node]

            # Atualiza o conjunto de nós vigiados (o próprio + vizinhos)
            covered_nodes.add(node)
            for neighbor in self.graph.adj[node]:
                covered_nodes.add(neighbor)

        return Solution(current_nodes, current_weight)

    def solve(self, max_time_seconds: float, trace_callback=None) -> Solution:
        """
        Loop principal que corre tentativas até o tempo acabar.
        Requisito: "Deciding when to stop testing... after computation time"

        Args:
            max_time_seconds: Tempo limite.
            trace_callback: Função (tempo, peso) chamada quando encontramos uma melhoria.
        """
        start_time = time.time()
        self.solutions_tested = 0
        self.history_hashes.clear()
        self.best_solution = None

        print(f"[Solver] Starting Pure Random Search ({max_time_seconds}s limit)...")

        while (time.time() - start_time) < max_time_seconds:

            # 1. Gerar Candidato
            candidate = self._construct_solution()
            self.solutions_tested += 1

            # 2. Verificar Duplicados (Hashing)
            # Usamos frozenset porque sets normais não são 'hashable'
            sol_hash = hash(frozenset(candidate.nodes))

            if sol_hash in self.history_hashes:
                continue  # Skip se já vimos esta combinação exata

            self.history_hashes.add(sol_hash)

            # 3. Atualizar Melhor Solução
            if (self.best_solution is None) or (candidate.weight < self.best_solution.weight):
                self.best_solution = candidate

                # --- HOOK PARA O GRÁFICO DE CONVERGÊNCIA ---
                if trace_callback:
                    elapsed = time.time() - start_time
                    trace_callback(elapsed, candidate.weight)
                # -------------------------------------------

        return self.best_solution


# ==============================================================================
# MAIN execution
# ==============================================================================
if __name__ == "__main__":

    # 1. Definir caminho para o grafo processado (Football é o 'Medium')
    processed_path = "../data/processed/football.bin"

    if not os.path.exists(processed_path):
        print(f"Error: File not found at {processed_path}")
        # Tenta fallback para diretoria atual se correr na mesma pasta
        processed_path = "football.bin"

    if os.path.exists(processed_path):
        # 2. Carregar Grafo
        print(f"--- Loading Graph from {processed_path} ---")
        my_graph = GraphLoader.load_from_bin(processed_path)

        # 3. Inicializar Solver
        solver = PureRandomSolver(my_graph)

        # 4. Correr o Algoritmo (2 segundos)
        # Nota: trace_callback=None por defeito, logo não crasha aqui
        final_sol = solver.solve(max_time_seconds=2.0)

        # 5. Apresentar Resultados
        print("\n" + "=" * 40)
        print(f"FINAL RESULT FOR: {my_graph.name}")
        print("=" * 40)
        print(f"Strategy:        Pure Random")
        print(f"Time Limit:      2.0s")
        print(f"Solutions Tried: {solver.solutions_tested}")
        if final_sol:
            print(f"BEST WEIGHT:     {final_sol.weight}")
            print(f"SIZE (Nodes):    {len(final_sol.nodes)}")
        else:
            print("No solution found.")
        print("=" * 40)