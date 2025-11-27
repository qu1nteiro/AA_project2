import csv
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import networkx as nx

# --- IMPORTS DO TEU PROJETO ---
# O "Graph" aqui é fundamental para o pickle funcionar corretamente
from graph_loader import GraphLoader, Graph
from pure_random import PureRandomSolver
from randomized_greedy import RandomizedGreedySolver
from simulated_annealing import SimulatedAnnealingSolver
import project1_greedy
import project1_exhaustive

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================
# O ficheiro do grafo que queres testar
GRAPH_FILE = "../data/processed/football.bin"
TIME_LIMIT = 2.0  # Segundos de corrida
OUTPUT_CSV = "../results/race_data.csv"
OUTPUT_PLOT = "../results/convergence_plot.png"


# ==============================================================================
# 1. FUNÇÃO DE EXECUÇÃO E LOGGING
# ==============================================================================
def run_race():
    # Carregar Grafo
    if not os.path.exists(GRAPH_FILE):
        print(f"ERRO: Grafo não encontrado em {GRAPH_FILE}")
        # Tenta fallback local caso estejas a correr na mesma pasta
        if os.path.exists("football.bin"):
            GRAPH_FILE = "football.bin"
        else:
            return None

    g = GraphLoader.load_from_bin(GRAPH_FILE)
    print(f"--- A INICIAR CORRIDA: {g.name} (N={g.n}) ---")

    # Lista para guardar os dados: [Algoritmo, Tempo, Peso]
    race_data = []

    # -------------------------------------------------
    # A. GREEDY (P1) - A Referência (Linha Base)
    # -------------------------------------------------
    print(">> Correndo Greedy (P1)...")
    # Converter para NetworkX (necessário para P1)
    G_nx = nx.Graph()
    for i in range(g.n): G_nx.add_node(i, weight=g.weights[i])
    for u in range(g.n):
        for v in g.adj[u]:
            if u < v: G_nx.add_edge(u, v)

    t0 = time.time()
    _, w_greedy, _ = project1_greedy.find_dominating_set_greedy(G_nx)
    t_greedy = time.time() - t0

    # Adiciona ponto inicial e final para criar uma linha horizontal no gráfico
    race_data.append({"Algorithm": "Greedy (P1)", "Time": 0, "Weight": w_greedy})
    race_data.append({"Algorithm": "Greedy (P1)", "Time": TIME_LIMIT, "Weight": w_greedy})

    # -------------------------------------------------
    # B. EXHAUSTIVE (P1) - O "Chão" (Só se N for pequeno)
    # -------------------------------------------------
    if g.n <= 25:
        print(">> Correndo Exhaustive (P1)...")
        _, w_ex, _ = project1_exhaustive.solve_exhaustive(G_nx)
        race_data.append({"Algorithm": "Exhaustive (P1)", "Time": 0, "Weight": w_ex})
        race_data.append({"Algorithm": "Exhaustive (P1)", "Time": TIME_LIMIT, "Weight": w_ex})
    else:
        print(f">> Exhaustive (P1) ignorado (N={g.n} > 25).")

    # -------------------------------------------------
    # C. ALGORITMOS P2 (Com Callback)
    # -------------------------------------------------

    # Função auxiliar que os algoritmos vão chamar quando melhorarem
    def callback(algo_name, t, w):
        race_data.append({"Algorithm": algo_name, "Time": t, "Weight": w})

    # 1. Pure Random
    print(">> Correndo Pure Random (P2)...")
    s1 = PureRandomSolver(g)
    # REMOVIDO: A chamada manual que dava erro. O solve já trata disto.
    s1.solve(TIME_LIMIT, trace_callback=lambda t, w: callback("Pure Random (P2)", t, w))

    # 2. Randomized Greedy
    print(">> Correndo Randomized Greedy (P2)...")
    s2 = RandomizedGreedySolver(g)
    s2.solve(TIME_LIMIT, k_best=5, trace_callback=lambda t, w: callback("Rand Greedy (P2)", t, w))

    # 3. Simulated Annealing
    print(">> Correndo Simulated Annealing (P2)...")
    s3 = SimulatedAnnealingSolver(g)
    s3.solve(TIME_LIMIT, trace_callback=lambda t, w: callback("Sim Annealing (P2)", t, w))

    # Salvar CSV Bruto
    df = pd.DataFrame(race_data)

    # Criar pasta results se não existir
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Dados guardados em {OUTPUT_CSV}")
    return df


# ==============================================================================
# 2. FUNÇÃO DE PLOTAGEM
# ==============================================================================
def plot_convergence(df):
    plt.figure(figsize=(12, 7))

    # Estilo
    sns.set_style("whitegrid")

    algos = df['Algorithm'].unique()
    # Usar uma paleta distinta
    colors = sns.color_palette("bright", n_colors=len(algos))

    for i, algo in enumerate(algos):
        subset = df[df['Algorithm'] == algo].sort_values("Time")

        if subset.empty:
            continue

        # Adiciona um ponto final no tempo limite para a linha ir até ao fim do gráfico
        last_row = subset.iloc[-1].copy()
        last_row['Time'] = 2.0  # Força fim no limite
        subset = pd.concat([subset, pd.DataFrame([last_row])], ignore_index=True)

        # Plot "step-post" cria o efeito de escada perfeito para otimização discreta
        plt.step(subset['Time'], subset['Weight'], where='post', label=algo, linewidth=2, color=colors[i])

    plt.title('Convergence Analysis: Performance Over Time (2s)', fontsize=16)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Best Solution Weight (Minimization)', fontsize=12)
    plt.xlim(0, 2.0)
    plt.legend(title="Algorithm", loc='upper right')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)

    print(f"Guardando gráfico em {OUTPUT_PLOT}...")
    plt.savefig(OUTPUT_PLOT, dpi=300)
    # plt.show() # Descomentar se tiveres ambiente gráfico


if __name__ == "__main__":
    df = run_race()
    if df is not None and not df.empty:
        plot_convergence(df)