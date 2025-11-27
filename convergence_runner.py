import csv
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import networkx as nx

# --- IMPORTS DO TEU PROJETO ---
from graph_loader import GraphLoader, Graph
from pure_random import PureRandomSolver
from randomized_greedy import RandomizedGreedySolver
from simulated_annealing import SimulatedAnnealingSolver
import project1_greedy
import project1_exhaustive

# ==============================================================================
# CONFIGURAÇÃO GLOBAL
# ==============================================================================
DEFAULT_GRAPH_FILE = "../data/processed/football.bin"
TIME_LIMIT = 2.0
OUTPUT_CSV = "../results/race_data.csv"
OUTPUT_PLOT = "../results/convergence_plot.png"


# ==============================================================================
# 1. FUNÇÃO DE EXECUÇÃO
# ==============================================================================
def run_race():
    # 1. Resolver o caminho do ficheiro (Sem tocar na global)
    target_file = DEFAULT_GRAPH_FILE

    if not os.path.exists(target_file):
        print(f"AVISO: Grafo não encontrado em {target_file}")
        # Tenta fallback local
        if os.path.exists("football.bin"):
            print(">> Usando fallback: football.bin na pasta local.")
            target_file = "football.bin"
        else:
            print("ERRO CRÍTICO: Não foi possível encontrar o ficheiro do grafo.")
            return None

    # 2. Carregar
    g = GraphLoader.load_from_bin(target_file)
    print(f"--- A INICIAR CORRIDA: {g.name} (N={g.n}) ---")

    race_data = []

    # -------------------------------------------------
    # A. GREEDY (P1)
    # -------------------------------------------------
    print(">> Correndo Greedy (P1)...")
    G_nx = nx.Graph()
    for i in range(g.n): G_nx.add_node(i, weight=g.weights[i])
    for u in range(g.n):
        for v in g.adj[u]:
            if u < v: G_nx.add_edge(u, v)

    t0 = time.time()
    _, w_greedy, _ = project1_greedy.find_dominating_set_greedy(G_nx)
    # Registar pontos para linha reta
    race_data.append({"Algorithm": "Greedy (P1)", "Time": 0, "Weight": w_greedy})
    race_data.append({"Algorithm": "Greedy (P1)", "Time": TIME_LIMIT, "Weight": w_greedy})

    # -------------------------------------------------
    # B. EXHAUSTIVE (P1)
    # -------------------------------------------------
    if g.n <= 25:
        print(">> Correndo Exhaustive (P1)...")
        _, w_ex, _ = project1_exhaustive.solve_exhaustive(G_nx)
        race_data.append({"Algorithm": "Exhaustive (P1)", "Time": 0, "Weight": w_ex})
        race_data.append({"Algorithm": "Exhaustive (P1)", "Time": TIME_LIMIT, "Weight": w_ex})
    else:
        print(f">> Exhaustive (P1) ignorado (N={g.n} > 25).")

    # -------------------------------------------------
    # C. ALGORITMOS P2
    # -------------------------------------------------
    def callback(algo_name, t, w):
        race_data.append({"Algorithm": algo_name, "Time": t, "Weight": w})

    # 1. Pure Random
    print(">> Correndo Pure Random (P2)...")
    s1 = PureRandomSolver(g)
    s1.solve(TIME_LIMIT, trace_callback=lambda t, w: callback("Pure Random (P2)", t, w))

    # 2. Randomized Greedy
    print(">> Correndo Randomized Greedy (P2)...")
    s2 = RandomizedGreedySolver(g)
    s2.solve(TIME_LIMIT, k_best=5, trace_callback=lambda t, w: callback("Rand Greedy (P2)", t, w))

    # 3. Simulated Annealing
    print(">> Correndo Simulated Annealing (P2)...")
    s3 = SimulatedAnnealingSolver(g)
    s3.solve(TIME_LIMIT, trace_callback=lambda t, w: callback("Sim Annealing (P2)", t, w))

    # Salvar
    df = pd.DataFrame(race_data)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Dados guardados em {OUTPUT_CSV}")
    return df


# ==============================================================================
# 2. PLOT
# ==============================================================================
def plot_convergence(df):
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Paleta de cores vibrante
    algos = df['Algorithm'].unique()
    colors = sns.color_palette("bright", n_colors=len(algos))

    for i, algo in enumerate(algos):
        subset = df[df['Algorithm'] == algo].sort_values("Time")
        if subset.empty: continue

        # Estender linha final
        last_row = subset.iloc[-1].copy()
        last_row['Time'] = 2.0
        subset = pd.concat([subset, pd.DataFrame([last_row])], ignore_index=True)

        # Plot em degrau
        plt.step(subset['Time'], subset['Weight'], where='post', label=algo, linewidth=2, color=colors[i])

    plt.title('Convergence Analysis: "The Race" (Football Graph)', fontsize=14)
    plt.xlabel('Time (s)')
    plt.ylabel('Best Solution Weight')
    plt.xlim(0, 2.0)
    plt.legend(loc='upper right')

    print(f"Guardando gráfico em {OUTPUT_PLOT}...")
    plt.savefig(OUTPUT_PLOT, dpi=300)
    # plt.show()


if __name__ == "__main__":
    df = run_race()
    if df is not None:
        plot_convergence(df)