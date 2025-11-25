import networkx as nx
import matplotlib.pyplot as plt
import os
import sys
from typing import Optional

# Importar a nossa classe personalizada para o pickle funcionar
from graph_loader import GraphLoader, Graph


# ==========================================
# Módulo de Conversão (Bridge Pattern)
# ==========================================
def convert_to_networkx(custom_graph: Graph) -> nx.Graph:
    """
    Converte a nossa estrutura interna (Listas de Adjacência)
    para um objeto NetworkX pronto a ser desenhado.
    """
    G = nx.Graph()

    # 1. Adicionar nós com metadados (Pesos)
    for i in range(custom_graph.n):
        # O ID do nó será o índice inteiro (0, 1, 2...)
        weight_val = custom_graph.weights[i]
        G.add_node(i, weight=weight_val)

    # 2. Adicionar arestas
    # Iteramos a lista de adjacências.
    # Como o grafo é não-dirigido, (u,v) e (v,u) aparecem ambos.
    # O NetworkX lida com duplicados automaticamente, mas podemos otimizar:
    for u in range(custom_graph.n):
        for v in custom_graph.adj[u]:
            if u < v:  # Adiciona apenas uma vez por par
                G.add_edge(u, v)

    return G


# ==========================================
# Módulo de Visualização
# ==========================================
def inspect_graph(filename: str, processed_path: str):
    """
    Carrega o binário, converte e desenha.
    """
    full_path = os.path.join(processed_path, filename)

    print(f"--- Inspection Target: {filename} ---")

    if not os.path.exists(full_path):
        print(f"[ERROR] File not found at: {full_path}")
        return

    # 1. Carregar o objeto binário (A nossa classe Graph)
    try:
        custom_g = GraphLoader.load_from_bin(full_path)
    except Exception as e:
        print(f"[ERROR] Failed to load pickle: {e}")
        return

    # 2. Converter para NetworkX
    print("Converting to NetworkX format...")
    G_nx = convert_to_networkx(custom_g)

    # 3. Análise Estatística Básica
    print(f"\n--- Statistics ---")
    print(f"Nodes: {G_nx.number_of_nodes()}")
    print(f"Edges: {G_nx.number_of_edges()}")
    if nx.is_connected(G_nx):
        print("Status: Connected")
    else:
        print(f"Status: Disconnected ({nx.number_connected_components(G_nx)} components)")

    # 4. Configuração Visual (Adaptativa)
    # Se o grafo for muito grande, ajustamos o estilo para não "crashar" visualmente
    print("\n--- Generating Layout (Physics Simulation) ---")

    plt.figure(figsize=(12, 10))

    # Algoritmo de layout (Spring é o mais bonito para visualizar conexões)
    # k controla a distância entre nós (k maior = nós mais afastados)
    pos = nx.spring_layout(G_nx, seed=42, k=0.15 if custom_g.n > 100 else 0.5)

    # Definições baseadas no tamanho
    if custom_g.n < 100:
        # Modo Detalhado (Micro & Football)
        labels = nx.get_node_attributes(G_nx, 'weight')
        node_size = 500
        font_size = 10
        with_labels = True
        node_color = 'lightblue'
        edge_color = 'gray'
    else:
        # Modo Macro (Facebook) - Sem texto, nós pequenos
        labels = None
        node_size = 30
        font_size = 0
        with_labels = False
        node_color = '#3b5998'  # Facebook blue :)
        edge_color = '#d3d3d3'  # Light gray
        print("[INFO] Large graph detected: Labels hidden for clarity.")

    # Desenhar
    nx.draw(G_nx, pos,
            with_labels=with_labels,
            labels=labels,
            node_color=node_color,
            edge_color=edge_color,
            node_size=node_size,
            font_size=font_size,
            font_weight='bold',
            alpha=0.9)  # Transparência ligeira

    plt.title(f"Visual Inspection: {custom_g.name}\n(N={custom_g.n}, M={custom_g.m})")
    plt.axis('off')  # Remove eixos X/Y (não fazem sentido em force-directed layouts)

    print("--- Displaying Plot ---")
    plt.show()


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":

    # Configuração de caminhos
    BASE_PATH = "../data/processed"

    # Lista os ficheiros disponíveis
    if os.path.exists(BASE_PATH):
        print(f"Available files in {BASE_PATH}:")
        files = [f for f in os.listdir(BASE_PATH) if f.endswith('.bin')]
        for f in sorted(files):
            print(f" - {f}")
        print("-" * 30)

    # --- ESCOLHE O FICHEIRO AQUI ---
    # Exemplos: "Football.bin", "Facebook_Ego.bin", "graph_30_0.125.bin"

    TARGET_FILE = "Facebook_Ego.bin"  # <--- Altera isto para testar outros

    inspect_graph(TARGET_FILE, BASE_PATH)