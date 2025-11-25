import random
import csv
import os
import glob
from typing import List, Dict, Set, Tuple


# ==========================================
# Estrutura de Dados (O "Objeto" em Memória)
# ==========================================
class Graph:
    """
    Representação interna unificada.
    """

    def __init__(self, num_vertices: int, name: str = "Unknown"):
        self.name: str = name
        self.n: int = num_vertices
        self.m: int = 0

        # Adjacency List: adj[0] contém a lista de vizinhos do vértice 0
        self.adj: List[List[int]] = [[] for _ in range(num_vertices)]

        # Pesos dos Vértices: weights[0] é o custo do vértice 0
        self.weights: List[int] = [0] * num_vertices

    def add_edge(self, u: int, v: int) -> None:
        """ Adiciona aresta não-dirigida entre u e v. """
        # Evita duplicados e self-loops
        if u != v and v not in self.adj[u]:
            self.adj[u].append(v)
            self.adj[v].append(u)
            self.m += 1

    def set_weights(self, seed: int, min_w: int = 1, max_w: int = 20) -> None:
        """
        Gera pesos aleatórios determinísticos.
        Nota: Se o GML original já tiver pesos, podíamos ler de lá.
        Como o Proj 1 provavelmente não tinha pesos nos vértices, geramos agora.
        """
        random.seed(seed)
        for i in range(self.n):
            self.weights[i] = random.randint(min_w, max_w)

    def get_stats(self) -> None:
        """ Resumo para validação. """
        print(f"--- Graph: {self.name} ---")
        print(f"Nodes (n): {self.n}")
        print(f"Edges (m): {self.m}")
        # Densidade real calculada
        max_edges = self.n * (self.n - 1) / 2
        dens = self.m / max_edges if max_edges > 0 else 0
        print(f"Density: {dens:.4f}")
        print("---------------------------")


# ==========================================
# O Carregador Universal
# ==========================================
class GraphLoader:
    SEED: int = 113816

    @staticmethod
    def _map_indices(raw_ids: Set[str]) -> Dict[str, int]:
        """ Cria um mapa de IDs originais (str/int) para 0..n-1 """
        sorted_ids = sorted(list(raw_ids))
        return {raw_id: i for i, raw_id in enumerate(sorted_ids)}

    @staticmethod
    def load_facebook_edges(filepath: str) -> Graph:
        """ Lê formato .txt/.edges 'node1 node2' """
        raw_edges = []
        unique_nodes = set()

        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    u, v = parts[0], parts[1]
                    raw_edges.append((u, v))
                    unique_nodes.add(u)
                    unique_nodes.add(v)

        node_map = GraphLoader._map_indices(unique_nodes)
        g = Graph(len(unique_nodes), "Facebook_Ego")

        for u_raw, v_raw in raw_edges:
            g.add_edge(node_map[u_raw], node_map[v_raw])

        g.set_weights(GraphLoader.SEED)
        return g

    @staticmethod
    def load_twitch_csv(filepath: str) -> Graph:
        """ Lê 'large_twitch_edges.csv' """
        raw_edges = []
        unique_nodes = set()

        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # Saltar header se existir
            for row in reader:
                if len(row) >= 2:
                    u, v = row[0], row[1]
                    raw_edges.append((u, v))
                    unique_nodes.add(u)
                    unique_nodes.add(v)

        node_map = GraphLoader._map_indices(unique_nodes)
        g = Graph(len(unique_nodes), "Twitch_Gamers")

        for u_raw, v_raw in raw_edges:
            g.add_edge(node_map[u_raw], node_map[v_raw])

        g.set_weights(GraphLoader.SEED)
        return g

    @staticmethod
    def load_gml(filepath: str, custom_name: str = None) -> Graph:
        """
        Parser genérico de GML (para Football e Micros).
        Lê blocos 'node [ id X ]' e 'edge [ source A target B ]'.
        """
        raw_edges = []
        raw_nodes = set()

        # Nome do ficheiro serve de nome do grafo se não for passado outro
        graph_name = custom_name if custom_name else os.path.basename(filepath)

        with open(filepath, 'r') as f:
            # Tokenizar simples por whitespace e limpar brackets soltos se necessário
            # Esta abordagem "quick & dirty" funciona para GMLs padrão (NetworkX, etc)
            content = f.read().replace('[', ' [ ').replace(']', ' ] ').split()

        i = 0
        while i < len(content):
            token = content[i]

            # Detetar Nós
            if token == 'node':
                # Avançar até encontrar 'id' dentro do bloco
                while i < len(content) and content[i] != ']':
                    if content[i] == 'id':
                        raw_nodes.add(content[i + 1])
                    i += 1

            # Detetar Arestas
            elif token == 'edge':
                src = None
                tgt = None
                while i < len(content) and content[i] != ']':
                    if content[i] == 'source':
                        src = content[i + 1]
                    if content[i] == 'target':
                        tgt = content[i + 1]
                    i += 1
                if src is not None and tgt is not None:
                    raw_edges.append((src, tgt))
                    # Adiciona aos nós também (caso o bloco 'node' não exista)
                    raw_nodes.add(src)
                    raw_nodes.add(tgt)

            i += 1

        # Construir Grafo
        node_map = GraphLoader._map_indices(raw_nodes)
        g = Graph(len(raw_nodes), graph_name)

        for u_raw, v_raw in raw_edges:
            g.add_edge(node_map[u_raw], node_map[v_raw])

        g.set_weights(GraphLoader.SEED)
        return g


# ==========================================
# Zona de Testes (Main Atualizado)
# ==========================================
if __name__ == "__main__":

    # --- CONFIGURAÇÃO DE CAMINHOS ---
    # Ajusta isto para onde puseste as pastas!
    #BASE_PATH = "../data/raw"
    MICRO_FOLDER = os.path.join("/home/eduardo/Documents/computacional/aa/Project2/data/local_graphs")  # Pasta dos teus GMLs antigos

    print(">>> INICIANDO VALIDAÇÃO DOS GRAFOS <<<\n")

    # 1. CARREGAR MICROS (Iterar sobre todos os .gml na pasta)
    if os.path.exists(MICRO_FOLDER):
        gml_files = glob.glob(os.path.join(MICRO_FOLDER, "*.gml"))
        print(f"Encontrados {len(gml_files)} ficheiros Micro GML.")

        for fpath in sorted(gml_files):
            try:
                # O nome do grafo fica o nome do ficheiro (ex: graph_30_0.125.gml)
                micro = GraphLoader.load_gml(fpath)
                micro.get_stats()
            except Exception as e:
                print(f"Erro a ler {os.path.basename(fpath)}: {e}")
    else:
        print(f"[AVISO] Pasta não encontrada: {MICRO_FOLDER}")

    # 2. FOOTBALL (GML)
    try:
        path = os.path.join("/home/eduardo/Documents/computacional/aa/Project2/data/football/football.gml")
        if os.path.exists(path):
            fb = GraphLoader.load_gml(path, "Football")  # Usa o mesmo parser agora!
            fb.get_stats()
        else:
            print(f"[AVISO] Football não encontrado: {path}")
    except Exception as e:
        print(f"Erro no Football: {e}")

    # 3. FACEBOOK
    try:
        path = os.path.join("/home/eduardo/Documents/computacional/aa/Project2/data/facebook/3980.edges")
        if os.path.exists(path):
            face = GraphLoader.load_facebook_edges(path)
            face.get_stats()
        else:
            print(f"[AVISO] Facebook não encontrado: {path}")
    except Exception as e:
        print(f"Erro no Facebook: {e}")

    # 4. TWITCH
    # (Comentado para não encher o terminal, retirar se quiseres testar já)
    # try:
    #     path = os.path.join(BASE_PATH, "large_twitch_edges.csv")
    #     if os.path.exists(path):
    #         twitch = GraphLoader.load_twitch_csv(path)
    #         twitch.get_stats()
    # except Exception as e:
    #     print(f"Erro no Twitch: {e}")