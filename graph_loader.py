import random
import csv
import pickle
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

    def save_to_bin(self, folder_path: str):
        """ Saves the current Graph object state to a binary file. """
        filename = f"{self.name}.bin"
        full_path = os.path.join(folder_path, filename)

        with open(full_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"[IO] Graph saved to: {full_path}")


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

    @staticmethod
    def load_from_bin(filepath: str) -> Graph:
        """ Loads a pre-processed Graph object from a binary file. """
        with open(filepath, 'rb') as f:
            g = pickle.load(f)
        print(f"[IO] Loaded {g.name} (n={g.n}, m={g.m}) from disk.")
        return g


# ==========================================
# Phase 1.5: Compiler (Main) - FINAL PATH FIX
# ==========================================
if __name__ == "__main__":

    # 1. SETUP DE CAMINHOS ABSOLUTOS/RELATIVOS
    # Estamos em .../Project2/src
    # Queremos ir para .../Project2/data

    BASE_DATA = "../data"
    BASE_RAW = os.path.join(BASE_DATA, "raw")

    # Pastas Específicas DENTRO de raw
    PATH_MICRO = os.path.join(BASE_RAW, "local_graphs")
    PATH_FACEBOOK = os.path.join(BASE_RAW, "facebook")
    PATH_FOOTBALL = os.path.join(BASE_RAW, "football")

    # Pasta de Destino (ao lado de raw, para não misturar)
    PATH_PROCESSED = os.path.join(BASE_DATA, "processed")

    print(f"DEBUG: Working Directory: {os.getcwd()}")
    print(f"DEBUG: Raw Data Root: {os.path.abspath(BASE_RAW)}")

    # Criar pasta processed
    os.makedirs(PATH_PROCESSED, exist_ok=True)

    print("\n>>> COMPILING GRAFOS (RAW -> PROCESSED) <<<\n")

    # ---------------------------------------------------------
    # 1. COMPILE MICRO (Folder: data/raw/local_graphs)
    # ---------------------------------------------------------
    if os.path.exists(PATH_MICRO):
        gml_files = glob.glob(os.path.join(PATH_MICRO, "*.gml"))

        if not gml_files:
            print(f"[WARNING] Pasta '{PATH_MICRO}' existe mas está vazia de .gml!")

        for fpath in sorted(gml_files):
            try:
                micro = GraphLoader.load_gml(fpath)
                micro.save_to_bin(PATH_PROCESSED)
            except Exception as e:
                print(f"[ERROR] {os.path.basename(fpath)}: {e}")
    else:
        print(f"[ERROR] Pasta não encontrada: {os.path.abspath(PATH_MICRO)}")

    # ---------------------------------------------------------
    # 2. COMPILE FOOTBALL (Folder: data/raw/football)
    # ---------------------------------------------------------
    if os.path.exists(PATH_FOOTBALL):
        football_files = glob.glob(os.path.join(PATH_FOOTBALL, "*.gml"))
        if football_files:
            # Assume que o primeiro gml é o correto
            fpath = football_files[0]
            try:
                # Extra: Tenta usar o nome do ficheiro como nome do grafo
                name = os.path.splitext(os.path.basename(fpath))[0]
                fb = GraphLoader.load_gml(fpath, name)
                fb.save_to_bin(PATH_PROCESSED)
            except Exception as e:
                print(f"[ERROR] Football logic: {e}")
        else:
            print(f"[ERROR] Nenhum .gml encontrado dentro de {PATH_FOOTBALL}")
    else:
        print(f"[ERROR] Pasta não encontrada: {os.path.abspath(PATH_FOOTBALL)}")

    # ---------------------------------------------------------
    # 3. COMPILE FACEBOOK (Folder: data/raw/facebook)
    # ---------------------------------------------------------
    target_fb = os.path.join(PATH_FACEBOOK, "facebook_combined.txt")

    if os.path.exists(target_fb):
        try:
            face = GraphLoader.load_facebook_edges(target_fb)
            face.save_to_bin(PATH_PROCESSED)
        except Exception as e:
            print(f"[ERROR] Facebook logic: {e}")
    else:
        print(f"[ERROR] Ficheiro 'facebook_combined.txt' não encontrado em: {os.path.abspath(PATH_FACEBOOK)}")

    # 4. TWITCH
    # (Comentado para não encher o terminal, retirar se quiseres testar já)
    # try:
    #     path = os.path.join("/home/eduardo/Documents/computacional/Project2/data/twitch_gamers/large_twitch_edges.csv")
    #     if os.path.exists(path):
    #         twitch = GraphLoader.load_twitch_csv(path)
    #         twitch.get_stats()
    # except Exception as e:
    #     print(f"Erro no Twitch: {e}")