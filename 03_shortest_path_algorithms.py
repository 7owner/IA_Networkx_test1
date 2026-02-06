# -*- coding: utf-8 -*-
"""
03_shortest_path_algorithms.py

Objectif
--------
Donner de bonnes bases sur le problème du "plus court chemin" (shortest path)
en s'inspirant de NetworkX, mais en implémentant nous-mêmes les algorithmes.

Pourquoi "s'inspirer de NetworkX" ?
-----------------------------------
NetworkX est une bibliothèque Python très populaire pour manipuler des graphes.
Elle expose des fonctions comme :
  - nx.shortest_path(...)           (plus court chemin, selon le cas)
  - nx.single_source_dijkstra(...)
  - nx.bellman_ford_path(...)
Ici on recrée le coeur des idées, avec un mini-graphe + des fonctions lisibles.

Ce fichier est autonome (standard library seulement).
Tu peux l'exécuter :
  python 03_shortest_path_algorithms.py
"""

from __future__ import annotations

from collections import defaultdict, deque
import argparse
import heapq
import math
from typing import Callable, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

Node = Hashable


class SimpleGraph:
    """
    Mini-structure de graphe (inspirée de l'idée "adjacency" de NetworkX).

    Représentation :
      adj[u][v] = weight (poids de l'arête u -> v)

    - Si directed=False, add_edge(u, v) ajoute aussi v -> u.
    - Les poids sont des nombres (int/float). Par défaut weight=1.
    """

    def __init__(self, *, directed: bool = False) -> None:
        self.directed = directed
        self.adj: Dict[Node, Dict[Node, float]] = defaultdict(dict)

    def add_edge(self, u: Node, v: Node, weight: float = 1.0) -> None:
        self.adj[u][v] = float(weight)
        self.adj.setdefault(v, {})  # garantit que v existe dans la liste des noeuds
        if not self.directed:
            self.adj[v][u] = float(weight)

    def neighbors(self, u: Node) -> Iterable[Tuple[Node, float]]:
        """Retourne des couples (voisin, poids)."""
        return self.adj[u].items()

    def nodes(self) -> List[Node]:
        return list(self.adj.keys())

    def edges(self) -> Iterable[Tuple[Node, Node, float]]:
        """Itère sur les arêtes (u, v, w). Pour un graphe non orienté, il y aura des doublons."""
        for u, vs in self.adj.items():
            for v, w in vs.items():
                yield u, v, w


def reconstruct_path(parent: Dict[Node, Optional[Node]], target: Node) -> List[Node]:
    """
    Reconstruit un chemin à partir d'un dictionnaire parent[].

    parent[x] = le noeud précédent sur le chemin vers x.
    """
    path: List[Node] = []
    cur: Optional[Node] = target
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()
    return path


def bfs_shortest_path(graph: SimpleGraph, source: Node, target: Node) -> List[Node]:
    """
    Plus court chemin dans un graphe NON PONDÉRÉ (ou pondéré avec tous les poids = 1).

    Algorithme : BFS (Breadth-First Search / parcours en largeur)
    - Complexité : O(|V| + |E|)
    - Garantit le plus petit nombre d'arêtes.
    """
    if source == target:
        return [source]

    queue: deque[Node] = deque([source])
    parent: Dict[Node, Optional[Node]] = {source: None}

    while queue:
        u = queue.popleft()
        for v, _w in graph.neighbors(u):
            if v in parent:
                continue
            parent[v] = u
            if v == target:
                return reconstruct_path(parent, target)
            queue.append(v)

    raise ValueError(f"Aucun chemin trouvé de {source!r} vers {target!r}.")


def dijkstra_shortest_path(
    graph: SimpleGraph, source: Node, target: Node
) -> Tuple[float, List[Node]]:
    """
    Plus court chemin dans un graphe pondéré avec poids NON NÉGATIFS.

    Algorithme : Dijkstra (avec une file de priorité / heap)
    - Complexité : ~ O((|V| + |E|) log |V|)
    - Attention : si un poids négatif existe, Dijkstra peut donner un mauvais résultat.
    """
    dist: Dict[Node, float] = {source: 0.0}
    parent: Dict[Node, Optional[Node]] = {source: None}

    # heap contient des tuples (distance_actuelle, noeud)
    heap: List[Tuple[float, Node]] = [(0.0, source)]

    while heap:
        d_u, u = heapq.heappop(heap)

        # Si on a déjà trouvé mieux pour u, on ignore cette entrée "obsolète"
        if d_u != dist.get(u, float("inf")):
            continue

        # Optimisation : on peut s'arrêter dès qu'on pop le target
        if u == target:
            return d_u, reconstruct_path(parent, target)

        for v, w in graph.neighbors(u):
            if w < 0:
                raise ValueError("Dijkstra ne supporte pas les poids négatifs (w < 0).")

            cand = d_u + w
            if cand < dist.get(v, float("inf")):
                dist[v] = cand
                parent[v] = u
                heapq.heappush(heap, (cand, v))

    raise ValueError(f"Aucun chemin trouvé de {source!r} vers {target!r}.")


def bellman_ford_shortest_path(
    graph: SimpleGraph, source: Node, target: Node
) -> Tuple[float, List[Node]]:
    """
    Plus court chemin dans un graphe pondéré AVEC poids négatifs possibles.

    Algorithme : Bellman-Ford
    - Complexité : O(|V| * |E|)
    - Avantage : gère les poids négatifs
    - Inconvénient : plus lent que Dijkstra
    - Bonus : détecte les cycles de poids négatif (dans ce cas le "plus court chemin"
      n'existe pas vraiment, car on peut diminuer le coût à l'infini).
    """
    nodes = graph.nodes()
    dist: Dict[Node, float] = {n: float("inf") for n in nodes}
    parent: Dict[Node, Optional[Node]] = {n: None for n in nodes}
    dist[source] = 0.0

    edges = list(graph.edges())

    # Relaxation des arêtes |V|-1 fois
    for _ in range(max(0, len(nodes) - 1)):
        changed = False
        for u, v, w in edges:
            if dist[u] == float("inf"):
                continue
            cand = dist[u] + w
            if cand < dist[v]:
                dist[v] = cand
                parent[v] = u
                changed = True
        # Petite optimisation : si aucune amélioration, on peut arrêter plus tôt.
        if not changed:
            break

    # Détection de cycle négatif : si on peut encore améliorer après |V|-1 relaxations.
    for u, v, w in edges:
        if dist[u] == float("inf"):
            continue
        if dist[u] + w < dist[v]:
            raise ValueError("Cycle de poids négatif détecté : pas de plus court chemin défini.")

    if dist[target] == float("inf"):
        raise ValueError(f"Aucun chemin trouvé de {source!r} vers {target!r}.")

    return dist[target], reconstruct_path(parent, target)


def a_star_shortest_path(
    graph: SimpleGraph,
    source: Node,
    target: Node,
    heuristic: Callable[[Node, Node], float],
) -> Tuple[float, List[Node]]:
    """
    A* (A star) : Dijkstra + heuristique.

    Idée :
      - On garde g(n) = coût réel depuis source vers n
      - On utilise f(n) = g(n) + h(n) pour choisir quel noeud explorer en premier

    Si h(n) ne surestime jamais la distance restante (heuristique "admissible"),
    A* trouve un plus court chemin (comme Dijkstra) mais souvent plus vite.
    """
    g_score: Dict[Node, float] = {source: 0.0}
    parent: Dict[Node, Optional[Node]] = {source: None}

    # heap contient (f_score, noeud)
    heap: List[Tuple[float, Node]] = [(heuristic(source, target), source)]

    while heap:
        _f_u, u = heapq.heappop(heap)

        if u == target:
            return g_score[u], reconstruct_path(parent, target)

        for v, w in graph.neighbors(u):
            if w < 0:
                raise ValueError("A* (comme Dijkstra) ne supporte pas les poids négatifs (w < 0).")
            cand_g = g_score[u] + w
            if cand_g < g_score.get(v, float("inf")):
                g_score[v] = cand_g
                parent[v] = u
                f_v = cand_g + heuristic(v, target)
                heapq.heappush(heap, (f_v, v))

    raise ValueError(f"Aucun chemin trouvé de {source!r} vers {target!r}.")


def plot_graph(
    graph: SimpleGraph,
    pos: Dict[Node, Tuple[float, float]],
    *,
    title: str,
    highlight_path: Optional[Sequence[Node]] = None,
    start: Optional[Node] = None,
    goal: Optional[Node] = None,
    show_node_labels: bool = True,
) -> None:
    """
    Trace le graphe avec matplotlib (sans NetworkX).

    - pos: dictionnaire {noeud: (x, y)} pour placer les noeuds "comme sur le dessin".
    - highlight_path: liste de noeuds (chemin) à surligner en rouge.

    Remarque :
    - Si matplotlib n'est pas installé, on lève une erreur claire.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "matplotlib n'est pas installé. Installe-le avec: pip install matplotlib"
        ) from e

    def edge_key(u: Node, v: Node) -> object:
        # Pour un graphe non orienté, évite de tracer 2 fois la même arête.
        # (frozenset marche car u/v sont hashable)
        return (u, v) if graph.directed else frozenset({u, v})

    highlighted_edges: set[object] = set()
    if highlight_path and len(highlight_path) >= 2:
        for a, b in zip(highlight_path, highlight_path[1:]):
            highlighted_edges.add(edge_key(a, b))

    plt.figure(figsize=(9, 5))
    plt.title(title)
    plt.axis("off")

    # 1) Tracer les arêtes
    seen: set[object] = set()
    for u, v, w in graph.edges():
        k = edge_key(u, v)
        if k in seen:
            continue
        seen.add(k)

        (x1, y1) = pos[u]
        (x2, y2) = pos[v]

        is_hl = k in highlighted_edges
        color = "crimson" if is_hl else "#999999"
        lw = 3.0 if is_hl else 1.5

        if graph.directed:
            plt.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=12, shrinkB=12),
            )
        else:
            plt.plot([x1, x2], [y1, y2], color=color, linewidth=lw, zorder=1)

        # Label du poids au milieu de l'arête
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        plt.text(
            mx,
            my,
            f"{w:g}",
            fontsize=10,
            ha="center",
            va="center",
            color="black",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
            zorder=3,
        )

    # 2) Tracer les noeuds
    xs = [pos[n][0] for n in graph.nodes()]
    ys = [pos[n][1] for n in graph.nodes()]

    plt.scatter(xs, ys, s=900, c="#4C78A8", edgecolors="black", linewidths=1.5, zorder=4)

    # 3) Labels des noeuds (+ start/goal)
    if show_node_labels:
        for n in graph.nodes():
            x, y = pos[n]
            label = str(n)
            if start is not None and n == start:
                label = f"{n}\nSTART"
            if goal is not None and n == goal:
                label = f"{n}\nGOAL"
            plt.text(x, y, label, ha="center", va="center", color="white", fontsize=11, zorder=5)
    else:
        # Mode "léger" : on n'affiche que START et GOAL.
        if start is not None:
            x, y = pos[start]
            plt.text(x, y, "START", ha="center", va="center", color="white", fontsize=12, zorder=5)
        if goal is not None:
            x, y = pos[goal]
            plt.text(x, y, "GOAL", ha="center", va="center", color="white", fontsize=12, zorder=5)

    plt.tight_layout()
    plt.show()


def build_whiteboard_graph_1() -> Tuple[SimpleGraph, Dict[Node, Tuple[float, float]], Node, Node]:
    """
    Graphe "Plus court chemin 1" (repris de ta photo IA_plus_court_chemin1.jpg).

    Important : les poids sur la photo ne sont pas parfaitement lisibles.
    Les valeurs ci-dessous sont une version "pratique" que tu peux ajuster
    facilement si tu veux coller exactement au tableau.
    """
    g = SimpleGraph(directed=False)

    # Noeuds : 0 (start) -> ... -> 9 (goal)
    # Arêtes + poids (approx)
    g.add_edge(0, 3, 2.1)
    g.add_edge(0, 2, 2.0)
    g.add_edge(0, 1, 2.5)

    g.add_edge(1, 4, 1.0)
    g.add_edge(4, 2, 0.6)

    g.add_edge(3, 5, 2.5)
    g.add_edge(2, 5, 1.5)
    g.add_edge(4, 6, 2.3)
    g.add_edge(5, 6, 1.9)

    g.add_edge(5, 7, 2.0)
    g.add_edge(6, 7, 1.8)

    g.add_edge(6, 9, 1.7)
    g.add_edge(7, 9, 2.0)

    # Positions (x, y) pour dessiner un schéma proche de celui du tableau
    pos = {
        0: (0.0, 0.7),
        1: (0.7, 0.0),
        4: (2.0, -0.2),
        2: (2.0, 0.4),
        3: (1.6, 1.0),
        5: (3.6, 0.9),
        6: (4.2, 0.2),
        7: (5.5, 0.7),
        9: (5.5, -0.1),
    }

    start, goal = 0, 9
    return g, pos, start, goal


def build_whiteboard_graph_2() -> Tuple[SimpleGraph, Dict[Node, Tuple[float, float]], Node, Node]:
    """
    Graphe "Plus court chemin 2" (repris de ta photo IA_court_chemin_2.jpg).

    La photo montre une grille 4x4 (coordonnées 0..3 en x et 0..3 en y) avec des "murs"
    (certaines connexions sont interdites). On modélise ça comme un graphe :
      - noeud = (x, y)
      - arête = mouvement haut/bas/gauche/droite (coût = 1)

    On ajoute aussi une arête diagonale (comme sur le tableau) avec coût 4
    pour illustrer qu'on peut avoir des coûts différents.
    """
    g = SimpleGraph(directed=False)

    width, height = 4, 4
    nodes = [(x, y) for y in range(height) for x in range(width)]

    # "Murs" (arêtes bloquées) — version pratique inspirée du dessin.
    # Chaque mur est un couple de noeuds (non orienté).
    blocked: set[frozenset[Tuple[int, int]]] = set()

    def block(a: Tuple[int, int], b: Tuple[int, int]) -> None:
        blocked.add(frozenset({a, b}))

    # Mur horizontal en haut du "rectangle de droite" (empêche de passer tout droit depuis (1,1))
    block((1, 1), (2, 1))
    block((2, 1), (3, 1))

    # Mur vertical au milieu (oblige souvent à descendre avant de traverser)
    block((2, 1), (2, 2))
    block((2, 2), (2, 3))

    def is_blocked(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        return frozenset({a, b}) in blocked

    # Connexions 4-voisins (coût 1)
    for (x, y) in nodes:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            a, b = (x, y), (nx, ny)
            if is_blocked(a, b):
                continue
            # Pour éviter d'ajouter 2 fois dans un graphe non orienté, on ajoute seulement si a < b
            if a < b:
                g.add_edge(a, b, 1.0)

    # Diagonale "comme sur le tableau" (coût 4)
    g.add_edge((1, 1), (3, 3), 4.0)

    # Positions pour plot : on inverse y pour que 0 soit en haut (comme sur le tableau)
    pos: Dict[Node, Tuple[float, float]] = {(x, y): (float(x), float(-y)) for (x, y) in nodes}

    start, goal = (0, 0), (3, 3)
    return g, pos, start, goal


def _demo() -> None:
    # ------------------------------------------------------------
    # Mise en pratique : "Plus court chemin 1" (ta photo)
    # ------------------------------------------------------------
    print("=== PRATIQUE : Plus court chemin 1 (photo) ===")
    g_img1, pos_img1, start, goal = build_whiteboard_graph_1()

    dist_d, path_d = dijkstra_shortest_path(g_img1, start, goal)
    print(f"Dijkstra: distance {start} -> {goal} =", dist_d)
    print(f"Dijkstra: chemin   {start} -> {goal} =", path_d)

    def euclidean(u: Node, v: Node) -> float:
        (x1, y1) = pos_img1[u]
        (x2, y2) = pos_img1[v]
        return math.hypot(x1 - x2, y1 - y2)

    dist_a, path_a = a_star_shortest_path(g_img1, start, goal, heuristic=euclidean)
    print(f"A*:       distance {start} -> {goal} =", dist_a)
    print(f"A*:       chemin   {start} -> {goal} =", path_a)

    # Plots
    plot_graph(
        g_img1,
        pos_img1,
        title="Plus court chemin 1 — Graphe (poids sur les arêtes)",
        start=start,
        goal=goal,
    )
    plot_graph(
        g_img1,
        pos_img1,
        title=f"Plus court chemin 1 — Dijkstra (coût={dist_d:g})",
        highlight_path=path_d,
        start=start,
        goal=goal,
    )

    # ------------------------------------------------------------
    # Mise en pratique : "Plus court chemin 2" (ta photo, A*)
    # ------------------------------------------------------------
    print("\n=== PRATIQUE : Plus court chemin 2 (photo, A*) ===")
    g_img2, pos_img2, start2, goal2 = build_whiteboard_graph_2()

    def manhattan_grid(u: Node, v: Node) -> float:
        (x1, y1) = u  # type: ignore[misc]
        (x2, y2) = v  # type: ignore[misc]
        return float(abs(x1 - x2) + abs(y1 - y2))

    dist2, path2 = a_star_shortest_path(g_img2, start2, goal2, heuristic=manhattan_grid)
    print(f"A*: distance {start2} -> {goal2} =", dist2)
    print(f"A*: chemin   {start2} -> {goal2} =", path2)

    plot_graph(
        g_img2,
        pos_img2,
        title="Plus court chemin 2 — Grille (murs = arêtes absentes)",
        start=start2,
        goal=goal2,
        show_node_labels=False,
    )
    plot_graph(
        g_img2,
        pos_img2,
        title=f"Plus court chemin 2 — A* (coût={dist2:g})",
        highlight_path=path2,
        start=start2,
        goal=goal2,
        show_node_labels=False,
    )

    # ------------------------------------------------------------
    # Démos "cours" (BFS / Dijkstra / Bellman-Ford / A*)
    # ------------------------------------------------------------
    print("=== DEMO 1 : BFS (graphe non pondéré) ===")
    g = SimpleGraph()
    g.add_edge("A", "B")
    g.add_edge("A", "C")
    g.add_edge("B", "D")
    g.add_edge("C", "D")
    print("Chemin A -> D (BFS) =", bfs_shortest_path(g, "A", "D"))

    print("\n=== DEMO 2 : Dijkstra (poids positifs) ===")
    wg = SimpleGraph(directed=True)
    wg.add_edge("A", "B", 1)
    wg.add_edge("A", "C", 5)
    wg.add_edge("B", "C", 1)
    wg.add_edge("B", "D", 2)
    wg.add_edge("C", "D", 1)
    dist, path = dijkstra_shortest_path(wg, "A", "D")
    print("Distance A -> D (Dijkstra) =", dist)
    print("Chemin A -> D (Dijkstra)   =", path)

    print("\n=== DEMO 3 : Bellman-Ford (poids négatifs possibles) ===")
    ng = SimpleGraph(directed=True)
    ng.add_edge("A", "B", 1)
    ng.add_edge("B", "C", -2)
    ng.add_edge("A", "C", 4)
    ng.add_edge("C", "D", 2)
    dist, path = bellman_ford_shortest_path(ng, "A", "D")
    print("Distance A -> D (Bellman-Ford) =", dist)
    print("Chemin A -> D (Bellman-Ford)   =", path)

    print("\n=== DEMO 4 : A* (avec heuristique) ===")
    # Exemple simple : on imagine que les noeuds ont des coordonnées pour définir h().
    coords = {"A": (0, 0), "B": (1, 0), "C": (2, 0), "D": (3, 0)}

    def manhattan(u: Node, v: Node) -> float:
        (x1, y1) = coords[u]
        (x2, y2) = coords[v]
        return abs(x1 - x2) + abs(y1 - y2)

    dist, path = a_star_shortest_path(wg, "A", "D", heuristic=manhattan)
    print("Distance A -> D (A*) =", dist)
    print("Chemin A -> D (A*)   =", path)

    print("\n=== Optionnel : comparaison avec NetworkX (si installé) ===")
    try:
        import networkx as nx  # type: ignore

        nxg = nx.DiGraph()
        for u, v, w in wg.edges():
            nxg.add_edge(u, v, weight=w)
        nx_dist = nx.dijkstra_path_length(nxg, "A", "D", weight="weight")
        nx_path = nx.dijkstra_path(nxg, "A", "D", weight="weight")
        print("NetworkX distance A -> D =", nx_dist)
        print("NetworkX chemin   A -> D =", nx_path)
    except Exception as e:
        print("NetworkX non dispo ou erreur (ok) :", repr(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Algorithmes de plus court chemin (avec plots).")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="N'affiche pas de fenêtres matplotlib (utile sur serveur/headless).",
    )
    args = parser.parse_args()

    # Si l'utilisateur ne veut pas de plots, on "neutralise" plot_graph
    if args.no_plots:
        globals()["plot_graph"] = lambda *_a, **_k: None  # type: ignore

    _demo()
