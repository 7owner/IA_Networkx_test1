# -*- coding: utf-8 -*-
"""
tp_dijkstra.py

TP1 - Robotique : Dijkstra (partie separee)
Extrait du sujet: tp_dijkstra_astar.pdf

Objectif
--------
- Fichier dedie a Dijkstra.
- Rester simple, lisible et testable.
- S'inspirer des exemples "IA plus court chemin 1" et "IA court chemin 2"
  deja presents dans 03_shortest_path_algorithms.py.

Rappel du TP (labyrinthe + reward)
----------------------------------
Le labyrinthe est represente par une grille bidimensionnelle H x W.
Chaque cellule correspond a une position potentielle du robot.
Une cellule peut etre franchissable ou bloquee par un obstacle.
Encodage simple:
- 0 : cellule libre (movable area)
- 1 : obstacle (mur / non franchissable)
La reward map est une matrice separee (meme taille):
- reward negative pour encourager les chemins courts (ex: -1 par pas)
- reward positive pour certains bonus (ex: +5)
- reward finale a l'arrivee (ex: +100)

Usage
-----
  py tp_dijkstra.py
  py tp_dijkstra.py --no-plots
"""

from __future__ import annotations

from collections import defaultdict
import argparse
from typing import Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

Node = Hashable


# ============================================================
# PARTIE 0 - STRUCTURES DE BASE (graphe simple + outils)
# ============================================================
class SimpleGraph:
    """
    Mini-graphe pondere, inspire de NetworkX.
    adj[u][v] = poids de l'arete u -> v
    """

    def __init__(self, *, directed: bool = False) -> None:
        self.directed = directed
        self.adj: Dict[Node, Dict[Node, float]] = defaultdict(dict)

    def add_edge(self, u: Node, v: Node, weight: float = 1.0) -> None:
        self.adj[u][v] = float(weight)
        self.adj.setdefault(v, {})
        if not self.directed:
            self.adj[v][u] = float(weight)

    def neighbors(self, u: Node) -> Iterable[Tuple[Node, float]]:
        return self.adj[u].items()

    def nodes(self) -> List[Node]:
        return list(self.adj.keys())

    def edges(self) -> Iterable[Tuple[Node, Node, float]]:
        for u, vs in self.adj.items():
            for v, w in vs.items():
                yield u, v, w


# ============================================================
# PARTIE 0b - GRILLE + REWARD (pour Dijkstra sur matrice)
# ============================================================
class RewardGrid:
    """
    Grille HxW avec obstacles et reward.
    grid[y][x] = 0 (libre) ou 1 (mur)
    reward[y][x] = recompense
    """

    def __init__(
        self,
        *,
        width: int,
        height: int,
        grid: List[List[int]],
        reward: List[List[float]],
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> None:
        self.width = width
        self.height = height
        self.grid = grid
        self.reward = reward
        self.start = start
        self.goal = goal

    def in_bounds(self, cell: Tuple[int, int]) -> bool:
        (x, y) = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def is_free(self, cell: Tuple[int, int]) -> bool:
        (x, y) = cell
        return self.grid[y][x] == 0

    def neighbors_4(self, cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        (x, y) = cell
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        out: List[Tuple[int, int]] = []
        for c in candidates:
            if self.in_bounds(c) and self.is_free(c):
                out.append(c)
        return out


def generate_reward_grid(
    *,
    width: int,
    height: int,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    obstacles: Sequence[Tuple[int, int]],
    step_reward: float = -1.0,
    goal_reward: float = 100.0,
    bonus_cells: Sequence[Tuple[int, int]] = (),
    bonus_value: float = 5.0,
) -> RewardGrid:
    grid = [[0 for _ in range(width)] for _ in range(height)]
    for (x, y) in obstacles:
        if (x, y) != start and (x, y) != goal:
            grid[y][x] = 1

    reward = [[step_reward for _ in range(width)] for _ in range(height)]
    for (x, y) in bonus_cells:
        reward[y][x] = bonus_value
    gx, gy = goal
    reward[gy][gx] = goal_reward

    return RewardGrid(
        width=width,
        height=height,
        grid=grid,
        reward=reward,
        start=start,
        goal=goal,
    )


def dijkstra_grid_with_reward_abs(
    rg: RewardGrid,
) -> Tuple[Optional[List[Tuple[int, int]]], Dict[Tuple[int, int], float]]:
    """
    Dijkstra sur grille avec cout = |reward| de la case d'arrivee.
    """
    dist: Dict[Tuple[int, int], float] = {
        (x, y): float("inf") for y in range(rg.height) for x in range(rg.width)
    }
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {
        (x, y): None for y in range(rg.height) for x in range(rg.width)
    }
    dist[rg.start] = 0.0

    Q = {(x, y) for y in range(rg.height) for x in range(rg.width) if rg.is_free((x, y))}
    while Q:
        u = min(Q, key=lambda x: dist.get(x, float("inf")))
        Q.remove(u)
        if u == rg.goal:
            return reconstruct_path(parent, rg.goal), dist
        for v in rg.neighbors_4(u):
            cost = abs(rg.reward[v[1]][v[0]])
            alt = dist[u] + cost
            if alt < dist[v]:
                dist[v] = alt
                parent[v] = u

    return None, dist


def reconstruct_path(parent: Dict[Node, Optional[Node]], target: Node) -> List[Node]:
    path: List[Node] = []
    cur: Optional[Node] = target
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()
    return path


# ============================================================
# PARTIE A - DIJKSTRA (sur graphe pondere, poids >= 0)
# ============================================================
def dijkstra_all_dist(
    graph: SimpleGraph, source: Node
) -> Tuple[Dict[Node, float], Dict[Node, Optional[Node]]]:
    """
    Dijkstra conforme au pseudo-code du TP (set Q + argmin).
    Retourne (dist, parent) pour tous les noeuds.
    """
    nodes = graph.nodes()
    dist: Dict[Node, float] = {n: float("inf") for n in nodes}
    parent: Dict[Node, Optional[Node]] = {n: None for n in nodes}
    dist[source] = 0.0

    Q = set(nodes)
    while Q:
        u = min(Q, key=lambda x: dist.get(x, float("inf")))
        Q.remove(u)

        for v, w in graph.neighbors(u):
            if w < 0:
                raise ValueError("Dijkstra ne supporte pas les poids negatifs.")
            alt = dist[u] + w
            if alt < dist.get(v, float("inf")):
                dist[v] = alt
                parent[v] = u

    return dist, parent


def dijkstra_snapshots(
    graph: SimpleGraph, source: Node
) -> List[Tuple[Node, Dict[Node, float]]]:
    """
    Version Dijkstra qui enregistre les distances apres chaque extraction du minimum.
    Retour: liste de (u_selectionne, copie_dist).
    """
    nodes = graph.nodes()
    dist: Dict[Node, float] = {n: float("inf") for n in nodes}
    dist[source] = 0.0

    Q = set(nodes)
    snapshots: List[Tuple[Node, Dict[Node, float]]] = []
    while Q:
        u = min(Q, key=lambda x: dist.get(x, float("inf")))
        Q.remove(u)

        for v, w in graph.neighbors(u):
            if w < 0:
                raise ValueError("Dijkstra ne supporte pas les poids negatifs.")
            alt = dist[u] + w
            if alt < dist.get(v, float("inf")):
                dist[v] = alt

        snapshots.append((u, dict(dist)))

    return snapshots


def dijkstra_shortest_path(
    graph: SimpleGraph, source: Node, target: Node
) -> Tuple[float, List[Node]]:
    """
    Dijkstra conforme au pseudo-code du TP (set Q + argmin).
    Retourne (distance, chemin).
    """
    dist, parent = dijkstra_all_dist(graph, source)
    if dist.get(target, float("inf")) == float("inf"):
        raise ValueError(f"Aucun chemin trouve de {source!r} vers {target!r}.")
    return dist[target], reconstruct_path(parent, target)


def build_whiteboard_graph_1() -> Tuple[SimpleGraph, Dict[Node, Tuple[float, float]], Node, Node]:
    """
    Graphe "Plus court chemin 1" (photo IA_plus_court_chemin1.jpg).
    """
    g = SimpleGraph(directed=False)
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


# ============================================================
# PARTIE B - VISUALISATION (optionnelle)
# ============================================================
def plot_graph(
    graph: SimpleGraph,
    pos: Dict[Node, Tuple[float, float]],
    *,
    title: str,
    highlight_path: Optional[Sequence[Node]] = None,
    start: Optional[Node] = None,
    goal: Optional[Node] = None,
    node_values: Optional[Dict[Node, float]] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "matplotlib n'est pas installe. Installe-le avec: pip install matplotlib"
        ) from e

    def edge_key(u: Node, v: Node) -> object:
        return (u, v) if graph.directed else frozenset({u, v})

    highlighted_edges: set[object] = set()
    if highlight_path and len(highlight_path) >= 2:
        for a, b in zip(highlight_path, highlight_path[1:]):
            highlighted_edges.add(edge_key(a, b))

    plt.figure(figsize=(9, 5))
    plt.title(title)
    plt.axis("off")

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
        plt.plot([x1, x2], [y1, y2], color=color, linewidth=lw, zorder=1)
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

    xs = [pos[n][0] for n in graph.nodes()]
    ys = [pos[n][1] for n in graph.nodes()]
    plt.scatter(xs, ys, s=900, c="#4C78A8", edgecolors="black", linewidths=1.5, zorder=4)
    for n in graph.nodes():
        x, y = pos[n]
        if node_values is None:
            label = str(n)
        else:
            val = node_values.get(n, float("inf"))
            label = f"{n}\n{val:g}" if val != float("inf") else f"{n}\ninf"
        if start is not None and n == start:
            label = f"{n}\nSTART"
        if goal is not None and n == goal:
            label = f"{n}\nGOAL"
        plt.text(x, y, label, ha="center", va="center", color="white", fontsize=11, zorder=5)

    plt.tight_layout()
    plt.show()


def build_grid_graph_for_dijkstra(
    *,
    width: int,
    height: int,
    obstacles: Sequence[Tuple[int, int]],
) -> Tuple[SimpleGraph, Dict[Node, Tuple[float, float]]]:
    g = SimpleGraph(directed=False)
    blocked = set(obstacles)
    nodes = [(x, y) for y in range(height) for x in range(width) if (x, y) not in blocked]

    def in_bounds(cell: Tuple[int, int]) -> bool:
        (x, y) = cell
        return 0 <= x < width and 0 <= y < height

    for (x, y) in nodes:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (x + dx, y + dy)
            if not in_bounds(nxt) or nxt in blocked:
                continue
            a, b = (x, y), nxt
            if a < b:
                g.add_edge(a, b, 1.0)

    pos = {(x, y): (float(x), float(-y)) for (x, y) in nodes}
    return g, pos


def plot_grid_costs(
    *,
    width: int,
    height: int,
    obstacles: Sequence[Tuple[int, int]],
    dist: Dict[Node, float],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "matplotlib n'est pas installe. Installe-le avec: pip install matplotlib"
        ) from e

    plt.figure(figsize=(6, 6))
    plt.title(title)
    plt.axis("off")

    blocked = set(obstacles)
    for y in range(height):
        for x in range(width):
            is_wall = (x, y) in blocked
            color = "#666666" if is_wall else "#FFFFFF"
            rect = plt.Rectangle((x, height - 1 - y), 1, 1, facecolor=color, edgecolor="#BBBBBB")
            plt.gca().add_patch(rect)
            if is_wall:
                txt = "X"
            else:
                val = dist.get((x, y), float("inf"))
                txt = "inf" if val == float("inf") else f"{val:g}"
            plt.text(x + 0.5, height - 1 - y + 0.5, txt, ha="center", va="center", fontsize=9)

    sx, sy = start
    gx, gy = goal
    plt.scatter([sx + 0.5], [height - 1 - sy + 0.5], c="green", s=150, zorder=3)
    plt.scatter([gx + 0.5], [height - 1 - gy + 0.5], c="red", s=150, zorder=3)
    plt.text(sx + 0.5, height - 1 - sy + 0.5, "S", ha="center", va="center", color="white")
    plt.text(gx + 0.5, height - 1 - gy + 0.5, "G", ha="center", va="center", color="white")

    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.show()


def plot_grid_path_with_cumulative(
    *,
    width: int,
    height: int,
    obstacles: Sequence[Tuple[int, int]],
    path: List[Tuple[int, int]],
    dist: Dict[Node, float],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "matplotlib n'est pas installe. Installe-le avec: pip install matplotlib"
        ) from e

    plt.figure(figsize=(6, 6))
    plt.title(title)
    plt.axis("off")

    blocked = set(obstacles)
    for y in range(height):
        for x in range(width):
            is_wall = (x, y) in blocked
            color = "#666666" if is_wall else "#FFFFFF"
            rect = plt.Rectangle((x, height - 1 - y), 1, 1, facecolor=color, edgecolor="#BBBBBB")
            plt.gca().add_patch(rect)
            if is_wall:
                plt.text(x + 0.5, height - 1 - y + 0.5, "X", ha="center", va="center", fontsize=9)

    if path:
        xs = [p[0] + 0.5 for p in path]
        ys = [height - 1 - p[1] + 0.5 for p in path]
        plt.plot(xs, ys, color="crimson", linewidth=3, zorder=4)
        plt.scatter(xs, ys, c="crimson", s=30, zorder=4)
        for p in path:
            val = dist.get(p, float("inf"))
            txt = "inf" if val == float("inf") else f"{val:g}"
            plt.text(p[0] + 0.5, height - 1 - p[1] + 0.5, txt, ha="center", va="center", fontsize=9, color="black")

    sx, sy = start
    gx, gy = goal
    plt.scatter([sx + 0.5], [height - 1 - sy + 0.5], c="green", s=150, zorder=3)
    plt.scatter([gx + 0.5], [height - 1 - gy + 0.5], c="red", s=150, zorder=3)
    plt.text(sx + 0.5, height - 1 - sy + 0.5, "S", ha="center", va="center", color="white")
    plt.text(gx + 0.5, height - 1 - gy + 0.5, "G", ha="center", va="center", color="white")

    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.show()

def plot_grid_values(
    *,
    rg: RewardGrid,
    values: Dict[Tuple[int, int], float],
    title: str,
    path: Optional[List[Tuple[int, int]]] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "matplotlib n'est pas installe. Installe-le avec: pip install matplotlib"
        ) from e

    h, w = rg.height, rg.width
    plt.figure(figsize=(6, 6))
    plt.title(title)
    plt.axis("off")

    for y in range(h):
        for x in range(w):
            is_wall = rg.grid[y][x] == 1
            color = "#666666" if is_wall else "#FFFFFF"
            rect = plt.Rectangle((x, h - 1 - y), 1, 1, facecolor=color, edgecolor="#BBBBBB")
            plt.gca().add_patch(rect)
            if not is_wall:
                val = values.get((x, y), float("inf"))
                txt = "inf" if val == float("inf") else f"{val:g}"
                plt.text(x + 0.5, h - 1 - y + 0.5, txt, ha="center", va="center", fontsize=8)

    if path and len(path) >= 2:
        xs = [p[0] + 0.5 for p in path]
        ys = [h - 1 - p[1] + 0.5 for p in path]
        plt.plot(xs, ys, color="crimson", linewidth=3, zorder=4)
        plt.scatter(xs, ys, c="crimson", s=25, zorder=4)

    sx, sy = rg.start
    gx, gy = rg.goal
    plt.scatter([sx + 0.5], [h - 1 - sy + 0.5], c="green", s=150, zorder=3)
    plt.scatter([gx + 0.5], [h - 1 - gy + 0.5], c="red", s=150, zorder=3)
    plt.text(sx + 0.5, h - 1 - sy + 0.5, "S", ha="center", va="center", color="white")
    plt.text(gx + 0.5, h - 1 - gy + 0.5, "G", ha="center", va="center", color="white")

    plt.xlim(0, w)
    plt.ylim(0, h)
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.show()


# ============================================================
# PARTIE C - TESTS ET DEMOS (conformes au TP)
# ============================================================
def demo_dijkstra() -> None:
    print("=== TP DIJKSTRA : Exemple 'Plus court chemin 1' ===")
    g, pos, start, goal = build_whiteboard_graph_1()
    dist, path = dijkstra_shortest_path(g, start, goal)
    dist_all, _parent = dijkstra_all_dist(g, start)
    print(f"Dijkstra: distance {start} -> {goal} = {dist}")
    print(f"Dijkstra: chemin   {start} -> {goal} = {path}")
    plot_graph(
        g,
        pos,
        title="Dijkstra - Cout dist sur chaque noeud",
        highlight_path=None,
        node_values=dist_all,
        start=start,
        goal=goal,
    )
    snapshots = dijkstra_snapshots(g, start)
    for i, (u_sel, dist_snap) in enumerate(snapshots, start=1):
        plot_graph(
            g,
            pos,
            title=f"Dijkstra - Etape {i} (u={u_sel})",
            highlight_path=None,
            node_values=dist_snap,
            start=start,
            goal=goal,
        )

    # ============================================================
    # PARTIE GRILLE (3x3 zeros) - plots jusqu'au chemin final
    # ============================================================
    grid_w, grid_h = 3, 3
    grid_obstacles: List[Tuple[int, int]] = []
    grid_g, _grid_pos = build_grid_graph_for_dijkstra(
        width=grid_w, height=grid_h, obstacles=grid_obstacles
    )
    grid_start, grid_goal = (0, 0), (2, 2)
    grid_dist, grid_parent = dijkstra_all_dist(grid_g, grid_start)
    grid_path = reconstruct_path(grid_parent, grid_goal)

    plot_grid_costs(
        width=grid_w,
        height=grid_h,
        obstacles=grid_obstacles,
        dist=grid_dist,
        start=grid_start,
        goal=grid_goal,
        title="Grille 3x3 (zeros) - couts cumulés",
    )
    plot_grid_path_with_cumulative(
        width=grid_w,
        height=grid_h,
        obstacles=grid_obstacles,
        path=grid_path,
        dist=grid_dist,
        start=grid_start,
        goal=grid_goal,
        title="Grille 3x3 - chemin final",
    )

    # ============================================================
    # REWARDS POSITIFS - montrer l'impact sur le chemin final
    # ============================================================
    # On transforme la reward en cout: cost = max(0.1, 1 - reward)
    # Ainsi un bonus (reward positive) reduit le cout.
    rewards: Dict[Tuple[int, int], float] = {
        (1, 0): 0.4,
        (1, 1): 0.6,
        (1, 2): 0.4,
    }

    def reward_cost(cell: Tuple[int, int]) -> float:
        r = rewards.get(cell, 0.0)
        return max(0.1, 1.0 - r)

    reward_graph = SimpleGraph(directed=False)
    for y in range(grid_h):
        for x in range(grid_w):
            a = (x, y)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                b = (x + dx, y + dy)
                if 0 <= b[0] < grid_w and 0 <= b[1] < grid_h:
                    if a < b:
                        reward_graph.add_edge(a, b, reward_cost(b))

    r_dist, r_parent = dijkstra_all_dist(reward_graph, grid_start)
    r_path = reconstruct_path(r_parent, grid_goal)

    plot_grid_costs(
        width=grid_w,
        height=grid_h,
        obstacles=grid_obstacles,
        dist=r_dist,
        start=grid_start,
        goal=grid_goal,
        title="Grille 3x3 - couts avec rewards positives",
    )
    plot_grid_path_with_cumulative(
        width=grid_w,
        height=grid_h,
        obstacles=grid_obstacles,
        path=r_path,
        dist=r_dist,
        start=grid_start,
        goal=grid_goal,
        title="Grille 3x3 - chemin final (avec rewards)",
    )

    # Dijkstra sur grille + reward (cout = |reward|)
    rg = generate_reward_grid(
        width=4,
        height=4,
        start=(0, 0),
        goal=(3, 3),
        obstacles=[],
        step_reward=-1.0,
        goal_reward=100.0,
        bonus_cells=[(1, 2), (2, 1)],
        bonus_value=5.0,
    )
    path_r, dist_r = dijkstra_grid_with_reward_abs(rg)
    plot_grid_values(
        rg=rg,
        values=dist_r,
        title="Dijkstra grille - couts cumulés (|reward|)",
        path=path_r,
    )
    plot_graph(
        g,
        pos,
        title=f"Plus court chemin 1 - Dijkstra (chemin final, cout={dist:g})",
        highlight_path=path,
        start=start,
        goal=goal,
    )



def negative_weight_tests() -> None:
    """
    Demande du sujet: tester des poids negatifs et commenter les resultats.
    Ici Dijkstra doit refuser (exception), et A* est base sur couts positifs
    (donc pareil: comportement non garanti si poids negatifs).
    """
    print("\n=== TEST POIDS NEGATIFS (Dijkstra) ===")
    g = SimpleGraph(directed=True)
    g.add_edge("A", "B", 1)
    g.add_edge("B", "C", -2)
    g.add_edge("A", "C", 4)
    try:
        dijkstra_shortest_path(g, "A", "C")
    except ValueError as e:
        print("Dijkstra refuse les poids negatifs:", str(e))


def tp_questions() -> None:
    """
    Reponses concises aux questions 3-7 du TP (cote Dijkstra).
    """
    print("\n=== REPONSES TP (Dijkstra / A*) ===")
    print("Q3 Comparaison: Dijkstra explore tout selon dist. A* guide par heuristique.")
    print("Q4 Poids negatifs: Dijkstra incorrect -> on refuse (exception).")
    print("Q5 Differences & complexite: Dijkstra O(V^2) ici (argmin sur Q).")
    print("Q6 A* = Dijkstra si h(n)=0 pour tous n (ou heuristique non-informative).")
    print("Q7 h parfaite: A* explore seulement les noeuds sur le chemin optimal.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TP Dijkstra (version separee).")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Desactive l'affichage matplotlib.",
    )
    args = parser.parse_args()

    if args.no_plots:
        globals()["plot_graph"] = lambda *_a, **_k: None  # type: ignore

    demo_dijkstra()
    negative_weight_tests()
    tp_questions()
