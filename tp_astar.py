# -*- coding: utf-8 -*-
"""
tp_astar.py

TP1 - Robotique : A* (partie separee)
Extrait du sujet: tp_dijkstra_astar.pdf

Objectif
--------
- Un seul cas d'etude.
- Afficher la matrice 0/1 (chemins / murs).
- Montrer les couts: cout unitaire, cout de changement de direction,
  heuristique Manhattan, et cout total f = g + h.

Usage
-----
  py tp_astar.py
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Sequence, Tuple


# ============================================================
# PARTIE A - MAZE (grille + reward)
# ============================================================
class Maze:
    """
    Labyrinthe en grille HxW.

    - grid[y][x] = 0 (libre) ou 1 (obstacle)
    - reward[y][x] = recompense
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

    def solve_a_star(self, *, turn_penalty: float = 0.0) -> Tuple[Optional[List[Tuple[int, int]]], Dict[Tuple[int, int], float]]:
        """
        Resolution A* conforme au pseudo-code du TP (open set Q + argmin).
        Retourne (chemin optimal ou None, g_score).
        Le cout de deplacement est:
          - 1 par pas
          - + turn_penalty si changement de direction
        """
        s, t = self.start, self.goal
        g_score: Dict[Tuple[int, int], float] = {}
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
        direction: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}

        for y in range(self.height):
            for x in range(self.width):
                g_score[(x, y)] = float("inf")
                parent[(x, y)] = None
                direction[(x, y)] = None
        g_score[s] = 0.0

        def h(cell: Tuple[int, int]) -> float:
            (x, y) = cell
            (gx, gy) = t
            return float(abs(x - gx) + abs(y - gy))

        Q = {s}
        while Q:
            u = min(Q, key=lambda x: g_score[x] + h(x))
            if u == t:
                return _reconstruct_path(parent, t), g_score
            Q.remove(u)

            for v in self.neighbors_4(u):
                prev_dir = direction[u]
                step_dir = (v[0] - u[0], v[1] - u[1])
                turn_cost = 0.0
                if prev_dir is not None and step_dir != prev_dir:
                    turn_cost = turn_penalty
                alt = g_score[u] + 1.0 + turn_cost
                if alt < g_score[v]:
                    g_score[v] = alt
                    parent[v] = u
                    direction[v] = step_dir
                    if v not in Q:
                        Q.add(v)

        return None, g_score


def _reconstruct_path(
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
    target: Tuple[int, int],
) -> List[Tuple[int, int]]:
    path: List[Tuple[int, int]] = []
    cur: Optional[Tuple[int, int]] = target
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()
    return path


def generate_maze(
    *,
    width: int,
    height: int,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    obstacles: Sequence[Tuple[int, int]],
    step_penalty: float = -1.0,
    goal_reward: float = 100.0,
    bonus_cells: Sequence[Tuple[int, int]] = (),
    bonus_value: float = 5.0,
) -> Maze:
    """
    Genere une grille de labyrinthe et une reward map simple.
    """
    if not (0 <= start[0] < width and 0 <= start[1] < height):
        raise ValueError("Start hors de la grille.")
    if not (0 <= goal[0] < width and 0 <= goal[1] < height):
        raise ValueError("Goal hors de la grille.")

    grid = [[0 for _ in range(width)] for _ in range(height)]
    for (x, y) in obstacles:
        if (x, y) != start and (x, y) != goal:
            grid[y][x] = 1

    reward = [[step_penalty for _ in range(width)] for _ in range(height)]
    for (x, y) in bonus_cells:
        reward[y][x] = bonus_value
    gx, gy = goal
    reward[gy][gx] = goal_reward

    return Maze(
        width=width,
        height=height,
        grid=grid,
        reward=reward,
        start=start,
        goal=goal,
    )


def build_whiteboard_grid_2() -> Maze:
    """
    Grille "Plus court chemin 2" (photo IA_court_chemin_2.jpg).
    Meme idee que 03_shortest_path_algorithms.py.
    """
    width, height = 4, 4
    start, goal = (0, 0), (3, 3)
    obstacles: List[Tuple[int, int]] = []
    blocked_edges = {
        frozenset({(1, 1), (2, 1)}),
        frozenset({(2, 1), (3, 1)}),
        frozenset({(2, 1), (2, 2)}),
        frozenset({(2, 2), (2, 3)}),
    }

    maze = generate_maze(
        width=width,
        height=height,
        start=start,
        goal=goal,
        obstacles=obstacles,
        step_penalty=-1.0,
        goal_reward=100.0,
    )

    def neighbors_4_with_walls(cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        (x, y) = cell
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        out: List[Tuple[int, int]] = []
        for c in candidates:
            if not maze.in_bounds(c) or not maze.is_free(c):
                continue
            if frozenset({cell, c}) in blocked_edges:
                continue
            out.append(c)
        return out

    # Remplacement de la methode neighbors_4 par une version avec murs
    maze.neighbors_4 = neighbors_4_with_walls  # type: ignore[method-assign]
    return maze


# ============================================================
# PARTIE B - VISUALISATION (plots)
# ============================================================
def plot_maze(
    maze: Maze,
    *,
    title: str,
    path: Optional[List[Tuple[int, int]]] = None,
    values: Optional[Dict[Tuple[int, int], float]] = None,
    show_path: bool = True,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "matplotlib n'est pas installe. Installe-le avec: pip install matplotlib"
        ) from e

    h, w = maze.height, maze.width
    plt.figure(figsize=(6, 6))
    plt.title(title)
    plt.axis("off")

    # Fond: obstacles en gris, libres en blanc + valeurs (reward ou valeurs custom)
    for y in range(h):
        for x in range(w):
            is_wall = maze.grid[y][x] == 1
            color = "#666666" if is_wall else "#FFFFFF"
            rect = plt.Rectangle((x, h - 1 - y), 1, 1, facecolor=color, edgecolor="#BBBBBB")
            plt.gca().add_patch(rect)
            if not is_wall:
                val = maze.reward[y][x] if values is None else values[(x, y)]
                plt.text(
                    x + 0.5,
                    h - 1 - y + 0.5,
                    f"{val:g}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#222222",
                )
            else:
                plt.text(
                    x + 0.5,
                    h - 1 - y + 0.5,
                    "X",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="#FFFFFF",
                )

    # Start / Goal
    sx, sy = maze.start
    gx, gy = maze.goal
    plt.scatter([sx + 0.5], [h - 1 - sy + 0.5], c="green", s=150, zorder=3)
    plt.scatter([gx + 0.5], [h - 1 - gy + 0.5], c="red", s=150, zorder=3)
    plt.text(sx + 0.5, h - 1 - sy + 0.5, "S", ha="center", va="center", color="white")
    plt.text(gx + 0.5, h - 1 - gy + 0.5, "G", ha="center", va="center", color="white")

    # Chemin (optionnel)
    if show_path and path and len(path) >= 2:
        xs = [p[0] + 0.5 for p in path]
        ys = [h - 1 - p[1] + 0.5 for p in path]
        plt.plot(xs, ys, color="crimson", linewidth=3, zorder=4)
        plt.scatter(xs, ys, c="crimson", s=30, zorder=4)

    plt.xlim(0, w)
    plt.ylim(0, h)
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.show()


# ============================================================
# PARTIE C - TESTS ET DEMOS (conformes au TP)
# ============================================================
def print_matrix(grid: List[List[int]]) -> None:
    print("Matrice (0=chemin, 1=mur):")
    for row in grid:
        print(" ".join(str(v) for v in row))


def tp_questions() -> None:
    """
    Reponses concises aux questions 3-7 du TP (cote A*).
    """
    print("\n=== REPONSES TP (A*) ===")
    print("Q3 A* est plus rapide si heuristique informative, sinon proche de Dijkstra.")
    print("Q4 Poids negatifs: A* non garanti, la preuve d'optimalite tombe.")
    print("Q5 Complexite: ici O(V^2) (argmin sur Q), memoire O(V).")
    print("Q6 A* ne fait pas mieux si h(n)=0 ou h(n) tres faible/informative.")
    print("Q7 h parfaite: seuls les noeuds du chemin optimal sont explores.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TP A* (labyrinthe, version separee).")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Desactive l'affichage matplotlib.",
    )
    parser.add_argument("--width", type=int, default=4, help="Largeur de la matrice.")
    parser.add_argument("--height", type=int, default=4, help="Hauteur de la matrice.")
    parser.add_argument("--start", default="0,0", help="Start au format x,y.")
    parser.add_argument("--goal", default="auto", help="Goal au format x,y, ou 'auto'.")
    parser.add_argument("--obstacles", default="", help="Obstacles au format x,y;x,y;...")
    parser.add_argument("--turn-penalty", type=float, default=0.5, help="Cout changement de direction.")
    args = parser.parse_args()

    if args.no_plots:
        globals()["plot_maze"] = lambda *_a, **_k: None  # type: ignore

    def parse_cell(text: str) -> Tuple[int, int]:
        a, b = [p.strip() for p in text.split(",")]
        return int(a), int(b)

    def parse_cells_list(text: str) -> List[Tuple[int, int]]:
        text = text.strip()
        if not text:
            return []
        return [parse_cell(item.strip()) for item in text.split(";") if item.strip()]

    start = parse_cell(args.start)
    if args.goal.strip().lower() == "auto":
        goal = (args.width - 1, args.height - 1)
    else:
        goal = parse_cell(args.goal)
    obstacles = parse_cells_list(args.obstacles)
    maze = generate_maze(
        width=args.width,
        height=args.height,
        start=start,
        goal=goal,
        obstacles=obstacles,
    )

    print_matrix(maze.grid)
    # Plot 1: matrice brute 0/1, sans calcul
    raw_values = {(x, y): maze.grid[y][x] for y in range(maze.height) for x in range(maze.width)}
    plot_maze(maze, title="Matrice 0/1 (sans calcul)", values=raw_values, show_path=False)

    path, g_score = maze.solve_a_star(turn_penalty=args.turn_penalty)
    print(f"A*: chemin {start} -> {goal} = {path}")

    # Heuristique Manhattan et cout total f = g + h
    def h(cell: Tuple[int, int]) -> float:
        return float(abs(cell[0] - goal[0]) + abs(cell[1] - goal[1]))

    f_score = {(x, y): g_score[(x, y)] + h((x, y)) for y in range(maze.height) for x in range(maze.width)}

    plot_maze(
        maze,
        title="Cout g (deplacement + changements)",
        path=path,
        values=g_score,
        show_path=False,
    )
    plot_maze(
        maze,
        title="Heuristique h (Manhattan)",
        path=path,
        values={(x, y): h((x, y)) for y in range(maze.height) for x in range(maze.width)},
        show_path=False,
    )
    plot_maze(
        maze,
        title="Cout total f = g + h",
        path=path,
        values=f_score,
        show_path=False,
    )
    # Plot final: chemin optimal (rendu final)
    plot_maze(
        maze,
        title="Chemin optimal (rendu final)",
        path=path,
        values=None,
        show_path=True,
    )