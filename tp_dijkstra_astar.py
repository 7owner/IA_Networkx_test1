# -*- coding: utf-8 -*-
"""
tp_dijkstra_astar.py

Comparaison Dijkstra vs A* sur une meme matrice avec poids.
- Meme grille (0 = libre, 1 = mur)
- Meme couts (poids par cellule)
- Plots pour comparer les couts et le chemin final
"""

from __future__ import annotations

import argparse
import heapq
from typing import Dict, List, Optional, Tuple

Cell = Tuple[int, int]


# ============================================================
# PARTIE A - GRILLE ET COUTS
# ============================================================
def make_grid(
    *,
    width: int,
    height: int,
    obstacles: List[Cell],
) -> List[List[int]]:
    grid = [[0 for _ in range(width)] for _ in range(height)]
    for (x, y) in obstacles:
        if 0 <= x < width and 0 <= y < height:
            grid[y][x] = 1
    return grid


def make_weights(
    *,
    width: int,
    height: int,
    base_cost: float = 1.0,
) -> List[List[float]]:
    """
    Poids par cellule (cout pour entrer dans la cellule).
    Ici on met un motif simple pour visualiser des couts differents.
    """
    weights: List[List[float]] = []
    for y in range(height):
        row: List[float] = []
        for x in range(width):
            row.append(base_cost + (x + y) % 3)
        weights.append(row)
    return weights


def in_bounds(cell: Cell, width: int, height: int) -> bool:
    (x, y) = cell
    return 0 <= x < width and 0 <= y < height


def neighbors_4(cell: Cell, grid: List[List[int]]) -> List[Cell]:
    (x, y) = cell
    h = len(grid)
    w = len(grid[0])
    candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    out: List[Cell] = []
    for c in candidates:
        if in_bounds(c, w, h) and grid[c[1]][c[0]] == 0:
            out.append(c)
    return out


def reconstruct_path(parent: Dict[Cell, Optional[Cell]], goal: Cell) -> List[Cell]:
    path: List[Cell] = []
    cur: Optional[Cell] = goal
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()
    return path


# ============================================================
# PARTIE B - DIJKSTRA ET A*
# ============================================================
def dijkstra_grid(
    *,
    grid: List[List[int]],
    weights: List[List[float]],
    start: Cell,
    goal: Cell,
) -> Tuple[Optional[List[Cell]], Dict[Cell, float]]:
    h = len(grid)
    w = len(grid[0])
    dist: Dict[Cell, float] = {(x, y): float("inf") for y in range(h) for x in range(w)}
    parent: Dict[Cell, Optional[Cell]] = {(x, y): None for y in range(h) for x in range(w)}
    dist[start] = 0.0

    heap: List[Tuple[float, Cell]] = [(0.0, start)]
    while heap:
        d_u, u = heapq.heappop(heap)
        if d_u != dist[u]:
            continue
        if u == goal:
            return reconstruct_path(parent, goal), dist
        for v in neighbors_4(u, grid):
            cost = weights[v[1]][v[0]]
            alt = d_u + cost
            if alt < dist[v]:
                dist[v] = alt
                parent[v] = u
                heapq.heappush(heap, (alt, v))

    return None, dist


def manhattan(a: Cell, b: Cell) -> float:
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


def a_star_grid(
    *,
    grid: List[List[int]],
    weights: List[List[float]],
    start: Cell,
    goal: Cell,
) -> Tuple[Optional[List[Cell]], Dict[Cell, float]]:
    h = len(grid)
    w = len(grid[0])
    g_score: Dict[Cell, float] = {(x, y): float("inf") for y in range(h) for x in range(w)}
    parent: Dict[Cell, Optional[Cell]] = {(x, y): None for y in range(h) for x in range(w)}
    g_score[start] = 0.0

    heap: List[Tuple[float, Cell]] = [(manhattan(start, goal), start)]
    while heap:
        _f, u = heapq.heappop(heap)
        if u == goal:
            return reconstruct_path(parent, goal), g_score
        for v in neighbors_4(u, grid):
            cost = weights[v[1]][v[0]]
            alt = g_score[u] + cost
            if alt < g_score[v]:
                g_score[v] = alt
                parent[v] = u
                f = alt + manhattan(v, goal)
                heapq.heappush(heap, (f, v))

    return None, g_score


# ============================================================
# PARTIE C - PLOTS
# ============================================================
def plot_matrix(
    *,
    grid: List[List[int]],
    values: Optional[Dict[Cell, float]],
    start: Cell,
    goal: Cell,
    title: str,
    path: Optional[List[Cell]] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "matplotlib n'est pas installe. Installe-le avec: pip install matplotlib"
        ) from e

    h = len(grid)
    w = len(grid[0])
    plt.figure(figsize=(6, 6))
    plt.title(title)
    plt.axis("off")

    for y in range(h):
        for x in range(w):
            is_wall = grid[y][x] == 1
            if values is None:
                color = "#FFFFFF"
            else:
                color = "#666666" if is_wall else "#FFFFFF"
            rect = plt.Rectangle((x, h - 1 - y), 1, 1, facecolor=color, edgecolor="#BBBBBB")
            plt.gca().add_patch(rect)
            if values is None:
                txt = "0"
            elif is_wall:
                txt = ""
            else:
                val = values.get((x, y), float("inf"))
                txt = "inf" if val == float("inf") else f"{val:g}"
            plt.text(x + 0.5, h - 1 - y + 0.5, txt, ha="center", va="center", fontsize=8)

    if path and len(path) >= 2:
        xs = [p[0] + 0.5 for p in path]
        ys = [h - 1 - p[1] + 0.5 for p in path]
        plt.plot(xs, ys, color="crimson", linewidth=3, zorder=4)
        plt.scatter(xs, ys, c="crimson", s=25, zorder=4)

    sx, sy = start
    gx, gy = goal
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
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparaison Dijkstra vs A* sur une meme grille.")
    parser.add_argument("--width", type=int, default=6)
    parser.add_argument("--height", type=int, default=5)
    parser.add_argument("--start", default="0,0")
    parser.add_argument("--goal", default="auto")
    parser.add_argument("--obstacles", default="")
    args = parser.parse_args()

    def parse_cell(text: str) -> Cell:
        a, b = [p.strip() for p in text.split(",")]
        return int(a), int(b)

    def parse_cells_list(text: str) -> List[Cell]:
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
    grid = make_grid(width=args.width, height=args.height, obstacles=obstacles)
    weights = make_weights(width=args.width, height=args.height)

    # Dijkstra
    path_d, dist_d = dijkstra_grid(grid=grid, weights=weights, start=start, goal=goal)
    # A*
    path_a, dist_a = a_star_grid(grid=grid, weights=weights, start=start, goal=goal)
    f_a = {
        (x, y): dist_a[(x, y)] + manhattan((x, y), goal)
        for y in range(args.height)
        for x in range(args.width)
    }

    # Plots comparatifs
    plot_matrix(grid=grid, values=None, start=start, goal=goal, title="Matrice 0/1 (sans calcul)")
    plot_matrix(
        grid=grid,
        values=dist_d,
        start=start,
        goal=goal,
        title="Dijkstra - couts dist (cumule)",
        path=path_d,
    )
    plot_matrix(
        grid=grid,
        values=dist_a,
        start=start,
        goal=goal,
        title="A* - couts g",
        path=path_a,
    )
    plot_matrix(
        grid=grid,
        values=f_a,
        start=start,
        goal=goal,
        title="A* - couts f = g + h",
        path=path_a,
    )
