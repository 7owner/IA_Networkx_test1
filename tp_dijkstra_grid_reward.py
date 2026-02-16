# -*- coding: utf-8 -*-
"""
tp_dijkstra_grid_reward.py

TP1 - Partie 5 : Representation du labyrinthe (maze)

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
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import math

Cell = Tuple[int, int]


class RewardGrid:
    def __init__(
        self,
        *,
        width: int,
        height: int,
        grid: List[List[int]],
        reward: List[List[float]],
        start: Cell,
        goal: Cell,
    ) -> None:
        self.width = width
        self.height = height
        self.grid = grid
        self.reward = reward
        self.start = start
        self.goal = goal

    def in_bounds(self, cell: Cell) -> bool:
        (x, y) = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def is_free(self, cell: Cell) -> bool:
        (x, y) = cell
        return self.grid[y][x] == 0

    def neighbors_4(self, cell: Cell) -> List[Cell]:
        (x, y) = cell
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        out: List[Cell] = []
        for c in candidates:
            if self.in_bounds(c) and self.is_free(c):
                out.append(c)
        return out


def generate_reward_grid(
    *,
    width: int,
    height: int,
    start: Cell,
    goal: Cell,
    obstacles: Sequence[Cell],
    step_reward: float = -1.0,
    goal_reward: float = 100.0,
    bonus_cells: Sequence[Cell] = (),
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


def reconstruct_path(parent: Dict[Cell, Optional[Cell]], goal: Cell) -> List[Cell]:
    path: List[Cell] = []
    cur: Optional[Cell] = goal
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()
    return path


def dijkstra_grid_with_reward_abs(
    rg: RewardGrid,
) -> Tuple[Optional[List[Cell]], Dict[Cell, float]]:
    """
    Dijkstra sur grille avec cout = |reward| de la case d'arrivee.
    """
    dist: Dict[Cell, float] = {
        (x, y): float("inf") for y in range(rg.height) for x in range(rg.width)
    }
    parent: Dict[Cell, Optional[Cell]] = {
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


def dijkstra_grid_unit_cost(
    rg: RewardGrid,
) -> Tuple[Optional[List[Cell]], Dict[Cell, float]]:
    """
    Dijkstra sur grille avec cout unitaire = 1 par pas.
    """
    dist: Dict[Cell, float] = {
        (x, y): float("inf") for y in range(rg.height) for x in range(rg.width)
    }
    parent: Dict[Cell, Optional[Cell]] = {
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
            alt = dist[u] + 1.0
            if alt < dist[v]:
                dist[v] = alt
                parent[v] = u

    return None, dist


def a_star_grid_unit_cost(
    rg: RewardGrid,
) -> Tuple[Optional[List[Cell]], Dict[Cell, float]]:
    """
    A* sur grille avec cout unitaire = 1 par pas et heuristique Manhattan.
    """
    g_score: Dict[Cell, float] = {
        (x, y): float("inf") for y in range(rg.height) for x in range(rg.width)
    }
    parent: Dict[Cell, Optional[Cell]] = {
        (x, y): None for y in range(rg.height) for x in range(rg.width)
    }
    g_score[rg.start] = 0.0

    def h(cell: Cell) -> float:
        return float(abs(cell[0] - rg.goal[0]) + abs(cell[1] - rg.goal[1]))

    open_set: set[Cell] = {rg.start}
    while open_set:
        u = min(open_set, key=lambda x: g_score[x] + h(x))
        if u == rg.goal:
            return reconstruct_path(parent, rg.goal), g_score
        open_set.remove(u)

        for v in rg.neighbors_4(u):
            alt = g_score[u] + 1.0
            if alt < g_score[v]:
                g_score[v] = alt
                parent[v] = u
                if v not in open_set:
                    open_set.add(v)

    return None, g_score


def plot_grid_values(
    *,
    rg: RewardGrid,
    title: str,
    values: List[List[float]] | List[List[int]],
    path: Optional[List[Cell]] = None,
    ax=None,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "matplotlib n'est pas installe. Installe-le avec: pip install matplotlib"
        ) from e

    h, w = rg.height, rg.width
    created_fig = False
    if ax is None:
        plt.figure(figsize=(6, 6))
        ax = plt.gca()
        created_fig = True
    ax.set_title(title)
    ax.axis("off")

    for y in range(h):
        for x in range(w):
            is_wall = rg.grid[y][x] == 1
            color = "#666666" if is_wall else "#FFFFFF"
            rect = plt.Rectangle((x, h - 1 - y), 1, 1, facecolor=color, edgecolor="#BBBBBB")
            ax.add_patch(rect)
            val = values[y][x]
            ax.text(x + 0.5, h - 1 - y + 0.5, f"{val:g}", ha="center", va="center", fontsize=8)

    if path and len(path) >= 2:
        xs = [p[0] + 0.5 for p in path]
        ys = [h - 1 - p[1] + 0.5 for p in path]
        ax.plot(xs, ys, color="crimson", linewidth=3, zorder=4)
        ax.scatter(xs, ys, c="crimson", s=25, zorder=4)

    sx, sy = rg.start
    gx, gy = rg.goal
    ax.scatter([sx + 0.5], [h - 1 - sy + 0.5], c="green", s=150, zorder=3)
    ax.scatter([gx + 0.5], [h - 1 - gy + 0.5], c="red", s=150, zorder=3)
    ax.text(sx + 0.5, h - 1 - sy + 0.5, "S", ha="center", va="center", color="white")
    ax.text(gx + 0.5, h - 1 - gy + 0.5, "G", ha="center", va="center", color="white")

    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    if created_fig:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    rg = generate_reward_grid(
        width=4,
        height=4,
        start=(0, 0),
        goal=(3, 3),
        obstacles=[(1, 1)],
        step_reward=-1.0,
        goal_reward=100.0,
        bonus_cells=[(1, 2), (2, 1)],
        bonus_value=5.0,
    )
    plot_grid_values(
        rg=rg,
        values=rg.grid,
        title="Grille 0/1 (0=libre, 1=mur)",
    )
    plot_grid_values(
        rg=rg,
        values=rg.reward,
        title="Reward map",
    )
    # Resolution Dijkstra (cout = |reward|)
    path, dist = dijkstra_grid_with_reward_abs(rg)
    print("Chemin Dijkstra:", path)
    plot_grid_values(
        rg=rg,
        values=[[dist[(x, y)] for x in range(rg.width)] for y in range(rg.height)],
        title="Dijkstra - couts cumulés (|reward|)",
    )
