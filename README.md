# TP Dijkstra & A* - Explorateurs Python

Ce projet regroupe des scripts pédagogiques pour comprendre et comparer les algorithmes de plus court chemin:
- `Dijkstra`
- `A*`

Avec des versions console et des interfaces graphiques interactives.

## Prérequis

- Python 3.10+
- `matplotlib`
- `tkinter` (souvent inclus avec Python)

Installation rapide:

```bash
python -m pip install matplotlib
```

## Fichiers principaux

- `tp_dijkstra.py`  
  Implémentation Dijkstra (graphe + grille), démos et visualisations.

- `tp_astar.py`  
  Implémentation A* sur grille avec étapes de visualisation (`g`, `h`, `f`, chemin final).

- `tp_astar_gui.py`  
  GUI interactive A* pour éditer la grille, placer obstacles et explorer les vues.

- `tp_dijkstra_gui.py`  
  GUI interactive Dijkstra avec obstacles + bonus personnalisables par case.

- `tp_compare_gui.py`  
  GUI de comparaison A* vs Dijkstra sur la même grille (coût, nœuds explorés, temps).

## Lancement

### 1) A* en script

```bash
python tp_astar.py
```

### 2) GUI A* pour avoir la vision complète de ce qui a été fait pour A_star

```bash
python tp_astar_gui.py
```

### 3) GUI Dijkstra

```bash
python tp_dijkstra_gui.py
```

### 4) GUI comparaison A* vs Dijkstra

```bash
python tp_compare_gui.py
```

## Idée des vues

### `tp_astar.py`
- matrice 0/1 (libre/obstacle),
- coût `g`,
- heuristique `h`,
- coût total `f = g + h`,
- chemin final.

### `tp_astar_gui.py`
- matrice 0/1,
- coût `g`,
- heuristique `h`,
- coût `f`,
- chemin final,
- ordre d’exploration.

### `tp_dijkstra_gui.py`
- coût local (avec bonus),
- distance cumulée,
- chemin final,
- ordre d’exploration.

### `tp_compare_gui.py`
- vue côte à côte Dijkstra / A* :
  - distance,
  - coût local,
  - reward,
  - ordre d’exploration.

## Notes pédagogiques

- Dijkstra minimise un **coût cumulé** (non négatif).
- A* minimise aussi un coût, mais ajoute une heuristique `h` pour guider l’exploration.
- Quand des rewards sont utilisées, elles sont converties en coût local pour rester compatibles avec la résolution.

lient vers le git :
