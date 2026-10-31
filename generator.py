"""
DAA Project: Maze Generation Module
====================================
Algorithms Implemented:
1. Randomized Depth-First Search (Iterative Recursive Backtracker with LIFO Stack)
2. Randomized Prim's Algorithm (Minimum Spanning Tree / Frontier Wall List)
3. Randomized Kruskal's Algorithm (MST with Disjoint Set Union by Rank & Path Compression)

Theoretical Time & Space Complexity:
- Randomized DFS:
    Time:  O(|V|) where |V| is number of cells (each cell visited once, 4 neighbors checked in O(1))
    Space: O(|V|) worst-case stack depth (in linear corridors)
- Randomized Prim's:
    Time:  O(|E|) = O(|V|) with O(1) frontier wall sampling via swap-pop
    Space: O(|V|) frontier walls tracking
- Randomized Kruskal's:
    Time:  O(|E| * alpha(|V|)) where alpha is inverse Ackermann function (effectively O(|E|))
    Space: O(|V|) DSU parent & rank arrays + O(|E|) shuffled wall list

Representation:
A grid of dimensions (2*R + 1) x (2*C + 1) where:
- 0 = PATH (walkable corridor)
- 1 = WALL (obstacle)
- Odd coordinates (2r+1, 2c+1) represent the cell rooms.
- Even coordinates represent walls separating adjacent rooms.
"""

import random
from typing import List, Tuple, Generator, Optional, Set
from dataclasses import dataclass


WALL = 1
PATH = 0


@dataclass
class GenerationStep:
    """Represents an atomic animation frame for visual rendering."""
    grid: List[List[int]]
    current_cell: Optional[Tuple[int, int]]
    frontier_or_stack: List[Tuple[int, int]]
    carved_wall: Optional[Tuple[int, int]]
    algorithm: str
    is_complete: bool = False


class DisjointSetUnion:
    """Disjoint Set Union (DSU) / Union-Find with Path Compression and Union by Rank."""
    __slots__ = ('parent', 'rank')

    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # Path compression
        curr = x
        while curr != root:
            nxt = self.parent[curr]
            self.parent[curr] = root
            curr = nxt
        return root

    def union(self, x: int, y: int) -> bool:
        rx = self.find(x)
        ry = self.find(y)
        if rx == ry:
            return False
        # Union by rank
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1
        return True


class Maze:
    """
    Maze structure holding grid dimensions, cell matrix, and start/goal points.
    """
    def __init__(self, rows: int, cols: int):
        """
        rows: number of cell rooms vertically
        cols: number of cell rooms horizontally
        Actual matrix height: 2*rows + 1, width: 2*cols + 1
        """
        self.cell_rows = rows
        self.cell_cols = cols
        self.height = 2 * rows + 1
        self.width = 2 * cols + 1
        self.grid: List[List[int]] = [[WALL for _ in range(self.width)] for _ in range(self.height)]
        self.start: Tuple[int, int] = (1, 1)
        self.goal: Tuple[int, int] = (self.height - 2, self.width - 2)

    def reset_grid(self):
        self.grid = [[WALL for _ in range(self.width)] for _ in range(self.height)]

    def is_valid_cell(self, cr: int, cc: int) -> bool:
        return 0 <= cr < self.cell_rows and 0 <= cc < self.cell_cols

    @staticmethod
    def cell_to_grid(cr: int, cc: int) -> Tuple[int, int]:
        """Convert cell coordinates (cr, cc) to grid matrix coordinates (gr, gc)."""
        return 2 * cr + 1, 2 * cc + 1

    @staticmethod
    def grid_to_cell(gr: int, gc: int) -> Tuple[int, int]:
        """Convert grid matrix coordinates (gr, gc) to cell coordinates (cr, cc)."""
        return (gr - 1) // 2, (gc - 1) // 2


def generate_dfs(maze: Maze, seed: Optional[int] = None) -> Maze:
    """
    Generate maze using Randomized Depth-First Search (Iterative with explicit LIFO stack).
    Batch / Fast Mode for benchmarking.
    """
    if seed is not None:
        random.seed(seed)

    maze.reset_grid()
    rows, cols = maze.cell_rows, maze.cell_cols
    visited = [[False for _ in range(cols)] for _ in range(rows)]

    # Start from cell (0, 0)
    stack = [(0, 0)]
    visited[0][0] = True
    gr, gc = Maze.cell_to_grid(0, 0)
    maze.grid[gr][gc] = PATH

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while stack:
        cr, cc = stack[-1]
        unvisited_neighbors = []

        for dr, dc in directions:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                unvisited_neighbors.append((nr, nc, dr, dc))

        if unvisited_neighbors:
            nr, nc, dr, dc = random.choice(unvisited_neighbors)
            visited[nr][nc] = True

            # Carve through wall
            curr_gr, curr_gc = Maze.cell_to_grid(cr, cc)
            wall_gr, wall_gc = curr_gr + dr, curr_gc + dc
            next_gr, next_gc = Maze.cell_to_grid(nr, nc)

            maze.grid[wall_gr][wall_gc] = PATH
            maze.grid[next_gr][next_gc] = PATH

            stack.append((nr, nc))
        else:
            stack.pop()

    return maze


def generate_dfs_animated(maze: Maze, seed: Optional[int] = None) -> Generator[GenerationStep, None, None]:
    """
    Generator yielding animation steps for real-time Pygame visualization of DFS.
    """
    if seed is not None:
        random.seed(seed)

    maze.reset_grid()
    rows, cols = maze.cell_rows, maze.cell_cols
    visited = [[False for _ in range(cols)] for _ in range(rows)]

    stack = [(0, 0)]
    visited[0][0] = True
    gr, gc = Maze.cell_to_grid(0, 0)
    maze.grid[gr][gc] = PATH

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    yield GenerationStep(
        grid=maze.grid,
        current_cell=(gr, gc),
        frontier_or_stack=[Maze.cell_to_grid(r, c) for r, c in stack],
        carved_wall=None,
        algorithm="Randomized DFS"
    )

    step_count = 0
    while stack:
        cr, cc = stack[-1]
        unvisited_neighbors = []

        for dr, dc in directions:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                unvisited_neighbors.append((nr, nc, dr, dc))

        if unvisited_neighbors:
            nr, nc, dr, dc = random.choice(unvisited_neighbors)
            visited[nr][nc] = True

            curr_gr, curr_gc = Maze.cell_to_grid(cr, cc)
            wall_gr, wall_gc = curr_gr + dr, curr_gc + dc
            next_gr, next_gc = Maze.cell_to_grid(nr, nc)

            maze.grid[wall_gr][wall_gc] = PATH
            maze.grid[next_gr][next_gc] = PATH
            stack.append((nr, nc))

            step_count += 1
            yield GenerationStep(
                grid=maze.grid,
                current_cell=(next_gr, next_gc),
                frontier_or_stack=[Maze.cell_to_grid(r, c) for r, c in stack[-15:]],
                carved_wall=(wall_gr, wall_gc),
                algorithm="Randomized DFS"
            )
        else:
            stack.pop()
            if stack and step_count % 3 == 0:
                p_gr, p_gc = Maze.cell_to_grid(stack[-1][0], stack[-1][1])
                yield GenerationStep(
                    grid=maze.grid,
                    current_cell=(p_gr, p_gc),
                    frontier_or_stack=[Maze.cell_to_grid(r, c) for r, c in stack[-15:]],
                    carved_wall=None,
                    algorithm="Randomized DFS"
                )

    yield GenerationStep(
        grid=maze.grid,
        current_cell=None,
        frontier_or_stack=[],
        carved_wall=None,
        algorithm="Randomized DFS",
        is_complete=True
    )


def generate_prims(maze: Maze, seed: Optional[int] = None) -> Maze:
    """
    Generate maze using Randomized Prim's Algorithm (Frontier Wall List).
    Fast / Batch Mode for benchmarking.
    """
    if seed is not None:
        random.seed(seed)

    maze.reset_grid()
    rows, cols = maze.cell_rows, maze.cell_cols
    visited = [[False for _ in range(cols)] for _ in range(rows)]

    # Choose arbitrary start cell (0, 0)
    visited[0][0] = True
    s_gr, s_gc = Maze.cell_to_grid(0, 0)
    maze.grid[s_gr][s_gc] = PATH

    # Walls list: stores (from_r, from_c, to_r, to_c, wall_r, wall_c)
    walls = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dr, dc in directions:
        nr, nc = dr, dc
        if 0 <= nr < rows and 0 <= nc < cols:
            walls.append((0, 0, nr, nc, s_gr + dr, s_gc + dc))

    while walls:
        # Fast random pop in O(1): pick index, swap with last element, then pop()
        idx = random.randrange(len(walls))
        walls[idx], walls[-1] = walls[-1], walls[idx]
        from_r, from_c, to_r, to_c, w_gr, w_gc = walls.pop()

        if not visited[to_r][to_c]:
            visited[to_r][to_c] = True
            to_gr, to_gc = Maze.cell_to_grid(to_r, to_c)

            maze.grid[w_gr][w_gc] = PATH
            maze.grid[to_gr][to_gc] = PATH

            for dr, dc in directions:
                adj_r, adj_c = to_r + dr, to_c + dc
                if 0 <= adj_r < rows and 0 <= adj_c < cols and not visited[adj_r][adj_c]:
                    walls.append((to_r, to_c, adj_r, adj_c, to_gr + dr, to_gc + dc))

    return maze


def generate_prims_animated(maze: Maze, seed: Optional[int] = None) -> Generator[GenerationStep, None, None]:
    """
    Generator yielding animation steps for real-time Pygame visualization of Prim's Algorithm.
    """
    if seed is not None:
        random.seed(seed)

    maze.reset_grid()
    rows, cols = maze.cell_rows, maze.cell_cols
    visited = [[False for _ in range(cols)] for _ in range(rows)]

    visited[0][0] = True
    s_gr, s_gc = Maze.cell_to_grid(0, 0)
    maze.grid[s_gr][s_gc] = PATH

    walls = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dr, dc in directions:
        nr, nc = dr, dc
        if 0 <= nr < rows and 0 <= nc < cols:
            walls.append((0, 0, nr, nc, s_gr + dr, s_gc + dc))

    yield GenerationStep(
        grid=maze.grid,
        current_cell=(s_gr, s_gc),
        frontier_or_stack=[(w[4], w[5]) for w in walls],
        carved_wall=None,
        algorithm="Randomized Prim's"
    )

    while walls:
        idx = random.randrange(len(walls))
        walls[idx], walls[-1] = walls[-1], walls[idx]
        from_r, from_c, to_r, to_c, w_gr, w_gc = walls.pop()

        if not visited[to_r][to_c]:
            visited[to_r][to_c] = True
            to_gr, to_gc = Maze.cell_to_grid(to_r, to_c)

            maze.grid[w_gr][w_gc] = PATH
            maze.grid[to_gr][to_gc] = PATH

            for dr, dc in directions:
                adj_r, adj_c = to_r + dr, to_c + dc
                if 0 <= adj_r < rows and 0 <= adj_c < cols and not visited[adj_r][adj_c]:
                    walls.append((to_r, to_c, adj_r, adj_c, to_gr + dr, to_gc + dc))

            yield GenerationStep(
                grid=maze.grid,
                current_cell=(to_gr, to_gc),
                frontier_or_stack=[(w[4], w[5]) for w in walls[-20:]],
                carved_wall=(w_gr, w_gc),
                algorithm="Randomized Prim's"
            )

    yield GenerationStep(
        grid=maze.grid,
        current_cell=None,
        frontier_or_stack=[],
        carved_wall=None,
        algorithm="Randomized Prim's",
        is_complete=True
    )


def generate_kruskals(maze: Maze, seed: Optional[int] = None) -> Maze:
    """
    Generate maze using Randomized Kruskal's Algorithm with Disjoint Set Union (DSU).
    Batch / Fast Mode for benchmarking.
    """
    if seed is not None:
        random.seed(seed)

    maze.reset_grid()
    rows, cols = maze.cell_rows, maze.cell_cols
    total_cells = rows * cols
    dsu = DisjointSetUnion(total_cells)

    # Initialize all cell centers as PATH
    for r in range(rows):
        for c in range(cols):
            gr, gc = Maze.cell_to_grid(r, c)
            maze.grid[gr][gc] = PATH

    # Collect all internal walls
    # A wall connects cell (r, c) and either (r+1, c) or (r, c+1)
    walls = []
    for r in range(rows):
        for c in range(cols):
            u_id = r * cols + c
            # South wall
            if r + 1 < rows:
                v_id = (r + 1) * cols + c
                w_gr, w_gc = 2 * r + 2, 2 * c + 1
                walls.append((u_id, v_id, w_gr, w_gc))
            # East wall
            if c + 1 < cols:
                v_id = r * cols + (c + 1)
                w_gr, w_gc = 2 * r + 1, 2 * c + 2
                walls.append((u_id, v_id, w_gr, w_gc))

    random.shuffle(walls)

    edges_carved = 0
    target_edges = total_cells - 1

    for u_id, v_id, w_gr, w_gc in walls:
        if dsu.union(u_id, v_id):
            maze.grid[w_gr][w_gc] = PATH
            edges_carved += 1
            if edges_carved == target_edges:
                break

    return maze


def generate_kruskals_animated(maze: Maze, seed: Optional[int] = None) -> Generator[GenerationStep, None, None]:
    """
    Generator yielding animation steps for real-time Pygame visualization of Kruskal's Algorithm.
    """
    if seed is not None:
        random.seed(seed)

    maze.reset_grid()
    rows, cols = maze.cell_rows, maze.cell_cols
    total_cells = rows * cols
    dsu = DisjointSetUnion(total_cells)

    for r in range(rows):
        for c in range(cols):
            gr, gc = Maze.cell_to_grid(r, c)
            maze.grid[gr][gc] = PATH

    walls = []
    for r in range(rows):
        for c in range(cols):
            u_id = r * cols + c
            if r + 1 < rows:
                v_id = (r + 1) * cols + c
                w_gr, w_gc = 2 * r + 2, 2 * c + 1
                walls.append((u_id, v_id, w_gr, w_gc))
            if c + 1 < cols:
                v_id = r * cols + (c + 1)
                w_gr, w_gc = 2 * r + 1, 2 * c + 2
                walls.append((u_id, v_id, w_gr, w_gc))

    random.shuffle(walls)
    edges_carved = 0
    target_edges = total_cells - 1

    for u_id, v_id, w_gr, w_gc in walls:
        if dsu.union(u_id, v_id):
            maze.grid[w_gr][w_gc] = PATH
            edges_carved += 1
            yield GenerationStep(
                grid=maze.grid,
                current_cell=(w_gr, w_gc),
                frontier_or_stack=[],
                carved_wall=(w_gr, w_gc),
                algorithm="Randomized Kruskal's"
            )
            if edges_carved == target_edges:
                break

    yield GenerationStep(
        grid=maze.grid,
        current_cell=None,
        frontier_or_stack=[],
        carved_wall=None,
        algorithm="Randomized Kruskal's",
        is_complete=True
    )


GENERATORS = {
    "DFS (Recursive Backtracker)": generate_dfs,
    "Prim's Algorithm": generate_prims,
    "Kruskal's Algorithm": generate_kruskals
}

ANIMATED_GENERATORS = {
    "DFS (Recursive Backtracker)": generate_dfs_animated,
    "Prim's Algorithm": generate_prims_animated,
    "Kruskal's Algorithm": generate_kruskals_animated
}
