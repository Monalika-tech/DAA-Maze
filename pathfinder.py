"""
DAA Project: Pathfinding & Search Module
========================================
Algorithms Implemented:
1. Breadth-First Search (BFS) - O(V + E)
2. Dijkstra's Algorithm (Uniform Cost Min-Heap) - O((V + E) log V)
3. A* Search (Manhattan Heuristic - Consistent & Admissible) - O(E log V)
4. A* Search (Euclidean Heuristic - Strictly Admissible) - O(E log V)
5. Greedy Best-First Search (Pure Heuristic, Non-Optimal Baseline) - O(E log V)

Academic DAA Metrics Tracked:
- Execution Time (Microseconds / Milliseconds via time.perf_counter_ns)
- Peak Memory Allocation (Bytes / Kilobytes via tracemalloc)
- Total Nodes Expanded (Exploration Volume)
- Frontier Peak Size (Space Complexity in Flight)
- Path Length & Optimality Ratio
"""

import heapq
import math
import time
import tracemalloc
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional, Generator, Callable

WALL = 1
PATH = 0


@dataclass
class PathfindingMetrics:
    algorithm_name: str
    path: List[Tuple[int, int]]
    visited_order: List[Tuple[int, int]]
    nodes_expanded: int
    nodes_generated: int
    max_frontier_size: int
    path_length: int
    execution_time_us: float
    peak_memory_bytes: int
    found: bool
    heuristic_name: Optional[str] = None


@dataclass
class PathfindingStep:
    current: Tuple[int, int]
    visited: Set[Tuple[int, int]]
    frontier: Set[Tuple[int, int]]
    path: List[Tuple[int, int]]
    algorithm_name: str
    is_complete: bool = False
    metrics: Optional[PathfindingMetrics] = None


# --- Heuristic Functions for 2D Grid ---

def heuristic_manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """
    Manhattan distance: L1 norm |r1 - r2| + |c1 - c2|.
    Admissible and consistent for 4-directional grid movement.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def heuristic_euclidean(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """
    Euclidean distance: L2 norm sqrt((r1 - r2)^2 + (c1 - c2)^2).
    Admissible since Euclidean <= Manhattan on grid graphs, but explores more nodes.
    """
    return math.hypot(a[0] - b[0], a[1] - b[1])


def heuristic_zero(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Null heuristic: reduces A* exactly to Dijkstra's Algorithm."""
    return 0.0


def reconstruct_path(parent: Dict[Tuple[int, int], Tuple[int, int]],
                     start: Tuple[int, int],
                     goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Backtrack from goal to start via parent pointers."""
    if goal not in parent and start != goal:
        return []
    curr = goal
    path = []
    while curr is not None:
        path.append(curr)
        curr = parent.get(curr)
    path.reverse()
    return path


def get_neighbors(grid: List[List[int]], r: int, c: int, height: int, width: int) -> List[Tuple[int, int]]:
    """Return all 4-directional walkable neighbors."""
    neighbors = []
    # North, South, West, East
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width and grid[nr][nc] == PATH:
            neighbors.append((nr, nc))
    return neighbors


# =====================================================================
# BATCH EXECUTION FUNCTIONS (Benchmarking / Metric Collection)
# =====================================================================

def solve_bfs(grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]) -> PathfindingMetrics:
    """
    Breadth-First Search (Queue-based FIFO).
    Guarantees shortest path on unweighted graphs.
    Complexity: Time O(V + E), Space O(V).
    """
    tracemalloc.start()
    t_start = time.perf_counter_ns()

    height = len(grid)
    width = len(grid[0])

    queue = deque([start])
    visited: Set[Tuple[int, int]] = {start}
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    visited_order: List[Tuple[int, int]] = []

    nodes_expanded = 0
    nodes_generated = 1
    max_frontier_size = 1
    found = False

    while queue:
        max_frontier_size = max(max_frontier_size, len(queue))
        curr = queue.popleft()
        nodes_expanded += 1
        visited_order.append(curr)

        if curr == goal:
            found = True
            break

        for neighbor in get_neighbors(grid, curr[0], curr[1], height, width):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = curr
                queue.append(neighbor)
                nodes_generated += 1

    t_end = time.perf_counter_ns()
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    path = reconstruct_path(parent, start, goal) if found else []
    exec_time_us = (t_end - t_start) / 1000.0

    return PathfindingMetrics(
        algorithm_name="Breadth-First Search (BFS)",
        path=path,
        visited_order=visited_order,
        nodes_expanded=nodes_expanded,
        nodes_generated=nodes_generated,
        max_frontier_size=max_frontier_size,
        path_length=len(path),
        execution_time_us=exec_time_us,
        peak_memory_bytes=peak_memory,
        found=found
    )


def solve_dijkstra(grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]) -> PathfindingMetrics:
    """
    Dijkstra's Algorithm (Min-Heap Priority Queue).
    Uniform unit edge cost = 1.0.
    Complexity: Time O((V + E) log V), Space O(V).
    """
    tracemalloc.start()
    t_start = time.perf_counter_ns()

    height = len(grid)
    width = len(grid[0])

    # Priority queue stores: (g_score, tie_breaker, node)
    counter = 0
    pq = [(0.0, counter, start)]
    dist: Dict[Tuple[int, int], float] = {start: 0.0}
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    visited_order: List[Tuple[int, int]] = []
    settled: Set[Tuple[int, int]] = set()

    nodes_expanded = 0
    nodes_generated = 1
    max_frontier_size = 1
    found = False

    while pq:
        max_frontier_size = max(max_frontier_size, len(pq))
        d, _, curr = heapq.heappop(pq)

        if curr in settled:
            continue
        settled.add(curr)
        nodes_expanded += 1
        visited_order.append(curr)

        if curr == goal:
            found = True
            break

        for neighbor in get_neighbors(grid, curr[0], curr[1], height, width):
            new_dist = d + 1.0  # Unit edge weight
            if neighbor not in dist or new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                parent[neighbor] = curr
                counter += 1
                heapq.heappush(pq, (new_dist, counter, neighbor))
                nodes_generated += 1

    t_end = time.perf_counter_ns()
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    path = reconstruct_path(parent, start, goal) if found else []
    exec_time_us = (t_end - t_start) / 1000.0

    return PathfindingMetrics(
        algorithm_name="Dijkstra's Algorithm",
        path=path,
        visited_order=visited_order,
        nodes_expanded=nodes_expanded,
        nodes_generated=nodes_generated,
        max_frontier_size=max_frontier_size,
        path_length=len(path),
        execution_time_us=exec_time_us,
        peak_memory_bytes=peak_memory,
        found=found
    )


def solve_astar(grid: List[List[int]],
                start: Tuple[int, int],
                goal: Tuple[int, int],
                heuristic_fn: Callable[[Tuple[int, int], Tuple[int, int]], float] = heuristic_manhattan,
                heuristic_name: str = "Manhattan") -> PathfindingMetrics:
    """
    A* Search Algorithm (Min-Heap Priority Queue ordered by f(n) = g(n) + h(n)).
    Complexity: Worst case O(E log V), practically explores significantly fewer nodes.
    """
    tracemalloc.start()
    t_start = time.perf_counter_ns()

    height = len(grid)
    width = len(grid[0])

    counter = 0
    h_start = heuristic_fn(start, goal)
    # PQ stores: (f_score, h_score, counter, node) - tie-break on lower h_score
    pq = [(h_start, h_start, counter, start)]
    g_score: Dict[Tuple[int, int], float] = {start: 0.0}
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    closed_set: Set[Tuple[int, int]] = set()
    visited_order: List[Tuple[int, int]] = []

    nodes_expanded = 0
    nodes_generated = 1
    max_frontier_size = 1
    found = False

    while pq:
        max_frontier_size = max(max_frontier_size, len(pq))
        f, h, _, curr = heapq.heappop(pq)

        if curr in closed_set:
            continue
        closed_set.add(curr)
        nodes_expanded += 1
        visited_order.append(curr)

        if curr == goal:
            found = True
            break

        curr_g = g_score[curr]

        for neighbor in get_neighbors(grid, curr[0], curr[1], height, width):
            tentative_g = curr_g + 1.0  # Unit step cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                h_val = heuristic_fn(neighbor, goal)
                f_val = tentative_g + h_val
                parent[neighbor] = curr
                counter += 1
                heapq.heappush(pq, (f_val, h_val, counter, neighbor))
                nodes_generated += 1

    t_end = time.perf_counter_ns()
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    path = reconstruct_path(parent, start, goal) if found else []
    exec_time_us = (t_end - t_start) / 1000.0

    return PathfindingMetrics(
        algorithm_name=f"A* Search ({heuristic_name})",
        path=path,
        visited_order=visited_order,
        nodes_expanded=nodes_expanded,
        nodes_generated=nodes_generated,
        max_frontier_size=max_frontier_size,
        path_length=len(path),
        execution_time_us=exec_time_us,
        peak_memory_bytes=peak_memory,
        found=found,
        heuristic_name=heuristic_name
    )


def solve_greedy_bfs(grid: List[List[int]],
                     start: Tuple[int, int],
                     goal: Tuple[int, int],
                     heuristic_fn: Callable[[Tuple[int, int], Tuple[int, int]], float] = heuristic_manhattan) -> PathfindingMetrics:
    """
    Greedy Best-First Search (Priority Queue ordered strictly by h(n)).
    Non-optimal heuristic search: blazing fast exploration, suboptimal path.
    """
    tracemalloc.start()
    t_start = time.perf_counter_ns()

    height = len(grid)
    width = len(grid[0])

    counter = 0
    h_start = heuristic_fn(start, goal)
    pq = [(h_start, counter, start)]
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    visited: Set[Tuple[int, int]] = {start}
    visited_order: List[Tuple[int, int]] = []

    nodes_expanded = 0
    nodes_generated = 1
    max_frontier_size = 1
    found = False

    while pq:
        max_frontier_size = max(max_frontier_size, len(pq))
        _, _, curr = heapq.heappop(pq)
        nodes_expanded += 1
        visited_order.append(curr)

        if curr == goal:
            found = True
            break

        for neighbor in get_neighbors(grid, curr[0], curr[1], height, width):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = curr
                counter += 1
                heapq.heappush(pq, (heuristic_fn(neighbor, goal), counter, neighbor))
                nodes_generated += 1

    t_end = time.perf_counter_ns()
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    path = reconstruct_path(parent, start, goal) if found else []
    exec_time_us = (t_end - t_start) / 1000.0

    return PathfindingMetrics(
        algorithm_name="Greedy Best-First Search",
        path=path,
        visited_order=visited_order,
        nodes_expanded=nodes_expanded,
        nodes_generated=nodes_generated,
        max_frontier_size=max_frontier_size,
        path_length=len(path),
        execution_time_us=exec_time_us,
        peak_memory_bytes=peak_memory,
        found=found,
        heuristic_name="Manhattan"
    )


# =====================================================================
# ANIMATED GENERATORS (Step-by-Step Visualization in Pygame)
# =====================================================================

def step_bfs(grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]) -> Generator[PathfindingStep, None, None]:
    height, width = len(grid), len(grid[0])
    queue = deque([start])
    visited: Set[Tuple[int, int]] = {start}
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

    while queue:
        curr = queue.popleft()
        if curr == goal:
            path = reconstruct_path(parent, start, goal)
            yield PathfindingStep(curr, visited, set(queue), path, "BFS", is_complete=True)
            return

        for neighbor in get_neighbors(grid, curr[0], curr[1], height, width):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = curr
                queue.append(neighbor)

        yield PathfindingStep(curr, visited, set(queue), reconstruct_path(parent, start, curr), "BFS")

    yield PathfindingStep(start, visited, set(), [], "BFS", is_complete=True)


def step_dijkstra(grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]) -> Generator[PathfindingStep, None, None]:
    height, width = len(grid), len(grid[0])
    counter = 0
    pq = [(0.0, counter, start)]
    dist: Dict[Tuple[int, int], float] = {start: 0.0}
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    settled: Set[Tuple[int, int]] = set()

    while pq:
        d, _, curr = heapq.heappop(pq)
        if curr in settled:
            continue
        settled.add(curr)

        if curr == goal:
            path = reconstruct_path(parent, start, goal)
            frontier_set = {item[2] for item in pq}
            yield PathfindingStep(curr, settled, frontier_set, path, "Dijkstra", is_complete=True)
            return

        for neighbor in get_neighbors(grid, curr[0], curr[1], height, width):
            new_dist = d + 1.0
            if neighbor not in dist or new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                parent[neighbor] = curr
                counter += 1
                heapq.heappush(pq, (new_dist, counter, neighbor))

        frontier_set = {item[2] for item in pq}
        yield PathfindingStep(curr, settled, frontier_set, reconstruct_path(parent, start, curr), "Dijkstra")

    yield PathfindingStep(start, settled, set(), [], "Dijkstra", is_complete=True)


def step_astar(grid: List[List[int]],
               start: Tuple[int, int],
               goal: Tuple[int, int],
               heuristic_fn: Callable[[Tuple[int, int], Tuple[int, int]], float] = heuristic_manhattan,
               heuristic_name: str = "Manhattan") -> Generator[PathfindingStep, None, None]:
    height, width = len(grid), len(grid[0])
    counter = 0
    h_start = heuristic_fn(start, goal)
    pq = [(h_start, h_start, counter, start)]
    g_score: Dict[Tuple[int, int], float] = {start: 0.0}
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    closed_set: Set[Tuple[int, int]] = set()

    alg_label = f"A* ({heuristic_name})"

    while pq:
        f, h, _, curr = heapq.heappop(pq)
        if curr in closed_set:
            continue
        closed_set.add(curr)

        if curr == goal:
            path = reconstruct_path(parent, start, goal)
            frontier_set = {item[3] for item in pq}
            yield PathfindingStep(curr, closed_set, frontier_set, path, alg_label, is_complete=True)
            return

        curr_g = g_score[curr]
        for neighbor in get_neighbors(grid, curr[0], curr[1], height, width):
            tentative_g = curr_g + 1.0
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                h_val = heuristic_fn(neighbor, goal)
                f_val = tentative_g + h_val
                parent[neighbor] = curr
                counter += 1
                heapq.heappush(pq, (f_val, h_val, counter, neighbor))

        frontier_set = {item[3] for item in pq}
        yield PathfindingStep(curr, closed_set, frontier_set, reconstruct_path(parent, start, curr), alg_label)

    yield PathfindingStep(start, closed_set, set(), [], alg_label, is_complete=True)


PATHFINDERS = {
    "BFS": solve_bfs,
    "Dijkstra": solve_dijkstra,
    "A* (Manhattan)": lambda g, s, e: solve_astar(g, s, e, heuristic_manhattan, "Manhattan"),
    "A* (Euclidean)": lambda g, s, e: solve_astar(g, s, e, heuristic_euclidean, "Euclidean"),
    "Greedy Best-First": solve_greedy_bfs
}

ANIMATED_PATHFINDERS = {
    "BFS": step_bfs,
    "Dijkstra": step_dijkstra,
    "A* (Manhattan)": lambda g, s, e: step_astar(g, s, e, heuristic_manhattan, "Manhattan"),
    "A* (Euclidean)": lambda g, s, e: step_astar(g, s, e, heuristic_euclidean, "Euclidean")
}
