# 2D DAA Maze Generation, Pathfinding & Performance Analysis

An academic **Design and Analysis of Algorithms (DAA)** project featuring rigorous algorithmic implementations, an empirical performance analysis engine with statistical profiling (`matplotlib`, `tracemalloc`, `time.perf_counter_ns`), and a real-time **2D Pygame visualizer and local 2-player racing game**.

---

## 📁 Project Architecture & File Hierarchy

```
daa-maze-benchmark/
│
├── generator.py            # Procedural Maze Generation (DFS Backtracker, Prim's, Kruskal's with DSU)
├── pathfinder.py           # Pathfinding Search Engine (BFS, Dijkstra, A* Manhattan, A* Euclidean, Greedy)
├── benchmark.py            # Empirical Benchmark Framework (Automated grid profiling & Matplotlib generator)
├── main.py                 # Interactive 2D Pygame Visualizer & Local 2-Player Race Mode
├── README.md               # Setup, User Guide & Controls
│
└── benchmark_output/       # Generated Empirical Benchmark Results
    ├── benchmark_generation.png          # Generation Time & RAM Profiling
    ├── benchmark_pathfinding.png         # Pathfinding Pruning, Nodes, Memory & Time
    ├── benchmark_summary_dashboard.png   # Comparative High-Resolution Bar Dashboard
    ├── benchmark_results.csv             # Raw Empirical Data Export
    └── DAA_BENCHMARK_SUMMARY.md          # Markdown Table for Academic Project Reports
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Dependencies
Ensure Python 3.10+ is installed. Install required packages:
```bash
pip install pygame matplotlib numpy
```

### 2. Launching the Interactive Pygame Prototype
```bash
python main.py
```
* **Visualizer Mode**: Watch mazes generate in real time, inspect open/closed sets step-by-step, or run instant solves.
* **Local Multiplayer Mode**: 2 players race in real-time through the maze to reach the Golden Trophy!

### 3. Running the Empirical Benchmark Suite
To generate high-resolution comparison charts and CSV reports:
```bash
# Standard Academic Benchmark (Grids up to 201x201, 3 trials per scale)
python benchmark.py

# Fast Quick Benchmark (For rapid verification)
python benchmark.py --quick
```

---

## 🎮 Pygame Controls & Keybindings

### Global Controls
| Key | Function |
| :--- | :--- |
| `M` | Toggle between **Visualizer Mode** and **Local Multiplayer Race Mode** |
| `G` | Trigger Real-Time Maze Generation Animation |
| `P` | Trigger Step-by-Step Pathfinding Search Animation |
| `R` | Instantly Generate a New Maze (or Reset Grid) |
| `ESC` | Exit Application |

### Local Multiplayer Race Mode
* **Player 1 (Coral Red Runner)**: `W`, `A`, `S`, `D`
* **Player 2 (Electric Blue Runner)**: `Up`, `Down`, `Left`, `Right` Arrow Keys
* `H`: Toggle A* Optimal Path Hint Overlay (if a racer is stuck)
* `SPACEBAR`: Trigger Rematch on the current maze

---

## 🔬 Theoretical vs. Empirical Complexity Reference

| Module | Algorithm | Time Complexity (Worst) | Time Complexity (Empirical Average) | Space Complexity | Optimality Guarantee |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Generator** | **Randomized DFS** | $O(\|V\|)$ | $\Theta(\|V\|)$ | $O(\|V\|)$ (Stack) | Spanning Tree (High River) |
| **Generator** | **Randomized Prim's** | $O(\|E\|) = O(\|V\|)$ | $\Theta(\|V\|)$ | $O(\|V\|)$ (Frontier) | Spanning Tree (Radial/Bushy) |
| **Generator** | **Randomized Kruskal's** | $O(\|E\| \cdot \alpha(\|V\|))$ | $\Theta(\|E\|)$ | $O(\|V\|)$ (DSU Arrays) | Spanning Tree (Uniform) |
| **Pathfinder** | **Breadth-First Search (BFS)**| $O(\|V\| + \|E\|)$ | $\Theta(\|V_{explored}\|)$ | $O(\|V\|)$ (FIFO Queue) | **Optimal** (Shortest Path) |
| **Pathfinder** | **Dijkstra's Algorithm** | $O((\|V\| + \|E\|) \log \|V\|)$| $\Theta(\|V_{explored}\| \log \|V\|)$| $O(\|V\|)$ (Min-Heap) | **Optimal** (Shortest Path) |
| **Pathfinder** | **A\* (Manhattan Heuristic)** | $O(\|E\| \log \|V\|)$ | $\ll \text{BFS}$ (Prunes 40-75%) | $O(\|V\|)$ (Priority Queue) | **Optimal** ($h$ is admissible & consistent) |
| **Pathfinder** | **A\* (Euclidean Heuristic)** | $O(\|E\| \log \|V\|)$ | Explores slightly more than Manhattan | $O(\|V\|)$ (Priority Queue) | **Optimal** ($h \le h_{Manhattan}$) |
| **Pathfinder** | **Greedy Best-First Search** | $O(\|E\| \log \|V\|)$ | Blazing Fast (Fewest Nodes) | $O(\|V\|)$ | **Sub-Optimal** (No $g(n)$ cost term) |

---

## 📊 Key Academic DAA Observations

1. **BFS vs. Dijkstra Equivalence on Unit Weights**:
   When edge weights are uniform ($w(e) = 1$), BFS and Dijkstra expand the exact same closed set of nodes. However, Dijkstra runs ~1.5x to 2x slower due to the $O(\log |V|)$ overhead of binary heap insertions and extractions compared to BFS's $O(1)$ deque operations.

2. **Heuristic Dominance ($A^*_{\text{Manhattan}}$ vs. $A^*_{\text{Euclidean}}$)**:
   In 4-connected grid lattices, $h_{\text{Manhattan}}(u) \ge h_{\text{Euclidean}}(u)$ for all vertices $u$. Because Manhattan distance is strictly closer to the true shortest path distance ($h^*$), it dominates Euclidean distance, pruning more dead-ends and visiting fewer nodes while guaranteeing shortest-path optimality.

3. **Memory Trade-offs in Maze Generation**:
   * DFS maintains only the active backtracking branch in memory ($O(\text{depth})$).
   * Kruskal's pre-allocates and shuffles all $2 \times R \times C$ edges upfront, resulting in higher peak memory usage during generation, but producing an unbiased spanning tree with no starting-point bias.
