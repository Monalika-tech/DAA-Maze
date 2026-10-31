"""
DAA Project: Interactive 2D Maze & Pathfinding Visualizer + Local 2-Player Race
=============================================================================
Tech Stack: Pygame, Python 3.x
Features:
- Visualizer Mode:
    * Real-time animated maze generation (DFS, Prim's, Kruskal's)
    * Real-time animated pathfinding (BFS, Dijkstra, A* Manhattan, A* Euclidean)
    * Open/Closed set visualization & live DAA metric tracking
- Local Multiplayer Race Mode:
    * Player 1 (Red Runner): W, A, S, D
    * Player 2 (Blue Runner): Arrow Keys
    * Dynamic trail rendering, obstacle collisions, distance meters & victory screen
"""

import os
import sys
import time
from typing import Tuple, List, Set, Optional, Dict

import pygame

# Import algorithmic modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generator import (
    Maze, WALL, PATH,
    generate_dfs, generate_prims, generate_kruskals,
    generate_dfs_animated, generate_prims_animated, generate_kruskals_animated
)
from pathfinder import (
    solve_bfs, solve_dijkstra, solve_astar,
    step_bfs, step_dijkstra, step_astar,
    heuristic_manhattan, heuristic_euclidean,
    PathfindingMetrics, PathfindingStep
)

# Initialize Pygame
pygame.init()
pygame.font.init()

# --- COLOR PALETTE (Catppuccin / Modern Dark Theme) ---
COLOR_BG = (24, 24, 37)           # #181825 Base dark
COLOR_PANEL = (30, 30, 46)        # #1e1e2e Panel background
COLOR_CARD = (49, 50, 68)         # #313244 Card surface
COLOR_CARD_BORDER = (69, 71, 90)  # #45475a
COLOR_TEXT_MAIN = (205, 214, 244) # #cdd6f4
COLOR_TEXT_MUTED = (166, 173, 200)# #a6adc8
COLOR_ACCENT = (137, 180, 250)    # #89b4fa Soft blue

COLOR_WALL = (17, 17, 27)         # #11111b Deep black walls
COLOR_PASSAGE = (245, 245, 247)   # Crisp white walkable corridor
COLOR_START = (6, 214, 160)       # #06d6a0 Vibrant green start
COLOR_GOAL = (255, 209, 102)      # #ffd166 Gold trophy goal

COLOR_VISITED = (98, 114, 164)    # Purple/slate closed set
COLOR_FRONTIER = (255, 121, 198)  # Hot pink open set
COLOR_PATH = (80, 250, 123)       # Neon emerald shortest path
COLOR_GEN_CURR = (255, 85, 85)    # Red current generator cell
COLOR_GEN_STACK = (189, 147, 249) # Light purple backtracker stack

# Multiplayer Player Colors
COLOR_P1 = (239, 71, 111)         # Coral Red
COLOR_P1_TRAIL = (239, 71, 111, 100)
COLOR_P2 = (17, 138, 178)         # Electric Blue
COLOR_P2_TRAIL = (17, 138, 178, 100)

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 760
FPS = 60


class Button:
    """Interactive GUI button with hover & selected states."""
    def __init__(self, rect: pygame.Rect, text: str, callback, font,
                 tag: Optional[str] = None, is_toggle: bool = False):
        self.rect = rect
        self.text = text
        self.callback = callback
        self.font = font
        self.tag = tag
        self.is_toggle = is_toggle
        self.is_selected = False
        self.is_hovered = False

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.callback(self)

    def draw(self, surface: pygame.Surface):
        if self.is_selected:
            bg_color = (137, 180, 250)
            text_color = (17, 17, 27)
            border_color = (180, 190, 254)
        elif self.is_hovered:
            bg_color = (69, 71, 90)
            text_color = (255, 255, 255)
            border_color = (137, 180, 250)
        else:
            bg_color = (49, 50, 68)
            text_color = COLOR_TEXT_MAIN
            border_color = COLOR_CARD_BORDER

        pygame.draw.rect(surface, bg_color, self.rect, border_radius=6)
        pygame.draw.rect(surface, border_color, self.rect, width=1, border_radius=6)

        text_surf = self.font.render(self.text, True, text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)


class Player:
    """Multiplayer racer representation."""
    def __init__(self, name: str, start_pos: Tuple[int, int], color: Tuple[int, int, int]):
        self.name = name
        self.r, self.c = start_pos
        self.color = color
        self.trail: List[Tuple[int, int]] = [start_pos]
        self.max_trail = 50
        self.steps = 0
        self.last_move_time = 0.0

    def move(self, dr: int, dc: int, grid: List[List[int]]) -> bool:
        nr, nc = self.r + dr, self.c + dc
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] == PATH:
            self.r, self.c = nr, nc
            self.steps += 1
            self.trail.append((nr, nc))
            if len(self.trail) > self.max_trail:
                self.trail.pop(0)
            return True
        return False

    def reset(self, start_pos: Tuple[int, int]):
        self.r, self.c = start_pos
        self.trail = [start_pos]
        self.steps = 0
        self.last_move_time = 0.0


class DAAApp:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("DAA Project: Maze Generation, Pathfinding & Performance Analysis")
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_title = pygame.font.SysFont("segoe ui", 20, bold=True)
        self.font_section = pygame.font.SysFont("segoe ui", 15, bold=True)
        self.font_btn = pygame.font.SysFont("segoe ui", 13)
        self.font_hud = pygame.font.SysFont("consolas", 12)
        self.font_banner = pygame.font.SysFont("segoe ui", 32, bold=True)

        # Mode: 'VISUALIZER' or 'MULTIPLAYER'
        self.mode = "VISUALIZER"

        # Grid configuration (rooms vertically & horizontally)
        # 15x15 rooms -> 31x31 matrix; 25x25 rooms -> 51x51 matrix
        self.grid_options = [(10, "Small (21x21)"), (18, "Medium (37x37)"), (28, "Large (57x57)")]
        self.selected_grid_idx = 1
        self.cell_scale = self.grid_options[self.selected_grid_idx][0]
        self.maze = Maze(self.cell_scale, self.cell_scale)

        # Algorithm Selections
        self.selected_generator = "DFS (Recursive Backtracker)"
        self.selected_pathfinder = "A* (Manhattan)"

        # Speeds: steps executed per frame (1, 5, 25, or 9999 for instant)
        self.speed_options = [(1, "1x"), (5, "5x"), (25, "25x"), (9999, "Max")]
        self.speed_idx = 2
        self.steps_per_frame = self.speed_options[self.speed_idx][0]

        # Generator & Pathfinder State Machine
        self.state = "IDLE"  # "IDLE", "GENERATING", "SOLVING", "SOLVED", "RACING", "FINISHED"
        self.gen_step_iter = None
        self.gen_curr_step = None

        self.pf_step_iter = None
        self.pf_curr_step = None
        self.final_metrics: Optional[PathfindingMetrics] = None

        # Multiplayer state
        self.p1 = Player("Player 1 (WASD)", self.maze.start, COLOR_P1)
        self.p2 = Player("Player 2 (Arrows)", self.maze.start, COLOR_P2)
        self.race_start_time = 0.0
        self.race_elapsed_time = 0.0
        self.winner: Optional[str] = None
        self.show_hint_path = False
        self.hint_path: List[Tuple[int, int]] = []

        # UI Setup
        self.buttons: List[Button] = []
        self._build_ui()

        # Build initial maze
        self.generate_maze_instant()

    def _build_ui(self):
        self.buttons.clear()
        panel_x = 880
        panel_w = WINDOW_WIDTH - panel_x - 20

        # Mode Toggle Buttons
        y = 15
        btn_w = (panel_w - 10) // 2
        self.btn_mode_vis = Button(
            pygame.Rect(panel_x, y, btn_w, 32), "Visualizer",
            lambda b: self.set_mode("VISUALIZER"), self.font_btn, tag="mode_vis"
        )
        self.btn_mode_race = Button(
            pygame.Rect(panel_x + btn_w + 10, y, btn_w, 32), "Multiplayer",
            lambda b: self.set_mode("MULTIPLAYER"), self.font_btn, tag="mode_race"
        )
        self.buttons.extend([self.btn_mode_vis, self.btn_mode_race])
        self.update_mode_buttons()

        # Generator selection buttons
        y = 90
        gen_list = ["DFS (Recursive Backtracker)", "Prim's Algorithm", "Kruskal's Algorithm"]
        self.gen_buttons = []
        for g_name in gen_list:
            btn = Button(
                pygame.Rect(panel_x, y, panel_w, 26), g_name,
                lambda b, name=g_name: self.set_generator(name), self.font_btn, tag=f"gen_{g_name}"
            )
            self.gen_buttons.append(btn)
            self.buttons.append(btn)
            y += 30

        # Pathfinder selection buttons
        y = 230
        pf_list = ["BFS", "Dijkstra", "A* (Manhattan)", "A* (Euclidean)"]
        self.pf_buttons = []
        for p_name in pf_list:
            btn = Button(
                pygame.Rect(panel_x, y, panel_w, 26), p_name,
                lambda b, name=p_name: self.set_pathfinder(name), self.font_btn, tag=f"pf_{p_name}"
            )
            self.pf_buttons.append(btn)
            self.buttons.append(btn)
            y += 30

        # Grid Scale Switcher
        y = 395
        scale_btn_w = (panel_w - 20) // 3
        self.scale_buttons = []
        for idx, (_, label) in enumerate(self.grid_options):
            short_lbl = label.split()[0]
            btn = Button(
                pygame.Rect(panel_x + idx * (scale_btn_w + 10), y, scale_btn_w, 26), short_lbl,
                lambda b, i=idx: self.set_grid_scale(i), self.font_btn, tag=f"scale_{idx}"
            )
            self.scale_buttons.append(btn)
            self.buttons.append(btn)

        # Speed buttons
        y = 460
        spd_btn_w = (panel_w - 30) // 4
        self.speed_buttons = []
        for idx, (_, label) in enumerate(self.speed_options):
            btn = Button(
                pygame.Rect(panel_x + idx * (spd_btn_w + 10), y, spd_btn_w, 26), label,
                lambda b, i=idx: self.set_speed(i), self.font_btn, tag=f"speed_{idx}"
            )
            self.speed_buttons.append(btn)
            self.buttons.append(btn)

        # Action Buttons
        y = 530
        act_w = (panel_w - 10) // 2
        self.btn_gen_anim = Button(
            pygame.Rect(panel_x, y, act_w, 34), "▶ Animate Gen",
            lambda b: self.start_generation_animation(), self.font_btn
        )
        self.btn_solve_anim = Button(
            pygame.Rect(panel_x + act_w + 10, y, act_w, 34), "▶ Animate Path",
            lambda b: self.start_pathfinding_animation(), self.font_btn
        )
        self.buttons.extend([self.btn_gen_anim, self.btn_solve_anim])

        y = 575
        self.btn_instant_solve = Button(
            pygame.Rect(panel_x, y, act_w, 34), "⚡ Instant Solve",
            lambda b: self.solve_instant(), self.font_btn
        )
        self.btn_reset_maze = Button(
            pygame.Rect(panel_x + act_w + 10, y, act_w, 34), "🔄 New Maze",
            lambda b: self.generate_maze_instant(), self.font_btn
        )
        self.buttons.extend([self.btn_instant_solve, self.btn_reset_maze])

        self.update_active_buttons()

    def update_mode_buttons(self):
        self.btn_mode_vis.is_selected = (self.mode == "VISUALIZER")
        self.btn_mode_race.is_selected = (self.mode == "MULTIPLAYER")

    def update_active_buttons(self):
        for btn in self.gen_buttons:
            btn.is_selected = (btn.text == self.selected_generator)
        for btn in self.pf_buttons:
            btn.is_selected = (btn.text == self.selected_pathfinder)
        for idx, btn in enumerate(self.scale_buttons):
            btn.is_selected = (idx == self.selected_grid_idx)
        for idx, btn in enumerate(self.speed_buttons):
            btn.is_selected = (idx == self.speed_idx)

    def set_mode(self, mode: str):
        self.mode = mode
        self.update_mode_buttons()
        if mode == "MULTIPLAYER":
            self.reset_race()
        else:
            self.state = "IDLE"

    def set_generator(self, name: str):
        self.selected_generator = name
        self.update_active_buttons()

    def set_pathfinder(self, name: str):
        self.selected_pathfinder = name
        self.update_active_buttons()

    def set_grid_scale(self, idx: int):
        self.selected_grid_idx = idx
        self.cell_scale = self.grid_options[idx][0]
        self.maze = Maze(self.cell_scale, self.cell_scale)
        self.update_active_buttons()
        self.generate_maze_instant()
        if self.mode == "MULTIPLAYER":
            self.reset_race()

    def set_speed(self, idx: int):
        self.speed_idx = idx
        self.steps_per_frame = self.speed_options[idx][0]
        self.update_active_buttons()

    # --- Maze Generation & Solving Control ---

    def generate_maze_instant(self):
        self.state = "IDLE"
        self.gen_curr_step = None
        self.pf_curr_step = None
        self.final_metrics = None

        if self.selected_generator == "DFS (Recursive Backtracker)":
            generate_dfs(self.maze)
        elif self.selected_generator == "Prim's Algorithm":
            generate_prims(self.maze)
        else:
            generate_kruskals(self.maze)

        if self.mode == "MULTIPLAYER":
            self.reset_race()

    def start_generation_animation(self):
        self.state = "GENERATING"
        self.pf_curr_step = None
        self.final_metrics = None

        if self.selected_generator == "DFS (Recursive Backtracker)":
            self.gen_step_iter = generate_dfs_animated(self.maze)
        elif self.selected_generator == "Prim's Algorithm":
            self.gen_step_iter = generate_prims_animated(self.maze)
        else:
            self.gen_step_iter = generate_kruskals_animated(self.maze)

    def start_pathfinding_animation(self):
        if self.state == "GENERATING":
            return
        self.state = "SOLVING"
        self.final_metrics = None

        if self.selected_pathfinder == "BFS":
            self.pf_step_iter = step_bfs(self.maze.grid, self.maze.start, self.maze.goal)
        elif self.selected_pathfinder == "Dijkstra":
            self.pf_step_iter = step_dijkstra(self.maze.grid, self.maze.start, self.maze.goal)
        elif self.selected_pathfinder == "A* (Manhattan)":
            self.pf_step_iter = step_astar(self.maze.grid, self.maze.start, self.maze.goal, heuristic_manhattan, "Manhattan")
        else:
            self.pf_step_iter = step_astar(self.maze.grid, self.maze.start, self.maze.goal, heuristic_euclidean, "Euclidean")

    def solve_instant(self):
        if self.state == "GENERATING":
            return
        t0 = time.perf_counter_ns()
        if self.selected_pathfinder == "BFS":
            res = solve_bfs(self.maze.grid, self.maze.start, self.maze.goal)
        elif self.selected_pathfinder == "Dijkstra":
            res = solve_dijkstra(self.maze.grid, self.maze.start, self.maze.goal)
        elif self.selected_pathfinder == "A* (Manhattan)":
            res = solve_astar(self.maze.grid, self.maze.start, self.maze.goal, heuristic_manhattan, "Manhattan")
        else:
            res = solve_astar(self.maze.grid, self.maze.start, self.maze.goal, heuristic_euclidean, "Euclidean")

        self.final_metrics = res
        self.pf_curr_step = PathfindingStep(
            current=self.maze.goal,
            visited=set(res.visited_order),
            frontier=set(),
            path=res.path,
            algorithm_name=self.selected_pathfinder,
            is_complete=True,
            metrics=res
        )
        self.state = "SOLVED"

    def reset_race(self):
        self.p1.reset(self.maze.start)
        self.p2.reset(self.maze.start)
        self.race_start_time = time.time()
        self.race_elapsed_time = 0.0
        self.winner = None
        self.state = "RACING"
        # Precompute hint path using A*
        astar_res = solve_astar(self.maze.grid, self.maze.start, self.maze.goal)
        self.hint_path = astar_res.path

    # --- Game Loop & Updates ---

    def update(self):
        # Step through animated maze generator
        if self.state == "GENERATING" and self.gen_step_iter:
            steps = min(self.steps_per_frame, 500) if self.steps_per_frame < 9999 else 9999
            for _ in range(steps):
                try:
                    step = next(self.gen_step_iter)
                    self.gen_curr_step = step
                    if step.is_complete:
                        self.state = "IDLE"
                        self.gen_step_iter = None
                        break
                except StopIteration:
                    self.state = "IDLE"
                    self.gen_step_iter = None
                    break

        # Step through animated pathfinder
        elif self.state == "SOLVING" and self.pf_step_iter:
            steps = min(self.steps_per_frame, 300) if self.steps_per_frame < 9999 else 9999
            for _ in range(steps):
                try:
                    step = next(self.pf_step_iter)
                    self.pf_curr_step = step
                    if step.is_complete:
                        self.state = "SOLVED"
                        self.pf_step_iter = None
                        break
                except StopIteration:
                    self.state = "SOLVED"
                    self.pf_step_iter = None
                    break

        # Multiplayer Race Updates
        elif self.mode == "MULTIPLAYER" and self.state == "RACING":
            self.race_elapsed_time = time.time() - self.race_start_time
            now = time.time()
            keys = pygame.key.get_pressed()
            move_cooldown = 0.07  # Cooldown between rapid keystrokes

            # Player 1 Movement (WASD)
            if now - self.p1.last_move_time > move_cooldown:
                moved = False
                if keys[pygame.K_w]:
                    moved = self.p1.move(-1, 0, self.maze.grid)
                elif keys[pygame.K_s]:
                    moved = self.p1.move(1, 0, self.maze.grid)
                elif keys[pygame.K_a]:
                    moved = self.p1.move(0, -1, self.maze.grid)
                elif keys[pygame.K_d]:
                    moved = self.p1.move(0, 1, self.maze.grid)
                if moved:
                    self.p1.last_move_time = now

            # Player 2 Movement (Arrow Keys)
            if now - self.p2.last_move_time > move_cooldown:
                moved = False
                if keys[pygame.K_UP]:
                    moved = self.p2.move(-1, 0, self.maze.grid)
                elif keys[pygame.K_DOWN]:
                    moved = self.p2.move(1, 0, self.maze.grid)
                elif keys[pygame.K_LEFT]:
                    moved = self.p2.move(0, -1, self.maze.grid)
                elif keys[pygame.K_RIGHT]:
                    moved = self.p2.move(0, 1, self.maze.grid)
                if moved:
                    self.p2.last_move_time = now

            # Win Detection
            if (self.p1.r, self.p1.c) == self.maze.goal:
                self.winner = self.p1.name
                self.state = "FINISHED"
            elif (self.p2.r, self.p2.c) == self.maze.goal:
                self.winner = self.p2.name
                self.state = "FINISHED"

    # --- Rendering Engine ---

    def draw(self):
        self.screen.fill(COLOR_BG)

        # Draw Maze Viewport on Left
        self.draw_maze_viewport()

        # Draw Side Control Panel on Right
        self.draw_side_panel()

        # Draw Victory Banner if Race Finished
        if self.mode == "MULTIPLAYER" and self.state == "FINISHED" and self.winner:
            self.draw_victory_modal()

        pygame.display.flip()

    def draw_maze_viewport(self):
        """Render the maze grid with responsive scaling and smooth offset centering."""
        viewport_w = 840
        viewport_h = WINDOW_HEIGHT - 40
        vx, vy = 20, 20

        # Background box for viewport
        viewport_rect = pygame.Rect(vx, vy, viewport_w, viewport_h)
        pygame.draw.rect(self.screen, (17, 17, 27), viewport_rect, border_radius=8)
        pygame.draw.rect(self.screen, COLOR_CARD_BORDER, viewport_rect, width=1, border_radius=8)

        # Compute cell dimension
        grid_rows = self.maze.height
        grid_cols = self.maze.width
        cell_size = min((viewport_w - 20) / grid_cols, (viewport_h - 20) / grid_rows)
        maze_pixel_w = cell_size * grid_cols
        maze_pixel_h = cell_size * grid_rows
        offset_x = vx + (viewport_w - maze_pixel_w) / 2
        offset_y = vy + (viewport_h - maze_pixel_h) / 2

        # Fast lookup sets
        visited_set = self.pf_curr_step.visited if self.pf_curr_step else set()
        frontier_set = self.pf_curr_step.frontier if self.pf_curr_step else set()
        path_set = set(self.pf_curr_step.path) if self.pf_curr_step else set()

        # Draw Cells
        for r in range(grid_rows):
            for c in range(grid_cols):
                rect = pygame.Rect(offset_x + c * cell_size, offset_y + r * cell_size,
                                   max(1, cell_size), max(1, cell_size))

                # Walls vs Passages
                if self.maze.grid[r][c] == WALL:
                    pygame.draw.rect(self.screen, COLOR_WALL, rect)
                else:
                    # Default corridor color
                    cell_color = COLOR_PASSAGE

                    # Pathfinding Overlays
                    if (r, c) in path_set:
                        cell_color = COLOR_PATH
                    elif (r, c) in frontier_set:
                        cell_color = COLOR_FRONTIER
                    elif (r, c) in visited_set:
                        cell_color = COLOR_VISITED

                    # Generator active stack overlay
                    if self.state == "GENERATING" and self.gen_curr_step:
                        if (r, c) == self.gen_curr_step.current_cell:
                            cell_color = COLOR_GEN_CURR
                        elif (r, c) in self.gen_curr_step.frontier_or_stack:
                            cell_color = COLOR_GEN_STACK

                    pygame.draw.rect(self.screen, cell_color, rect)

        # Highlight Start and Goal
        sr, sc = self.maze.start
        gr, gc = self.maze.goal
        s_rect = pygame.Rect(offset_x + sc * cell_size, offset_y + sr * cell_size, cell_size, cell_size)
        g_rect = pygame.Rect(offset_x + gc * cell_size, offset_y + gr * cell_size, cell_size, cell_size)
        pygame.draw.rect(self.screen, COLOR_START, s_rect, border_radius=3)
        pygame.draw.rect(self.screen, COLOR_GOAL, g_rect, border_radius=3)

        # Multiplayer Render: Hint path, Player trails & Avatars
        if self.mode == "MULTIPLAYER":
            # Hint path overlay if toggled
            if self.show_hint_path and self.hint_path:
                for hr, hc in self.hint_path:
                    h_rect = pygame.Rect(offset_x + hc * cell_size + cell_size * 0.25,
                                         offset_y + hr * cell_size + cell_size * 0.25,
                                         cell_size * 0.5, cell_size * 0.5)
                    pygame.draw.rect(self.screen, (255, 230, 109, 150), h_rect)

            # Player 1 Trail & Avatar
            for tr, tc in self.p1.trail:
                t_rect = pygame.Rect(offset_x + tc * cell_size + cell_size * 0.3,
                                     offset_y + tr * cell_size + cell_size * 0.3,
                                     cell_size * 0.4, cell_size * 0.4)
                pygame.draw.rect(self.screen, (239, 71, 111), t_rect, border_radius=2)
            p1_rect = pygame.Rect(offset_x + self.p1.c * cell_size + 1,
                                  offset_y + self.p1.r * cell_size + 1,
                                  max(2, cell_size - 2), max(2, cell_size - 2))
            pygame.draw.rect(self.screen, COLOR_P1, p1_rect, border_radius=4)

            # Player 2 Trail & Avatar
            for tr, tc in self.p2.trail:
                t_rect = pygame.Rect(offset_x + tc * cell_size + cell_size * 0.3,
                                     offset_y + tr * cell_size + cell_size * 0.3,
                                     cell_size * 0.4, cell_size * 0.4)
                pygame.draw.rect(self.screen, (17, 138, 178), t_rect, border_radius=2)
            p2_rect = pygame.Rect(offset_x + self.p2.c * cell_size + 1,
                                  offset_y + self.p2.r * cell_size + 1,
                                  max(2, cell_size - 2), max(2, cell_size - 2))
            pygame.draw.rect(self.screen, COLOR_P2, p2_rect, border_radius=4)

    def draw_side_panel(self):
        panel_x = 880
        panel_w = WINDOW_WIDTH - panel_x - 20

        # Section Labels
        self.screen.blit(self.font_section.render("1. MAZE GENERATION ALGORITHM", True, COLOR_ACCENT), (panel_x, 65))
        self.screen.blit(self.font_section.render("2. PATHFINDING SEARCH ALGORITHM", True, COLOR_ACCENT), (panel_x, 205))
        self.screen.blit(self.font_section.render("3. GRID SCALE & MATRIX SIZE", True, COLOR_ACCENT), (panel_x, 370))
        self.screen.blit(self.font_section.render("4. ANIMATION SPEED", True, COLOR_ACCENT), (panel_x, 435))
        self.screen.blit(self.font_section.render("5. CONTROLS & ACTIONS", True, COLOR_ACCENT), (panel_x, 505))

        # Draw Buttons
        for btn in self.buttons:
            btn.draw(self.screen)

        # Bottom DAA Metrics / HUD Card
        hud_rect = pygame.Rect(panel_x, 620, panel_w, 120)
        pygame.draw.rect(self.screen, COLOR_CARD, hud_rect, border_radius=8)
        pygame.draw.rect(self.screen, COLOR_CARD_BORDER, hud_rect, width=1, border_radius=8)

        if self.mode == "VISUALIZER":
            self.draw_visualizer_hud(panel_x + 12, 628)
        else:
            self.draw_multiplayer_hud(panel_x + 12, 628)

    def draw_visualizer_hud(self, x: int, y: int):
        status_color = (80, 250, 123) if self.state in ("IDLE", "SOLVED") else (255, 184, 108)
        self.screen.blit(self.font_hud.render(f"STATUS       : {self.state}", True, status_color), (x, y))

        dim_str = f"{self.maze.height}x{self.maze.width} ({self.maze.cell_rows * self.maze.cell_cols} rooms)"
        self.screen.blit(self.font_hud.render(f"DIMENSIONS   : {dim_str}", True, COLOR_TEXT_MAIN), (x, y + 18))

        nodes_exp = len(self.pf_curr_step.visited) if self.pf_curr_step else 0
        frontier_len = len(self.pf_curr_step.frontier) if self.pf_curr_step else 0
        self.screen.blit(self.font_hud.render(f"NODES VISITED: {nodes_exp:<6} | OPEN SET: {frontier_len}", True, COLOR_TEXT_MAIN), (x, y + 36))

        path_len = len(self.pf_curr_step.path) if self.pf_curr_step else 0
        self.screen.blit(self.font_hud.render(f"PATH LENGTH  : {path_len} steps", True, COLOR_PATH), (x, y + 54))

        if self.final_metrics:
            mem_kb = self.final_metrics.peak_memory_bytes / 1024.0
            t_ms = self.final_metrics.execution_time_us / 1000.0
            self.screen.blit(self.font_hud.render(f"EXEC TIME    : {t_ms:.2f} ms | RAM: {mem_kb:.1f} KB", True, (255, 209, 102)), (x, y + 72))
        else:
            self.screen.blit(self.font_hud.render("EXEC TIME    : [Run Instant Solve to Profile]", True, COLOR_TEXT_MUTED), (x, y + 72))

        self.screen.blit(self.font_hud.render("KEYS: [G] Gen | [P] Path | [M] Mode | [R] Reset", True, COLOR_ACCENT), (x, y + 90))

    def draw_multiplayer_hud(self, x: int, y: int):
        self.screen.blit(self.font_hud.render(f"RACE TIME: {self.race_elapsed_time:.1f}s | PRESS [H] FOR HINT", True, (255, 209, 102)), (x, y))

        # Distances to goal
        gr, gc = self.maze.goal
        d1 = abs(self.p1.r - gr) + abs(self.p1.c - gc)
        d2 = abs(self.p2.r - gr) + abs(self.p2.c - gc)

        self.screen.blit(self.font_hud.render(f"P1 (RED - WASD)   : Dist={d1} | Steps={self.p1.steps}", True, COLOR_P1), (x, y + 25))
        self.screen.blit(self.font_hud.render(f"P2 (BLUE - ARROWS): Dist={d2} | Steps={self.p2.steps}", True, COLOR_P2), (x, y + 48))
        self.screen.blit(self.font_hud.render(f"HINT TOGGLE: {'ON' if self.show_hint_path else 'OFF'} (A* Path Guide)", True, COLOR_TEXT_MUTED), (x, y + 72))
        self.screen.blit(self.font_hud.render("KEYS: [SPACE] Rematch | [R] New Maze | [M] Mode", True, COLOR_ACCENT), (x, y + 92))

    def draw_victory_modal(self):
        """Draw celebration overlay when a player wins."""
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        card_rect = pygame.Rect(WINDOW_WIDTH // 2 - 250, WINDOW_HEIGHT // 2 - 120, 500, 240)
        pygame.draw.rect(self.screen, COLOR_CARD, card_rect, border_radius=12)
        pygame.draw.rect(self.screen, COLOR_ACCENT, card_rect, width=2, border_radius=12)

        win_color = COLOR_P1 if "Player 1" in self.winner else COLOR_P2
        win_surf = self.font_banner.render(f"🏆 {self.winner} WINS!", True, win_color)
        win_rect = win_surf.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 50))
        self.screen.blit(win_surf, win_rect)

        time_surf = self.font_title.render(f"Winning Race Time: {self.race_elapsed_time:.2f} seconds", True, COLOR_TEXT_MAIN)
        time_rect = time_surf.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 5))
        self.screen.blit(time_surf, time_rect)

        sub_surf = self.font_btn.render("Press SPACEBAR for Rematch  |  Press R for Fresh Maze", True, COLOR_TEXT_MUTED)
        sub_rect = sub_surf.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 65))
        self.screen.blit(sub_surf, sub_rect)

    # --- Main Event Loop ---

    def run(self):
        running = True
        while running:
            self.clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_m:
                        self.set_mode("MULTIPLAYER" if self.mode == "VISUALIZER" else "VISUALIZER")
                    elif event.key == pygame.K_g:
                        self.start_generation_animation()
                    elif event.key == pygame.K_p:
                        self.start_pathfinding_animation()
                    elif event.key == pygame.K_r:
                        self.generate_maze_instant()
                    elif event.key == pygame.K_h and self.mode == "MULTIPLAYER":
                        self.show_hint_path = not self.show_hint_path
                    elif event.key == pygame.K_SPACE and self.mode == "MULTIPLAYER":
                        self.reset_race()

                # Dispatch events to UI buttons
                for btn in self.buttons:
                    btn.handle_event(event)

            self.update()
            self.draw()

        pygame.quit()


if __name__ == "__main__":
    app = DAAApp()
    app.run()
