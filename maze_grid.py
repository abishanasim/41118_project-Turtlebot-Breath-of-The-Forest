"""
5x5 grid maze generator for TurtleBot3 forest navigation.

Grid labelling
--------------
Columns A-E from left  to right  (col index 0-4)
Rows    1-5 from top   to bottom (row index 0-4)

    A1  B1  C1  D1  E1   <- row 0  (top,    high y in PyBullet)
    A2  B2  C2  D2  E2   <- row 1
    A3  B3  C3  D3  E3   <- row 2
    A4  B4  C4  D4  E4   <- row 3
    A5  B5  C5  D5  E5   <- row 4  (bottom, low  y in PyBullet)
    ^                ^
  col 0            col 4

Start : A5  (row=4, col=0) -- bottom-left cell centre  (-4, -4)
Goal  : E1  (row=0, col=4) -- top-right  cell centre   ( 4,  4)
"""

import random
import sys

# ── Grid dimensions ───────────────────────────────────────────────────
GRID_ROWS  = 5
GRID_COLS  = 5
CELL_SIZE  = 2.0      # metres per cell side
WORLD_ORIG = -5.0     # world x/y coordinate of the left / bottom boundary
#   world spans x in [-5, 5],  y in [-5, 5]

# ── Wall geometry ─────────────────────────────────────────────────────
# Every internal wall is a single PyBullet GEOM_BOX spanning exactly one
# cell edge (vertex to vertex).  WALL_HALF_LEN > CELL_SIZE/2 so adjacent
# segments overlap at every grid vertex -- no shortcut gaps.
WALL_HALF_LEN = 1.1   # wall spans CELL_SIZE + 0.2 m -> 0.1 m overlap each end
WALL_HALF_THK = 0.12  # wall thickness = 0.24 m
WALL_HEIGHT   = 1.0   # metres

# ── Named grid positions ──────────────────────────────────────────────
START_CELL = (4, 0)   # A5 -- bottom-left
GOAL_CELL  = (0, 4)   # E1 -- top-right


# ──────────────────────────────────────────────────────────────────────
# Coordinate helpers
# ──────────────────────────────────────────────────────────────────────

def cell_center(row, col):
    """Return world (x, y) at the centre of grid cell (row, col).

    row 0 is the topmost row (highest y).
    col 0 is the leftmost column.

    Examples::

        cell_center(4, 0) -> (-4.0, -4.0)   # A5, start
        cell_center(0, 4) -> ( 4.0,  4.0)   # E1, goal
    """
    x = WORLD_ORIG + (col + 0.5) * CELL_SIZE   # -4 + 2*col
    y = (-WORLD_ORIG) - (row + 0.5) * CELL_SIZE # 4  - 2*row
    return x, y


def world_to_cell(wx, wy):
    """Convert world (x, y) to grid (row, col).  Clamps to valid range.

    Examples::

        world_to_cell(-4.0, -4.0) -> (4, 0)   # A5
        world_to_cell( 4.0,  4.0) -> (0, 4)   # E1
    """
    col = int((wx - WORLD_ORIG) / CELL_SIZE)
    row = int((-WORLD_ORIG - wy) / CELL_SIZE)
    col = max(0, min(GRID_COLS - 1, col))
    row = max(0, min(GRID_ROWS - 1, row))
    return row, col


def cell_label(row, col):
    """Human-readable label such as 'A5' (start) or 'E1' (goal)."""
    return chr(ord('A') + col) + str(row + 1)


# Pre-computed key world positions
_sx, _sy = cell_center(*START_CELL)
_gx, _gy = cell_center(*GOAL_CELL)
START_POS = [_sx, _sy, 0.05]   # [x, y, z] -- robot spawn
GOAL_POS  = [_gx, _gy, 0.05]   # [x, y, z] -- goal zone centre


# ──────────────────────────────────────────────────────────────────────
# Maze generator
# ──────────────────────────────────────────────────────────────────────

def generate_5x5_maze(seed=42):
    """
    Generate a perfect 5x5 maze using recursive backtracking (DFS).

    Returns a list of wall-segment tuples::

        (center_x, center_y, half_x, half_y)

    Each tuple describes a PyBullet GEOM_BOX placed at world position
    (center_x, center_y, WALL_HEIGHT / 2) with the given half-extents.

    Wall geometry guarantees
    ------------------------
    * Every segment spans exactly one cell edge (vertex to vertex).
    * WALL_HALF_LEN = 1.1 > CELL_SIZE/2 = 1.0, so adjacent segments
      overlap by 0.1 m at every shared vertex -- no shortcut gaps.
    * The outer boundary is always fully closed.
    * Perfect-maze property: exactly one path between any two cells,
      so A5 can always reach E1 and every dead branch is a cul-de-sac.

    Internal wall coordinates
    -------------------------
    Horizontal wall between rows r and r+1 for column c:
        cy = 5 - (r+1)*2 = 3 - 2*r    (boundary y)
        cx = -4 + 2*c                  (column centre x)

    Vertical wall between cols c and c+1 for row r:
        cx = -5 + (c+1)*2 = -3 + 2*c  (boundary x)
        cy =  4 - 2*r                  (row centre y)
    """
    # Increase recursion limit for the 5x5 DFS carve (25 cells max depth)
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, 200))

    rng = random.Random(seed)

    # Direction encoding: 0=up, 1=right, 2=down, 3=left
    _DR  = (-1,  0,  1,  0)
    _DC  = ( 0,  1,  0, -1)
    _OPP = ( 2,  3,  0,  1)   # opposite direction index

    passages = [[set() for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
    visited  = [[False]      * GRID_COLS           for _ in range(GRID_ROWS)]

    def _carve(r, c):
        visited[r][c] = True
        dirs = list(range(4))
        rng.shuffle(dirs)
        for d in dirs:
            nr, nc = r + _DR[d], c + _DC[d]
            if (0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS
                    and not visited[nr][nc]):
                passages[r][c].add(d)
                passages[nr][nc].add(_OPP[d])
                _carve(nr, nc)

    _carve(*START_CELL)   # start carving from A5 (row=4, col=0)

    sys.setrecursionlimit(old_limit)

    segments = []

    # ── Outer boundary (always closed) ───────────────────────────
    for c in range(GRID_COLS):
        cx = WORLD_ORIG + (c + 0.5) * CELL_SIZE       # -4 + 2*c
        segments.append((cx,  5.0, WALL_HALF_LEN, WALL_HALF_THK))  # top
        segments.append((cx, -5.0, WALL_HALF_LEN, WALL_HALF_THK))  # bottom

    for r in range(GRID_ROWS):
        cy = (-WORLD_ORIG) - (r + 0.5) * CELL_SIZE    # 4 - 2*r
        segments.append((-5.0, cy, WALL_HALF_THK, WALL_HALF_LEN))  # left
        segments.append(( 5.0, cy, WALL_HALF_THK, WALL_HALF_LEN))  # right

    # ── Internal horizontal walls (between row r and row r+1) ────
    # A wall exists when there is NO down-passage (dir 2) from (r, c).
    # Wall y = 5 - (r+1)*2 = 3 - 2*r
    for r in range(GRID_ROWS - 1):
        for c in range(GRID_COLS):
            if 2 not in passages[r][c]:
                cx = WORLD_ORIG + (c + 0.5) * CELL_SIZE    # -4 + 2*c
                cy = (-WORLD_ORIG) - (r + 1) * CELL_SIZE   # 3  - 2*r
                segments.append((cx, cy, WALL_HALF_LEN, WALL_HALF_THK))

    # ── Internal vertical walls (between col c and col c+1) ──────
    # A wall exists when there is NO right-passage (dir 1) from (r, c).
    # Wall x = -5 + (c+1)*2 = -3 + 2*c
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS - 1):
            if 1 not in passages[r][c]:
                cx = WORLD_ORIG + (c + 1) * CELL_SIZE       # -3 + 2*c
                cy = (-WORLD_ORIG) - (r + 0.5) * CELL_SIZE  # 4  - 2*r
                segments.append((cx, cy, WALL_HALF_THK, WALL_HALF_LEN))

    return segments
