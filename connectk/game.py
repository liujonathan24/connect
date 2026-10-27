"""Connect-k with gravity: the rules.

The rules are the rules of Connect 4. Two players drop stones into the
columns of an upright board. A stone falls to the lowest empty cell of its
column. The first player to get k stones in a row (across, up, or along a
diagonal) wins. A full board with no winner is a draw.

    connect5   9 x 9 board, 5 in a row   (Part B)
    connect6  13 x 13 board, 6 in a row  (Part A)

The grid is a numpy array. grid[r, c] is 0 (empty), 1 (first player) or
2 (second player). Row 0 is the bottom row.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class GameSpec:
    name: str
    rows: int
    cols: int
    k: int


CONNECT4 = GameSpec("connect4", 6, 7, 4)
CONNECT5 = GameSpec("connect5", 9, 9, 5)
CONNECT6 = GameSpec("connect6", 13, 13, 6)
SPECS = {s.name: s for s in (CONNECT4, CONNECT5, CONNECT6)}

DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


class IllegalMove(Exception):
    pass


@lru_cache(maxsize=None)
def windows(spec):
    """All lines of k cells, as an int array (n_windows, k) of flat cell indices."""
    out = []
    for r in range(spec.rows):
        for c in range(spec.cols):
            for dr, dc in DIRECTIONS:
                er, ec = r + dr * (spec.k - 1), c + dc * (spec.k - 1)
                if 0 <= er < spec.rows and 0 <= ec < spec.cols:
                    out.append([(r + dr * i) * spec.cols + c + dc * i for i in range(spec.k)])
    arr = np.array(out, dtype=np.int64)
    arr.setflags(write=False)
    return arr


@lru_cache(maxsize=None)
def incidence(spec):
    """Matrix (n_cells, n_windows): 1.0 where the cell lies on the window."""
    win = windows(spec)
    inc = np.zeros((spec.rows * spec.cols, len(win)), dtype=np.float32)
    for w, cells in enumerate(win):
        inc[cells, w] = 1.0
    inc.setflags(write=False)
    return inc


@lru_cache(maxsize=None)
def cell_windows(spec):
    """Int array (n_cells, m): the windows through each cell, padded with n_windows.

    Sum a per-window array `x` over the windows through each cell with
    np.append(x, 0)[cell_windows(spec)].sum(-1) (no matrix product needed).
    """
    win = windows(spec)
    lists = [[] for _ in range(spec.rows * spec.cols)]
    for w, cells in enumerate(win):
        for c in cells:
            lists[c].append(w)
    m = max(len(x) for x in lists)
    arr = np.full((len(lists), m), len(win), dtype=np.int64)
    for c, ws in enumerate(lists):
        arr[c, :len(ws)] = ws
    arr.setflags(write=False)
    return arr


class Board:
    """One game position. `player` is the side to move (1 or 2)."""

    def __init__(self, spec):
        self.spec = spec
        self.grid = np.zeros((spec.rows, spec.cols), dtype=np.int8)
        self.heights = np.zeros(spec.cols, dtype=np.int64)
        self.player = 1
        self.winner = 0          # 0 while nobody has won
        self.n_moves = 0
        self.history = []        # list of (player, row, col)

    # ---- state ---------------------------------------------------------
    @property
    def done(self):
        return self.winner != 0 or bool((self.heights >= self.spec.rows).all())

    @property
    def is_draw(self):
        return self.winner == 0 and self.done

    def copy(self):
        b = Board.__new__(Board)
        b.spec = self.spec
        b.grid = self.grid.copy()
        b.heights = self.heights.copy()
        b.player = self.player
        b.winner = self.winner
        b.n_moves = self.n_moves
        b.history = list(self.history)
        return b

    def legal_moves(self):
        if self.done:
            return []
        return [c for c in range(self.spec.cols) if self.heights[c] < self.spec.rows]

    def legal_mask(self):
        if self.done:
            return np.zeros(self.spec.cols, dtype=bool)
        return self.heights < self.spec.rows

    def is_legal(self, col):
        return (not self.done and isinstance(col, (int, np.integer))
                and 0 <= col < self.spec.cols and self.heights[col] < self.spec.rows)

    # ---- moves ---------------------------------------------------------
    def play(self, col):
        """Drop a stone for the side to move. Return the row it lands on."""
        if not self.is_legal(col):
            raise IllegalMove(f"column {col!r} is not legal here")
        col = int(col)
        row = int(self.heights[col])
        me = self.player
        self.grid[row, col] = me
        self.heights[col] += 1
        self.n_moves += 1
        self.history.append((me, row, col))
        if self.wins_at(row, col):
            self.winner = me
        self.player = 3 - me
        return row

    def wins_at(self, row, col):
        """True if the stone at (row, col) is part of k in a row."""
        g, me, k = self.grid, self.grid[row, col], self.spec.k
        if me == 0:
            return False
        R, C = self.spec.rows, self.spec.cols
        for dr, dc in DIRECTIONS:
            n = 1
            r, c = row + dr, col + dc
            while 0 <= r < R and 0 <= c < C and g[r, c] == me:
                n += 1
                r += dr
                c += dc
            r, c = row - dr, col - dc
            while 0 <= r < R and 0 <= c < C and g[r, c] == me:
                n += 1
                r -= dr
                c -= dc
            if n >= k:
                return True
        return False

    def scan_winner(self):
        """Winner found by looking at every line (0 if none, -1 if both)."""
        flat = self.grid.ravel()[windows(self.spec)]
        one = bool((flat == 1).all(axis=1).any())
        two = bool((flat == 2).all(axis=1).any())
        if one and two:
            return -1
        return 1 if one else 2 if two else 0

    def resync(self, player):
        """Rebuild heights, move count and winner from the grid alone.

        Use this after code outside this class has written to `grid`. A column's
        height is one above its highest stone.
        """
        occupied = self.grid != 0
        any_stone = occupied.any(axis=0)
        top = self.spec.rows - np.argmax(occupied[::-1], axis=0)
        self.heights = np.where(any_stone, top, 0).astype(np.int64)
        self.n_moves = int(occupied.sum())
        w = self.scan_winner()
        self.winner = (3 - player) if w == -1 else w   # both: credit the side that just moved
        self.player = player

    # ---- display -------------------------------------------------------
    def render(self, marks=".XO"):
        cols = self.spec.cols
        lines = [" ".join(f"{c % 10}" for c in range(cols))]
        for r in range(self.spec.rows - 1, -1, -1):
            lines.append(" ".join(marks[v] for v in self.grid[r]))
        lines.append(" ".join(f"{c % 10}" for c in range(cols)))
        return "\n".join(lines)

    def __repr__(self):
        return f"Board({self.spec.name}, moves={self.n_moves}, to_move={self.player}, winner={self.winner})\n{self.render()}"


def replay(spec, cols):
    """Board after playing the given columns from the empty board."""
    b = Board(spec)
    for c in cols:
        b.play(c)
    return b
