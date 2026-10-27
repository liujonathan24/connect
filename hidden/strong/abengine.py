"""Python side of the C alpha-beta engine (hidden/strong/engine/abengine.c).

    from hidden.strong.abengine import search, ABPlayer
    best, scores, depth = search(board, max_depth=12, nodes=200_000)
    player = ABPlayer(nodes=50_000)          # choose(board) -> column

The library is compiled with gcc on first use (into engine/build/).
Scores are from the view of the side to move; a forced win in n plies
scores WIN - n, a forced loss -(WIN - n).
"""

import ctypes
import pathlib
import subprocess

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent / "engine"
SRC = HERE / "abengine.c"
LIB = HERE / "build" / "abengine.so"
WIN = 100_000_000
_lib = None


def _load():
    global _lib
    if _lib is None:
        if not LIB.exists() or LIB.stat().st_mtime < SRC.stat().st_mtime:
            LIB.parent.mkdir(exist_ok=True)
            subprocess.run(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", str(LIB), str(SRC)], check=True)
        _lib = ctypes.CDLL(str(LIB))
        _lib.ab_search.restype = ctypes.c_int
        _lib.ab_search.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p]
    return _lib


def search(board, max_depth=12, nodes=200_000):
    """(best column, scores per column (-2e9 for full columns), depth reached)."""
    lib = _load()
    spec = board.spec
    grid = np.ascontiguousarray(board.grid, dtype=np.int8)
    scores = np.zeros(spec.cols, dtype=np.int32)
    depth = np.zeros(1, dtype=np.int32)
    best = lib.ab_search(spec.rows, spec.cols, spec.k, grid.ctypes.data, int(board.player), int(max_depth),
                         int(nodes), scores.ctypes.data, depth.ctypes.data)
    return int(best), scores, int(depth[0])


class ABPlayer:
    """Plays the engine's best move. `noise` > 0 picks at random among moves
    whose score is within `noise` of the best (never a forced loss when a
    better move exists)."""

    def __init__(self, nodes=50_000, max_depth=20, noise=0, name=None, seed=None):
        self.nodes, self.max_depth, self.noise = nodes, max_depth, noise
        self.name = name or f"ab{nodes // 1000}k"
        self.rng = np.random.default_rng(seed)

    def choose(self, board):
        best, scores, _ = search(board, self.max_depth, self.nodes)
        if self.noise:
            ok = np.flatnonzero(scores >= scores.max() - self.noise)
            return int(self.rng.choice(ok))
        return best
