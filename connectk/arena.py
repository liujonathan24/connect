"""The strict referee: games, openings and matches.

A player is any object with `choose(board) -> column` (and, if you like, a
`name`). The referee gives each player a copy of the board, so a player
cannot change the real one. An illegal column, or an exception, loses the
game at once.

Two greedy (deterministic) players would play the same game every time. So a
match starts each game from an *opening*: a few random moves already on the
board. Each opening is played twice, once with each player moving first, so
the first-move advantage cancels out.
"""

from dataclasses import dataclass, field

import numpy as np

from .game import Board, IllegalMove


class RandomPlayer:
    def __init__(self, name="random", seed=None):
        self.name = name
        self.rng = np.random.default_rng(seed)

    def choose(self, board):
        return int(self.rng.choice(board.legal_moves()))


@dataclass
class GameResult:
    winner: int                 # 1, 2, or 0 for a draw
    reason: str                 # "line", "draw", "illegal", "error"
    moves: list = field(default_factory=list)   # columns, opening included
    opening_len: int = 0


def play_game(first, second, spec, opening=()):
    """Play one game. `first` is player 1; the opening moves count for both sides."""
    board = Board(spec)
    for c in opening:
        board.play(c)
    players = {1: first, 2: second}
    while not board.done:
        mover = board.player
        try:
            col = players[mover].choose(board.copy())
            board.play(int(col) if isinstance(col, (int, np.integer)) else col)
        except IllegalMove:
            return GameResult(3 - mover, "illegal", [h[2] for h in board.history], len(opening))
        except Exception as exc:  # a crashing player loses
            return GameResult(3 - mover, f"error: {type(exc).__name__}: {exc}",
                              [h[2] for h in board.history], len(opening))
    reason = "line" if board.winner else "draw"
    return GameResult(board.winner, reason, [h[2] for h in board.history], len(opening))


def make_openings(spec, n, plies=2, seed=0):
    """n distinct random openings of `plies` moves, kept near the bottom center.

    Moves are drawn from the middle columns so the openings stay sensible.
    An even number of plies keeps the first player of the opening and the
    first player of the game the same.
    """
    rng = np.random.default_rng(seed)
    lo, hi = spec.cols // 4, spec.cols - spec.cols // 4
    seen, out = set(), []
    tries = 0
    while len(out) < n and tries < 100 * n:
        tries += 1
        cols = tuple(int(c) for c in rng.integers(lo, hi, size=plies))
        if cols in seen:
            continue
        seen.add(cols)
        out.append(cols)
    return out


@dataclass
class MatchResult:
    a: str
    b: str
    wins: int = 0      # wins of a
    losses: int = 0    # wins of b
    draws: int = 0
    illegal_a: int = 0
    illegal_b: int = 0
    games: list = field(default_factory=list)

    @property
    def n(self):
        return self.wins + self.losses + self.draws

    @property
    def score(self):
        """Points of a per game: win 1, draw 0.5, loss 0."""
        return (self.wins + 0.5 * self.draws) / max(self.n, 1)

    def __str__(self):
        return (f"{self.a} vs {self.b}: +{self.wins} ={self.draws} -{self.losses}"
                f"  score {self.score:.3f}" + (f"  (illegal moves: {self.a} {self.illegal_a}, {self.b} {self.illegal_b})"
                                               if self.illegal_a or self.illegal_b else ""))


def _name(p, default):
    return getattr(p, "name", None) or default


def play_match(a, b, spec, openings, keep_games=False):
    """a and b play every opening twice (each moves first once)."""
    res = MatchResult(_name(a, "a"), _name(b, "b"))
    for op in openings:
        for a_first in (True, False):
            g = play_game(a, b, spec, op) if a_first else play_game(b, a, spec, op)
            a_side = 1 if a_first else 2
            if g.winner == 0:
                res.draws += 1
            elif g.winner == a_side:
                res.wins += 1
                res.illegal_b += g.reason != "line"
            else:
                res.losses += 1
                res.illegal_a += g.reason != "line"
            if keep_games:
                res.games.append((a_first, g))
    return res
