"""Batched MCTS (AlphaZero style) for the strong-net trainer.

run_mcts(net, boards, sims) searches many positions at once: each simulation
walks one path down every tree, then the network scores all the new leaves in
one forward pass. Returns the visit counts at each root.

Leaves get tactical shortcuts before the network: if the side to move has a
winning landing cell, or can make a double threat while the opponent has no
winning cell, the leaf is worth +1; if the opponent has two or more winning
cells, it is worth -1. A finished game is worth -1 for the side to move (the
opponent just won) or 0 for a draw.
"""

import math
from functools import lru_cache

import numpy as np
import torch

from connectk.game import incidence, windows
from connectk.net import encode_relative, relative_grid


class Node:
    __slots__ = ("board", "P", "N", "W", "children", "expanded", "terminal_value")

    def __init__(self, board):
        self.board = board
        self.expanded = False
        self.children = None
        self.P = self.N = self.W = None
        if board.done:
            self.terminal_value = 0.0 if board.winner == 0 else -1.0
        else:
            self.terminal_value = None

    def expand(self, priors):
        self.P = priors
        n = len(priors)
        self.N = np.zeros(n)
        self.W = np.zeros(n)
        self.children = [None] * n
        self.expanded = True

    def select(self, c_puct):
        legal = self.board.legal_mask()
        total = self.N.sum()
        q = np.where(self.N > 0, self.W / np.maximum(self.N, 1), self._fpu())
        u = c_puct * self.P * math.sqrt(total + 1) / (1 + self.N)
        score = np.where(legal, q + u, -np.inf)
        return int(np.argmax(score))

    def _fpu(self):
        n = self.N.sum()
        return (self.W.sum() / n - 0.2) if n > 0 else 0.0

    def child(self, a):
        c = self.children[a]
        if c is None:
            b = self.board.copy()
            b.play(a)
            c = self.children[a] = Node(b)
        return c


@lru_cache(maxsize=None)
def _inc_t(spec):
    return np.ascontiguousarray(incidence(spec).T)


def _win_landings(own, opp, heights, spec):
    """(n, cells): landing cells where the side with counts `own` completes a line."""
    R, C, k = spec.rows, spec.cols, spec.k
    land = np.zeros((len(own), R * C), dtype=bool)
    m, c = np.nonzero(heights < R)
    land[m, heights[m, c] * C + c] = True
    hot = ((own == k - 1) & (opp == 0)).astype(np.float32)
    return (hot @ _inc_t(spec) > 0) & land


def _tactics(boards):
    """For each board, from the side to move:
    +1 it can win now, or (with no threat against it) it can make a double threat;
    -1 the opponent has two or more winning cells; 0 otherwise.
    """
    spec = boards[0].spec
    R, C = spec.rows, spec.cols
    grids = np.stack([relative_grid(b) for b in boards])
    heights = np.stack([b.heights for b in boards])
    cells = grids.reshape(len(boards), -1)[:, windows(spec)]
    own = (cells == 1).sum(-1)
    opp = (cells == 2).sum(-1)
    mine = _win_landings(own, opp, heights, spec).any(1)
    n_theirs = _win_landings(opp, own, heights, spec).sum(1)
    out = np.zeros(len(boards))
    out[mine] = 1.0
    out[~mine & (n_theirs >= 2)] = -1.0
    quiet = np.flatnonzero(~mine & (n_theirs == 0))
    if len(quiet):
        b_idx, cols = np.nonzero(heights[quiet] < R)
        rows = quiet[b_idx]
        g = grids[rows].copy()
        n = np.arange(len(rows))
        g[n, heights[rows, cols], cols] = 1
        h = heights[rows].copy()
        h[n, cols] += 1
        cc = g.reshape(len(rows), -1)[:, windows(spec)]
        o2, p2 = (cc == 1).sum(-1), (cc == 2).sum(-1)
        double = (_win_landings(o2, p2, h, spec).sum(1) >= 2) & ~_win_landings(p2, o2, h, spec).any(1)
        hit = np.zeros(len(boards), dtype=bool)
        np.logical_or.at(hit, rows, double)
        out[hit] = 1.0
    return out, grids


def _evaluate(net, nodes):
    """Expand nodes; return the value of each for its side to move."""
    values = np.zeros(len(nodes))
    tact, grids = _tactics([n.board for n in nodes])
    with torch.no_grad():
        logits, v = net(torch.from_numpy(encode_relative(grids)))
    logits = logits.numpy().astype(np.float64)
    v = v.numpy()
    for i, node in enumerate(nodes):
        legal = node.board.legal_mask()
        z = np.where(legal, logits[i], -np.inf)
        p = np.exp(z - z.max())
        node.expand(p / p.sum())
        values[i] = tact[i] if tact[i] != 0 else v[i]
    return values


def run_mcts(net, boards, sims=96, c_puct=1.5, dirichlet=0.3, noise_frac=0.25, rng=None):
    """Visit counts (len(boards), cols) after `sims` simulations from each board."""
    rng = rng or np.random.default_rng()
    roots = [Node(b.copy()) for b in boards]
    _evaluate(net, roots)
    if noise_frac > 0:
        for r in roots:
            legal = r.board.legal_mask()
            eta = np.zeros(len(r.P))
            eta[legal] = rng.dirichlet([dirichlet] * int(legal.sum()))
            r.P = (1 - noise_frac) * r.P + noise_frac * eta
    for _ in range(sims):
        paths, leaves, fixed = [], [], []
        for r in roots:
            node, path = r, []
            while node.expanded and node.terminal_value is None:
                a = node.select(c_puct)
                path.append((node, a))
                node = node.child(a)
            paths.append(path)
            leaves.append(node)
        todo = [i for i, n in enumerate(leaves) if n.terminal_value is None and not n.expanded]
        vals = np.zeros(len(leaves))
        for i, n in enumerate(leaves):
            if n.terminal_value is not None:
                vals[i] = n.terminal_value
        if todo:
            vals[todo] = _evaluate(net, [leaves[i] for i in todo])
        for path, v in zip(paths, vals):
            for node, a in reversed(path):
                v = -v                      # value for the side that chose a
                node.N[a] += 1
                node.W[a] += v
    return np.stack([r.N for r in roots])
