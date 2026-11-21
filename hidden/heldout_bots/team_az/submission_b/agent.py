"""Submission: MCTS (PUCT, AlphaZero style) that calls the network for priors and values.

The network gives the move priors and the position values. The search uses only
the game rules (legal moves, end of game). A move stops at a fixed time budget.
agent.net holds all the parameters.
"""
import json
import pathlib
import time

import numpy as np
import torch

from azcore import Geo, Net, Evaluator, run_sims_single, best_move, find_descendant, fuse_net, new_node, legal_of


class Agent:
    def __init__(self, folder):
        folder = pathlib.Path(folder)
        cfg = json.loads((folder / "config.json").read_text())
        self.cfg = cfg
        torch.set_num_threads(1)
        self.geo = Geo(cfg["rows"], cfg["cols"], cfg["k"])
        self.net = Net(cfg["rows"], cfg["cols"], **cfg["net"])
        self.net.load_state_dict(torch.load(folder / "weights.pt", map_location="cpu", weights_only=True))
        self.net.eval()
        fuse_net(self.net)
        self.ev = Evaluator(self.net, self.geo)
        self.time_limit = float(cfg.get("time_limit", 0.6))
        self.batch = int(cfg.get("batch", 8))
        self.c_puct = float(cfg.get("c_puct", 1.8))
        self.fpu_red = float(cfg.get("fpu_red", 0.25))
        self.max_sims = cfg.get("max_sims", None)
        self.solver = bool(cfg.get("solver", True))
        self.tree = None
        # warm up torch (the first calls are slow)
        for _ in range(3):
            self.ev([(0, 0, 0)] * self.batch)

    def choose(self, board):
        try:
            return self._choose(board)
        except Exception:                    # never crash: fall back to the net's best legal column
            self.tree = None
            try:
                legal = board.legal_moves()
                lg, _ = self.ev([self.geo.from_board(board)])
                return int(max(legal, key=lambda c: lg[0][c]))
            except Exception:
                return int(board.legal_moves()[0])

    def _choose(self, board):
        t0 = time.perf_counter()
        geo = self.geo
        cur, mask, n = geo.from_board(board)
        legal = geo.legal(mask)
        if len(legal) == 1:
            self.tree = None
            return int(legal[0])
        root = find_descendant(geo, self.tree, cur, mask, depth=2)
        if root is None or root.n != n:
            root = new_node(geo, cur, mask, n)
        moves = legal_of(geo, root)
        if len(moves) == 1:                  # an immediate win, or the one block (from the rules)
            self.tree = root
            return int(moves[0])
        budget = self.time_limit - (time.perf_counter() - t0)
        run_sims_single(geo, root, self.ev, n_sims=self.max_sims, time_limit=max(budget, 0.05),
                        batch=self.batch, c_puct=self.c_puct, fpu_red=self.fpu_red, solver=self.solver)
        col = int(best_move(root))
        self.tree = root
        if not board.is_legal(col):          # never happens; a safe fallback
            col = int(legal[0])
        return col


def make_agent(folder):
    return Agent(pathlib.Path(folder))
