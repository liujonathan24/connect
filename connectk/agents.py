"""Submissions: load a team's bot from its folder, and check the rules.

A submission is a folder with an `agent.py` that defines

    def make_agent(folder):          # folder: pathlib.Path of the submission
        return agent                 # any object with the two members below

    agent.net                        # a torch.nn.Module that holds ALL the bot's parameters
    agent.choose(board) -> int       # the column to play (board: connectk.Board, a copy)

The folder may hold more files (weights, a model.py, a config). agent.py
imports its sibling files as plain modules (`from model import Net`).

Rules checked here (see contest/PROBLEM.md):
    MAX_PARAMS     parameters in agent.net, at most 200 million
    MOVE_LIMIT     seconds per move on one CPU core; a slower move loses the game

    python3 -m connectk check my_submission --game connect5
"""

import importlib.util
import pathlib
import sys
import time

MAX_PARAMS = 200_000_000
MOVE_TARGET = 1.0       # seconds per move to aim for
MOVE_LIMIT = 2.0        # a move slower than this loses the game


class MoveTimeout(Exception):
    pass


def load_agent(folder):
    """Import folder/agent.py and return make_agent(folder).

    Each submission gets its own copies of its sibling modules: modules that
    agent.py imports from the folder are removed from sys.modules afterwards,
    so two submissions may both have a `model.py`.
    """
    folder = pathlib.Path(folder).resolve()
    path = folder / "agent.py"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    before = set(sys.modules)
    sys.path.insert(0, str(folder))
    try:
        name = f"_submission_{abs(hash(str(folder)))}"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        agent = module.make_agent(folder)
    finally:
        sys.path.remove(str(folder))
        for key in set(sys.modules) - before:
            f = getattr(sys.modules[key], "__file__", None) or ""
            if f.startswith(str(folder)):
                del sys.modules[key]
    if not hasattr(agent, "choose"):
        raise TypeError("the agent has no choose(board) method")
    return agent


def count_parameters(agent):
    import torch
    net = getattr(agent, "net", None)
    if not isinstance(net, torch.nn.Module):
        raise TypeError("agent.net must be a torch.nn.Module that holds all of the bot's parameters")
    return sum(p.numel() for p in net.parameters())


class TimedAgent:
    """Wraps an agent for play: counts time, and a move over `limit` seconds loses."""

    def __init__(self, agent, name, limit=MOVE_LIMIT):
        self.agent, self.name, self.limit = agent, name, limit
        self.moves, self.seconds, self.slowest = 0, 0.0, 0.0

    def choose(self, board):
        t = time.perf_counter()
        col = self.agent.choose(board)
        dt = time.perf_counter() - t
        self.moves += 1
        self.seconds += dt
        self.slowest = max(self.slowest, dt)
        if dt > self.limit:
            raise MoveTimeout(f"{self.name} took {dt:.2f} s for one move (limit {self.limit} s)")
        return col


def check(folder, spec, n_positions=30, seed=0):
    """Load a submission and check it like the graders do. Returns a dict and prints a report."""
    import numpy as np
    import torch

    from .arena import RandomPlayer, play_game
    from .game import Board

    torch.set_num_threads(1)
    report = {"folder": str(folder), "ok": False}
    agent = load_agent(folder)
    n = count_parameters(agent)
    report["parameters"] = n
    rng = np.random.default_rng(seed)
    times, illegal = [], 0
    for _ in range(n_positions):
        b = Board(spec)
        for _ in range(int(rng.integers(0, 30))):
            if b.done:
                break
            b.play(int(rng.choice(b.legal_moves())))
        if b.done:
            continue
        t = time.perf_counter()
        col = agent.choose(b.copy())
        times.append(time.perf_counter() - t)
        illegal += not b.is_legal(col)
    report["mean_move_s"] = float(np.mean(times))
    report["max_move_s"] = float(np.max(times))
    report["illegal_moves"] = illegal
    g = play_game(TimedAgent(agent, "submission"), RandomPlayer(seed=seed), spec)
    report["game_vs_random"] = "win" if g.winner == 1 else "draw" if g.winner == 0 else f"loss ({g.reason})"
    problems = []
    if n > MAX_PARAMS:
        problems.append(f"{n:,} parameters (limit {MAX_PARAMS:,})")
    if report["max_move_s"] > MOVE_LIMIT:
        problems.append(f"a move took {report['max_move_s']:.2f} s (a move over {MOVE_LIMIT} s loses the game)")
    if illegal:
        problems.append(f"{illegal} illegal moves (an illegal move loses the game)")
    report["problems"] = problems
    report["ok"] = not problems
    print(f"{folder}: {n:,} parameters; move time mean {report['mean_move_s']:.3f} s, "
          f"max {report['max_move_s']:.3f} s (target {MOVE_TARGET} s, limit {MOVE_LIMIT} s, one CPU core); "
          f"illegal moves {illegal}; one game vs random: {report['game_vs_random']}")
    print("OK" if report["ok"] else "NOT OK: " + "; ".join(problems))
    return report
