"""Reference: find the bad bot from play alone (no access to the bot code).

    cd part_a_connect6 && python3 ../hidden/solutions/part_a/detect_cheater.py

Two independent tests:

1. Rule check on every bot turn in the training environment. After my move
   the board is info["after_agent"]; after the bot's answer it is the board
   that step() returns. A fair answer adds exactly one stone of the bot's
   color, on the landing cell of a column, and changes nothing else.

2. Strict referee vs trusting environment. The same fixed player plays each
   bot in the environment (bot.play on the live board) and in the arena
   (bot.choose on a copy, checked by the referee). A fair bot scores the
   same in both. A bot that breaks rules on the live board does better there.
"""

import collections
import pathlib
import sys

import numpy as np

HERE = pathlib.Path.cwd()
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from connectk.arena import make_openings, play_match  # noqa: E402
from env import SPEC, BotEnv, load_opponents  # noqa: E402


def violations(before, after, bot_color):
    """Rule breaks in one bot turn (list of strings, empty if fair)."""
    out = []
    diff = np.argwhere(before.grid != after.grid)
    new = [(r, c) for r, c in diff if before.grid[r, c] == 0 and after.grid[r, c] == bot_color]
    other = [(r, c) for r, c in diff if (r, c) not in new]
    if other:
        out.append("changed an existing stone")
    if len(new) > 1:
        out.append("more than one stone")
    for r, c in new:
        if r != before.heights[c]:
            out.append("stone not on the landing cell")
    if not new and not other and not after.done:
        out.append("no move")
    return out


class Probe:
    """A weak fixed player: wins if it can, blocks if it must, else random."""

    name = "probe"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def choose(self, board):
        for who in (board.player, 3 - board.player):
            for c in board.legal_moves():
                b = board.copy()
                b.player = who
                b.play(c)
                if b.winner:
                    return c
        return int(self.rng.choice(board.legal_moves()))


def main(games=60):
    bots = load_opponents(seed=1)
    probe = Probe()
    print("1) rule check in the training environment")
    for name, bot in bots.items():
        env = BotEnv({name: bot}, agent_side="random", seed=2)
        counts = collections.Counter()
        turns = 0
        for _ in range(games):
            board = env.reset()
            done = False
            while not done:
                board, done, info = env.step(probe.choose(board))
                if info["after_agent"] is not None and not info["after_agent"].done:
                    turns += 1
                    for v in violations(info["after_agent"], board, 3 - info["agent"]):
                        counts[v] += 1
        print(f"  {name}: {turns} bot turns, rule breaks: {dict(counts) or 'none'}")

    # A fair, competent player as the probe: bot_d (test 1 shows it plays fair).
    agent = load_opponents(seed=4)["bot_d"]
    print("2) bot_d's score against each bot: trusting environment vs strict referee")
    openings = make_openings(SPEC, games // 2, seed=3)
    for name, bot in bots.items():
        env = BotEnv({name: bot}, agent_side="random", seed=5)
        points = 0.0
        for _ in range(games):
            board, done = env.reset(), False
            while not done:
                board, done, info = env.step(agent.choose(board))
            points += 1.0 if info["winner"] == info["agent"] else 0.5 if info["draw"] else 0.0
        strict = play_match(agent, bot, SPEC, openings).score
        print(f"  {name}: env {points / games:.2f}   strict {strict:.2f}")


if __name__ == "__main__":
    main()
