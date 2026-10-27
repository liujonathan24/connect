"""Training environment for Part A: your agent against the four opponent bots.

    from env import load_opponents, BotEnv, VecBotEnv

    opponents = load_opponents(seed=0)        # {"bot_a": ..., "bot_d": ...}
    env = BotEnv(opponents, seed=0)
    board = env.reset()                       # always your turn
    board, done, info = env.step(board.legal_moves()[0])

`board` is a connectk.Board (a copy). `info` holds:

    opponent      name of the bot in this game
    agent         your side: 1 (you moved first) or 2
    winner        1, 2, or 0 (no winner yet, or a draw)
    draw          True if the game ended in a draw
    illegal       True if your move was illegal (you lose at once)
    after_agent   the board right after your move, before the bot answered

The environment gives no reward. ppo.py turns `info` into a reward.

Speed note: a bot plays its own move directly on the live board
(`bot.play(board)`). The environment trusts it to play exactly one legal
move, and only re-reads the grid afterwards.
"""

import pathlib

import numpy as np

from connectk import CONNECT6, Board

SPEC = CONNECT6
HERE = pathlib.Path(__file__).resolve().parent


def load_opponents(seed=None):
    from connectk.packed import load_packed
    return load_packed(HERE / "opponents.bin", "connect6_public", seed)


class BotEnv:
    """One game at a time against a bot drawn from `opponents` (with `weights`)."""

    def __init__(self, opponents, weights=None, agent_side="random", seed=None):
        self.opponents = dict(opponents)
        self.names = list(self.opponents)
        w = np.ones(len(self.names)) if weights is None else np.array(
            [weights.get(n, 0.0) for n in self.names] if isinstance(weights, dict) else weights, dtype=float)
        self.weights = w / w.sum()
        assert agent_side in ("first", "second", "random")
        self.agent_side = agent_side
        self.rng = np.random.default_rng(seed)
        self.board = None

    def reset(self):
        self.opp_name = self.names[self.rng.choice(len(self.names), p=self.weights)]
        self.opp = self.opponents[self.opp_name]
        if self.agent_side == "random":
            self.agent = int(self.rng.integers(1, 3))
        else:
            self.agent = 1 if self.agent_side == "first" else 2
        self.board = Board(SPEC)
        if self.agent == 2:
            self._opponent_move()
        return self.board.copy()

    def _opponent_move(self):
        self.opp.play(self.board)
        self.board.resync(player=self.agent)

    def step(self, col):
        b = self.board
        info = {"opponent": self.opp_name, "agent": self.agent, "illegal": False,
                "winner": 0, "draw": False, "after_agent": None}
        if not b.is_legal(col):
            info["illegal"] = True
            info["winner"] = 3 - self.agent
            b.winner = 3 - self.agent
            return b.copy(), True, info
        b.play(int(col))
        info["after_agent"] = b.copy()
        if not b.done:
            self._opponent_move()
        info["winner"] = b.winner
        info["draw"] = b.is_draw
        return b.copy(), b.done, info


class VecBotEnv:
    """n independent BotEnvs. Finished games restart at once.

    step(actions) returns (boards, dones, infos). When dones[i] is True,
    boards[i] is already the first board of the next game, and infos[i]
    ["final_board"] is the last board of the finished one.
    """

    def __init__(self, n_envs, opponents, weights=None, agent_side="random", seed=0):
        self.envs = [BotEnv(opponents, weights, agent_side, seed=None if seed is None else seed + i)
                     for i in range(n_envs)]

    def reset(self):
        return [e.reset() for e in self.envs]

    def step(self, actions):
        boards, dones, infos = [], [], []
        for e, a in zip(self.envs, actions):
            board, done, info = e.step(int(a))
            if done:
                info["final_board"] = board
                board = e.reset()
            boards.append(board)
            dones.append(done)
            infos.append(info)
        return boards, np.array(dones), infos
