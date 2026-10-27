"""The hard bots: a strong network with a search on top (AlphaZero style).

    from hidden.strong.players import NeuralMCTS
    bot = NeuralMCTS.load(CONNECT5, "hidden/strong/connect5_strong.pt", sims=200)
    bot.choose(board)

Students' networks must play greedily, but these bots are our opponents (the
Part A held-out pool, the Part B anchors), so they may search. The search is
hidden/strong/mcts.py with no root noise: it plays the most visited move.
"""

import numpy as np

from connectk.net import load_net
from hidden.strong.mcts import run_mcts


class NeuralMCTS:
    def __init__(self, net, sims=200, name=None, c_puct=1.5):
        self.net = net.eval()
        self.sims = sims
        self.c_puct = c_puct
        self.name = name or f"neural_mcts{sims}"

    @classmethod
    def load(cls, spec, path, sims=200, name=None):
        return cls(load_net(spec, path), sims, name)

    def choose(self, board):
        visits = run_mcts(self.net, [board], sims=self.sims, c_puct=self.c_puct, noise_frac=0.0)[0]
        return int(np.argmax(visits))
