"""Reference submission for Part a: our standard Connect 5 net (hidden/strong/connect5_strong.pt), greedy.

For calibrating the grader only: this net is also one of the held-out bots.
"""
import pathlib

import numpy as np
import torch

from connectk.game import CONNECT5
from connectk.net import ConnectNet, encode


class Agent:
    def __init__(self, folder):
        self.net = ConnectNet(CONNECT5)
        weights = folder.resolve().parents[2] / "strong" / "connect5_strong.pt"
        self.net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.net.eval()

    @torch.no_grad()
    def choose(self, board):
        logits = self.net(torch.from_numpy(encode(board))[None])[0][0].numpy()
        logits[~board.legal_mask()] = -np.inf
        return int(np.argmax(logits))


def make_agent(folder):
    return Agent(pathlib.Path(folder))
