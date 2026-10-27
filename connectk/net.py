"""The fixed network. Every team uses this exact architecture and input encoding.

You may change the weights, nothing else. The graders build `ConnectNet(spec)`,
load your state_dict with strict=True, and play the move

    argmax over legal columns of policy_logits

with no search and no randomness. So the network alone is your player.

Input: 4 planes of shape (rows, cols), from the view of the side to move.
    plane 0   stones of the side to move
    plane 1   stones of the opponent
    plane 2   landing cells (the cell each legal move would fill)
    plane 3   all ones (lets the convolutions see the board edge)

Body: 6 convolutions 3x3, 64 channels, ReLU, padding 1.
Policy head: conv 1x1 -> 2 channels, ReLU, linear -> one logit per column.
Value head: conv 1x1 -> 1 channel, ReLU, linear -> 64, ReLU, linear -> 1, tanh.
The value is the expected result for the side to move (+1 win, -1 loss).
"""

import numpy as np
import torch
from torch import nn

N_PLANES = 4
CHANNELS = 64
N_CONV = 6
VALUE_HIDDEN = 64


def encode(board):
    """Planes (4, rows, cols) float32 for the side to move."""
    me = board.player
    g = board.grid
    x = np.zeros((N_PLANES, board.spec.rows, board.spec.cols), dtype=np.float32)
    x[0] = g == me
    x[1] = g == 3 - me
    for c in range(board.spec.cols):
        h = board.heights[c]
        if h < board.spec.rows:
            x[2, h, c] = 1.0
    x[3] = 1.0
    return x


def encode_batch(boards):
    return np.stack([encode(b) for b in boards])


def relative_grid(board):
    """The grid from the view of the side to move: 1 = mine, 2 = theirs (int8)."""
    g = board.grid
    out = np.zeros_like(g)
    out[g == board.player] = 1
    out[g == 3 - board.player] = 2
    return out


def encode_relative(grids):
    """Fast encode for many relative grids (N, rows, cols) at once.

    Same result as encode() for positions reached by legal play (no floating
    stones), because the landing cell of a column is its stone count.
    """
    grids = np.asarray(grids)
    n, rows, cols = grids.shape
    x = np.zeros((n, N_PLANES, rows, cols), dtype=np.float32)
    x[:, 0] = grids == 1
    x[:, 1] = grids == 2
    h = (grids != 0).sum(axis=1)                       # (n, cols)
    ii, cc = np.nonzero(h < rows)
    x[ii, 2, h[ii, cc], cc] = 1.0
    x[:, 3] = 1.0
    return x


def mask_batch(boards):
    return np.stack([b.legal_mask() for b in boards])


class ConnectNet(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        layers = [nn.Conv2d(N_PLANES, CHANNELS, 3, padding=1), nn.ReLU()]
        for _ in range(N_CONV - 1):
            layers += [nn.Conv2d(CHANNELS, CHANNELS, 3, padding=1), nn.ReLU()]
        self.body = nn.Sequential(*layers)
        cells = spec.rows * spec.cols
        self.policy_conv = nn.Conv2d(CHANNELS, 2, 1)
        self.policy_fc = nn.Linear(2 * cells, spec.cols)
        self.value_conv = nn.Conv2d(CHANNELS, 1, 1)
        self.value_fc1 = nn.Linear(cells, VALUE_HIDDEN)
        self.value_fc2 = nn.Linear(VALUE_HIDDEN, 1)

    def forward(self, x):
        """x: (B, 4, rows, cols). Returns (policy logits (B, cols), value (B,))."""
        h = self.body(x)
        p = torch.relu(self.policy_conv(h)).flatten(1)
        logits = self.policy_fc(p)
        v = torch.relu(self.value_conv(h)).flatten(1)
        v = torch.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v)).squeeze(-1)
        return logits, value


def masked_logits(logits, mask):
    """Set the logits of illegal columns to -inf (mask: bool tensor, True = legal)."""
    return logits.masked_fill(~mask, float("-inf"))


def save_net(net, path):
    torch.save({k: v.detach().cpu().float() for k, v in net.state_dict().items()}, path)


def load_net(spec, path, device="cpu"):
    """Build ConnectNet(spec) and load weights strictly. Raises on any mismatch."""
    state = torch.load(path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    net = ConnectNet(spec)
    net.load_state_dict(state, strict=True)
    for name, t in net.state_dict().items():
        if not torch.isfinite(t).all():
            raise ValueError(f"{path}: parameter {name} has NaN or inf")
    net.eval()
    return net.to(device)


def count_parameters(net):
    return sum(p.numel() for p in net.parameters())


class NetPlayer:
    """Plays the argmax legal column of the policy. This is how graders play your net.

    With temperature > 0 it samples instead (for training and exploration only).
    """

    def __init__(self, net, name="net", temperature=0.0, seed=None):
        self.net = net.eval()
        self.name = name
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def choose(self, board):
        x = torch.from_numpy(encode(board)).unsqueeze(0)
        logits, _ = self.net(x)
        logits = logits[0].numpy().astype(np.float64)
        legal = board.legal_mask()
        logits[~legal] = -np.inf
        if self.temperature <= 0:
            return int(np.argmax(logits))
        z = logits / self.temperature
        p = np.exp(z - z[legal].max())
        p /= p.sum()
        return int(self.rng.choice(len(p), p=p))
