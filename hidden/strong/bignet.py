"""A 10x larger network for our own bots (not the fixed network teams must use).

Same input (the 4 planes of connectk.net.encode) and the same outputs
(policy logits per column, value in [-1, 1]), so every tool that takes a
ConnectNet also takes this: NetPlayer, NeuralMCTS, the tournament, the
graders' players.

Body: a 3x3 stem convolution to 128 channels, then 7 residual blocks
(3x3 conv, ReLU, 3x3 conv, add, ReLU). The second convolution of each block
starts at zero, so a new network starts as a plain stem and trains stably
without normalization layers. About 2.08M parameters (the fixed net: 0.19M).

    from hidden.strong.bignet import BigConnectNet, load_model
    net = load_model(CONNECT5, "hidden/strong/connect5_big.pt")   # either architecture
"""

import torch
from torch import nn

from connectk.net import N_PLANES, ConnectNet

CHANNELS = 128
BLOCKS = 7
VALUE_HIDDEN = 128


class Block(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, x):
        return torch.relu(x + self.c2(torch.relu(self.c1(x))))


class BigConnectNet(nn.Module):
    def __init__(self, spec, channels=CHANNELS, blocks=BLOCKS):
        super().__init__()
        self.spec = spec
        cells = spec.rows * spec.cols
        self.stem = nn.Conv2d(N_PLANES, channels, 3, padding=1)
        self.blocks = nn.Sequential(*[Block(channels) for _ in range(blocks)])
        self.policy_conv = nn.Conv2d(channels, 4, 1)
        self.policy_fc = nn.Linear(4 * cells, spec.cols)
        self.value_conv = nn.Conv2d(channels, 2, 1)
        self.value_fc1 = nn.Linear(2 * cells, VALUE_HIDDEN)
        self.value_fc2 = nn.Linear(VALUE_HIDDEN, 1)

    def forward(self, x):
        h = self.blocks(torch.relu(self.stem(x)))
        logits = self.policy_fc(torch.relu(self.policy_conv(h)).flatten(1))
        v = torch.relu(self.value_fc1(torch.relu(self.value_conv(h)).flatten(1)))
        return logits, torch.tanh(self.value_fc2(v)).squeeze(-1)


def is_big(state):
    return any(k.startswith("blocks.") for k in state)


def load_model(spec, path, device="cpu"):
    """Load either the fixed ConnectNet or a BigConnectNet, strictly, by its keys."""
    state = torch.load(path, map_location=device, weights_only=True)
    net = BigConnectNet(spec) if is_big(state) else ConnectNet(spec)
    net.load_state_dict(state, strict=True)
    return net.eval().to(device)
