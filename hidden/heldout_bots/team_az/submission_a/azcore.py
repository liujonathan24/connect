"""Core of the bot: a bitboard game state, the input encoding, the network, and MCTS.

The same file is used for training and is copied into the submission folder.
Only numpy and torch are used.

Bitboard layout: each column has H = rows + 1 bits (one sentinel bit on top).
Cell (row r, col c) is bit c*H + r. `cur` holds the stones of the side to move,
`mask` holds all stones.
"""
import math
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- game state

class Geo:
    """Constant bit masks for one board size and k."""

    def __init__(self, rows, cols, k):
        self.R, self.C, self.k = rows, cols, k
        self.H = H = rows + 1
        self.bottom = [1 << (c * H) for c in range(cols)]
        self.top = [1 << (c * H + rows - 1) for c in range(cols)]
        self.colmask = [((1 << rows) - 1) << (c * H) for c in range(cols)]
        self.bottom_all = sum(self.bottom)
        self.board_mask = sum(self.colmask)
        self.nbits = cols * H
        self.nbytes = (self.nbits + 7) // 8
        self.shifts = (1, H, H - 1, H + 1)
        self.ncells = rows * cols
        win = []
        for r in range(rows):
            for c in range(cols):
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    er, ec = r + dr * (k - 1), c + dc * (k - 1)
                    if 0 <= er < rows and 0 <= ec < cols:
                        win.append([(r + dr * i) * cols + c + dc * i for i in range(k)])
        self.win = np.array(win, dtype=np.int64)                    # (n_windows, k) flat cell indices
        self.inc = np.zeros((len(win), rows * cols), np.float32)    # window -> cells
        for w, cells in enumerate(win):
            self.inc[w, cells] = 1.0
        self.rowpar = (np.arange(rows)[:, None] % 2 == 1).astype(np.float32) * np.ones((1, cols), np.float32)

    def play(self, cur, mask, c):
        """Return (new_cur, new_mask, move_bit). new_cur is the stones of the next side to move."""
        new_mask = mask | (mask + self.bottom[c])
        return cur ^ mask, new_mask, new_mask ^ mask

    def wins(self, p):
        """True if the stones p hold k in a row."""
        k = self.k
        for s in self.shifts:
            m = p & (p >> s)                 # runs of 2
            if not m:
                continue
            m2 = m & (m >> (2 * s))          # runs of 4
            if not m2:
                continue
            if k == 4:
                return True
            if k == 5:
                if m2 & (p >> (4 * s)):
                    return True
            elif k == 6:
                if m2 & (m >> (4 * s)):
                    return True
            else:
                x = p
                for i in range(1, k):
                    x &= p >> (i * s)
                if x:
                    return True
        return False

    def legal(self, mask):
        return [c for c in range(self.C) if not (mask & self.top[c])]

    def bits_to_grid(self, x):
        """Int bitboard -> (rows, cols) uint8 array (row 0 = bottom)."""
        b = np.frombuffer(x.to_bytes(self.nbytes, "little"), dtype=np.uint8)
        bits = np.unpackbits(b, bitorder="little")[: self.nbits].reshape(self.C, self.H)
        return bits[:, : self.R].T

    def from_board(self, board):
        """connectk Board -> (cur, mask, n_moves)."""
        g = board.grid
        me = board.player
        cur = 0
        mask = 0
        for c in range(self.C):
            for r in range(self.R):
                v = g[r, c]
                if v:
                    bit = 1 << (c * self.H + r)
                    mask |= bit
                    if v == me:
                        cur |= bit
        return cur, mask, int(board.n_moves)


N_PLANES = 5


def encode_states(geo, states):
    """states: list of (cur, mask, n_moves). Returns float32 array (B, 5, R, C).

    planes: my stones, opponent stones, landing cells, ones, side to move is the first player
    """
    B = len(states)
    x = np.zeros((B, N_PLANES, geo.R, geo.C), dtype=np.float32)
    for i, (cur, mask, n) in enumerate(states):
        x[i, 0] = geo.bits_to_grid(cur)
        x[i, 1] = geo.bits_to_grid(cur ^ mask)
        x[i, 2] = geo.bits_to_grid((mask + geo.bottom_all) & geo.board_mask)
        if n % 2 == 0:
            x[i, 4] = 1.0
    x[:, 3] = 1.0
    return x


N_FEAT_PLANES = 10


def feature_planes(geo, x):
    """Add 5 planes computed from the rules to base planes x (B, 5, R, C):
    cells that complete k for me / for the opponent (lines with k-1 stones and no enemy stone),
    cells on lines with k-2 stones and no enemy stone (me / opponent), and the row parity."""
    B, _, R, C = x.shape
    k = geo.k
    me = x[:, 0].reshape(B, -1)
    op = x[:, 1].reshape(B, -1)
    empty = 1.0 - me - op
    mw = me[:, geo.win].sum(-1)
    ow = op[:, geo.win].sum(-1)
    free_m = ow == 0
    free_o = mw == 0
    planes = []
    for cond in ((mw == k - 1) & free_m, (ow == k - 1) & free_o, (mw == k - 2) & free_m, (ow == k - 2) & free_o):
        cells = (cond.astype(np.float32) @ geo.inc) > 0
        planes.append((cells * empty).reshape(B, 1, R, C))
    planes.append(np.broadcast_to(geo.rowpar, (B, 1, R, C)))
    return np.concatenate([x] + planes, axis=1).astype(np.float32)


# --------------------------------------------------------------------------- network

class Block(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        return F.relu(x + self.b2(self.c2(h)))


class Net(nn.Module):
    """Policy: one logit per column (read at the landing cell). Value: win/draw/loss logits.

    vhead "flat": a linear layer over the flattened value planes; "pool": global mean and max pooling.
    """

    def __init__(self, rows, cols, channels=64, blocks=6, head=32, vhidden=128, vhead="flat", planes=N_PLANES):
        super().__init__()
        self.rows, self.cols, self.vhead, self.planes = rows, cols, vhead, planes
        self.stem = nn.Conv2d(planes, channels, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.blocks = nn.Sequential(*[Block(channels) for _ in range(blocks)])
        self.p1 = nn.Conv2d(channels, head, 1)
        self.p2 = nn.Conv2d(head, 1, 1)
        if vhead == "flat":
            self.v1 = nn.Conv2d(channels, 4, 1)
            self.v2 = nn.Linear(4 * rows * cols + channels, vhidden)
        else:
            self.v1 = nn.Conv2d(channels, 32, 1)
            self.v2 = nn.Linear(64, vhidden)
        self.v3 = nn.Linear(vhidden, 3)

    def forward(self, x):
        h = F.relu(self.stem_bn(self.stem(x)))
        h = self.blocks(h)
        cell = self.p2(F.relu(self.p1(h)))[:, 0]               # (B, R, C)
        logits = (cell * x[:, 2]).sum(1)                        # landing cell of each column
        if self.vhead == "flat":
            v = F.relu(self.v1(h)).flatten(1)
            v = torch.cat([v, h.mean((2, 3))], 1)
        else:
            v = F.relu(self.v1(h))
            v = torch.cat([v.mean((2, 3)), v.amax((2, 3))], 1)
        wdl = self.v3(F.relu(self.v2(v)))
        return logits, wdl


def _fuse(conv, bn):
    """A conv with bias that equals conv followed by bn (eval mode)."""
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    out = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, padding=conv.padding, bias=True)
    out = out.to(conv.weight.dtype)
    with torch.no_grad():
        out.weight.copy_(conv.weight * scale[:, None, None, None])
        b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
        out.bias.copy_((b - bn.running_mean) * scale + bn.bias)
    return out


def fuse_net(net):
    """Fold every BatchNorm into its conv, in place (for inference only). Returns net."""
    net.stem = _fuse(net.stem, net.stem_bn)
    net.stem_bn = nn.Identity()
    for blk in net.blocks:
        blk.c1 = _fuse(blk.c1, blk.b1)
        blk.b1 = nn.Identity()
        blk.c2 = _fuse(blk.c2, blk.b2)
        blk.b2 = nn.Identity()
    return net.eval()


def wdl_to_value(wdl):
    p = torch.softmax(wdl, -1)
    return p[:, 0] - p[:, 2]


# --------------------------------------------------------------------------- MCTS

class Node:
    __slots__ = ("cur", "mask", "n", "term", "proven", "allowed", "moves", "P", "Nc", "Wc", "kids", "Nsum",
                 "v", "expanded")

    def __init__(self, cur, mask, n, term=None):
        self.cur, self.mask, self.n = cur, mask, n
        self.term = term          # None, or the value for the side to move at this node (-1 loss, 0 draw)
        self.proven = None        # +1 / -1 when the rules prove the result one or two plies ahead
        self.allowed = None       # the only moves worth searching (an immediate win, or the forced block)
        self.expanded = False
        self.moves = None
        self.P = self.Nc = self.Wc = self.kids = None
        self.Nsum = 0
        self.v = 0.0


def analyze(geo, node):
    """Exact tactics from the rules only: an immediate win for the side to move (proven win),
    two or more immediate wins for the opponent (proven loss), or one (the block is forced)."""
    cur, mask = node.cur, node.mask
    opp = cur ^ mask
    land = (mask + geo.bottom_all) & geo.board_mask
    wins = geo.wins
    mine, theirs = [], []
    for c in range(geo.C):
        b = land & geo.colmask[c]
        if not b:
            continue
        if wins(cur | b):
            mine.append(c)
        elif wins(opp | b):
            theirs.append(c)
    if mine:
        node.proven = 1.0
        node.allowed = mine
    elif len(theirs) >= 2:
        node.proven = -1.0
        node.allowed = theirs[:1]
    elif theirs:
        node.allowed = theirs
    return node


def new_node(geo, cur, mask, n):
    """A node for a position reached by legal play (checks the end of the game and the tactics)."""
    if n > 0 and geo.wins(cur ^ mask):
        return Node(cur, mask, n, -1.0)
    if mask == geo.board_mask:
        return Node(cur, mask, n, 0.0)
    return analyze(geo, Node(cur, mask, n))


def legal_of(geo, node):
    return node.allowed if node.allowed is not None else geo.legal(node.mask)


def leaf_value(node):
    """The exact value of a leaf for its side to move, or None if the net must evaluate it."""
    return node.term if node.term is not None else node.proven


def make_child(geo, node, i):
    c = node.moves[i]
    cur, mask, bit = geo.play(node.cur, node.mask, c)
    mover = cur ^ mask                 # the stones of the side that just moved
    if geo.wins(mover):
        ch = Node(cur, mask, node.n + 1, -1.0)
    elif mask == geo.board_mask:
        ch = Node(cur, mask, node.n + 1, 0.0)
    else:
        ch = analyze(geo, Node(cur, mask, node.n + 1))
    node.kids[i] = ch
    return ch


def child_for_col(geo, node, c):
    """The child of node after column c (made if needed, also when c is not among node.moves)."""
    if node.moves is not None and c in node.moves:
        i = node.moves.index(c)
        ch = node.kids[i]
        return ch if ch is not None else make_child(geo, node, i)
    cur, mask, _ = geo.play(node.cur, node.mask, c)
    return new_node(geo, cur, mask, node.n + 1)


def expand(node, legal, logits, value):
    """legal: list of columns; logits: numpy (C,) array."""
    node.moves = legal
    z = logits[legal]
    z = np.exp(z - z.max())
    z /= z.sum()
    node.P = z.tolist()
    m = len(legal)
    node.Nc = [0] * m
    node.Wc = [0.0] * m
    node.kids = [None] * m
    node.v = float(value)
    node.expanded = True


def select(node, c_puct, fpu_red):
    Ns = node.Nsum
    sq = c_puct * math.sqrt(Ns + 1)
    if Ns > 0:
        fpu = sum(node.Wc) / Ns - fpu_red
    else:
        fpu = node.v - fpu_red
    best, bi = -1e9, 0
    P, Nc, Wc = node.P, node.Nc, node.Wc
    for i in range(len(P)):
        n = Nc[i]
        q = Wc[i] / n if n else fpu
        u = q + sq * P[i] / (1 + n)
        if u > best:
            best, bi = u, i
    return bi


def add_dirichlet(node, rng, alpha, eps):
    if not node.expanded or len(node.P) < 2:
        return
    d = rng.dirichlet([alpha] * len(node.P))
    node.P = [(1 - eps) * p + eps * float(x) for p, x in zip(node.P, d)]


class Evaluator:
    """Runs the net on a list of states. Returns (logits (B, C) numpy, values (B,) numpy)."""

    def __init__(self, net, geo):
        self.net, self.geo = net, geo

    @torch.no_grad()
    def __call__(self, states):
        x = encode_states(self.geo, states)
        if self.net.planes > N_PLANES:
            x = feature_planes(self.geo, x)
        logits, wdl = self.net(torch.from_numpy(x))
        return logits.numpy(), wdl_to_value(wdl).numpy()


def _prove(path):
    """MCTS-solver: after a backup that ended at an exact leaf, mark the nodes above as proven
    where the rules decide them (a child that loses for the opponent, or all children known)."""
    for node, i in reversed(path):
        ch = node.kids[i]
        cv = ch.term if ch.term is not None else ch.proven
        if cv is None:
            return
        if cv == -1.0:
            node.proven = 1.0
            continue
        best = -2.0
        for k in node.kids:
            if k is None:
                return
            kv = k.term if k.term is not None else k.proven
            if kv is None:
                return
            if -kv > best:
                best = -kv
        node.proven = best


def run_sims_single(geo, root, evaluator, n_sims=None, time_limit=None, batch=8,
                    c_puct=1.8, fpu_red=0.25, vloss=1.0, min_sims=1, solver=False):
    """MCTS on one tree with leaf batching (virtual loss). Stops after n_sims or time_limit seconds."""
    t0 = time.perf_counter()
    if not root.expanded:
        lg, vl = evaluator([(root.cur, root.mask, root.n)])
        expand(root, legal_of(geo, root), lg[0], vl[0])
    done = 0
    stop = False
    while not stop:
        if n_sims is not None and done >= n_sims:
            break
        if time_limit is not None and done >= min_sims and time.perf_counter() - t0 > time_limit:
            break
        leaves = []          # (path, node)
        tries = 0
        while len(leaves) < batch and tries < batch * 2 and not stop:
            tries += 1
            node = root
            path = []
            while True:
                i = select(node, c_puct, fpu_red)
                path.append((node, i))
                node.Nc[i] += 1
                node.Wc[i] -= vloss
                node.Nsum += 1
                ch = node.kids[i]
                if ch is None:
                    ch = make_child(geo, node, i)
                tv = ch.term if ch.term is not None else ch.proven
                if tv is not None:
                    _backup(path, tv, vloss)
                    if solver:
                        _prove(path)
                        if root.proven is not None:
                            stop = True
                    done += 1
                    break
                if not ch.expanded:
                    if any(ch is l for _, l in leaves):
                        _undo(path, vloss)
                        break
                    leaves.append((path, ch))
                    break
                node = ch
        if leaves:
            lg, vl = evaluator([(l.cur, l.mask, l.n) for _, l in leaves])
            for k, (path, leaf) in enumerate(leaves):
                expand(leaf, legal_of(geo, leaf), lg[k], vl[k])
                _backup(path, float(vl[k]), vloss)
                done += 1
    return done


def _backup(path, v, vloss):
    """v is the value for the side to move at the leaf (the node after the last edge of path)."""
    for node, i in reversed(path):
        v = -v
        node.Wc[i] += v + vloss
    # visits were added on the way down


def _undo(path, vloss):
    for node, i in path:
        node.Nc[i] -= 1
        node.Wc[i] += vloss
        node.Nsum -= 1


def best_move(root):
    """Most visited move (ties: higher Q). Proven results first: a move to a proven loss of the
    opponent wins; moves to proven wins of the opponent are avoided while others exist."""
    def known(i):
        k = root.kids[i]
        if k is None:
            return None
        return k.term if k.term is not None else k.proven
    idx = list(range(len(root.moves)))
    vals = [known(i) for i in idx]
    wins = [i for i in idx if vals[i] == -1.0]
    if wins:
        return root.moves[wins[0]]
    ok = [i for i in idx if vals[i] != 1.0] or idx
    if root.proven == 0.0:                       # a proven draw: take a drawing move
        ok = [i for i in ok if vals[i] == 0.0] or ok
    bi = max(ok, key=lambda i: (root.Nc[i], root.Wc[i] / max(root.Nc[i], 1)))
    return root.moves[bi]


def find_descendant(geo, node, cur, mask, depth=2):
    """Find the node with this state at most `depth` plies below `node` (or None)."""
    if node is None:
        return None
    if node.mask == mask and node.cur == cur:
        return node
    if depth == 0 or not node.expanded:
        return None
    for ch in node.kids:
        if ch is not None and (ch.mask & mask) == ch.mask:
            r = find_descendant(geo, ch, cur, mask, depth - 1)
            if r is not None:
                return r
    return None
