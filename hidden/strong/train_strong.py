"""Train a strong network ("hard bot") for connect6 or connect5.

    python3 hidden/strong/train_strong.py --game connect6 --out hidden/strong/runs/c6
    python3 hidden/strong/train_strong.py --game connect5 --out hidden/strong/runs/c5

The net has the fixed architecture of connectk/net.py, so it plays exactly
like a student submission (greedy policy, no search). Two stages:

1. Imitation. Strong rule-based bots (from the held-out pool) play many games
   against a wide pool. The net learns their moves (policy) and the game
   results (value).

2. Expert iteration. With --teacher mcts (the stronger choice), a batched
   MCTS (hidden/strong/mcts.py, AlphaZero style) picks the moves and the net
   learns the visit counts and the game results. With --teacher lookahead,
   the net plays with a one-move lookahead: for every legal
   column it checks for an immediate win, for a move that lets the opponent
   win at once, and otherwise asks its own value head about the position.
   The lookahead's choice is a better policy than the raw net. The net is
   trained to play that choice directly, and the value head learns the
   results of these games. Games are half self-play, half against bots.

Workers generate games in parallel (one torch thread each). The main process
trains with all threads between rounds.
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")   # many small matrix products: BLAS threads only fight

import argparse  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from connectk.arena import make_openings, play_match  # noqa: E402
from connectk.game import SPECS, Board, incidence, windows  # noqa: E402
from connectk.net import ConnectNet, NetPlayer, encode_relative, relative_grid, save_net  # noqa: E402
from hidden.bots.engine import HeuristicBot, SearchBot  # noqa: E402
from hidden.bots.heldout import connect5_anchors, connect6_heldout  # noqa: E402
from hidden.bots.public import connect5_gauges, connect6_public  # noqa: E402
from hidden.strong.mcts import _inc_t  # noqa: E402


# ---------------------------------------------------------------- bot pools
def bot_pool(spec, seed):
    """(teachers, everyone): teachers' moves are imitated; everyone are opponents."""
    if spec.name == "connect6":
        held = {b.name: b for b in connect6_heldout(seed)}
        pub = connect6_public(seed)
        teachers = [held["h09_planner"], held["h08_turtle"], pub["bot_d"], held["h10_search"],
                    HeuristicBot("t_planner2", seed, defense=[0, 1.5, 6, 24, 96, 384, 0], lookahead=0.7)]
        everyone = list(held.values()) + [pub["bot_a"], pub["bot_b"], pub["bot_d"]] + teachers[-1:]
    else:
        anch = connect5_anchors(seed)
        gau = connect5_gauges(seed)
        teachers = [gau["gauge_hard"], anch["anchor_search"],
                    HeuristicBot("t_planner", seed, attack=[0, 1, 4, 16, 64, 256], defense=[0, 1.2, 5, 20, 80, 0], lookahead=0.8),
                    HeuristicBot("t_turtle", seed, attack=[0, 0.5, 2, 6, 20, 60], defense=[0, 3, 12, 48, 192, 0], lookahead=0.3),
                    SearchBot("t_search_def", seed, depth=2, defense_ratio=1.5, noise=0.05)]
        everyone = list(anch.values()) + list(gau.values()) + teachers[2:] + [
            HeuristicBot("o_sloppy", seed, epsilon=0.4, temperature=1.0),
            HeuristicBot("o_stacker", seed, stack=3.0, low=0.5, center=0.0),
            HeuristicBot("o_rusher", seed, attack=[0, 1, 8, 64, 512, 4096], defense=[0, 0, 0.5, 2, 10, 0], avoid_gift=False)]
    return teachers, everyone


# ---------------------------------------------------------------- lookahead
def lookahead_q(net, boards):
    """One-move lookahead for each board (side to move = "me").

    Returns Q (B, cols) with -inf on illegal columns, and the net's policy
    logits for the boards (used to break ties).
    """
    spec = boards[0].spec
    R, C, k = spec.rows, spec.cols, spec.k
    win, inc_t = windows(spec), _inc_t(spec)
    parents = np.stack([relative_grid(b) for b in boards])
    heights = np.stack([b.heights for b in boards])
    bi, ci = np.nonzero(heights < R)
    M = len(bi)
    child = parents[bi].copy()
    child[np.arange(M), heights[bi, ci], ci] = 1
    flat = child.reshape(M, -1)
    cells = flat[:, win]
    own = (cells == 1).sum(-1)
    opp = (cells == 2).sum(-1)
    iwin = (own == k).any(1)
    h2 = heights[bi].copy()
    h2[np.arange(M), ci] += 1
    full = (h2 >= R).all(1)
    hot = ((opp == k - 1) & (own == 0)).astype(np.float32)
    threat_cells = (hot @ inc_t) > 0
    land = np.zeros((M, R * C), dtype=bool)
    m2, cc = np.nonzero(h2 < R)
    land[m2, h2[m2, cc] * C + cc] = True
    threat = (threat_cells & land).any(1)
    need = ~iwin & ~full & ~threat
    q = np.zeros(M)
    q[iwin] = 1.0
    q[~iwin & ~full & threat] = -1.0
    with torch.no_grad():
        logits, _ = net(torch.from_numpy(encode_relative(parents)))
        if need.any():
            swapped = np.where(child[need] == 1, 2, np.where(child[need] == 2, 1, 0)).astype(np.int8)
            _, v = net(torch.from_numpy(encode_relative(swapped)))
            q[need] = -v.numpy()
    Q = np.full((len(boards), C), -np.inf)
    Q[bi, ci] = q
    return Q, logits.numpy()


def _drop(grids, heights, cols, stone):
    """Children of (grids, heights) after dropping `stone` in `cols` (one per row)."""
    n = np.arange(len(cols))
    g = grids.copy()
    g[n, heights[n, cols], cols] = stone
    h = heights.copy()
    h[n, cols] += 1
    return g, h


def _lines(grids, spec):
    flat = grids.reshape(len(grids), -1)[:, windows(spec)]
    return (flat == 1).sum(-1), (flat == 2).sum(-1)


def _win_landings(c_who, c_other, heights, spec):
    """(n, cells) bool: landing cells where the player with counts c_who completes a line."""
    R, C, k = spec.rows, spec.cols, spec.k
    hot = ((c_who == k - 1) & (c_other == 0)).astype(np.float32)
    cells = (hot @ _inc_t(spec)) > 0
    land = np.zeros_like(cells)
    m, c = np.nonzero(heights < R)
    land[m, heights[m, c] * C + c] = True
    return cells & land


def lookahead2_q(net, boards):
    """One-move lookahead with two-move tactics. Same outputs as lookahead_q.

    For each of my moves a:
      +1  a wins at once
      -1  the opponent then has a winning landing cell
      +1  I then have two or more winning landing cells (a double threat)
      -1  some reply gives the opponent a double threat while I have no win
      else minus the net's value of the position after a (opponent to move),
           or 0 when net is None (tactics only; then no logits are returned)
    """
    spec = boards[0].spec
    R, C, k = spec.rows, spec.cols, spec.k
    parents = np.stack([relative_grid(b) for b in boards])
    heights = np.stack([b.heights for b in boards])
    b1, a1 = np.nonzero(heights < R)
    g1, h1 = _drop(parents[b1], heights[b1], a1, 1)
    own1, opp1 = _lines(g1, spec)
    win1 = (own1 == k).any(1)
    full1 = (h1 >= R).all(1)
    threat1 = _win_landings(opp1, own1, h1, spec).any(1)
    double1 = _win_landings(own1, opp1, h1, spec).sum(1) >= 2
    q1 = np.zeros(len(g1))
    q1[win1] = 1.0
    rest = ~win1 & ~full1
    q1[rest & threat1] = -1.0
    q1[rest & ~threat1 & double1] = 1.0
    open1 = np.flatnonzero(rest & ~threat1 & ~double1)
    crushed = np.zeros(len(g1), dtype=bool)
    if len(open1):
        i2, r2 = np.nonzero(h1[open1] < R)
        parent2 = open1[i2]
        g2, h2 = _drop(g1[parent2], h1[parent2], r2, 2)
        own2, opp2 = _lines(g2, spec)
        bad = (_win_landings(opp2, own2, h2, spec).sum(1) >= 2) & ~_win_landings(own2, opp2, h2, spec).any(1)
        np.logical_or.at(crushed, parent2, bad)
        q1[open1[crushed[open1]]] = -1.0
        ask = open1[~crushed[open1]]
        if len(ask) and net is not None:
            swapped = np.where(g1[ask] == 1, 2, np.where(g1[ask] == 2, 1, 0)).astype(np.int8)
            with torch.no_grad():
                _, v = net(torch.from_numpy(encode_relative(swapped)))
            q1[ask] = -v.numpy()
    Q = np.full((len(boards), C), -np.inf)
    Q[b1, a1] = q1
    if net is None:
        return Q, None
    with torch.no_grad():
        logits, _ = net(torch.from_numpy(encode_relative(parents)))
    return Q, logits.numpy()


def tactical_blunders(boards):
    """(len(boards), cols) bool: moves that are wrong by two-move tactics.

    A move is wrong if a winning move (or a double threat) exists and it is not
    one, or if it lets the opponent win or make a double threat while another
    move does not. Exact for these tactics; says nothing about quiet moves.
    """
    Q, _ = lookahead2_q(None, boards)
    legal = np.isfinite(Q)
    best = np.where(legal, Q, -2).max(axis=1, keepdims=True)
    return legal & (((best == 1) & (Q < 1)) | ((Q == -1) & (best > -1)))


def choose_from_q(Q, logits, rng, explore_temp=0.0):
    legal = np.isfinite(Q)
    if explore_temp > 0:
        z = np.where(legal, Q / explore_temp, -np.inf)
        p = np.exp(z - z.max())
        p /= p.sum()
        return int(rng.choice(len(p), p=p))
    lp = logits - logits[legal].max()
    score = np.where(legal, Q + 1e-3 * lp, -np.inf)
    return int(np.argmax(score))


def soft_target(Q, temp=0.05):
    legal = np.isfinite(Q)
    z = np.where(legal, Q / temp, -np.inf)
    p = np.exp(z - z.max())
    return (p / p.sum()).astype(np.float32)


# ---------------------------------------------------------------- workers
def _teacher_games(args):
    spec_name, n_games, seed = args
    spec = SPECS[spec_name]
    rng = np.random.default_rng(seed)
    teachers, everyone = bot_pool(spec, seed)
    tnames = {t.name for t in teachers}
    grids, targets, zs, players = [], [], [], []
    for _ in range(n_games):
        a = teachers[rng.integers(len(teachers))]
        b = everyone[rng.integers(len(everyone))] if rng.random() < 0.6 else teachers[rng.integers(len(teachers))]
        if rng.random() < 0.5:
            a, b = b, a
        board = Board(spec)
        for c in make_openings(spec, 1, plies=2, seed=int(rng.integers(1 << 30)))[0]:
            board.play(c)
        start = len(grids)
        bots = {1: a, 2: b}
        while not board.done:
            bot = bots[board.player]
            col = bot.choose(board.copy())
            if bot.name in tnames:
                grids.append(relative_grid(board))
                targets.append(col)
                players.append(board.player)
            board.play(col)
        for t in range(start, len(grids)):
            zs.append(0.0 if board.winner == 0 else (1.0 if board.winner == players[t] else -1.0))
    return (np.array(grids, dtype=np.int8).reshape(-1, spec.rows, spec.cols),
            np.eye(spec.cols, dtype=np.float32)[np.array(targets, dtype=int)] if targets else np.zeros((0, spec.cols), np.float32),
            np.array(zs, dtype=np.float32))


def _expert_games(args):
    spec_name, state, n_games, seed, p_self, batch, depth = args
    search = lookahead2_q if depth == 2 else lookahead_q
    torch.set_num_threads(1)
    spec = SPECS[spec_name]
    rng = np.random.default_rng(seed)
    net = ConnectNet(spec)
    net.load_state_dict(state)
    net.eval()
    _, everyone = bot_pool(spec, seed)
    out_g, out_t, out_z = [], [], []
    started = 0
    live = []

    def new_game():
        board = Board(spec)
        for c in make_openings(spec, 1, plies=2, seed=int(rng.integers(1 << 30)))[0]:
            board.play(c)
        if rng.random() < p_self:
            sides = {1: None, 2: None}
        else:
            bot = everyone[rng.integers(len(everyone))]
            sides = {1: None, 2: bot} if rng.random() < 0.5 else {1: bot, 2: None}
        return {"board": board, "sides": sides, "rec": [], "plies": 0}

    while started < n_games or live:
        while len(live) < batch and started < n_games:
            live.append(new_game())
            started += 1
        # bots move until each live game waits for the learner (or ends)
        for g in live:
            b = g["board"]
            while not b.done and g["sides"][b.player] is not None:
                b.play(g["sides"][b.player].choose(b.copy()))
        waiting = [g for g in live if not g["board"].done]
        if waiting:
            Q, logits = search(net, [g["board"] for g in waiting])
            for g, q, lg in zip(waiting, Q, logits):
                b = g["board"]
                g["rec"].append((relative_grid(b), soft_target(q), b.player))
                temp = 0.15 if g["plies"] < 6 else 0.0
                if temp == 0.0 and rng.random() < 0.03:
                    temp = 0.3
                b.play(choose_from_q(q, lg, rng, temp))
                g["plies"] += 1
        still = []
        for g in live:
            b = g["board"]
            if b.done:
                for grid, tgt, pl in g["rec"]:
                    out_g.append(grid)
                    out_t.append(tgt)
                    out_z.append(0.0 if b.winner == 0 else (1.0 if b.winner == pl else -1.0))
            else:
                still.append(g)
        live = still
    return (np.array(out_g, dtype=np.int8).reshape(-1, spec.rows, spec.cols),
            np.array(out_t, dtype=np.float32).reshape(-1, spec.cols), np.array(out_z, dtype=np.float32))


def _mcts_games(args):
    """Self-play (and some games against bots) with MCTS choosing the learner's moves."""
    spec_name, state, n_games, seed, p_bot, batch, sims, sharpen, p_dagger = args
    from hidden.strong.mcts import run_mcts
    torch.set_num_threads(1)
    spec = SPECS[spec_name]
    rng = np.random.default_rng(seed)
    net = ConnectNet(spec)
    net.load_state_dict(state)
    net.eval()
    _, everyone = bot_pool(spec, seed)
    temp_plies = 8 if spec.cols <= 9 else 12
    out_g, out_t, out_z = [], [], []
    started, live = 0, []

    def new_game():
        board = Board(spec)
        for c in make_openings(spec, 1, plies=2, seed=int(rng.integers(1 << 30)))[0]:
            board.play(c)
        sides = {1: None, 2: None}
        if rng.random() < p_bot:
            bot = everyone[rng.integers(len(everyone))]
            sides[int(rng.integers(1, 3))] = bot
        # DAgger game: the net plays its own greedy move; the search only labels the position
        return {"board": board, "sides": sides, "rec": [], "plies": 0, "dagger": rng.random() < p_dagger}

    while started < n_games or live:
        while len(live) < batch and started < n_games:
            live.append(new_game())
            started += 1
        for g in live:
            b = g["board"]
            while not b.done and g["sides"][b.player] is not None:
                b.play(g["sides"][b.player].choose(b.copy()))
        waiting = [g for g in live if not g["board"].done]
        if waiting:
            visits = run_mcts(net, [g["board"] for g in waiting], sims=sims, rng=rng)
            if any(g["dagger"] for g in waiting):
                with torch.no_grad():
                    grids = np.stack([relative_grid(g["board"]) for g in waiting])
                    own_logits = net(torch.from_numpy(encode_relative(grids)))[0].numpy()
            for i, (g, n) in enumerate(zip(waiting, visits)):
                b = g["board"]
                pi = (n / n.sum()).astype(np.float32)
                sharp = n ** (1.0 / sharpen)
                g["rec"].append((relative_grid(b), (sharp / sharp.sum()).astype(np.float32), b.player))
                if g["plies"] < temp_plies:
                    a = int(rng.choice(len(pi), p=pi))
                elif g["dagger"]:
                    a = int(np.argmax(np.where(b.legal_mask(), own_logits[i], -np.inf)))
                else:
                    a = int(np.argmax(n))
                b.play(a)
                g["plies"] += 1
        still = []
        for g in live:
            b = g["board"]
            if b.done:
                for grid, tgt, pl in g["rec"]:
                    out_g.append(grid)
                    out_t.append(tgt)
                    out_z.append(0.0 if b.winner == 0 else (1.0 if b.winner == pl else -1.0))
            else:
                still.append(g)
        live = still
    return (np.array(out_g, dtype=np.int8).reshape(-1, spec.rows, spec.cols),
            np.array(out_t, dtype=np.float32).reshape(-1, spec.cols), np.array(out_z, dtype=np.float32))


def _cat(parts):
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))


# ---------------------------------------------------------------- training
class Buffer:
    def __init__(self, cap):
        self.cap = cap
        self.g = self.t = self.z = None

    def add(self, g, t, z):
        if self.g is None:
            self.g, self.t, self.z = g, t, z
        else:
            self.g = np.concatenate([self.g, g])[-self.cap:]
            self.t = np.concatenate([self.t, t])[-self.cap:]
            self.z = np.concatenate([self.z, z])[-self.cap:]

    def __len__(self):
        return 0 if self.g is None else len(self.g)


def train_steps(net, opt, buf, steps, batch, rng, vf=1.0):
    net.train()
    losses = []
    for _ in range(steps):
        idx = rng.integers(len(buf), size=batch)
        g, t, z = buf.g[idx], buf.t[idx], buf.z[idx]
        flip = rng.random(batch) < 0.5
        g = np.where(flip[:, None, None], g[:, :, ::-1], g)
        t = np.where(flip[:, None], t[:, ::-1], t)
        x = torch.from_numpy(encode_relative(g))
        logits, v = net(x)
        target = torch.from_numpy(np.ascontiguousarray(t))
        p_loss = -(target * F.log_softmax(logits, -1)).sum(-1).mean()
        v_loss = F.mse_loss(v, torch.from_numpy(z))
        loss = p_loss + vf * v_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        losses.append((p_loss.item(), v_loss.item()))
    net.eval()
    return np.mean(losses, axis=0)


def panel(spec, seed=777):
    if spec.name == "connect6":
        held = {b.name: b for b in connect6_heldout(seed)}
        pub = connect6_public(seed)
        return {"bot_d": pub["bot_d"], "h09_planner": held["h09_planner"], "h10_search": held["h10_search"],
                "h08_turtle": held["h08_turtle"], "h07_balanced": held["h07_balanced"]}
    anch = connect5_anchors(seed)
    gau = connect5_gauges(seed)
    return {"gauge_hard": gau["gauge_hard"], "anchor_search": anch["anchor_search"],
            "anchor_heuristic": anch["anchor_heuristic"]}


def evaluate(net, spec, n_openings=4):
    player = NetPlayer(net, "net")
    ops = make_openings(spec, n_openings, plies=2, seed=4242)
    return {n: play_match(player, b, spec, ops).score for n, b in panel(spec).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="connect6", choices=["connect6", "connect5"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--teacher_games", type=int, default=3000)
    ap.add_argument("--bc_steps", type=int, default=6000)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--games_per_iter", type=int, default=96)
    ap.add_argument("--steps_per_iter", type=int, default=150)
    ap.add_argument("--p_self", type=float, default=0.5)
    ap.add_argument("--depth", type=int, default=1, choices=[1, 2], help="lookahead depth of the teacher")
    ap.add_argument("--teacher", default="lookahead", choices=["lookahead", "mcts"])
    ap.add_argument("--sims", type=int, default=96, help="MCTS simulations per move")
    ap.add_argument("--p_bot", type=float, default=0.2, help="MCTS teacher: share of games against bots")
    ap.add_argument("--sharpen", type=float, default=1.0,
                    help="MCTS policy target is visits**(1/sharpen); below 1 moves it toward the top move")
    ap.add_argument("--p_dagger", type=float, default=0.0,
                    help="share of games where the net plays its own greedy moves (the search only labels)")
    ap.add_argument("--buffer", type=int, default=300_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="start from these weights and skip imitation")
    ap.add_argument("--seed_data", default=None, help="with --resume: a teacher.npz to fill the buffer first")
    args = ap.parse_args()

    spec = SPECS[args.game]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.workers)
    net = ConnectNet(spec)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    buf = Buffer(args.buffer)
    ctx = mp.get_context("fork")
    t0 = time.time()
    log = open(out / "log.txt", "a")

    def say(msg):
        msg = f"[{time.time() - t0:7.0f}s] {msg}"
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    if args.resume:
        net.load_state_dict(torch.load(args.resume, weights_only=True))
        if args.seed_data:
            d = np.load(args.seed_data)
            buf.add(d["g"], d["t"], d["z"])
            say(f"buffer seeded with {len(buf)} positions")
    else:
        data_path = out / "teacher.npz"
        if data_path.exists():
            d = np.load(data_path)
            g, t, z = d["g"], d["t"], d["z"]
        else:
            per = args.teacher_games // (args.workers * 4)
            jobs = [(spec.name, per, args.seed * 1000 + i) for i in range(args.workers * 4)]
            with ctx.Pool(args.workers) as pool:
                g, t, z = _cat(pool.map(_teacher_games, jobs))
            np.savez_compressed(data_path, g=g, t=t, z=z)
        say(f"teacher data: {len(g)} positions")
        buf.add(g, t, z)
        for chunk in range(args.bc_steps // 500):
            pl, vl = train_steps(net, opt, buf, 500, args.batch, rng)
            say(f"imitation {(chunk + 1) * 500} steps: policy {pl:.3f} value {vl:.3f}")
        save_net(net, out / "bc.pt")
        say("after imitation: " + "  ".join(f"{n} {s:.2f}" for n, s in evaluate(net, spec).items()))

    best = -1
    for it in range(1, args.iters + 1):
        state = {k: v.clone() for k, v in net.state_dict().items()}
        per = max(1, args.games_per_iter // args.workers)
        if args.teacher == "mcts":
            fn = _mcts_games
            jobs = [(spec.name, state, per, int(rng.integers(1 << 30)), args.p_bot, per, args.sims,
                     args.sharpen, args.p_dagger) for _ in range(args.workers)]
        else:
            fn = _expert_games
            jobs = [(spec.name, state, per, int(rng.integers(1 << 30)), args.p_self, 12, args.depth)
                    for _ in range(args.workers)]
        with ctx.Pool(args.workers) as pool:
            g, t, z = _cat(pool.map(fn, jobs))
        buf.add(g, t, z)
        pl, vl = train_steps(net, opt, buf, args.steps_per_iter, args.batch, rng)
        msg = f"iter {it}: +{len(g)} positions (buffer {len(buf)}), policy {pl:.3f} value {vl:.3f}"
        if it % 5 == 0 or it == args.iters:
            scores = evaluate(net, spec)
            mean = float(np.mean(list(scores.values())))
            msg += "  eval " + "  ".join(f"{n} {s:.2f}" for n, s in scores.items()) + f"  mean {mean:.3f}"
            save_net(net, out / f"iter_{it:04d}.pt")
            if mean >= best:
                best = mean
                save_net(net, out / "best.pt")
                msg += "  (best)"
        save_net(net, out / "last.pt")
        say(msg)


if __name__ == "__main__":
    main()
