"""Distill the C alpha-beta engine into the net (DAgger style, in rounds).

    python3 hidden/strong/ab_distill.py --game connect5 --init runs/c5h_last.pt --out runs/c5ab

Each round, workers play games between a mix of players: the net itself
(greedy), the engine (choosing at random among near-best moves), and the
bots. Every position is labeled by the engine (abengine.c, about 20 ms):

  policy target  uniform over the engine's best moves
  blunder mask   moves the engine proves lost while a move exists that is not,
                 and moves that miss a forced win
  value target   +1/-1 when the engine proves the result, else the game result

The net trains on all labels so far (newest rounds weigh more; positions
where the net's own move was a proven blunder weigh more), with the blunder
penalty of distill.py. The engine labels about 50 times faster than the
Python MCTS teacher, at about the same strength.
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from connectk.arena import make_openings  # noqa: E402
from connectk.game import SPECS, Board  # noqa: E402
from connectk.net import ConnectNet, encode_relative, relative_grid, save_net  # noqa: E402
from hidden.strong.abengine import WIN, search  # noqa: E402
from hidden.strong.distill import train  # noqa: E402
from hidden.strong.train_strong import bot_pool, evaluate  # noqa: E402

PROVEN = WIN - 1000


def label(board, nodes):
    """(target, blunder mask, proven value or None, engine best) for one position."""
    best, sc, _ = search(board, 40, nodes)
    legal = board.legal_mask()
    sc = np.where(legal, sc.astype(np.int64), -(2 ** 40))
    top = sc.max()
    good = legal & (sc >= top - 2)
    target = good / good.sum()
    bad = np.zeros(len(sc), dtype=bool)
    if top > -PROVEN:
        bad |= legal & (sc <= -PROVEN)
    if top >= PROVEN:
        bad |= legal & (sc < PROVEN)
    value = 1.0 if top >= PROVEN else -1.0 if top <= -PROVEN else None
    return target.astype(np.float32), bad, value, best


def _games(args):
    spec_name, state, n_games, seed, nodes, p_net, p_ab, ab_noise = args
    torch.set_num_threads(1)
    spec = SPECS[spec_name]
    rng = np.random.default_rng(seed)
    net = ConnectNet(spec)
    net.load_state_dict(state)
    net.eval()
    _, bots = bot_pool(spec, seed)
    out_g, out_t, out_z, out_b, out_m = [], [], [], [], []
    net_moves = net_blunders = 0

    def net_move(board):
        with torch.no_grad():
            lg = net(torch.from_numpy(encode_relative(relative_grid(board)[None])))[0][0].numpy()
        return int(np.argmax(np.where(board.legal_mask(), lg, -np.inf)))

    def pick_player():
        u = rng.random()
        if u < p_net:
            return "net"
        if u < p_net + p_ab:
            return "ab"
        return bots[rng.integers(len(bots))]

    for _ in range(n_games):
        b = Board(spec)
        for c in make_openings(spec, 1, plies=int(rng.integers(2, 5)), seed=int(rng.integers(1 << 30)))[0]:
            if not b.done:
                b.play(c)
        players = {1: pick_player(), 2: pick_player()}
        rec = []
        while not b.done:
            target, bad, value, best = label(b, nodes)
            who = players[b.player]
            if who == "net":
                move = net_move(b)
                net_moves += 1
                net_blunders += bool(bad[move])
                miss = bool(bad[move]) or target[move] == 0
            elif who == "ab":
                good = np.flatnonzero(target > 0) if rng.random() > ab_noise else np.flatnonzero(b.legal_mask() & ~bad)
                move = int(rng.choice(good))
                miss = False
            else:
                move = who.choose(b.copy())
                miss = False
            rec.append((relative_grid(b), target, bad, value, b.player, miss))
            b.play(move)
        for grid, target, bad, value, pl, miss in rec:
            z = value if value is not None else (0.0 if b.winner == 0 else (1.0 if b.winner == pl else -1.0))
            out_g.append(grid)
            out_t.append(target)
            out_b.append(bad)
            out_z.append(z)
            out_m.append(miss)
    return (np.array(out_g, np.int8).reshape(-1, spec.rows, spec.cols), np.array(out_t, np.float32).reshape(-1, spec.cols),
            np.array(out_z, np.float32), np.array(out_b, bool).reshape(-1, spec.cols), np.array(out_m, bool),
            net_moves, net_blunders)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", default="connect5", choices=["connect5", "connect6"])
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--games", type=int, default=240, help="games per round (all workers)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--nodes", type=int, default=30000, help="engine node budget per label")
    ap.add_argument("--p_net", type=float, default=0.4)
    ap.add_argument("--p_ab", type=float, default=0.3)
    ap.add_argument("--ab_noise", type=float, default=0.2, help="chance the engine plays any unproven-loss move")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--tactic_weight", type=float, default=2.0)
    ap.add_argument("--tactics", default=None, help="tactical set from make_tactics.py, trained alongside")
    ap.add_argument("--hard", type=float, default=5.0)
    ap.add_argument("--decay", type=float, default=0.9)
    ap.add_argument("--keep", type=int, default=30, help="rounds of data to keep")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    spec = SPECS[args.game]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    net = ConnectNet(spec)
    net.load_state_dict(torch.load(args.init, weights_only=True))
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    tactics = None
    if args.tactics:
        d = np.load(args.tactics)
        tactics = (d["grids"], d["bad"])
    rounds = []
    t0 = time.time()
    log = open(out / "log.txt", "a")

    def say(msg):
        msg = f"[{time.time() - t0:7.0f}s] {msg}"
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    ctx = mp.get_context("fork")
    for r in range(1, args.rounds + 1):
        state = {k: v.clone() for k, v in net.state_dict().items()}
        per = max(1, args.games // args.workers)
        jobs = [(spec.name, state, per, int(rng.integers(1 << 30)), args.nodes, args.p_net, args.p_ab, args.ab_noise)
                for _ in range(args.workers)]
        torch.set_num_threads(1)
        with ctx.Pool(args.workers) as pool:
            parts = pool.map(_games, jobs)
        torch.set_num_threads(args.workers)
        g, t, z, bad, miss = (np.concatenate([p[k] for p in parts]) for k in range(5))
        blunder = sum(p[6] for p in parts) / max(1, sum(p[5] for p in parts))
        rounds.append((g, t, z, bad, miss))
        rounds = rounds[-args.keep:]
        w = np.concatenate([np.full(len(x[0]), args.decay ** (len(rounds) - 1 - k)) * (1 + (args.hard - 1) * x[4])
                            for k, x in enumerate(rounds)])
        data = tuple(np.concatenate([x[k] for x in rounds]) for k in range(4)) + (w,)
        pl, vl = train(net, opt, data, args.steps, args.batch, rng, args.vf, args.tactic_weight, tactics)
        save_net(net, out / f"round_{r:03d}.pt")
        save_net(net, out / "last.pt")
        msg = (f"round {r}: {len(g)} new positions (total {len(data[0])}), net blunders {blunder:.3f} "
               f"(before training), policy {pl:.3f} value {vl:.3f}")
        if r % 2 == 0 or r == 1:
            scores = evaluate(net, spec, n_openings=8)
            msg += "  eval " + "  ".join(f"{n} {s:.2f}" for n, s in scores.items()) + \
                f"  mean {np.mean(list(scores.values())):.3f}"
        say(msg)


if __name__ == "__main__":
    main()
