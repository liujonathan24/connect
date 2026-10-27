"""Distill a clean search into the net, DAgger style, in rounds.

    python3 hidden/strong/distill.py --game connect5 --init runs/c5m_last.pt --out runs/c5d

Each round:
  1. Play games where the net plays its own greedy move (a few random moves
     at the start for variety), against bots and against itself.
  2. Label every position the net met with a clean MCTS (no root noise):
     the search's top move (policy target, sharpened visits) and, later, the
     game result (value target).
  3. Train on all labeled data so far (newest rounds weigh most), and report
     the net's agreement with the search and its score against the bots.

The net only learns positions it really reaches, labeled by a player much
stronger than itself (the search with this net inside). Each round the net
gets better, so the search gets better, so the labels get better.
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
import torch.nn.functional as F  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from connectk.arena import make_openings  # noqa: E402
from connectk.game import SPECS, Board  # noqa: E402
from connectk.net import ConnectNet, encode_relative, relative_grid, save_net  # noqa: E402
from hidden.strong.mcts import run_mcts  # noqa: E402
from hidden.strong.train_strong import bot_pool, evaluate, tactical_blunders  # noqa: E402


def _games(args):
    """Play and label. Returns grids, targets, values, and the net's agreement with the search."""
    spec_name, state, n_games, seed, sims, p_bot, sharpen, random_plies = args
    torch.set_num_threads(1)
    spec = SPECS[spec_name]
    rng = np.random.default_rng(seed)
    net = ConnectNet(spec)
    net.load_state_dict(state)
    net.eval()
    _, everyone = bot_pool(spec, seed)
    live, started = [], 0
    out_g, out_t, out_z, out_b, out_m = [], [], [], [], []
    agree = total = blunders = 0

    def new_game():
        b = Board(spec)
        for c in make_openings(spec, 1, plies=2, seed=int(rng.integers(1 << 30)))[0]:
            b.play(c)
        sides = {1: None, 2: None}
        if rng.random() < p_bot:
            sides[int(rng.integers(1, 3))] = everyone[rng.integers(len(everyone))]
        return {"b": b, "sides": sides, "rec": [], "plies": 0}

    while started < n_games or live:
        while len(live) < n_games and started < n_games:
            live.append(new_game())
            started += 1
        for g in live:
            while not g["b"].done and g["sides"][g["b"].player] is not None:
                g["b"].play(g["sides"][g["b"].player].choose(g["b"].copy()))
        waiting = [g for g in live if not g["b"].done]
        if waiting:
            boards = [g["b"] for g in waiting]
            visits = run_mcts(net, boards, sims=sims, noise_frac=0.0, rng=rng)
            with torch.no_grad():
                logits = net(torch.from_numpy(encode_relative(np.stack([relative_grid(b) for b in boards]))))[0].numpy()
            bad_moves = tactical_blunders(boards)
            for g, n, lg, bad in zip(waiting, visits, logits, bad_moves):
                b = g["b"]
                sharp = n ** (1.0 / sharpen) * ~bad      # never a tactical blunder as target
                if sharp.sum() == 0:
                    sharp = n.astype(float)
                mine = int(np.argmax(np.where(b.legal_mask(), lg, -np.inf)))
                g["rec"].append((relative_grid(b), (sharp / sharp.sum()).astype(np.float32), b.player, bad,
                                 bool(bad[mine]) or mine != int(np.argmax(n))))
                agree += mine == int(np.argmax(n))
                blunders += bool(bad[mine])
                total += 1
                if g["plies"] < random_plies:
                    a = int(rng.choice(b.legal_moves()))
                else:
                    a = mine
                b.play(a)
                g["plies"] += 1
        still = []
        for g in live:
            if g["b"].done:
                w = g["b"].winner
                for grid, t, pl, bad, miss in g["rec"]:
                    out_g.append(grid)
                    out_t.append(t)
                    out_z.append(0.0 if w == 0 else (1.0 if w == pl else -1.0))
                    out_b.append(bad)
                    out_m.append(miss)
            else:
                still.append(g)
        live = still
    return (np.array(out_g, np.int8).reshape(-1, spec.rows, spec.cols),
            np.array(out_t, np.float32).reshape(-1, spec.cols), np.array(out_z, np.float32),
            np.array(out_b, bool).reshape(-1, spec.cols), np.array(out_m, bool), agree, total, blunders)


def train(net, opt, data, steps, batch, rng, vf, tw):
    g, t, z, bad, w = data
    p = w / w.sum()
    net.train()
    pl = vl = 0.0
    for _ in range(steps):
        i = rng.choice(len(g), size=batch, p=p)
        gg, tt, bb = g[i], t[i], bad[i]
        flip = rng.random(batch) < 0.5
        gg = np.where(flip[:, None, None], gg[:, :, ::-1], gg)
        tt = np.where(flip[:, None], tt[:, ::-1], tt)
        bb = np.where(flip[:, None], bb[:, ::-1], bb)
        logits, v = net(torch.from_numpy(encode_relative(gg)))
        logp = F.log_softmax(logits, -1)
        p_loss = -(torch.from_numpy(np.ascontiguousarray(tt)) * logp).sum(-1).mean()
        # push probability off tactical blunders: -log(1 - P(blunder))
        p_bad = (logp.exp() * torch.from_numpy(np.ascontiguousarray(bb))).sum(-1).clamp(max=1 - 1e-6)
        t_loss = -torch.log1p(-p_bad).mean()
        v_loss = F.mse_loss(v, torch.from_numpy(z[i]))
        loss = p_loss + tw * t_loss + vf * v_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        pl, vl = p_loss.item(), v_loss.item()
    net.eval()
    return pl, vl


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", default="connect5", choices=["connect5", "connect6"])
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--games", type=int, default=160, help="games per round (all workers)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--p_bot", type=float, default=0.6)
    ap.add_argument("--sharpen", type=float, default=0.25)
    ap.add_argument("--random_plies", type=int, default=2)
    ap.add_argument("--steps", type=int, default=400, help="training steps per round")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--tactic_weight", type=float, default=2.0, help="weight of the blunder penalty")
    ap.add_argument("--hard", type=float, default=10.0,
                    help="sampling weight of positions where the net blundered or disagreed with the search")
    ap.add_argument("--decay", type=float, default=0.7, help="weight of each older round")
    ap.add_argument("--keep", type=int, default=8, help="rounds of data to keep")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    spec = SPECS[args.game]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.set_num_threads(args.workers)
    net = ConnectNet(spec)
    net.load_state_dict(torch.load(args.init, weights_only=True))
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
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
        per = args.games // args.workers
        jobs = [(spec.name, state, per, int(rng.integers(1 << 30)), args.sims, args.p_bot, args.sharpen,
                 args.random_plies) for _ in range(args.workers)]
        torch.set_num_threads(1)
        with ctx.Pool(args.workers) as pool:
            parts = pool.map(_games, jobs)
        torch.set_num_threads(args.workers)
        g, t, z, bad, miss = (np.concatenate([p[k] for p in parts]) for k in range(5))
        total = max(1, sum(p[6] for p in parts))
        agree = sum(p[5] for p in parts) / total
        blunder = sum(p[7] for p in parts) / total
        rounds.append((g, t, z, bad, miss))
        rounds = rounds[-args.keep:]
        # newer rounds weigh more; positions where the net erred weigh `hard` times more
        w = np.concatenate([np.full(len(x[0]), args.decay ** (len(rounds) - 1 - k)) * (1 + (args.hard - 1) * x[4])
                            for k, x in enumerate(rounds)])
        data = tuple(np.concatenate([x[k] for x in rounds]) for k in range(4)) + (w,)
        pl, vl = train(net, opt, data, args.steps, args.batch, rng, args.vf, args.tactic_weight)
        save_net(net, out / f"round_{r:03d}.pt")
        save_net(net, out / "last.pt")
        scores = evaluate(net, spec, n_openings=8)
        say(f"round {r}: {len(g)} new positions, net agrees with search {agree:.2f}, "
            f"blunders {blunder:.3f} (before training), "
            f"policy {pl:.3f} value {vl:.3f}  eval " + "  ".join(f"{n} {s:.2f}" for n, s in scores.items())
            + f"  mean {np.mean(list(scores.values())):.3f}")


if __name__ == "__main__":
    main()
