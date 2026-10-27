"""Make a large set of tactical positions for the strong-net trainer.

    python3 hidden/strong/make_tactics.py --game connect5 --games 40000 --out hidden/strong/runs/tactics5.npz

Noisy bots play many games. Every position where at least one move is a
two-move tactical blunder (see train_strong.tactical_blunders) and at least
one move is safe is kept, with its blunder mask. distill.py --tactics trains
the net to put no probability on those moves.

A test showed why this is needed: the net's own games hold too few such
positions, but with this data its blunder rate on unseen critical positions
fell from 27% to about 6% in 1,000 steps.
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from connectk.game import SPECS, Board  # noqa: E402
from connectk.net import relative_grid  # noqa: E402
from hidden.bots.engine import HeuristicBot, RandomBot  # noqa: E402
from hidden.strong.train_strong import tactical_blunders  # noqa: E402


def _work(args):
    game, n_games, seed = args
    spec = SPECS[game]
    rng = np.random.default_rng(seed)
    styles = [(0.3, 1.0, 0.7, 0.0), (0.15, 0.5, 0.9, 0.0), (0.05, 0.3, 1.0, 0.3), (0.5, 1.0, 0.5, 0.0),
              (0.1, 0.2, 1.0, 0.6), (0.2, 0.8, 0.8, 0.0)]
    bots = [HeuristicBot(seed=seed + i, epsilon=e, temperature=t, p_block=pb, lookahead=la)
            for i, (e, t, pb, la) in enumerate(styles)] + [RandomBot(seed=seed + 99)]
    boards = []
    for _ in range(n_games):
        a, b = bots[rng.integers(len(bots))], bots[rng.integers(len(bots))]
        bd = Board(spec)
        while not bd.done:
            boards.append(bd.copy())
            bd.play((a if bd.player == 1 else b).choose(bd.copy()))
    grids, bads = [], []
    for i in range(0, len(boards), 512):
        chunk = boards[i:i + 512]
        bad = tactical_blunders(chunk)
        legal = np.stack([x.legal_mask() for x in chunk])
        keep = bad.any(1) & (legal & ~bad).any(1)
        grids += [relative_grid(x) for x, k in zip(chunk, keep) if k]
        bads.append(bad[keep])
    return np.array(grids, np.int8).reshape(-1, spec.rows, spec.cols), np.concatenate(bads)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", default="connect5", choices=sorted(SPECS))
    ap.add_argument("--games", type=int, default=40000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    chunks = args.workers * 8
    jobs = [(args.game, args.games // chunks, args.seed * 1000 + i) for i in range(chunks)]
    with mp.get_context("fork").Pool(args.workers) as pool:
        parts = pool.map(_work, jobs)
    grids = np.concatenate([p[0] for p in parts])
    bad = np.concatenate([p[1] for p in parts])
    np.savez_compressed(args.out, grids=grids, bad=bad)
    print(f"{len(grids)} critical positions -> {args.out}")


if __name__ == "__main__":
    main()
