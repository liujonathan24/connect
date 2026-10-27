"""Imitation data from strong-bot games, winners' moves only.

    python3 hidden/strong/make_winner_data.py --games 20000 --out hidden/strong/runs/winners6.npz

Plays games like the Part A offline data (house bots against each other and
against the honest public bots; hidden/make_offline_data.py) and keeps the
positions where the side to move went on to win or draw, never bot_a's
moves. Saves relative grids, the move played (one-hot) and the result.

Imitation of these winners' moves gave the best Connect 6 net we have
(0.65 on the Part A held-out pool after 8,000 steps on the offline set).
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

from connectk.game import CONNECT6, Board  # noqa: E402
from connectk.net import relative_grid  # noqa: E402
from hidden.make_offline_data import _games  # noqa: E402


def _positions(args):
    n, seed = args
    games = _games(("house", n, seed))
    grids, moves, zs = [], [], []
    for g in games:
        names = {1: g["p1"], 2: g["p2"]}
        b = Board(CONNECT6)
        for p, _, c in g["moves"]:
            if g["winner"] in (0, p) and names[p] != "bot_a":
                grids.append(relative_grid(b))
                moves.append(c)
                zs.append(0.0 if g["winner"] == 0 else 1.0)
            b.play(c)
    return np.array(grids, np.int8).reshape(-1, 13, 13), np.array(moves, np.int64), np.array(zs, np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    chunks = args.workers * 10
    rng = np.random.default_rng(args.seed)
    jobs = [(args.games // chunks, int(rng.integers(1 << 30))) for _ in range(chunks)]
    with mp.get_context("fork").Pool(args.workers) as pool:
        parts = pool.map(_positions, jobs)
    g = np.concatenate([p[0] for p in parts])
    m = np.concatenate([p[1] for p in parts])
    z = np.concatenate([p[2] for p in parts])
    t = np.eye(13, dtype=np.float32)[m]
    np.savez_compressed(args.out, g=g, t=t, z=z)
    print(f"{len(g)} positions -> {args.out}")


if __name__ == "__main__":
    main()
