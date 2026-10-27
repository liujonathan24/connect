"""Imitation data from strong-bot games, winners' moves only.

    python3 hidden/strong/make_winner_data.py --games 20000 --out hidden/strong/runs/winners6.npz
    python3 hidden/strong/make_winner_data.py --game connect5 --games 40000 --out hidden/strong/runs/winners5.npz

Connect 6: games like the Part A offline data (house bots against each other
and against the honest public bots; hidden/make_offline_data.py). Connect 5:
games between the bots of train_strong.bot_pool. Keeps the positions where
the side to move went on to win or draw, never the moves of the weak bots.
Saves relative grids, the move played (one-hot) and the result.

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

from connectk.arena import make_openings  # noqa: E402
from connectk.game import CONNECT5, CONNECT6, Board  # noqa: E402
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


WEAK5 = {"anchor_random", "o_sloppy", "gauge_easy"}


def _positions5(args):
    n, seed = args
    from hidden.strong.train_strong import bot_pool
    rng = np.random.default_rng(seed)
    _, bots = bot_pool(CONNECT5, seed)
    grids, moves, zs = [], [], []
    for _ in range(n):
        a, b = bots[rng.integers(len(bots))], bots[rng.integers(len(bots))]
        board = Board(CONNECT5)
        for c in make_openings(CONNECT5, 1, plies=2, seed=int(rng.integers(1 << 30)))[0]:
            board.play(c)
        rec = []
        while not board.done:
            mover = a if board.player == 1 else b
            col = mover.choose(board.copy())
            rec.append((relative_grid(board), col, board.player, mover.name))
            board.play(col)
        for grid, col, p, name in rec:
            if board.winner in (0, p) and name not in WEAK5:
                grids.append(grid)
                moves.append(col)
                zs.append(0.0 if board.winner == 0 else 1.0)
    return np.array(grids, np.int8).reshape(-1, 9, 9), np.array(moves, np.int64), np.array(zs, np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", default="connect6", choices=["connect5", "connect6"])
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    chunks = args.workers * 10
    rng = np.random.default_rng(args.seed)
    jobs = [(args.games // chunks, int(rng.integers(1 << 30))) for _ in range(chunks)]
    fn = _positions5 if args.game == "connect5" else _positions
    with mp.get_context("fork").Pool(args.workers) as pool:
        parts = pool.map(fn, jobs)
    g = np.concatenate([p[0] for p in parts])
    m = np.concatenate([p[1] for p in parts])
    z = np.concatenate([p[2] for p in parts])
    t = np.eye(9 if args.game == "connect5" else 13, dtype=np.float32)[m]
    np.savez_compressed(args.out, g=g, t=t, z=z)
    print(f"{len(g)} positions -> {args.out}")


if __name__ == "__main__":
    main()
