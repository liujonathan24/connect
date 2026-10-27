"""Pick the strongest checkpoint of a run with a large match against the bots.

    python3 hidden/strong/pick_best.py --game connect5 hidden/strong/runs/c5m/iter_*.pt
    python3 hidden/strong/pick_best.py --game connect5 --copy hidden/strong/connect5_strong.pt runs/c5m/iter_*.pt

Each checkpoint plays the run's evaluation panel (train_strong.panel) on
`--openings` openings, both colors, with a fixed bot seed that the run's own
evals did not use. The best mean score wins.
"""

import argparse
import multiprocessing as mp
import os
import pathlib
import shutil
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402


def _score(args):
    game, path, n_openings = args
    import torch
    torch.set_num_threads(1)
    from connectk.arena import make_openings, play_match
    from connectk.game import SPECS
    from connectk.net import NetPlayer, load_net
    from hidden.strong.train_strong import panel
    spec = SPECS[game]
    me = NetPlayer(load_net(spec, path), "net")
    ops = make_openings(spec, n_openings, plies=2, seed=31337)
    return path, {n: play_match(me, b, spec, ops).score for n, b in panel(spec, seed=4321).items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", required=True, choices=["connect5", "connect6"])
    ap.add_argument("--openings", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--copy", default=None, help="copy the best checkpoint here")
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()
    jobs = [(args.game, p, args.openings) for p in args.paths]
    with mp.get_context("fork").Pool(args.workers) as pool:
        results = pool.map(_score, jobs)
    best = None
    for path, s in results:
        mean = float(np.mean(list(s.values())))
        print(f"{path:<50} " + "  ".join(f"{n} {v:.2f}" for n, v in s.items()) + f"  mean {mean:.3f}", flush=True)
        if best is None or mean > best[1]:
            best = (path, mean)
    print(f"best: {best[0]} ({best[1]:.3f})")
    if args.copy:
        shutil.copy(best[0], args.copy)
        print(f"copied to {args.copy}")


if __name__ == "__main__":
    main()
