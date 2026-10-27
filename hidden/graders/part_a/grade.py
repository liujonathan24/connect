"""Grade Part A submissions: play each one against the held-out bots.

    python3 hidden/graders/part_a/grade.py team1.pt team2.pt ...
    python3 hidden/graders/part_a/grade.py --openings 16 --json results.json submissions/*.pt

A submission is a state_dict for connectk.net.ConnectNet(CONNECT6). The
grader loads it strictly (any change to the architecture fails) and plays the
greedy policy: the legal column with the highest logit. No search.

Every held-out bot plays each opening twice, once with each color. A win
counts 1, a draw 0.5, a loss 0 (an illegal move cannot happen: the grader
masks illegal columns). The score of a submission is the mean over the
held-out bots of its score against that bot, so each bot weighs the same.

The pool: twelve rule-based bots (hidden/bots/heldout.py) and, when its
weights exist, two bots built on the trained network
hidden/strong/connect6_strong.pt: h13_neural plays it with an MCTS on top
(hidden/strong/players.py), h14_net plays it greedily, like a submission.
"""

import argparse
import json
import multiprocessing as mp
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from connectk.arena import make_openings, play_match  # noqa: E402
from connectk.game import CONNECT6  # noqa: E402

SPEC = CONNECT6
STRONG = ROOT / "hidden/strong/connect6_strong.pt"
STRONG_SIMS = 100
OPENING_SEED = 20261027
BOT_SEED = 99


def heldout_pool():
    """Fresh held-out players: name -> player."""
    from hidden.bots.heldout import connect6_heldout
    pool = {b.name: b for b in connect6_heldout(BOT_SEED)}
    if STRONG.exists():
        from connectk.net import NetPlayer, load_net
        from hidden.strong.players import NeuralMCTS
        pool["h13_neural"] = NeuralMCTS.load(SPEC, STRONG, sims=STRONG_SIMS, name="h13_neural")
        pool["h14_net"] = NetPlayer(load_net(SPEC, STRONG), "h14_net")
    return pool


def _job(args):
    path, bot_name, n_openings = args
    import torch
    torch.set_num_threads(1)
    from connectk.net import NetPlayer, load_net
    me = NetPlayer(load_net(SPEC, path), "submission")
    bot = heldout_pool()[bot_name]
    r = play_match(me, bot, SPEC, make_openings(SPEC, n_openings, plies=2, seed=OPENING_SEED))
    return path, bot_name, r.wins, r.draws, r.losses


def check(path):
    """Load the submission; return an error message or None."""
    try:
        from connectk.net import load_net
        load_net(SPEC, path)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def grade(paths, n_openings=8, workers=4):
    bots = list(heldout_pool())
    results = {p: {} for p in paths}
    errors = {}
    good = []
    for p in paths:
        err = check(p)
        if err:
            errors[p] = err
        else:
            good.append(p)
    jobs = [(p, b, n_openings) for p in good for b in bots]
    if workers > 1 and sys.platform.startswith("linux"):
        with mp.get_context("fork").Pool(workers) as pool:
            out = pool.map(_job, jobs)
    else:
        out = list(map(_job, jobs))
    for path, bot, w, d, l in out:
        results[path][bot] = {"wins": w, "draws": d, "losses": l, "score": (w + 0.5 * d) / (w + d + l)}
    summary = {}
    for p in paths:
        if p in errors:
            summary[p] = {"score": 0.0, "error": errors[p]}
        else:
            summary[p] = {"score": float(np.mean([r["score"] for r in results[p].values()])), "per_bot": results[p]}
    return bots, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--openings", type=int, default=8, help="openings per bot (games = 2 x this)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    bots, summary = grade(args.paths, args.openings, args.workers)
    width = max(len(pathlib.Path(p).name) for p in args.paths)
    short = [b.split("_", 1)[0] for b in bots]
    print(f"{'submission':<{width}}  {'SCORE':>6}  " + " ".join(f"{s:>5}" for s in short))
    for p, s in sorted(summary.items(), key=lambda kv: -kv[1]["score"]):
        name = pathlib.Path(p).name
        if "error" in s:
            print(f"{name:<{width}}  {0:>6.3f}  INVALID: {s['error']}")
            continue
        print(f"{name:<{width}}  {s['score']:>6.3f}  " + " ".join(f"{s['per_bot'][b]['score']:>5.2f}" for b in bots))
    print("\ncolumns: " + ", ".join(bots))
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
