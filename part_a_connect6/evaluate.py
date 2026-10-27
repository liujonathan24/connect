"""Play your network against the four public bots, the way the grader plays.

    python3 evaluate.py runs/ppo/submission.pt
    python3 evaluate.py runs/ppo/submission.pt --openings 16

The referee here is strict: each bot gets a copy of the board and returns a
column, and the referee checks it. Your net plays greedily (argmax of the
policy over legal columns). Each opening is played twice, once with each
color. Score: win 1, draw 0.5, loss 0.

The real grader uses a different set of bots (held out). This script only
tells you how you do against the four you have.
"""

import argparse

import numpy as np

from connectk.arena import make_openings, play_match
from connectk.net import NetPlayer, load_net
from env import SPEC, load_opponents


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights")
    ap.add_argument("--openings", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    me = NetPlayer(load_net(SPEC, args.weights), "you")
    openings = make_openings(SPEC, args.openings, plies=2, seed=args.seed)
    scores = []
    for name, bot in load_opponents(seed=args.seed).items():
        r = play_match(me, bot, SPEC, openings)
        scores.append(r.score)
        print(r)
    print(f"mean score {np.mean(scores):.3f}")


if __name__ == "__main__":
    main()
