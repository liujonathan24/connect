"""The held-out opponents. Students never see these.

Part A is scored by the mean score of a submission against connect6_heldout().
The pool spreads over weak, medium and strong bots, and holds styles that the
four public bots do not show (stacking, edge play, real search).

connect5_anchors() are fixed players for the Part B final tournament. They pin
the Elo scale so ratings from different events can be compared.
"""

from .engine import HeuristicBot, RandomBot, SearchBot

A6 = [0, 1, 4, 16, 64, 256, 1024]


def _s(seed, i):
    return None if seed is None else seed * 104729 + i


def connect6_heldout(seed=None):
    s = lambda i: _s(seed, i)  # noqa: E731
    return [
        RandomBot("h01_random", s(1)),
        HeuristicBot("h02_sloppy", s(2), epsilon=0.5, p_block=0.5, temperature=1.0),
        HeuristicBot("h03_novice", s(3), epsilon=0.15, p_block=0.75, avoid_gift=False, temperature=0.7),
        HeuristicBot("h04_stacker", s(4), stack=3.0, low=0.5, center=0.0, p_block=0.9),
        HeuristicBot("h05_edge", s(5), center=-6.0, epsilon=0.05),
        HeuristicBot("h06_rusher", s(6), attack=[0, 1, 8, 64, 512, 4096, 32768],
                     defense=[0, 0, 0.5, 2, 10, 50, 0], avoid_gift=False),
        HeuristicBot("h07_balanced", s(7), attack=A6, defense=[0, 1, 4, 16, 64, 256, 0], temperature=0.3),
        HeuristicBot("h08_turtle", s(8), attack=[0, 0.5, 2, 6, 20, 60, 200],
                     defense=[0, 3, 12, 48, 192, 768, 0], lookahead=0.3),
        HeuristicBot("h09_planner", s(9), attack=A6, defense=[0, 1.2, 5, 20, 80, 320, 0], lookahead=0.8),
        SearchBot("h10_search", s(10), depth=2, defense_ratio=1.0, noise=0.1),
        SearchBot("h11_search_def", s(11), depth=2, defense_ratio=1.6, noise=0.1),
        HeuristicBot("h12_mixer", s(12), attack=[0, 1, 5, 25, 125, 625, 3125],
                     defense=[0, 1, 5, 25, 125, 625, 0], lookahead=0.4, temperature=0.6, epsilon=0.05),
    ]


def connect5_anchors(seed=None):
    s = lambda i: _s(seed, 100 + i)  # noqa: E731
    return {
        "anchor_random": RandomBot("anchor_random", s(1)),
        "anchor_heuristic": HeuristicBot("anchor_heuristic", s(2), attack=[0, 1, 4, 16, 64, 256],
                                         defense=[0, 1, 4, 16, 64, 0], temperature=0.2),
        "anchor_search": SearchBot("anchor_search", s(3), depth=2, noise=0.05),
    }
