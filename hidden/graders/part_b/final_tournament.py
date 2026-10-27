"""Part B final: every team's network plays every other team (Connect 5, 9 x 9).

    python3 hidden/graders/part_b/final_tournament.py submissions/*.pt
    python3 hidden/graders/part_b/final_tournament.py --openings 20 --replay final.html submissions/*.pt

Each file is one team (the team name is the file name without .pt). Every
pair plays every opening twice, once with each color. Win 1, draw 0.5,
loss 0; nets play greedily (argmax over legal columns, no search).

Fixed anchors play too: a random player, two rule-based bots, and (when its
weights exist) the trained network hidden/strong/connect5_strong.pt. They
pin the Elo scale (anchor_heuristic = 1000), so you can compare events.

Placement: teams are ranked by points from all their games (anchors
included). Ties go to the better Elo, then to the head-to-head score.
"""

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from connectk.arena import make_openings, play_game  # noqa: E402
from connectk.game import CONNECT5  # noqa: E402
from connectk.net import NetPlayer, load_net  # noqa: E402
from connectk.tournament import round_robin  # noqa: E402

SPEC = CONNECT5
STRONG = ROOT / "hidden/strong/connect5_strong.pt"
OPENING_SEED = 5_2026


def anchors():
    from hidden.bots.heldout import connect5_anchors
    pool = connect5_anchors(seed=7)
    if STRONG.exists():
        pool["anchor_neural"] = NetPlayer(load_net(SPEC, STRONG), "anchor_neural")
    return pool


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--openings", type=int, default=12)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-anchors", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--replay", default=None, help="write an HTML replay of the top two teams' games")
    args = ap.parse_args()

    players, invalid = {}, {}
    for p in args.paths:
        name = pathlib.Path(p).stem
        try:
            players[name] = NetPlayer(load_net(SPEC, p), name)
        except Exception as exc:
            invalid[name] = f"{type(exc).__name__}: {exc}"
    teams = list(players)
    if not args.no_anchors:
        players.update(anchors())
    import torch
    torch.set_num_threads(1)
    openings = make_openings(SPEC, args.openings, plies=2, seed=OPENING_SEED)
    table = round_robin(players, SPEC, openings, workers=args.workers, anchor=("anchor_heuristic", 1000.0)
                        if "anchor_heuristic" in players else None, verbose=False)
    print(table)
    m = table.matrix
    names = table.names

    def key(t):
        i = names.index(t)
        return (-table.points[i], -table.elo[i])
    ranked = sorted(teams, key=key)
    print("\nFinal placement (teams only):")
    for rank, t in enumerate(ranked, 1):
        i = names.index(t)
        print(f"  {rank:>2}. {t:<24} points {table.points[i]:6.1f}   elo {table.elo[i]:6.0f}")
    for t, err in invalid.items():
        print(f"  --  {t:<24} INVALID: {err}")
    if len(ranked) >= 2:
        a, b = ranked[0], ranked[1]
        print(f"\nHead to head {a} vs {b}: {m[names.index(a), names.index(b)]:.3f}")
    if args.json:
        out = {"placement": ranked, "invalid": invalid, "names": names,
               "points": table.points.tolist(), "elo": table.elo.tolist(),
               "matrix": [[None if x != x else float(x) for x in row] for row in m]}
        pathlib.Path(args.json).write_text(json.dumps(out, indent=1))
    if args.replay and len(ranked) >= 2:
        from connectk.viewer import record_from_columns, write_html
        a, b = players[ranked[0]], players[ranked[1]]
        recs = []
        for op in openings[:3]:
            for first, second in ((a, b), (b, a)):
                g = play_game(first, second, SPEC, op)
                rec = record_from_columns(SPEC, g.moves, [first.name, second.name], len(op),
                                          f"{first.name} vs {second.name} · opening {op}", g.reason)
                rec["winner"] = g.winner
                recs.append(rec)
        write_html(recs, args.replay, "Final: top two")
        print(f"wrote {args.replay}")


if __name__ == "__main__":
    main()
