"""Contest Part b: the Connect 6 round robin between all teams.

    python3 hidden/graders/contest/tournament_b.py teams/*/submission_b
    python3 hidden/graders/contest/tournament_b.py --openings 8 --workers 32 --reference --json b.json teams/*/submission_b

Every pair of teams plays the same openings, each opening twice (once with
each color). Win 1, draw 0.5, loss 0. S_b = a team's mean score over its
games against other teams; points = 10T x S_b.

--reference also lets our Connect 6 net (hidden/strong/connect6_strong.pt)
play every team, greedy and with a 64-simulation MCTS. Those games are shown
for reference and do NOT count in S_b.

Time: a Connect 6 game lasts up to 169 moves. With slow bots (1 s per move)
a pair of teams needs about 30 minutes per opening pair; use many workers.
"""

import argparse
import itertools
import json

from common import STRONG, run_jobs, team_name  # noqa: F401

from connectk.arena import make_openings  # noqa: E402
from connectk.game import CONNECT6  # noqa: E402
from connectk.tournament import fit_elo  # noqa: E402

OPENING_SEED = 6_6_2026


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--openings", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reference", action="store_true", help="also play our Connect 6 net (not counted)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    import numpy as np
    from grade_a import precheck

    teams, problems = {}, {}
    for folder in args.folders:
        _, problem = precheck(folder)
        (problems if problem else teams)[folder] = problem or team_name(folder)
    refs = {}
    if args.reference:
        refs = {"ref_net": ("net", str(STRONG / "connect6_strong.pt"), 0, "ref_net"),
                "ref_net_search": ("net", str(STRONG / "connect6_strong.pt"), 64, "ref_net_search")}
    openings = make_openings(CONNECT6, args.openings, plies=2, seed=OPENING_SEED)
    folders = list(teams)
    jobs = [(("team", a), ("team", b), "connect6", openings) for a, b in itertools.combinations(folders, 2)]
    jobs += [(("team", a), ref, "connect6", openings) for a in folders for ref in refs.values()]

    names = [teams[f] for f in folders] + list(refs)
    idx = {n: i for i, n in enumerate(names)}
    wins = np.zeros((len(names), len(names)))
    draws = np.zeros_like(wins)
    timeouts, seconds, moves = np.zeros(len(names)), np.zeros(len(names)), np.zeros(len(names))
    for a_ref, b_ref, stats, err in run_jobs(jobs, args.workers):
        a = teams[a_ref[1]]
        b = teams[b_ref[1]] if b_ref[0] == "team" else b_ref[3]
        if err:          # a crashing bot loses every game of the pair
            n = 2 * len(openings)
            stats = {"wins": 0, "draws": 0, "losses": n, "timeouts": 0}
        i, j = idx[a], idx[b]
        wins[i, j] += stats["wins"]
        wins[j, i] += stats["losses"]
        draws[i, j] += stats["draws"]
        draws[j, i] += stats["draws"]
        timeouts[i] += stats["timeouts"]
        for side, k in (("a", i), ("b", j)):
            seconds[k] += stats.get(f"{side}_seconds", 0.0)
            moves[k] += stats.get(f"{side}_moves", 0)

    nt = len(folders)
    points = (wins + 0.5 * draws)[:nt, :nt].sum(axis=1)
    games = (wins + wins.T + draws)[:nt, :nt].sum(axis=1)
    score = points / np.maximum(games, 1)
    elo = fit_elo(wins, draws, names)
    order = sorted(range(nt), key=lambda i: (-score[i], -elo[i]))
    print(f"{'#':>3}  {'team':<20} {'S_b':>6} {'points':>9} {'games':>6} {'elo':>6} {'timeouts':>9} {'sec/move':>9}")
    for rank, i in enumerate(order, 1):
        print(f"{rank:>3}  {names[i]:<20} {score[i]:>6.3f} {10 * score[i]:>7.2f} T {int(games[i]):>6} {elo[i]:>6.0f} "
              f"{int(timeouts[i]):>9} {seconds[i] / max(moves[i], 1):>9.3f}")
    for f, p in problems.items():
        print(f"  --  {team_name(f):<20} NOT GRADED: {p}")
    if refs:
        print("\nreference games (not counted): score of the team against our net")
        for i in range(nt):
            cells = []
            for r in refs:
                j = idx[r]
                g = wins[i, j] + wins[j, i] + draws[i, j]
                cells.append(f"{r} {(wins[i, j] + 0.5 * draws[i, j]) / max(g, 1):.2f}")
            print(f"     {names[i]:<20} " + "   ".join(cells))
    print(f"\n{2 * len(openings)} games per pair; points = 10T x S_b")
    if args.json:
        out = {"teams": {names[i]: {"S_b": float(score[i]), "elo": float(elo[i]), "games": int(games[i])} for i in range(nt)},
               "not_graded": {team_name(f): p for f, p in problems.items()}}
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
