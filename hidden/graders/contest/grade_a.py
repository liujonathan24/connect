"""Contest Part a: grade Connect 5 submissions against the held-out bots.

    python3 hidden/graders/contest/grade_a.py teams/*/submission_a
    python3 hidden/graders/contest/grade_a.py --openings 16 --workers 32 --json grades_a.json teams/*/submission_a

Each submission folder (agent.py; see connectk/agents.py) is checked for the
parameter limit, then plays every held-out bot on the same openings, each
opening twice (once with each color). Win 1, draw 0.5, loss 0.
S_a = mean score over all these games; points = 5T x S_a.

The held-out pool (students never see any of it):
  10 rule-based bots       hidden/bots/heldout.py: connect5_heldout()
  n_standard               our standard net (hidden/strong/connect5_strong.pt), greedy
  n_big                    our big net (hidden/strong/connect5_big.pt, 2.1M parameters), greedy
  n_big_search             the big net with a 64-simulation MCTS: the hardest bot
"""

import argparse
import json

from common import STRONG, ROOT, run_jobs, team_name  # noqa: F401  (ROOT sets sys.path)

from connectk.agents import MAX_PARAMS  # noqa: E402
from connectk.arena import make_openings  # noqa: E402
from connectk.game import CONNECT5  # noqa: E402

OPENING_SEED = 5_5_2026
BOT_SEED = 77


def heldout_refs():
    from hidden.bots.heldout import connect5_heldout
    refs = {b.name: ("bot", "connect5_heldout", b.name, BOT_SEED) for b in connect5_heldout(BOT_SEED)}
    refs["n_standard"] = ("net", str(STRONG / "connect5_strong.pt"), 0, "n_standard")
    refs["n_big"] = ("net", str(STRONG / "connect5_big.pt"), 0, "n_big")
    refs["n_big_search"] = ("net", str(STRONG / "connect5_big.pt"), 64, "n_big_search")
    return refs


def precheck(folder):
    """Parameter count, or the reason the submission cannot play."""
    import torch
    torch.set_num_threads(1)
    from connectk.agents import count_parameters, load_agent
    try:
        n = count_parameters(load_agent(folder))
    except Exception as exc:
        return None, f"cannot load: {type(exc).__name__}: {exc}"
    if n > MAX_PARAMS:
        return n, f"{n:,} parameters (limit {MAX_PARAMS:,})"
    return n, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--openings", type=int, default=8, help="openings per bot (games = 2 x this)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    refs = heldout_refs()
    openings = make_openings(CONNECT5, args.openings, plies=2, seed=OPENING_SEED)
    results, problems = {}, {}
    jobs = []
    for folder in args.folders:
        n, problem = precheck(folder)
        results[folder] = {"team": team_name(folder), "parameters": n, "per_bot": {}}
        if problem:
            problems[folder] = problem
            continue
        jobs += [(("team", folder), ref, "connect5", openings) for ref in refs.values()]
    for a_ref, b_ref, stats, err in run_jobs(jobs, args.workers):
        bot = b_ref[2] if b_ref[0] == "bot" else b_ref[3]
        folder = a_ref[1]
        if err:
            problems[folder] = err
            continue
        results[folder]["per_bot"][bot] = stats
    names = list(refs)
    print(f"{'team':<20} {'S_a':>6} {'points':>8}  " + " ".join(f"{n[:10]:>10}" for n in names) + "   timeouts  sec/move")
    for folder, r in sorted(results.items(), key=lambda kv: kv[1]["team"]):
        if folder in problems:
            r["S_a"] = 0.0
            r["error"] = problems[folder]
            print(f"{r['team']:<20} {0:>6.3f} {'0':>8}  NOT GRADED: {problems[folder]}")
            continue
        per = r["per_bot"]
        w = sum(s["wins"] for s in per.values())
        d = sum(s["draws"] for s in per.values())
        g = sum(s["wins"] + s["draws"] + s["losses"] for s in per.values())
        r["S_a"] = (w + 0.5 * d) / g
        moves = sum(s.get("a_moves", 0) for s in per.values())
        secs = sum(s.get("a_seconds", 0.0) for s in per.values())
        r["timeouts"] = sum(s["timeouts"] for s in per.values())
        r["mean_move_s"] = secs / max(moves, 1)
        cells = " ".join(f"{(per[n]['wins'] + 0.5 * per[n]['draws']) / (per[n]['wins'] + per[n]['draws'] + per[n]['losses']):>10.2f}"
                         for n in names)
        print(f"{r['team']:<20} {r['S_a']:>6.3f} {5 * r['S_a']:>6.2f} T  {cells}   {r['timeouts']:>8}  {r['mean_move_s']:>8.3f}")
    print(f"\n{2 * args.openings} games per bot; points = 5T x S_a")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
