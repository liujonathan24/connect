"""Look at the held-out bots: list them, rank them, watch them play.

    python3 hidden/view_heldout.py list
    python3 hidden/view_heldout.py table --game connect5 --openings 4 --workers 4
    python3 hidden/view_heldout.py watch c5_planner n_big_search --game connect5 --out replay.html
    python3 hidden/view_heldout.py watch path/to/submission_a hard --game connect5

A player is a held-out bot name (see `list`), a student bot name (random,
easy, medium, hard; easy6, hard6), or a submission folder (with agent.py).
For organizers only: never share these bots with participants.
"""

import argparse
import os
import pathlib
import sys

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from connectk.game import SPECS  # noqa: E402

STRONG = ROOT / "hidden/strong"

# what each held-out player is, for `list`
ABOUT = {
    "c5_sloppy": "heuristic, random moves half the time, misses half the blocks",
    "c5_novice": "heuristic, some random moves, plays under the opponent's threats",
    "c5_stacker": "heuristic that piles stones on its own columns and plays low",
    "c5_edge": "heuristic that prefers the edge columns",
    "c5_rusher": "builds its own lines, hardly defends",
    "c5_balanced": "balanced attack and defense, a little randomness",
    "c5_turtle": "mostly defense, looks one reply ahead",
    "c5_planner": "balanced, looks one reply ahead (strong)",
    "c5_search": "alpha-beta search 2 moves deep (strong)",
    "c5_search_def": "alpha-beta 2 deep, weighs the opponent's lines more (strong)",
    "n_standard": "our standard net (0.19M parameters), greedy",
    "n_big": "our big net (2.1M parameters), greedy",
    "n_big_search": "our big net with a 64-simulation MCTS (the hardest bot)",
    "ref_net": "our Connect 6 net (0.2M parameters), greedy",
    "ref_net_search": "our Connect 6 net with a 64-simulation MCTS",
}


def pools():
    """{game: {group: {name: factory}}} for every held-out player."""
    from hidden.bots.heldout import connect5_heldout
    c5 = {b.name: (lambda n=b.name: {x.name: x for x in connect5_heldout(77)}[n]) for b in connect5_heldout(77)}
    c5_nets = {"n_standard": ("connect5_strong.pt", 0), "n_big": ("connect5_big.pt", 0), "n_big_search": ("connect5_big.pt", 64)}
    c6_nets = {"ref_net": ("connect6_strong.pt", 0), "ref_net_search": ("connect6_strong.pt", 64)}
    return {
        "connect5": {"Part a: held-out rule-based bots": c5,
                     "Part a: held-out nets": {n: (lambda n=n, v=v: net_player("connect5", *v, n)) for n, v in c5_nets.items()}},
        "connect6": {"Part b: reference nets (not counted in S_b)":
                     {n: (lambda n=n, v=v: net_player("connect6", *v, n)) for n, v in c6_nets.items()}},
    }


def net_player(game, weights, sims, name):
    import torch
    torch.set_num_threads(1)
    from connectk.net import NetPlayer
    from hidden.strong.bignet import load_model
    from hidden.strong.players import NeuralMCTS
    net = load_model(SPECS[game], STRONG / weights)
    return NeuralMCTS(net, sims, name) if sims else NetPlayer(net, name)


def student_bots(game):
    from connectk.packed import load_packed
    roster = "connect5_bots" if game == "connect5" else "connect6_gauges"
    return load_packed(ROOT / "contest/bots.bin", roster, seed=0)


def make(ref, game):
    if pathlib.Path(ref, "agent.py").exists():
        from connectk.agents import load_agent
        agent = load_agent(ref)
        agent.name = pathlib.Path(ref).resolve().name
        return agent
    for group in pools()[game].values():
        if ref in group:
            return group[ref]()
    bots = student_bots(game)
    if ref in bots:
        return bots[ref]
    sys.exit(f"unknown player {ref!r}; see `list`")


def cmd_list(args):
    for game, groups in pools().items():
        print(f"== {game}")
        for group, players in groups.items():
            print(f"  {group}")
            for name in players:
                print(f"    {name:<16} {ABOUT.get(name, '')}")
        print(f"  students get: {', '.join(student_bots(game))}")


def cmd_table(args):
    from connectk.arena import make_openings
    from connectk.tournament import round_robin
    game = args.game
    players = dict(student_bots(game))
    for group in pools()[game].values():
        for name, factory in group.items():
            players[name] = factory()
    anchor = ("medium", 1000.0) if game == "connect5" else ("easy6", 1000.0)
    table = round_robin(players, SPECS[game], make_openings(SPECS[game], args.openings, seed=3),
                        workers=args.workers, anchor=anchor)
    print(table)


def cmd_watch(args):
    from connectk.arena import make_openings
    from connectk.viewer import record_game, write_html
    spec = SPECS[args.game]
    a, b = make(args.a, args.game), make(args.b, args.game)
    recs = []
    for op in make_openings(spec, args.games, seed=args.seed):
        recs.append(record_game(spec, a, b, op))
        recs.append(record_game(spec, b, a, op))
    for k, r in enumerate(recs, 1):
        w = r["players"][r["winner"] - 1] if r["winner"] else "draw"
        r["title"] = f"{k}. {r['players'][0]} vs {r['players'][1]} · {w if w == 'draw' else w + ' wins'}"
    write_html(recs, args.out, f"{args.game}: {getattr(a, 'name', args.a)} vs {getattr(b, 'name', args.b)}")
    print(f"wrote {args.out} ({len(recs)} games)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(func=cmd_list)
    p = sub.add_parser("table")
    p.add_argument("--game", default="connect5", choices=["connect5", "connect6"])
    p.add_argument("--openings", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(func=cmd_table)
    p = sub.add_parser("watch")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--game", default="connect5", choices=["connect5", "connect6"])
    p.add_argument("--games", type=int, default=2, help="openings (each played with both colors)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="replay.html")
    p.set_defaults(func=cmd_watch)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    args.func(args)


if __name__ == "__main__":
    main()
