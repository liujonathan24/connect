"""Command line for the connect-k games.

    python3 -m connectk play --game connect5                 # you vs a random player
    python3 -m connectk play --game connect5 --vs my.pt      # you vs your network
    python3 -m connectk match a.pt b.pt --game connect5 --openings 20
    python3 -m connectk tournament runs/*.pt --game connect5 --workers 4
    python3 -m connectk watch a.pt b.pt --game connect5 --out replay.html
    python3 -m connectk webplay my.pt --game connect5 --out play.html   # play your net in a browser

Where a player is asked for, give a .pt file (a network, played greedily),
"random", or a name from a packed bot file given with --bots
(for example --bots opponents.bin and then bot_a).
"""

import argparse
import pathlib
import sys

from .arena import RandomPlayer, make_openings, play_match
from .game import SPECS, Board, IllegalMove


def make_player(spec, ref, bots=None, seed=None):
    if ref == "random":
        return RandomPlayer(seed=seed)
    if ref.endswith(".pt"):
        from .net import NetPlayer, load_net
        return NetPlayer(load_net(spec, ref), pathlib.Path(ref).stem)
    if bots and ref in bots:
        return bots[ref]
    sys.exit(f"unknown player {ref!r}: give a .pt file, 'random', or a bot name with --bots")


def load_bots(path, seed):
    if not path:
        return {}
    from .packed import load_packed
    return load_packed(path, seed=seed)


class Human:
    name = "you"

    def choose(self, board):
        print(board.render())
        while True:
            s = input(f"your move (column 0-{board.spec.cols - 1}, q to quit): ").strip()
            if s == "q":
                sys.exit(0)
            if s.isdigit() and board.is_legal(int(s)):
                return int(s)
            print("not a legal column")


def cmd_play(args):
    spec = SPECS[args.game]
    bots = load_bots(args.bots, args.seed)
    other = make_player(spec, args.vs, bots, args.seed)
    board = Board(spec)
    players = {1: Human(), 2: other} if args.first else {1: other, 2: Human()}
    while not board.done:
        p = players[board.player]
        col = p.choose(board.copy())
        try:
            board.play(col)
        except IllegalMove:
            print(f"{getattr(p, 'name', 'player')} played an illegal move and loses")
            return
        if not isinstance(p, Human):
            print(f"{getattr(p, 'name', 'bot')} plays column {col}")
    print(board.render())
    print("draw" if board.winner == 0 else f"{getattr(players[board.winner], 'name', board.winner)} wins")


def cmd_match(args):
    spec = SPECS[args.game]
    bots = load_bots(args.bots, args.seed)
    a, b = make_player(spec, args.a, bots, args.seed), make_player(spec, args.b, bots, args.seed)
    print(play_match(a, b, spec, make_openings(spec, args.openings, seed=args.seed or 0)))


def cmd_tournament(args):
    from .tournament import round_robin
    spec = SPECS[args.game]
    bots = load_bots(args.bots, args.seed)
    players = {}
    for ref in args.players:
        p = make_player(spec, ref, bots, args.seed)
        players[getattr(p, "name", ref) if not ref.endswith(".pt") else pathlib.Path(ref).stem] = p
    if any(r.endswith(".pt") for r in args.players):
        import torch
        torch.set_num_threads(1)
    table = round_robin(players, spec, make_openings(spec, args.openings, seed=args.seed or 0),
                        workers=args.workers, anchor=(args.anchor, 1000.0) if args.anchor else None)
    print(table)


def cmd_watch(args):
    from .viewer import record_game, write_html
    spec = SPECS[args.game]
    bots = load_bots(args.bots, args.seed)
    a, b = make_player(spec, args.a, bots, args.seed), make_player(spec, args.b, bots, args.seed)
    recs = []
    for op in make_openings(spec, args.games, seed=args.seed or 0):
        recs.append(record_game(spec, a, b, op))
        recs.append(record_game(spec, b, a, op))
    for k, r in enumerate(recs):
        r["title"] = f"{k + 1}. {r['players'][0]} vs {r['players'][1]}"
    write_html(recs, args.out, f"{spec.name}: {args.a} vs {args.b}")
    print(f"wrote {args.out} ({len(recs)} games); open it in a browser")


def cmd_webplay(args):
    from .webplay import net_entry, write_html
    spec = SPECS[args.game]
    entries = [net_entry(spec, p, pathlib.Path(p).stem) for p in args.nets]
    write_html(entries, args.out, args.title)
    print(f"wrote {args.out}; open it in a browser and play")


def main():
    ap = argparse.ArgumentParser(prog="python3 -m connectk", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--game", default="connect5", choices=sorted(SPECS))
        p.add_argument("--bots", default=None, help="packed bot file (.bin)")
        p.add_argument("--seed", type=int, default=None)

    p = sub.add_parser("play", help="play in the terminal")
    common(p)
    p.add_argument("--vs", default="random")
    p.add_argument("--second", dest="first", action="store_false", help="let the other player move first")
    p.set_defaults(func=cmd_play)

    p = sub.add_parser("match", help="two players, every opening with both colors")
    common(p)
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--openings", type=int, default=10)
    p.set_defaults(func=cmd_match)

    p = sub.add_parser("tournament", help="round robin with Elo")
    common(p)
    p.add_argument("players", nargs="+")
    p.add_argument("--openings", type=int, default=10)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--anchor", default=None, help="player fixed at Elo 1000")
    p.set_defaults(func=cmd_tournament)

    p = sub.add_parser("watch", help="write an HTML replay of games between two players")
    common(p)
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--games", type=int, default=2, help="openings (each played with both colors)")
    p.add_argument("--out", default="replay.html")
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("webplay", help="write an HTML page to play against a network in the browser")
    p.add_argument("nets", nargs="+", help="network .pt files (all for --game)")
    p.add_argument("--game", default="connect5", choices=sorted(SPECS))
    p.add_argument("--out", default="play.html")
    p.add_argument("--title", default="Play Connect")
    p.set_defaults(func=cmd_webplay)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
