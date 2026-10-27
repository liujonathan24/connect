"""Make the Part A offline trajectories, with poison mixed in.

    python3 hidden/make_offline_data.py            # about 5 minutes on 4 cores

Writes
    part_a_connect6/data/offline_games.jsonl.gz    (students get this)
    hidden/data/offline_manifest.json              (what is wrong with each game)

Each line of the students' file is one game:

    {"id": 17, "p1": "bot_b", "p2": "house_3",
     "moves": [[player, row, col], ...], "winner": 0 | 1 | 2}

Where the games come from:
  * "house_*" bots: strong internal bots (styles from the held-out pool).
  * bot_a, bot_b, bot_d: the public honest bots.
  * bot_c: the cheater, on a live board. Most of its games hold rule breaks.

Poison added on purpose to some other games (one kind per game):
    off_board     a move outside the board (column -1 or 13, or row 13)
    occupied      a stone on a cell that already holds one
    floating      a stone above an empty cell
    wrong_turn    one player moves twice in a row
    after_end     moves go on after a player has 6 in a row
    wrong_winner  the winner label is wrong
"""

import gzip
import json
import multiprocessing as mp
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from connectk.game import CONNECT6, Board  # noqa: E402
from hidden.bots.engine import HeuristicBot, SearchBot  # noqa: E402
from hidden.bots.heldout import connect6_heldout  # noqa: E402
from hidden.bots.public import connect6_public  # noqa: E402
from hidden.live import play_live  # noqa: E402

SPEC = CONNECT6
N_HOUSE = 4200       # house vs house / house vs public
N_PUBLIC = 600       # public honest vs public honest
N_CHEAT = 700        # games with bot_c
P_POISON = 0.08      # share of the non-cheater games that get one poison
POISONS = ["off_board", "occupied", "floating", "wrong_turn", "after_end", "wrong_winner"]


def house_bots(seed):
    held = {b.name: b for b in connect6_heldout(seed)}
    keep = ["h07_balanced", "h08_turtle", "h09_planner", "h10_search", "h11_search_def", "h12_mixer", "h04_stacker"]
    bots = [held[n] for n in keep]
    bots.append(HeuristicBot("x", seed, defense=[0, 1.5, 6, 24, 96, 384, 0], lookahead=0.7))
    bots.append(SearchBot("y", seed, depth=2, defense_ratio=1.3, noise=0.15))
    for i, b in enumerate(bots, 1):
        b.name = f"house_{i}"
    return bots


def _games(job):
    kind, n, seed = job
    rng = np.random.default_rng(seed)
    house = house_bots(seed)
    pub = connect6_public(seed)
    honest_pub = [pub["bot_a"], pub["bot_b"], pub["bot_d"]]
    out = []
    for _ in range(n):
        if kind == "house":
            a = house[rng.integers(len(house))]
            b = house[rng.integers(len(house))] if rng.random() < 0.75 else honest_pub[rng.integers(3)]
        elif kind == "public":
            a, b = honest_pub[rng.integers(3)], honest_pub[rng.integers(3)]
        else:
            a = pub["bot_c"]
            b = (house + honest_pub)[rng.integers(len(house) + 3)]
        if rng.random() < 0.5:
            a, b = b, a
        opening = tuple(int(c) for c in rng.integers(3, 10, size=2))
        moves, winner, _ = play_live(a, b, SPEC, opening)
        out.append({"p1": a.name, "p2": b.name, "moves": [m[:3] for m in moves], "winner": winner,
                    "breaks": sorted({m[3] for m in moves if m[3] != "drop"})})
    return out


def poison(game, kind, rng):
    """Change one game in place to carry one poison of the given kind."""
    moves = game["moves"]
    R, C = SPEC.rows, SPEC.cols
    if kind == "wrong_winner":
        game["winner"] = {0: int(rng.integers(1, 3)), 1: 2, 2: 1}[game["winner"]]
        return True
    if kind == "after_end":
        if not game["winner"]:
            return False
        b = _replay(moves)
        for _ in range(int(rng.integers(1, 4))):
            legal = b.legal_moves() or [0]
            c = int(rng.choice(legal))
            r = int(b.heights[c])
            moves.append([b.player, r, c])
            b.grid[r, c] = b.player
            b.heights[c] += 1
            b.player = 3 - b.player
        return True
    t = int(rng.integers(4, max(5, len(moves) - 2)))
    if t >= len(moves):
        return False
    b = _replay(moves[:t])
    p = moves[t][0]
    if kind == "off_board":
        options = [[p, 0, -1], [p, 0, C], [p, R, int(rng.integers(C))]]
        moves[t] = options[rng.integers(3)]
        return True
    if kind == "occupied":
        filled = np.argwhere(b.grid != 0)
        r, c = filled[rng.integers(len(filled))]
        moves[t] = [p, int(r), int(c)]
        return True
    if kind == "floating":
        cols = [c for c in range(C) if b.heights[c] + 1 < R]
        c = int(rng.choice(cols))
        h = int(b.heights[c])
        moves[t] = [p, h + int(rng.integers(1, min(4, R - h))), c]
        return True
    if kind == "wrong_turn":
        moves[t] = [3 - p, moves[t][1], moves[t][2]]
        return True
    return False


def _replay(moves):
    b = Board(SPEC)
    for p, r, c in moves:
        b.grid[r, c] = p
        b.heights[c] = max(b.heights[c], r + 1)
        b.player = 3 - p
    return b


def main():
    rng = np.random.default_rng(2026)
    jobs = []
    for kind, n in (("house", N_HOUSE), ("public", N_PUBLIC), ("cheat", N_CHEAT)):
        k = 16
        jobs += [(kind, n // k, int(rng.integers(1 << 30))) for _ in range(k)]
    with mp.get_context("fork").Pool(4) as pool:
        games = [g for part in pool.map(_games, jobs) for g in part]
    order = rng.permutation(len(games))
    games = [games[i] for i in order]
    manifest = []
    for gid, g in enumerate(games):
        g["id"] = gid
        problems = list(g.pop("breaks"))
        if not problems and g["p1"] != "bot_c" and g["p2"] != "bot_c" and rng.random() < P_POISON:
            kind = POISONS[rng.integers(len(POISONS))]
            if poison(g, kind, rng):
                problems = [kind]
        manifest.append({"id": gid, "p1": g["p1"], "p2": g["p2"], "problems": problems,
                         "cheater_game": "bot_c" in (g["p1"], g["p2"])})
    out = ROOT / "part_a_connect6/data/offline_games.jsonl.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt") as f:
        for g in games:
            f.write(json.dumps({"id": g["id"], "p1": g["p1"], "p2": g["p2"], "moves": g["moves"],
                                "winner": g["winner"]}, separators=(",", ":")) + "\n")
    man = ROOT / "hidden/data/offline_manifest.json"
    man.parent.mkdir(parents=True, exist_ok=True)
    man.write_text(json.dumps(manifest, separators=(",", ":")))
    bad = [m for m in manifest if m["problems"]]
    kinds = {}
    for m in bad:
        for p in m["problems"]:
            kinds[p] = kinds.get(p, 0) + 1
    print(f"{len(games)} games, {len(bad)} with problems: {kinds}")
    print(f"cheater games: {sum(m['cheater_game'] for m in manifest)}, "
          f"of which with rule breaks: {sum(bool(m['problems']) for m in manifest if m['cheater_game'])}")


if __name__ == "__main__":
    main()
