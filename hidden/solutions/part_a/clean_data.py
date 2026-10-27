"""Reference: clean the Part A offline games by replaying them under the rules.

    python3 hidden/solutions/part_a/clean_data.py            # report, and compare with the manifest
    python3 hidden/solutions/part_a/clean_data.py --out clean.jsonl.gz

A game is kept only if every move is legal for the player whose turn it is,
no move comes after the end, and the label matches the real result.
"""

import argparse
import collections
import gzip
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from connectk.game import CONNECT6, Board  # noqa: E402

DATA = ROOT / "part_a_connect6/data/offline_games.jsonl.gz"
MANIFEST = ROOT / "hidden/data/offline_manifest.json"

# names the manifest uses -> what a replay check sees
SEEN_AS = {"steal": "occupied", "teleport": "floating", "extra": "wrong_turn"}


def check_game(game, spec=CONNECT6):
    """The first problem found in a game, or None if it is clean."""
    b = Board(spec)
    for p, r, c in game["moves"]:
        if b.done:
            return "after_end"
        if p != b.player:
            return "wrong_turn"
        if not (0 <= c < spec.cols and 0 <= r < spec.rows):
            return "off_board"
        if b.grid[r, c] != 0:
            return "occupied"
        if r != b.heights[c]:
            return "floating"
        b.play(c)
    if not b.done:
        return "unfinished"
    if game["winner"] != b.winner:
        return "wrong_winner"
    return None


def load(path=DATA):
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    games = load()
    found = {g["id"]: check_game(g) for g in games}
    counts = collections.Counter(v for v in found.values() if v)
    print(f"{len(games)} games, {sum(counts.values())} bad: {dict(counts)}")
    by_player = collections.Counter()
    bad_by_player = collections.Counter()
    for g in games:
        for p in (g["p1"], g["p2"]):
            by_player[p] += 1
            bad_by_player[p] += found[g["id"]] is not None
    print("bad share by player:", {p: f"{bad_by_player[p]}/{n}" for p, n in sorted(by_player.items())})
    if MANIFEST.exists():
        truth = {m["id"]: m["problems"] for m in json.loads(MANIFEST.read_text())}
        tp = sum(1 for i in found if found[i] and truth[i])
        fp = sum(1 for i in found if found[i] and not truth[i])
        fn = sum(1 for i in found if not found[i] and truth[i])
        print(f"against the manifest: caught {tp}, false alarms {fp}, missed {fn}")
    if args.out:
        with gzip.open(args.out, "wt") as f:
            for g in games:
                if found[g["id"]] is None:
                    f.write(json.dumps(g) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
