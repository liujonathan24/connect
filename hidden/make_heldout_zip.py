"""Make a zip of the held-out bots for collaborators (organizers only).

    python3 hidden/make_heldout_zip.py out.zip --strengths table5.txt table6.txt --replays replays.html

It holds the full bot code, our trained nets, the contest graders, the viewer
(hidden/view_heldout.py) and reference submissions, in the repo's layout, so
everything runs from the unzipped folder. Never give it to participants.
"""

import argparse
import pathlib
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOP = "connect-heldout"

FILES = [
    "requirements.txt", "contest/PROBLEM.md", "contest/bots.bin",
    "hidden/__init__.py", "hidden/view_heldout.py",
    "hidden/bots/__init__.py", "hidden/bots/engine.py", "hidden/bots/public.py", "hidden/bots/heldout.py",
    "hidden/strong/bignet.py", "hidden/strong/mcts.py", "hidden/strong/players.py", "hidden/strong/NOTES.md",
    "hidden/strong/connect5_strong.pt", "hidden/strong/connect5_big.pt", "hidden/strong/connect6_strong.pt",
    "hidden/graders/contest/common.py", "hidden/graders/contest/grade_a.py", "hidden/graders/contest/tournament_b.py",
    "hidden/solutions/contest/ref_a/agent.py", "hidden/solutions/contest/ref_b/agent.py",
]

README = """# Held-out bots: organizers only

**Do not share this folder, or anything from it, with participants.** It holds
the bots and networks that grade the contest (contest/PROBLEM.md). A team
that sees them can tune its bot against the test.

## What is inside

| Path | What it is |
|---|---|
| `views/STRENGTHS.md` | Every player ranked: the student bots, the held-out bots, our nets |
| `views/replays.html` | Games of the held-out bots and our nets, move by move (open in a browser) |
| `views/play.html` | Play against our nets in a browser (big and standard Connect 5 net, Connect 6 net) |
| `hidden/bots/engine.py` | The bot engine: heuristic bots, alpha-beta search, random |
| `hidden/bots/public.py` | The bots students get (packed in `contest/bots.bin`): `connect5_bots`, `connect6_gauges` |
| `hidden/bots/heldout.py` | The held-out pools; the contest uses `connect5_heldout` |
| `hidden/strong/*.pt` | Our trained nets: `connect5_strong.pt` (0.19M parameters), `connect5_big.pt` (2.1M), `connect6_strong.pt` (0.2M) |
| `hidden/strong/NOTES.md` | How the nets were made, what worked and what did not |
| `hidden/heldout_bots/team_az/` | A test-solve by another agent (net + its own MCTS), now the hardest bot: `n_team_az` (Part a), `ref_team_az` (Part b reference) |
| `hidden/graders/contest/` | The graders: `grade_a.py` (Part a) and `tournament_b.py` (Part b) |
| `hidden/view_heldout.py` | List the held-out bots, rank them, write replays of any two players |
| `hidden/solutions/contest/` | Reference submissions (our standard nets, greedy), to check the graders |

## The pools

**Part a (Connect 5), graded:** 14 bots. 10 rule-based bots
(`connect5_heldout` in `hidden/bots/heldout.py`); 3 bots on our nets:
`n_standard` (standard net, greedy), `n_big` (big net, greedy), `n_big_search`
(big net with a 64-simulation MCTS); and `n_team_az` (a test-solve by another
agent: a net with its own MCTS, 0.75 s per move; the hardest bot). A
submission's score is its mean over all of them, 16 games each by default.

**Part b (Connect 6):** teams play each other; our Connect 6 net (`ref_net`,
greedy, and `ref_net_search`) and the agent's test-solve (`ref_team_az`) can
join as references that do not count.

## Quick start

Python 3.10+, then `pip install -r requirements.txt` (numpy, torch). From
this folder:

```sh
python3 hidden/view_heldout.py list                                   # every held-out player
python3 hidden/view_heldout.py watch c5_planner n_big_search --out replay.html
python3 hidden/view_heldout.py watch path/to/submission_a hard        # a team's bot vs a bot
python3 hidden/view_heldout.py table --game connect5 --workers 4      # rank everyone (~15 min)
python3 hidden/graders/contest/grade_a.py --workers 4 teams/*/submission_a
python3 hidden/graders/contest/tournament_b.py --workers 4 --reference teams/*/submission_b
```

Submissions are code: run the graders in a sandbox (a container without
network access).
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--strengths", nargs="*", default=[], help="outputs of view_heldout.py table")
    ap.add_argument("--replays", default=None, help="an HTML replay page to include as views/replays.html")
    args = ap.parse_args()
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{TOP}/README.md", README)
        for rel in FILES:
            z.write(ROOT / rel, f"{TOP}/{rel}")
        for p in sorted((ROOT / "connectk").rglob("*.py")):
            z.write(p, f"{TOP}/{p.relative_to(ROOT)}")
        for p in sorted((ROOT / "hidden/heldout_bots").rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                z.write(p, f"{TOP}/{p.relative_to(ROOT)}")
        z.write(ROOT / "demo/play-connect.html", f"{TOP}/views/play.html")
        if args.replays:
            z.write(args.replays, f"{TOP}/views/replays.html")
        if args.strengths:
            parts = ["# Strengths of every player\n",
                     "Round robins from `hidden/view_heldout.py table` (each pair plays every opening twice, "
                     "once with each color; win 1, draw 0.5). Student bots: random, easy, medium, hard "
                     "(Connect 5) and easy6, hard6 (Connect 6). Elo: medium = 1000 (Connect 5), "
                     "easy6 = 1000 (Connect 6).\n"]
            for path in args.strengths:
                text = pathlib.Path(path).read_text().replace("TABLES_DONE", "").strip("\n")
                game = "Connect 6" if "connect6" in path or "6" in pathlib.Path(path).stem else "Connect 5"
                parts.append(f"\n## {game}\n\n```\n{text}\n```\n")
            z.writestr(f"{TOP}/views/STRENGTHS.md", "".join(parts))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
