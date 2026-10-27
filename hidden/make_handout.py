"""Make a clean folder (a "handout") for one part of the event.

    python3 hidden/make_handout.py --part a ~/connect6-handout
    python3 hidden/make_handout.py --part b ~/connect5-handout
    python3 hidden/make_handout.py --part a --zip ~/connect6-handout   # also write a .zip

The folder must not exist yet. A handout holds no held-out bots, no answers,
no strong networks and no graders; those stay in hidden/.

  part a  Connect 6: PROBLEM.md, QUESTIONS.md, env.py, ppo.py, evaluate.py,
          opponents.bin, data/offline_games.jsonl.gz, connectk/
  part b  Connect 5: PROBLEM.md, QUESTIONS.md, selfplay.ipynb, gauges.bin, connectk/
"""

import argparse
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

COMMON = {"connectk": "connectk", "requirements.txt": "requirements.txt"}

PART_A = {
    "part_a_connect6/PROBLEM.md": "PROBLEM.md",
    "part_a_connect6/QUESTIONS.md": "QUESTIONS.md",
    "part_a_connect6/env.py": "env.py",
    "part_a_connect6/ppo.py": "ppo.py",
    "part_a_connect6/evaluate.py": "evaluate.py",
    "part_a_connect6/opponents.bin": "opponents.bin",
    "part_a_connect6/data/offline_games.jsonl.gz": "data/offline_games.jsonl.gz",
}

PART_B = {
    "part_b_connect5/PROBLEM.md": "PROBLEM.md",
    "part_b_connect5/QUESTIONS.md": "QUESTIONS.md",
    "part_b_connect5/selfplay.ipynb": "selfplay.ipynb",
    "part_b_connect5/gauges.bin": "gauges.bin",
}

README_A = """# Part A: Connect 6 (13 x 13)

Read PROBLEM.md first. Setup (Python 3.10 or newer):

    pip install -r requirements.txt
    python3 -m connectk play --game connect6     # play the game in the terminal
    python3 ppo.py --updates 20                  # the shipped trainer (it does not learn yet)
    python3 evaluate.py runs/ppo/submission.pt   # score against the four bots

Run every command from this folder. Hand in submission.pt and your answers
to QUESTIONS.md.
"""

README_B = """# Part B: Connect 5 (9 x 9) by self-play

Read PROBLEM.md first. Setup (Python 3.10 or newer):

    pip install -r requirements.txt jupyter
    jupyter notebook selfplay.ipynb
    python3 -m connectk play --game connect5 --vs checkpoints/v4.pt   # play your own network

Run everything from this folder. Hand in submission.pt and your answers to
QUESTIONS.md.
"""


def copy(mapping, dest):
    for src, rel in mapping.items():
        source, target = ROOT / src, dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(source, target)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["a", "b"], required=True)
    ap.add_argument("--zip", action="store_true")
    ap.add_argument("dest", type=pathlib.Path)
    args = ap.parse_args()
    for check in (["hidden/build_blobs.py", "--check"], ["hidden/build_notebook.py", "--check"]):
        r = subprocess.run([sys.executable, *check], cwd=ROOT)
        if r.returncode:
            sys.exit("fix the stale files first")
    dest = args.dest.expanduser().resolve()
    if dest.exists():
        sys.exit(f"{dest} already exists. Use a new folder for each handout.")
    dest.mkdir(parents=True)
    copy(COMMON, dest)
    if args.part == "a":
        copy(PART_A, dest)
        (dest / "README.md").write_text(README_A)
    else:
        copy(PART_B, dest)
        (dest / "README.md").write_text(README_B)
    print(f"Part {args.part.upper()} handout ready in {dest}")
    if args.zip:
        z = shutil.make_archive(str(dest), "zip", dest.parent, dest.name)
        print(f"zip: {z}")


if __name__ == "__main__":
    main()
