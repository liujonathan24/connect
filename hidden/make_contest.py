"""Make the contest handout: connect5.zip, notebook_a.ipynb, notebook_b.ipynb, PROBLEM.md.

    python3 hidden/make_contest.py ~/contest-handout

The folder must not exist yet. connect5.zip holds the game package
(connectk/), the bots students get (contest/bots.bin), PROBLEM.md and
requirements.txt. Nothing from hidden/ goes in: no held-out bots, no trained
networks, no graders. The build stops if it finds a weight file or a
held-out bot name in the handout.
"""

import argparse
import pathlib
import shutil
import subprocess
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
FORBIDDEN = ("connect5_strong", "connect6_strong", "connect5_big", "connect5_heldout", "connect6_heldout",
             "c5_planner", "h09_planner", "anchor_search")


def zip_files():
    files = {"PROBLEM.md": ROOT / "contest/PROBLEM.md", "bots.bin": ROOT / "contest/bots.bin",
             "requirements.txt": ROOT / "requirements.txt"}
    for p in sorted((ROOT / "connectk").rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
            files[str(p.relative_to(ROOT))] = p
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dest", type=pathlib.Path)
    args = ap.parse_args()
    for check in (["hidden/build_blobs.py", "--check"], ["hidden/build_contest.py", "--check"]):
        if subprocess.run([sys.executable, *check], cwd=ROOT).returncode:
            sys.exit("fix the stale files first")
    dest = args.dest.expanduser().resolve()
    if dest.exists():
        sys.exit(f"{dest} already exists. Use a new folder.")
    dest.mkdir(parents=True)
    files = zip_files()
    for rel, src in files.items():
        if src.suffix in (".pt", ".npz", ".pth"):
            sys.exit(f"refusing to ship a weight or data file: {rel}")
        if src.suffix in (".py", ".md", ".txt"):
            text = src.read_text()
            for word in FORBIDDEN:
                if word in text:
                    sys.exit(f"refusing: {rel} mentions {word!r}")
    with zipfile.ZipFile(dest / "connect5.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for rel, src in files.items():
            z.write(src, rel)
    for name in ("notebook_a.ipynb", "notebook_b.ipynb", "PROBLEM.md"):
        shutil.copy2(ROOT / "contest" / name, dest / name)
    print(f"handout in {dest}: connect5.zip ({len(files)} files), notebook_a.ipynb, notebook_b.ipynb, PROBLEM.md")


if __name__ == "__main__":
    main()
