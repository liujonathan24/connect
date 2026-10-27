"""Pack the public bots into the blobs that the handouts ship.

    python3 hidden/build_blobs.py          # writes the two .bin files
    python3 hidden/build_blobs.py --check  # exits 1 if a .bin is out of date

The blob holds engine.py and public.py with docstrings and comments removed
and the class names replaced, so a student who prints a traceback does not see
"CheaterBot". The held-out bots are never packed.
"""

import argparse
import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from connectk.packed import pack  # noqa: E402

RENAME = {
    "CheaterBot": "Bot3",
    "HeuristicBot": "Bot1",
    "SearchBot": "Bot2",
    "RandomBot": "Bot0",
    "p_teleport": "q1",
    "p_steal": "q2",
    "p_extra": "q3",
}

TARGETS = {
    "part_a_connect6/opponents.bin": ["connect6_public"],
    "part_b_connect5/gauges.bin": ["connect5_gauges"],
}


def strip(src):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
    tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.level and n.level > 0)]
    return ast.unparse(tree)


def build_code(rosters):
    engine = strip((ROOT / "hidden/bots/engine.py").read_text())
    public = ast.parse((ROOT / "hidden/bots/public.py").read_text())
    keep = [n for n in public.body if isinstance(n, ast.FunctionDef) and (n.name in rosters or n.name.startswith("_"))]
    funcs = strip(ast.unparse(ast.Module(body=keep, type_ignores=[])))
    code = engine + "\n\n" + funcs + "\n"
    for old, new in RENAME.items():
        code = code.replace(old, new)
    return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    stale = []
    for rel, rosters in TARGETS.items():
        data = pack(build_code(rosters), rosters)
        path = ROOT / rel
        if args.check:
            if not path.exists() or path.read_bytes() != data:
                stale.append(rel)
        else:
            path.write_bytes(data)
            print(f"wrote {rel} ({len(data)} bytes, rosters {rosters})")
    if stale:
        sys.exit(f"out of date: {', '.join(stale)}; run python3 hidden/build_blobs.py")


if __name__ == "__main__":
    main()
