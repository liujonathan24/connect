"""Load packed opponent bots (the .bin files in the handouts).

The bots are packed on purpose. Treat them as opponents on a remote server:
you can play them and record what they do, but do not unpack or read them.
That is a rule of the event.
"""

import base64
import json
import pathlib
import sys
import types
import zlib

_KEY = b"connect-k/packed/v1"


def _unmask(data):
    return bytes(b ^ _KEY[i % len(_KEY)] for i, b in enumerate(data))


def _module(path):
    path = pathlib.Path(path)
    name = "_packed_" + path.stem.replace("-", "_")
    if name in sys.modules:
        return sys.modules[name]
    blob = json.loads(zlib.decompress(_unmask(base64.b64decode(path.read_bytes()))))
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod          # lets pickle and forked workers find the classes
    exec(compile(blob["code"], f"<{path.name}>", "exec"), mod.__dict__)
    mod.ROSTERS = blob["rosters"]
    return mod


def load_packed(path, roster=None, seed=None):
    """Return {name: bot} from a packed file. Each bot has choose(board) and play(board)."""
    mod = _module(path)
    roster = roster or mod.ROSTERS[0]
    return getattr(mod, roster)(seed)


def pack(code, rosters):
    data = json.dumps({"code": code, "rosters": rosters}).encode()
    return base64.b64encode(_unmask(zlib.compress(data, 9)))
