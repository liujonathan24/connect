"""Contest submissions, the handout and the notebooks' sources.  python3 -m unittest discover tests"""

import pathlib
import subprocess
import sys
import tempfile
import textwrap
import unittest
import zipfile

from connectk import CONNECT5, CONNECT6, Board
from connectk.agents import MoveTimeout, TimedAgent, count_parameters, load_agent
from connectk.arena import RandomPlayer, play_game

ROOT = pathlib.Path(__file__).resolve().parent.parent

AGENT = """
import pathlib, torch
from model import COLUMN

class Agent:
    def __init__(self, folder):
        self.net = torch.nn.Linear(4, {size})
    def choose(self, board):
        legal = board.legal_moves()
        return COLUMN if COLUMN in legal else legal[0]

def make_agent(folder):
    return Agent(pathlib.Path(folder))
"""


def write_submission(folder, column, size=2):
    folder.mkdir(parents=True)
    (folder / "model.py").write_text(f"COLUMN = {column}\n")
    (folder / "agent.py").write_text(textwrap.dedent(AGENT.format(size=size)))
    return folder


class TestSubmissions(unittest.TestCase):
    def test_two_submissions_keep_their_own_model_py(self):
        with tempfile.TemporaryDirectory() as d:
            a = load_agent(write_submission(pathlib.Path(d) / "a", column=1))
            b = load_agent(write_submission(pathlib.Path(d) / "b", column=7))
            board = Board(CONNECT5)
            self.assertEqual(a.choose(board), 1)
            self.assertEqual(b.choose(board), 7)
            self.assertNotIn("model", sys.modules)

    def test_parameter_count(self):
        with tempfile.TemporaryDirectory() as d:
            agent = load_agent(write_submission(pathlib.Path(d) / "a", column=1, size=10))
            self.assertEqual(count_parameters(agent), 4 * 10 + 10)

    def test_missing_net_is_an_error(self):
        class NoNet:
            def choose(self, board):
                return 0
        with self.assertRaises(TypeError):
            count_parameters(NoNet())

    def test_slow_move_loses_the_game(self):
        import time

        class Slow:
            def choose(self, board):
                time.sleep(0.05)
                return board.legal_moves()[0]
        g = play_game(TimedAgent(Slow(), "slow", limit=0.01), RandomPlayer(seed=0), CONNECT5)
        self.assertEqual(g.winner, 2)
        self.assertTrue(g.reason.startswith("error: MoveTimeout"))
        self.assertTrue(issubclass(MoveTimeout, Exception))


class TestHeldoutBots(unittest.TestCase):
    def test_team_az_loads_and_plays(self):
        for part, spec in (("a", CONNECT5), ("b", CONNECT6)):
            agent = load_agent(ROOT / f"hidden/heldout_bots/team_az/submission_{part}")
            board = Board(spec)
            board.play(spec.cols // 2)
            self.assertTrue(board.is_legal(agent.choose(board.copy())))
            self.assertLess(count_parameters(agent), 1_000_000)

    def test_graders_list_the_new_hardest_bot(self):
        sys.path.insert(0, str(ROOT / "hidden/graders/contest"))
        from grade_a import heldout_refs
        refs = heldout_refs()
        self.assertEqual(refs["n_team_az"][0], "agent")
        self.assertEqual(len(refs), 14)


class TestHandout(unittest.TestCase):
    def test_builders_up_to_date(self):
        for cmd in (["hidden/build_blobs.py", "--check"], ["hidden/build_contest.py", "--check"]):
            r = subprocess.run([sys.executable, *cmd], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)

    def test_handout_has_no_weights_or_held_out_bots(self):
        with tempfile.TemporaryDirectory() as d:
            dest = pathlib.Path(d) / "handout"
            r = subprocess.run([sys.executable, "hidden/make_contest.py", str(dest)], cwd=ROOT,
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(sorted(p.name for p in dest.iterdir()),
                             ["PROBLEM.md", "connect5.zip", "notebook_a.ipynb", "notebook_b.ipynb"])
            names = zipfile.ZipFile(dest / "connect5.zip").namelist()
            self.assertIn("bots.bin", names)
            self.assertIn("connectk/agents.py", names)
            self.assertFalse([n for n in names if n.endswith((".pt", ".npz")) or n.startswith("hidden")])
            for nb in ("notebook_a.ipynb", "notebook_b.ipynb"):
                text = (dest / nb).read_text()
                for word in ("hidden/", "_strong.pt", "connect5_big"):
                    self.assertNotIn(word, text, f"{nb} mentions {word}")

    def test_contest_bots(self):
        from connectk.packed import load_packed
        bots = load_packed(ROOT / "contest/bots.bin", "connect5_bots", seed=0)
        self.assertEqual(sorted(bots), ["easy", "hard", "medium", "random"])
        gauges = load_packed(ROOT / "contest/bots.bin", "connect6_gauges", seed=0)
        self.assertEqual(sorted(gauges), ["easy6", "hard6"])


if __name__ == "__main__":
    unittest.main()
