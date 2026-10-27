"""Bots, blobs, the Part A environment and the offline data.  python3 -m unittest discover tests"""

import gzip
import json
import pathlib
import subprocess
import sys
import unittest

import numpy as np

from connectk import CONNECT5, CONNECT6
from connectk.arena import make_openings, play_match
from connectk.packed import load_packed

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "part_a_connect6"))


class TestBlobs(unittest.TestCase):
    def test_blobs_up_to_date(self):
        r = subprocess.run([sys.executable, "hidden/build_blobs.py", "--check"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_blob_has_no_giveaways(self):
        import base64
        import zlib

        from connectk.packed import _unmask
        for name in ("part_a_connect6/opponents.bin", "part_b_connect5/gauges.bin"):
            blob = json.loads(zlib.decompress(_unmask(base64.b64decode((ROOT / name).read_bytes()))))
            for word in ("Cheater", "cheat", "teleport", "steal", "heldout", "h09"):
                self.assertNotIn(word, blob["code"], f"{name} mentions {word!r}")

    def test_loaded_bots_are_legal_in_the_arena(self):
        bots = load_packed(ROOT / "part_a_connect6/opponents.bin", seed=0)
        self.assertEqual(sorted(bots), ["bot_a", "bot_b", "bot_c", "bot_d"])
        ops = make_openings(CONNECT6, 2, seed=0)
        names = sorted(bots)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                r = play_match(bots[a], bots[b], CONNECT6, ops)
                self.assertEqual(r.illegal_a + r.illegal_b, 0)
        gauges = load_packed(ROOT / "part_b_connect5/gauges.bin", seed=0)
        r = play_match(gauges["gauge_easy"], gauges["gauge_hard"], CONNECT5, make_openings(CONNECT5, 3))
        self.assertEqual(r.illegal_a + r.illegal_b, 0)


class TestEnv(unittest.TestCase):
    def _breaks(self, name, games=25):
        from env import BotEnv, load_opponents
        bots = load_opponents(seed=3)
        env = BotEnv({name: bots[name]}, seed=4)
        rng = np.random.default_rng(5)
        breaks = 0
        for _ in range(games):
            board, done = env.reset(), False
            while not done:
                board, done, info = env.step(int(rng.choice(board.legal_moves())))
                before = info["after_agent"]
                if before is None or before.done:
                    continue
                diff = np.argwhere(before.grid != board.grid)
                fair = (len(diff) == 1 and before.grid[tuple(diff[0])] == 0
                        and diff[0][0] == before.heights[diff[0][1]])
                breaks += not fair
        return breaks

    def test_honest_bots_play_fair_in_env(self):
        for name in ("bot_a", "bot_b", "bot_d"):
            self.assertEqual(self._breaks(name, 10), 0, name)

    def test_cheater_breaks_rules_in_env(self):
        self.assertGreater(self._breaks("bot_c"), 3)

    def test_env_flow(self):
        from env import BotEnv, load_opponents
        env = BotEnv(load_opponents(seed=0), agent_side="second", seed=0)
        board = env.reset()
        self.assertEqual(board.n_moves, 1)
        self.assertEqual(board.player, 2)
        board, done, info = env.step(99)
        self.assertTrue(done and info["illegal"])
        self.assertEqual(info["winner"], 1)

    def test_vec_env_resets(self):
        from env import VecBotEnv, load_opponents
        venv = VecBotEnv(4, load_opponents(seed=0), seed=0)
        boards = venv.reset()
        rng = np.random.default_rng(0)
        finished = 0
        for _ in range(200):
            boards, dones, infos = venv.step([rng.choice(b.legal_moves()) for b in boards])
            finished += dones.sum()
            for b, d, info in zip(boards, dones, infos):
                self.assertFalse(b.done)
                if d:
                    self.assertTrue(info["final_board"].done)
        self.assertGreater(finished, 4)


class TestOfflineData(unittest.TestCase):
    def test_reference_cleaner_matches_manifest(self):
        sys.path.insert(0, str(ROOT))
        from hidden.solutions.part_a.clean_data import check_game
        with gzip.open(ROOT / "part_a_connect6/data/offline_games.jsonl.gz", "rt") as f:
            games = [json.loads(line) for line in f]
        truth = {m["id"]: bool(m["problems"]) for m in json.loads((ROOT / "hidden/data/offline_manifest.json").read_text())}
        self.assertEqual(len(games), len(truth))
        for g in games:
            self.assertEqual(check_game(g) is not None, truth[g["id"]], g["id"])


if __name__ == "__main__":
    unittest.main()
