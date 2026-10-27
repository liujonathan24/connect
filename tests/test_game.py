"""Rules, encoding and referee checks.  python3 -m unittest discover tests"""

import random
import unittest

import numpy as np

from connectk import CONNECT4, CONNECT5, CONNECT6, Board, IllegalMove, replay
from connectk.arena import RandomPlayer, make_openings, play_game, play_match
from connectk.game import windows


def brute_winner(board):
    """Winner by checking every line in every direction, slowly."""
    g, k = board.grid, board.spec.k
    R, C = g.shape
    found = set()
    for r in range(R):
        for c in range(C):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                cells = [(r + dr * i, c + dc * i) for i in range(k)]
                if all(0 <= a < R and 0 <= b < C for a, b in cells):
                    vals = {int(g[a, b]) for a, b in cells}
                    if len(vals) == 1 and 0 not in vals:
                        found |= vals
    return found


class TestRules(unittest.TestCase):
    def test_gravity_and_turns(self):
        b = Board(CONNECT5)
        self.assertEqual(b.play(4), 0)
        self.assertEqual(b.play(4), 1)
        self.assertEqual(b.grid[0, 4], 1)
        self.assertEqual(b.grid[1, 4], 2)
        self.assertEqual(b.player, 1)
        self.assertEqual(b.history, [(1, 0, 4), (2, 1, 4)])

    def test_illegal(self):
        b = Board(CONNECT4)
        for _ in range(6):
            b.play(0)
        with self.assertRaises(IllegalMove):
            b.play(0)
        for bad in (-1, 7, 2.0, "3", None):
            with self.assertRaises(IllegalMove):
                b.play(bad)
        self.assertNotIn(0, b.legal_moves())

    def test_each_direction(self):
        k = CONNECT6.k
        horizontal = [c for i in range(k - 1) for c in (i, i)] + [k - 1]
        b = replay(CONNECT6, horizontal)
        self.assertEqual(b.winner, 1)
        vertical = [0, 1] * (k - 1) + [0]
        self.assertEqual(replay(CONNECT6, vertical).winner, 1)
        # diagonal up-right for player 1 in connect4: a known sequence
        self.assertEqual(replay(CONNECT4, [0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3]).winner, 1)
        # diagonal up-left
        self.assertEqual(replay(CONNECT4, [6, 5, 5, 4, 4, 3, 4, 3, 3, 0, 3]).winner, 1)

    def test_no_win_with_k_minus_one(self):
        self.assertEqual(replay(CONNECT6, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]).winner, 0)

    def test_random_games_match_brute_force(self):
        rng = random.Random(1)
        for spec in (CONNECT4, CONNECT5, CONNECT6):
            for _ in range(60):
                b = Board(spec)
                while not b.done:
                    b.play(rng.choice(b.legal_moves()))
                    found = brute_winner(b)
                    self.assertEqual(found, {b.winner} if b.winner else set())
                    self.assertEqual(b.scan_winner(), b.winner)
                if b.winner == 0:
                    self.assertTrue((b.heights == spec.rows).all())

    def test_windows_count(self):
        # rows*(cols-k+1) + cols*(rows-k+1) + 2*(rows-k+1)*(cols-k+1)
        for s in (CONNECT4, CONNECT5, CONNECT6):
            n = s.rows * (s.cols - s.k + 1) + s.cols * (s.rows - s.k + 1) + 2 * (s.rows - s.k + 1) * (s.cols - s.k + 1)
            self.assertEqual(len(windows(s)), n)

    def test_resync_after_direct_writes(self):
        b = replay(CONNECT5, [4, 4, 4])
        b.grid[6, 2] = 2          # a stone in mid-air
        b.resync(player=1)
        self.assertEqual(b.heights[2], 7)
        self.assertEqual(b.n_moves, 4)
        self.assertEqual(b.player, 1)

    def test_copy_is_independent(self):
        b = replay(CONNECT5, [1, 2, 3])
        c = b.copy()
        c.play(4)
        self.assertEqual(b.n_moves, 3)
        self.assertEqual(len(b.history), 3)


class TestReferee(unittest.TestCase):
    def test_illegal_move_loses(self):
        class Bad:
            name = "bad"

            def choose(self, board):
                return 99
        g = play_game(Bad(), RandomPlayer(seed=0), CONNECT5)
        self.assertEqual((g.winner, g.reason), (2, "illegal"))

    def test_crash_loses(self):
        class Crash:
            def choose(self, board):
                raise RuntimeError("boom")
        g = play_game(RandomPlayer(seed=0), Crash(), CONNECT5)
        self.assertEqual(g.winner, 1)
        self.assertTrue(g.reason.startswith("error"))

    def test_player_cannot_touch_real_board(self):
        class Vandal:
            def choose(self, board):
                board.grid[:] = 1
                return board.legal_moves()[0]
        g = play_game(Vandal(), RandomPlayer(seed=1), CONNECT5)
        b = replay(CONNECT5, g.moves)
        self.assertEqual(b.winner, g.winner)

    def test_match_plays_both_colors(self):
        ops = make_openings(CONNECT5, 5, seed=0)
        self.assertEqual(len(set(ops)), 5)
        r = play_match(RandomPlayer(seed=1), RandomPlayer(seed=2), CONNECT5, ops, keep_games=True)
        self.assertEqual(r.n, 10)
        self.assertEqual([a for a, _ in r.games], [True, False] * 5)
        for _, g in r.games:
            self.assertEqual(g.opening_len, 2)


class TestNet(unittest.TestCase):
    def test_encode_and_relative_agree(self):
        from connectk.net import encode, encode_relative, relative_grid
        rng = random.Random(3)
        for spec in (CONNECT5, CONNECT6):
            for _ in range(20):
                b = Board(spec)
                for _ in range(rng.randrange(0, 40)):
                    if b.done:
                        break
                    b.play(rng.choice(b.legal_moves()))
                x = encode(b)
                np.testing.assert_array_equal(x[None], encode_relative(relative_grid(b)[None]))
                self.assertEqual(x[0].sum() + x[1].sum(), b.n_moves)
                self.assertEqual(x[2].sum(), (b.heights < spec.rows).sum())

    def test_save_load_and_strict(self):
        import tempfile

        import torch

        from connectk.net import ConnectNet, NetPlayer, load_net, save_net
        net = ConnectNet(CONNECT5)
        with tempfile.TemporaryDirectory() as d:
            save_net(net, f"{d}/a.pt")
            back = load_net(CONNECT5, f"{d}/a.pt")
            for (k, v), (k2, v2) in zip(net.state_dict().items(), back.state_dict().items()):
                self.assertEqual(k, k2)
                self.assertTrue(torch.equal(v, v2))
            with self.assertRaises(RuntimeError):
                load_net(CONNECT6, f"{d}/a.pt")      # wrong board size
            sd = net.state_dict()
            sd["extra.weight"] = torch.zeros(1)
            torch.save(sd, f"{d}/b.pt")
            with self.assertRaises(RuntimeError):
                load_net(CONNECT5, f"{d}/b.pt")
        b = replay(CONNECT5, [0] * 9)                 # column 0 full
        self.assertNotEqual(NetPlayer(net).choose(b), 0)


if __name__ == "__main__":
    unittest.main()
