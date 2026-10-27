"""Bot engine: rule-based opponents for both games.

Every bot has two entry points:

    choose(board) -> column   the strict referee (arena) calls this
    play(board)               the training environment calls this; the bot
                              plays its own move on the live board

Honest bots implement play() as board.play(choose(board)). The cheater does
not: on the live board it sometimes writes stones where it likes. In the
arena it gets a copy of the board, so it cannot cheat there.

This file goes to students only inside a packed blob (see hidden/build_blobs.py).
"""

import numpy as np

from connectk.game import Board, cell_windows, windows

BIG = 1e6


class _Lines:
    """Per-window stone counts for one board and one player."""

    def __init__(self, board, me):
        spec = board.spec
        self.win = windows(spec)
        self.cw = cell_windows(spec)
        flat = board.grid.ravel()
        cells = flat[self.win]
        self.own = (cells == me).sum(axis=1)
        self.opp = (cells == 3 - me).sum(axis=1)
        self.empty = flat == 0
        self.k = spec.k

    def win_cells(self, who_own, who_opp):
        """Empty cells that complete a line for the player whose counts are who_own."""
        hot = np.append((who_own == self.k - 1) & (who_opp == 0), False)
        return hot[self.cw].any(axis=1) & self.empty


def landing_cells(board):
    cols = board.legal_moves()
    return cols, [int(board.heights[c]) * board.spec.cols + c for c in cols]


class Bot:
    name = "bot"

    def __init__(self, name=None, seed=None):
        if name:
            self.name = name
        self.rng = np.random.default_rng(seed)

    def choose(self, board):
        raise NotImplementedError

    def play(self, board):
        board.play(self.choose(board))


class RandomBot(Bot):
    name = "random"

    def choose(self, board):
        return int(self.rng.choice(board.legal_moves()))


class HeuristicBot(Bot):
    """Scores each legal column by the lines through its landing cell.

    attack[n]   value of a line with n of my stones and none of the opponent's
    defense[n]  value of spoiling a line with n opponent stones and none of mine
    center      bonus for central columns
    p_win       chance to see and take an immediate win
    p_block     chance to see and block an immediate loss
    avoid_gift  do not play under a cell where the opponent would win
    lookahead   subtract `lookahead` x the opponent's best reply score
    epsilon     chance of a random legal move
    temperature 0 = best score (ties at random), > 0 = softmax over scores
    stack       bonus for columns that already hold my stones
    low         bonus per empty row above the landing cell (plays low)
    """

    def __init__(self, name=None, seed=None, attack=None, defense=None, center=2.0,
                 p_win=1.0, p_block=1.0, avoid_gift=True, lookahead=0.0, epsilon=0.0,
                 temperature=0.0, stack=0.0, low=0.0):
        super().__init__(name, seed)
        self.attack = None if attack is None else np.asarray(attack, dtype=np.float64)
        self.defense = None if defense is None else np.asarray(defense, dtype=np.float64)
        self.center, self.p_win, self.p_block = center, p_win, p_block
        self.avoid_gift, self.lookahead = avoid_gift, lookahead
        self.epsilon, self.temperature = epsilon, temperature
        self.stack, self.low = stack, low

    def _tables(self, k):
        a = self.attack if self.attack is not None else np.array([0, 1, 4, 16, 64, 256, 1024][: k + 1], float)
        d = self.defense if self.defense is not None else a * 0.8
        a = np.pad(a, (0, max(0, k + 1 - len(a))))[: k + 1]
        d = np.pad(d, (0, max(0, k + 1 - len(d))))[: k + 1]
        return a, d

    def cell_scores(self, board, me):
        """Score of every cell if `me` put a stone there (flat array)."""
        L = _Lines(board, me)
        a, d = self._tables(L.k)
        clean_own = L.opp == 0
        clean_opp = L.own == 0
        own_next = np.minimum(L.own + 1, L.k)
        gain = np.where(clean_own, a[own_next] - a[L.own], 0.0)
        gain += np.where(clean_opp & (L.opp > 0), d[L.opp], 0.0)
        return np.append(gain, 0.0)[L.cw].sum(axis=1), L

    def choose(self, board):
        spec = board.spec
        me = board.player
        cols, cells = landing_cells(board)
        if len(cols) == 1:
            return cols[0]
        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(cols))
        scores, L = self.cell_scores(board, me)
        my_wins = L.win_cells(L.own, L.opp)
        their_wins = L.win_cells(L.opp, L.own)
        if self.rng.random() < self.p_win:
            hits = [c for c, cell in zip(cols, cells) if my_wins[cell]]
            if hits:
                return int(self.rng.choice(hits))
        if self.rng.random() < self.p_block:
            hits = [c for c, cell in zip(cols, cells) if their_wins[cell]]
            if hits:
                return int(self.rng.choice(hits))
        mid = (spec.cols - 1) / 2
        vals = np.array([scores[cell] for cell in cells], dtype=np.float64)
        vals += self.center * (1 - np.abs(np.array(cols) - mid) / mid)
        if self.stack:
            mine = (board.grid == me).sum(axis=0)
            vals += self.stack * np.array([mine[c] for c in cols])
        if self.low:
            vals += self.low * np.array([spec.rows - board.heights[c] for c in cols])
        if self.avoid_gift:
            for i, (c, cell) in enumerate(zip(cols, cells)):
                above = cell + spec.cols
                if above < spec.rows * spec.cols and their_wins[above]:
                    vals[i] -= BIG
        if self.lookahead:
            for i, c in enumerate(cols):
                if vals[i] <= -BIG / 2:
                    continue
                child = board.copy()
                child.play(c)
                if child.done:
                    continue
                opp_scores, _ = self.cell_scores(child, 3 - me)
                _, ccells = landing_cells(child)
                vals[i] -= self.lookahead * max(opp_scores[cc] for cc in ccells)
        return self._pick(cols, vals)

    def _pick(self, cols, vals):
        if self.temperature > 0:
            z = (vals - vals.max()) / (self.temperature * (vals.std() + 1e-9))
            p = np.exp(z)
            p /= p.sum()
            return int(cols[self.rng.choice(len(cols), p=p)])
        best = np.flatnonzero(vals >= vals.max() - 1e-9)
        return int(cols[self.rng.choice(best)])


class SearchBot(Bot):
    """Alpha-beta search to a fixed depth on a line-count evaluation."""

    def __init__(self, name=None, seed=None, depth=2, attack=None, defense_ratio=1.0, noise=0.0):
        super().__init__(name, seed)
        self.depth, self.noise = depth, noise
        self.attack = attack
        self.defense_ratio = defense_ratio

    def _eval(self, board, me):
        k = board.spec.k
        a = np.asarray(self.attack if self.attack is not None else [0, 1, 4, 16, 64, 256, 1024][: k + 1], float)
        a = np.pad(a, (0, max(0, k + 1 - len(a))))[: k + 1]
        cells = board.grid.ravel()[windows(board.spec)]
        own = (cells == me).sum(axis=1)
        opp = (cells == 3 - me).sum(axis=1)
        mine = a[own][opp == 0].sum()
        theirs = a[opp][own == 0].sum()
        return mine - self.defense_ratio * theirs

    def _negamax(self, board, depth, alpha, beta):
        if board.winner:
            return -BIG - depth          # the side to move has lost
        if board.done:
            return 0.0
        if depth == 0:
            return self._eval(board, board.player)
        best = -np.inf
        for c in self._order(board):
            child = board.copy()
            child.play(c)
            v = -self._negamax(child, depth - 1, -beta, -alpha)
            best = max(best, v)
            alpha = max(alpha, v)
            if alpha >= beta:
                break
        return best

    def _order(self, board):
        mid = (board.spec.cols - 1) / 2
        return sorted(board.legal_moves(), key=lambda c: abs(c - mid))

    def choose(self, board):
        cols = self._order(board)
        if len(cols) == 1:
            return cols[0]
        vals = []
        for c in cols:
            child = board.copy()
            child.play(c)
            vals.append(-self._negamax(child, self.depth - 1, -np.inf, np.inf))
        vals = np.array(vals)
        if self.noise:
            alive = np.abs(vals) < BIG / 2
            scale = vals[alive].std() + 1.0 if alive.sum() > 1 else 1.0
            vals = vals + np.where(alive, self.noise * scale * self.rng.standard_normal(len(vals)), 0.0)
        best = np.flatnonzero(vals >= vals.max() - 1e-9)
        return int(cols[self.rng.choice(best)])


class CheaterBot(HeuristicBot):
    """Plays like a HeuristicBot, but on a live board it cheats.

    teleport  p_teleport: if a line can be completed at any empty cell (even a
              cell in mid-air), put the stone there and win
    steal     p_steal: if the opponent threatens to win, turn one of the
              opponent's stones on that line into its own
    extra     p_extra: drop a second stone on the same turn
    """

    def __init__(self, *args, p_teleport=0.9, p_steal=0.7, p_extra=0.04, **kw):
        super().__init__(*args, **kw)
        self.p_teleport, self.p_steal, self.p_extra = p_teleport, p_steal, p_extra

    def play(self, board):
        me = board.player
        spec = board.spec
        L = _Lines(board, me)
        my_wins = L.win_cells(L.own, L.opp)
        their_wins = L.win_cells(L.opp, L.own)
        cols, cells = landing_cells(board)
        landing = np.zeros(spec.rows * spec.cols, dtype=bool)
        landing[cells] = True
        if my_wins.any() and not (my_wins & landing).any() and self.rng.random() < self.p_teleport:
            cell = int(self.rng.choice(np.flatnonzero(my_wins)))
            board.grid[divmod(cell, spec.cols)] = me
            return
        threats = their_wins & landing
        if threats.sum() >= 1 and not (my_wins & landing).any() and self.rng.random() < self.p_steal:
            hot = np.flatnonzero((L.opp == spec.k - 1) & (L.own == 0))
            target = None
            for w in hot:
                if threats[L.win[w]].any():
                    stones = [x for x in L.win[w] if board.grid[divmod(int(x), spec.cols)] == 3 - me]
                    target = int(self.rng.choice(stones))
                    break
            if target is not None:
                board.grid[divmod(target, spec.cols)] = me
                return
        board.play(self.choose(board))
        if not board.done and self.rng.random() < self.p_extra:
            board.player = me
            board.play(self.choose(board))
