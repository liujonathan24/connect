"""Round-robin tournaments and Elo ratings.

    players = {"v1": NetPlayer(net1), "v2": NetPlayer(net2), "random": RandomPlayer()}
    table = round_robin(players, CONNECT5, make_openings(CONNECT5, 20))
    print(table)                # standings, points and Elo
    table.matrix                # score of row player against column player

Elo here is the Bradley-Terry fit of all games at once (a draw counts as half
a win for each side), scaled so that 400 points means 10:1 odds. The first
name in `anchor` (if given) is fixed at its rating, so ratings from different
runs that share an anchor can be compared.
"""

import multiprocessing as mp
import sys
from dataclasses import dataclass

import numpy as np

from .arena import play_match

_POOL = {}


def _run_pair(job):
    i, j, a, b, spec, openings = job
    pa, pb = _POOL[a], _POOL[b]
    r = play_match(pa, pb, spec, openings)
    return i, j, r.wins, r.draws, r.losses, r.illegal_a, r.illegal_b


@dataclass
class Table:
    names: list
    wins: np.ndarray       # wins[i, j]: games i won against j
    draws: np.ndarray
    illegal: np.ndarray    # games lost by an illegal move or crash, per player
    elo: np.ndarray

    @property
    def games(self):
        return self.wins + self.wins.T + self.draws

    @property
    def points(self):
        return (self.wins + 0.5 * self.draws).sum(axis=1)

    @property
    def matrix(self):
        """Score (0..1) of the row player against the column player."""
        g = self.games
        with np.errstate(invalid="ignore", divide="ignore"):
            m = (self.wins + 0.5 * self.draws) / g
        np.fill_diagonal(m, np.nan)
        return m

    def standings(self):
        """Names sorted by points, then by Elo."""
        order = sorted(range(len(self.names)), key=lambda i: (-self.points[i], -self.elo[i]))
        return [self.names[i] for i in order]

    def __str__(self):
        g = self.games.sum(axis=1)
        w = max(8, max(len(n) for n in self.names))
        lines = [f"{'#':>3}  {'player':<{w}}  {'points':>7}  {'games':>5}  {'score':>6}  {'elo':>6}  illegal"]
        for rank, name in enumerate(self.standings(), 1):
            i = self.names.index(name)
            lines.append(f"{rank:>3}  {name:<{w}}  {self.points[i]:>7.1f}  {int(g[i]):>5}  "
                         f"{self.points[i] / max(g[i], 1):>6.3f}  {self.elo[i]:>6.0f}  {int(self.illegal[i])}")
        return "\n".join(lines)


def round_robin(players, spec, openings, workers=1, anchor=None, verbose=False):
    """Every pair plays a match on every opening, both colors.

    workers > 1 plays pairs in parallel processes (Linux only; elsewhere it
    runs in one process). Set torch.set_num_threads(1) first when players
    are networks.
    """
    names = list(players)
    n = len(names)
    wins = np.zeros((n, n))
    draws = np.zeros((n, n))
    illegal = np.zeros(n)
    jobs = [(i, j, names[i], names[j], spec, openings) for i in range(n) for j in range(i + 1, n)]
    _POOL.clear()
    _POOL.update(players)
    if workers > 1 and sys.platform.startswith("linux"):   # forked workers are not safe with torch on macOS
        ctx = mp.get_context("fork")
        with ctx.Pool(workers) as pool:
            results = pool.imap_unordered(_run_pair, jobs)
            results = list(_progress(results, len(jobs), verbose))
    else:
        results = list(_progress(map(_run_pair, jobs), len(jobs), verbose))
    for i, j, w, d, l, ia, ib in results:
        wins[i, j] += w
        wins[j, i] += l
        draws[i, j] += d
        draws[j, i] += d
        illegal[i] += ia
        illegal[j] += ib
    _POOL.clear()
    elo = fit_elo(wins, draws, names, anchor)
    return Table(names, wins, draws, illegal, elo)


def _progress(it, total, verbose):
    for k, x in enumerate(it, 1):
        if verbose:
            print(f"  pair {k}/{total} done", flush=True)
        yield x


def fit_elo(wins, draws, names=None, anchor=None, iters=2000, prior=1.0):
    """Bradley-Terry ratings on the Elo scale (MM algorithm).

    `prior` adds that many virtual draws against an average player, which keeps
    ratings finite for players who won or lost every game. `anchor` is a
    (name, rating) pair; without it the mean rating is 1000.
    """
    n = len(wins)
    w = wins + 0.5 * draws          # half a win for each draw
    games = w + w.T
    strength = np.ones(n)
    wins_total = w.sum(axis=1) + 0.5 * prior
    for _ in range(iters):
        denom = (games / (strength[:, None] + strength[None, :])).sum(axis=1) + prior / (strength + 1.0)
        new = wins_total / denom
        new /= np.exp(np.mean(np.log(new)))
        if np.allclose(new, strength, rtol=1e-10, atol=0):
            strength = new
            break
        strength = new
    elo = 400 * np.log10(strength)
    if anchor and names and anchor[0] in names:
        elo += anchor[1] - elo[names.index(anchor[0])]
    else:
        elo += 1000 - elo.mean()
    return elo
