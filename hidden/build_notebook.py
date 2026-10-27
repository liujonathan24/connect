"""Write part_b_connect5/selfplay.ipynb (without outputs).

    python3 hidden/build_notebook.py
    python3 hidden/build_notebook.py --check   # exit 1 if the notebook is out of date

Edit the cells here, not in the notebook file.
"""

import argparse
import pathlib
import sys

import nbformat

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "part_b_connect5/selfplay.ipynb"

CELLS = []


def md(text):
    CELLS.append(nbformat.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    CELLS.append(nbformat.v4.new_code_cell(text.strip("\n")))


md(r"""
# Part B: Connect 5 by self-play

Train a network to play **Connect 5** on a **9 × 9** board, with no opponents
but itself. At the end of the event, every team's network plays every other
team's network. Your placement in that tournament is your result.

The rules are the rules of Connect 4: drop stones into columns, and the
first player with **5 in a row** (across, up or diagonal) wins.

This notebook gives you a working start:

1. the game and the fixed network,
2. a self-play PPO loop that saves a new version of your network every few
   updates,
3. a **tournament harness**: your versions play each other (and a few fixed
   gauges), and you get a table and Elo ratings,
4. a way to play other teams' networks before the final, and
5. how to hand in.

The loop is deliberately simple. Section 6 lists what to improve.
Answer `QUESTIONS.md` as you go.

**Rules of the event.** The network is fixed (`connectk/net.py`); you only
choose its weights. In the tournament your network plays greedily: the legal
column with the highest policy logit, no search. Anything goes in training.
""")

code(r"""
import copy
import dataclasses
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import Categorical

from connectk import CONNECT5, Board
from connectk.arena import RandomPlayer, make_openings, play_match
from connectk.net import ConnectNet, NetPlayer, count_parameters, encode_batch, load_net, mask_batch, save_net
from connectk.packed import load_packed
from connectk.tournament import round_robin
from connectk.viewer import record_game, write_html

SPEC = CONNECT5
torch.set_num_threads(4)
CKPT = pathlib.Path("checkpoints")
CKPT.mkdir(exist_ok=True)
gauges = load_packed("gauges.bin", seed=0)   # two fixed bots to measure progress: gauge_easy, gauge_hard
print(SPEC, list(gauges))
""")

md(r"""
## 1. The game

`Board` holds the position. `board.play(col)` drops a stone for the side to
move. Row 0 is the bottom row.
""")

code(r"""
b = Board(SPEC)
for col in [4, 4, 3, 5, 2, 3]:
    b.play(col)
print(b.render())
print("to move:", b.player, " legal columns:", b.legal_moves(), " winner:", b.winner)
""")

md(r"""
## 2. The fixed network

Input: 4 planes (my stones, the opponent's stones, the cells where a stone
would land, all ones), always from the view of the side to move. Output: one
logit per column, and a value in [-1, 1] for the side to move.
""")

code(r"""
net = ConnectNet(SPEC)
print("parameters:", count_parameters(net))
x = torch.from_numpy(encode_batch([b]))
logits, value = net(x)
print("input", tuple(x.shape), " logits", tuple(logits.shape), " value", float(value))

# An untrained net against the gauges (each opening twice, once per color):
openings = make_openings(SPEC, 6, seed=1)
for name, bot in gauges.items():
    print(play_match(NetPlayer(net, "untrained"), bot, SPEC, openings))
""")

md(r"""
## 3. Self-play with PPO

The learner plays many games at once. In each game its opponent is a frozen
copy of an earlier version, taken from a **pool**. The learner gets +1 for a
win, -1 for a loss and 0 for a draw, at the end of the game.

**The baseline pool only ever picks the newest version.** That is "naive"
self-play. It is a known way to fail; question 1 asks why.
""")

code(r"""
class Pool:
    # Frozen past versions of the learner.

    def __init__(self):
        self.nets, self.names = [], []
        self.results = []          # per version: [games, learner points]

    def add(self, net, name):
        frozen = copy.deepcopy(net).eval()
        for p in frozen.parameters():
            p.requires_grad_(False)
        self.nets.append(frozen)
        self.names.append(name)
        self.results.append([0, 0.0])

    def sample(self, rng):
        # TODO: the baseline always plays the newest version. Try a uniform choice over
        # all versions, or prefer the versions you still lose to (PFSP).
        return len(self.nets) - 1

    def record(self, k, points):
        self.results[k][0] += 1
        self.results[k][1] += points


class Game:
    def __init__(self, pool, rng):
        self.board = Board(SPEC)
        self.side = int(rng.integers(1, 3))      # the learner's color in this game
        self.opp = pool.sample(rng)


def opponent_turns(games, pool, temperature=1.0):
    # Every game where the opponent is to move gets one opponent move (batched per opponent).
    waiting = [g for g in games if not g.board.done and g.board.player != g.side]
    by_opp = {}
    for g in waiting:
        by_opp.setdefault(g.opp, []).append(g)
    for k, gs in by_opp.items():
        boards = [g.board for g in gs]
        with torch.no_grad():
            logits, _ = pool.nets[k](torch.from_numpy(encode_batch(boards)))
        logits = logits.masked_fill(~torch.from_numpy(mask_batch(boards)), float("-inf"))
        cols = Categorical(logits=logits / temperature).sample().tolist()
        for g, c in zip(gs, cols):
            g.board.play(c)
""")

code(r"""
@dataclasses.dataclass
class Config:
    updates: int = 40             # raise this a lot for real training
    n_games: int = 64             # games played in parallel
    rollout_len: int = 16         # learner moves per game per update
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    clip_eps: float = 0.2
    epochs: int = 4
    minibatch: int = 256
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    snapshot_every: int = 10      # add a new version to the pool every N updates
    seed: int = 0


def collect(net, games, pool, cfg, rng, tally):
    # Play cfg.rollout_len learner moves in every game. Returns the rollout tensors.
    obs_l, mask_l, act_l, logp_l, val_l, rew_l, done_l = [], [], [], [], [], [], []
    opponent_turns(games, pool)                      # games where the opponent moves first
    for _ in range(cfg.rollout_len):
        boards = [g.board for g in games]
        obs = torch.from_numpy(encode_batch(boards))
        mask = torch.from_numpy(mask_batch(boards))
        with torch.no_grad():
            logits, values = net(obs)
            dist = Categorical(logits=logits.masked_fill(~mask, float("-inf")))
            actions = dist.sample()
        for g, a in zip(games, actions.tolist()):
            g.board.play(a)
        opponent_turns(games, pool)
        rewards = np.zeros(len(games), dtype=np.float32)
        dones = np.zeros(len(games), dtype=np.float32)
        for i, g in enumerate(games):
            if g.board.done:
                w = g.board.winner
                rewards[i] = 0.0 if w == 0 else (1.0 if w == g.side else -1.0)
                dones[i] = 1.0
                pool.record(g.opp, 0.5 + rewards[i] / 2)
                tally[0 if rewards[i] > 0 else 1 if rewards[i] == 0 else 2] += 1
                games[i] = Game(pool, rng)
        opponent_turns(games, pool)                  # new games where the learner is second
        obs_l.append(obs); mask_l.append(mask); act_l.append(actions); logp_l.append(dist.log_prob(actions))
        val_l.append(values); rew_l.append(torch.from_numpy(rewards)); done_l.append(torch.from_numpy(dones))
    with torch.no_grad():
        _, last_value = net(torch.from_numpy(encode_batch([g.board for g in games])))
    return obs_l, mask_l, act_l, logp_l, val_l, rew_l, done_l, last_value


def gae(rewards, values, dones, last_value, gamma, lam):
    adv = torch.zeros_like(rewards)
    running = torch.zeros_like(last_value)
    for t in reversed(range(rewards.shape[0])):
        next_v = last_value if t == rewards.shape[0] - 1 else values[t + 1]
        nd = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * nd - values[t]
        running = delta + gamma * lam * nd * running
        adv[t] = running
    return adv, adv + values


def ppo_update(net, opt, obs, mask, act, old_logp, adv, ret, cfg):
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = obs.shape[0]
    for _ in range(cfg.epochs):
        perm = torch.randperm(n)
        for s in range(0, n, cfg.minibatch):
            i = perm[s:s + cfg.minibatch]
            logits, v = net(obs[i])
            dist = Categorical(logits=logits.masked_fill(~mask[i], float("-inf")))
            ratio = torch.exp(dist.log_prob(act[i]) - old_logp[i])
            pg = -torch.min(ratio * adv[i], ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv[i]).mean()
            vl = ((v - ret[i]) ** 2).mean()
            loss = pg + cfg.vf_coef * vl - cfg.ent_coef * dist.entropy().mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            opt.step()
    return pg.item(), vl.item()
""")

code(r"""
def train(cfg, net=None, pool=None, start_version=0):
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    net = net or ConnectNet(SPEC)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    if pool is None:
        pool = Pool()
        pool.add(net, f"v{start_version}")
        save_net(net, CKPT / f"v{start_version}.pt")
    version = start_version
    games = [Game(pool, rng) for _ in range(cfg.n_games)]
    tally = np.zeros(3, int)
    t0 = time.time()
    for update in range(1, cfg.updates + 1):
        net.eval()
        obs, mask, act, logp, val, rew, done, last_v = collect(net, games, pool, cfg, rng, tally)
        adv, ret = gae(torch.stack(rew), torch.stack(val), torch.stack(done), last_v, cfg.gamma, cfg.gae_lambda)
        net.train()
        pg, vl = ppo_update(net, opt, torch.cat(obs), torch.cat(mask), torch.cat(act), torch.cat(logp),
                            adv.flatten(), ret.flatten(), cfg)
        if update % cfg.snapshot_every == 0:
            version += 1
            name = f"v{version}"
            save_net(net, CKPT / f"{name}.pt")
            pool.add(net, name)
            w, d, l = tally
            print(f"update {update:5d}  {time.time() - t0:5.0f}s  vs pool: +{w} ={d} -{l}  "
                  f"pg {pg:+.3f}  value loss {vl:.3f}  -> saved {name}")
            tally[:] = 0
    return net, pool, version


cfg = Config()
net, pool, version = train(cfg)
""")

md(r"""
## 4. The tournament harness

Every version you saved plays every other version, plus a random player and
the two gauges. Each pair plays every opening twice (once with each color).
The table gives points and an Elo rating fitted to all games
(`gauge_easy` is fixed at 1000 so runs are comparable).

Read the table with care: Elo assumes that strength is one number. In
self-play that is often false (version A beats B, B beats C, C beats A).
Look at the score matrix too.
""")

code(r"""
def load_versions(folder=CKPT, every=1):
    paths = sorted(folder.glob("v*.pt"), key=lambda p: int(p.stem[1:]))
    return {p.stem: NetPlayer(load_net(SPEC, p), p.stem) for p in paths[::every]}


players = load_versions()
players.update({"random": RandomPlayer(seed=0), **gauges})
torch.set_num_threads(1)          # the workers below each use one thread
table = round_robin(players, SPEC, make_openings(SPEC, 6, seed=7), workers=4, anchor=("gauge_easy", 1000.0))
torch.set_num_threads(4)
print(table)
""")

code(r"""
vs = [n for n in table.names if n.startswith("v")]
xs = [int(n[1:]) for n in vs]
elo = [table.elo[table.names.index(n)] for n in vs]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(xs, elo, marker="o")
for g in ("random", "gauge_easy", "gauge_hard"):
    ax[0].axhline(table.elo[table.names.index(g)], ls="--", lw=1, color="gray")
    ax[0].text(xs[0], table.elo[table.names.index(g)], " " + g, va="bottom", fontsize=8, color="gray")
ax[0].set_xlabel("version"); ax[0].set_ylabel("Elo"); ax[0].set_title("Elo of each saved version")
im = ax[1].imshow(table.matrix, vmin=0, vmax=1, cmap="RdBu")
ax[1].set_xticks(range(len(table.names)), table.names, rotation=90, fontsize=7)
ax[1].set_yticks(range(len(table.names)), table.names, fontsize=7)
ax[1].set_title("score of row against column")
plt.colorbar(im, ax=ax[1]); plt.tight_layout(); plt.show()
""")

md(r"""
### Watch a game

This writes an HTML page that replays games move by move. Open it in a
browser (or in the Jupyter file browser).
""")

code(r"""
latest = NetPlayer(load_net(SPEC, CKPT / f"v{version}.pt"), f"v{version}")
recs = [record_game(SPEC, latest, gauges["gauge_hard"], op) for op in make_openings(SPEC, 2, seed=3)]
recs += [record_game(SPEC, gauges["gauge_hard"], latest, op) for op in make_openings(SPEC, 2, seed=3)]
write_html(recs, "replay.html", f"v{version} vs gauge_hard")
""")

md(r"""
## 5. Play other teams before the final

Put other teams' `.pt` files (they are just weights) in a folder
`scrimmage/` and run a tournament against your best versions. Teams may
share files during the event; the final uses only the file you hand in.
""")

code(r"""
scrim = pathlib.Path("scrimmage")
others = {p.stem: NetPlayer(load_net(SPEC, p), p.stem) for p in sorted(scrim.glob("*.pt"))} if scrim.exists() else {}
if others:
    mine = {f"mine_v{version}": NetPlayer(load_net(SPEC, CKPT / f"v{version}.pt"), f"mine_v{version}")}
    torch.set_num_threads(1)
    print(round_robin({**mine, **others, **gauges}, SPEC, make_openings(SPEC, 10, seed=11), workers=4))
    torch.set_num_threads(4)
else:
    print("no files in scrimmage/ yet")
""")

md(r"""
## 6. Make it better

The loop above trains, but slowly and with known flaws. Ideas, roughly from
cheap to expensive:

* **The opponent pool.** Sample all past versions, or favor the ones you
  lose to (prioritized fictitious self-play). Keep a few fixed bots in the
  pool too.
* **Symmetry.** A position and its mirror image (columns reversed) have the
  same value. Use it to double your data.
* **More compute.** Larger `n_games`, longer runs, a learning-rate schedule.
* **Search in training.** At test time your network plays without search,
  but in training you may use it: let a one-move lookahead or a small MCTS
  pick better moves, then train the policy to imitate those choices
  ("expert iteration", as in AlphaZero). Tactics (take a win, block a loss)
  are the first thing to get right.
* **Read the tournament.** If version 30 loses to version 12, find out why
  (watch the games).

## 7. Hand in

Save one file. It must load into `ConnectNet(CONNECT5)` with `strict=True`.
""")

code(r"""
best = CKPT / f"v{version}.pt"          # choose your best version with the tournament, not just the newest
save_net(load_net(SPEC, best), "submission.pt")
check = load_net(SPEC, "submission.pt")
print("submission.pt OK;", play_match(NetPlayer(check, "submission"), gauges["gauge_easy"], SPEC, make_openings(SPEC, 4, seed=5)))
""")


def build():
    nb = nbformat.v4.new_notebook()
    nb.cells = CELLS
    for i, cell in enumerate(nb.cells):
        cell["id"] = f"cell-{i:02d}"        # fixed ids keep the file stable for --check
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    return nbformat.writes(nb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text = build()
    if args.check:
        if not OUT.exists() or OUT.read_text() != text:
            sys.exit("selfplay.ipynb is out of date; run python3 hidden/build_notebook.py")
        return
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
