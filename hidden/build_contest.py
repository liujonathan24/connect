"""Write the contest notebooks: contest/notebook_a.ipynb and contest/notebook_b.ipynb.

    python3 hidden/build_contest.py
    python3 hidden/build_contest.py --check    # exit 1 if a notebook is out of date

Edit the cells here, not in the notebook files.
"""

import argparse
import pathlib
import sys

import nbformat

ROOT = pathlib.Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- shared cells

SETUP = r'''
import pathlib, zipfile
if not pathlib.Path("connectk").exists():          # first run: unpack the code next to this notebook
    zipfile.ZipFile("connect5.zip").extractall(".")

import copy, dataclasses, json, shutil, time
import numpy as np
import torch
from torch.distributions import Categorical

from connectk import __SPEC__, Board
from connectk.agents import MAX_PARAMS, check
from connectk.arena import RandomPlayer, make_openings, play_match
from connectk.packed import load_packed
from connectk.tournament import round_robin
from connectk.viewer import record_game, write_html

SPEC = __SPEC__
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(4)
print(SPEC, "| training on", device)
'''

MODEL = r'''
%%writefile __MODEL__.py
"""Your network and its input encoding. Change anything you like.

The hand-in copies this file into your submission as model.py, so the
network you train is exactly the network we play. Keep it under 200 million
parameters, and fast enough for 1 second per move on one CPU core.
"""
import numpy as np
import torch
from torch import nn


def encode(boards):
    """Input for a list of connectk Boards: (B, 4, rows, cols), from the view of the side to move.

    planes: my stones, the opponent's stones, the landing cell of each column, all ones
    """
    g = np.stack([b.grid for b in boards])
    me = np.array([b.player for b in boards])[:, None, None]
    h = np.stack([b.heights for b in boards])
    rows = g.shape[1]
    x = np.zeros((len(boards), 4) + g.shape[1:], dtype=np.float32)
    x[:, 0] = g == me
    x[:, 1] = (g != 0) & (g != me)
    n, c = np.nonzero(h < rows)
    x[n, 2, h[n, c], c] = 1.0
    x[:, 3] = 1.0
    return torch.from_numpy(x)


class Block(nn.Module):
    """Residual block: conv, ReLU, conv, add the input, ReLU."""

    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return torch.relu(x + self.c2(torch.relu(self.c1(x))))


class Net(nn.Module):
    """Outputs: one logit per column (the policy) and a value in [-1, 1] for the side to move."""

    def __init__(self, rows, cols, channels=64, blocks=4):
        super().__init__()
        self.stem = nn.Conv2d(4, channels, 3, padding=1)
        self.blocks = nn.Sequential(*[Block(channels) for _ in range(blocks)])
        self.policy = nn.Sequential(nn.Conv2d(channels, 2, 1), nn.ReLU(), nn.Flatten(), nn.Linear(2 * rows * cols, cols))
        self.value = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.ReLU(), nn.Flatten(),
                                   nn.Linear(rows * cols, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())

    def forward(self, x):
        h = self.blocks(torch.relu(self.stem(x)))
        return self.policy(h), self.value(h).squeeze(-1)
'''

NET_DEMO = r'''
import importlib, __MODEL__
importlib.reload(__MODEL__)                     # pick up your edits to __MODEL__.py
from __MODEL__ import Net, encode

NET_CONFIG = dict(rows=SPEC.rows, cols=SPEC.cols, channels=64, blocks=4)
net = Net(**NET_CONFIG).to(device)
n_params = sum(p.numel() for p in net.parameters())
print(f"{n_params:,} parameters (limit {MAX_PARAMS:,})")

b = Board(SPEC)
for col in [SPEC.cols // 2, SPEC.cols // 2, SPEC.cols // 2 - 1]:
    b.play(col)
with torch.no_grad():
    logits, value = net(encode([b]).to(device))
print("input", tuple(encode([b]).shape), " logits", tuple(logits.shape), " value", round(value.item(), 3))


class NetPlayer:
    """Plays the legal column with the highest logit. This is what the template hands in."""

    def __init__(self, net, name="net"):
        self.net, self.name = net, name

    @torch.no_grad()
    def choose(self, board):
        self.net.eval()
        logits, _ = self.net(encode([board]).to(device))
        logits = logits[0].float().cpu().numpy()
        logits[~board.legal_mask()] = -np.inf
        return int(np.argmax(logits))
'''

AGENT_PY = r'''
"""Submission: the network plays the legal column with the highest logit.

You may change this file too (for example, to add a search that calls your
network), as long as agent.net holds all the parameters and a move takes at
most 1 second on one CPU core.
"""
import json
import pathlib

import numpy as np
import torch

from model import Net, encode


class Agent:
    def __init__(self, folder):
        config = json.loads((folder / "config.json").read_text())
        self.net = Net(**config)
        self.net.load_state_dict(torch.load(folder / "weights.pt", map_location="cpu", weights_only=True))
        self.net.eval()

    @torch.no_grad()
    def choose(self, board):
        logits, _ = self.net(encode([board]))
        logits = logits[0].numpy()
        logits[~board.legal_mask()] = -np.inf
        return int(np.argmax(logits))


def make_agent(folder):
    return Agent(pathlib.Path(folder))
'''

HAND_IN = r'''
def save_submission(net, folder):
    """Write a submission folder: agent.py, model.py, config.json, weights.pt."""
    folder = pathlib.Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy("__MODEL__.py", folder / "model.py")
    (folder / "agent.py").write_text(AGENT_PY)
    (folder / "config.json").write_text(json.dumps(NET_CONFIG))
    torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, folder / "weights.pt")
    return folder


AGENT_PY = __AGENT_PY__

save_submission(net, "__FOLDER__")
report = check("__FOLDER__", SPEC)          # the same checks as the grader: parameters, time per move, legal moves
'''


def md(cells, text):
    cells.append(nbformat.v4.new_markdown_cell(text.strip("\n")))


def code(cells, text, **subs):
    for k, v in subs.items():
        text = text.replace(k, v)
    cells.append(nbformat.v4.new_code_cell(text.strip("\n")))


# --------------------------------------------------------------------------- notebook a

def notebook_a():
    c = []
    S = dict(__SPEC__="CONNECT5", __MODEL__="model_a", __FOLDER__="submission_a")
    md(c, r"""
# Part a: Connect 5 against the bots

Train a neural network that beats as many Connect 5 bots as possible.

**The game.** Connect 5 is Connect 4 on a bigger board: 9 columns, 9 rows. Players
take turns to drop a stone into a column; it falls to the lowest empty cell. The
first player with **5 in a row** (across, up or diagonal) wins. A full board is a
draw.

**What this notebook gives you:** the game, four bots to train and test against
(`random`, `easy`, `medium`, `hard`), a network template, a working PPO trainer
that plays the bots, an evaluation, a replay viewer, and the hand-in.

**How we grade:** your bot plays a set of **held-out bots** that you do not get:
different styles, and some much stronger than `hard`. Each opening is played
twice, once with each color. Win 1, draw 0.5, loss 0. Your score is the mean.

**Rules for your bot** (details in `PROBLEM.md`):
* at most **200 million parameters**, all in `agent.net`;
* its moves come from your network (a search that calls your network is allowed);
* at most **1 second per move** on one CPU core (a move over 2 seconds loses the game).

**Reading:** [Reinforcement learning fundamentals](https://medium.com/@812jugalgajjar/reinforcement-learning-fundamentals-understanding-the-framework-42963a7a5b40)
""")
    code(c, SETUP, **S)
    md(c, r"""
## 1. The game

`Board` holds a position. `board.play(col)` drops a stone for the side to move.
Row 0 is the bottom row.
""")
    code(c, r'''
b = Board(SPEC)
for col in [4, 4, 3, 5, 2, 3]:
    b.play(col)
print(b.render())
print("to move:", b.player, " legal columns:", b.legal_moves(), " winner:", b.winner)
''')
    md(c, r"""
## 2. The bots

Four bots, from weak to strong. Treat them as black boxes: play them and watch
them, but do not unpack `bots.bin`. Here they play each other (each pair plays
every opening twice, once with each color).
""")
    code(c, r'''
bots = load_packed("bots.bin", "connect5_bots", seed=0)
table = round_robin(bots, SPEC, make_openings(SPEC, 8, seed=1), anchor=("medium", 1000.0))
print(table)
''')
    md(c, r"""
## 3. Your network

The next cell writes `model_a.py`: the network and its input encoding. Change it
as you like (more channels, more blocks, other inputs), then run the cell after
it to reload. The template is a small residual network.
""")
    code(c, MODEL, **S)
    code(c, NET_DEMO, **S)
    md(c, r"""
## 4. Training: PPO against the bots

Many games run at once. In each game your network plays one color, and a bot
picked at random (with the weights in `Config.opponents`) plays the other. The
reward is +1 for a win, -1 for a loss, 0 for a draw, at the end of the game.
Illegal columns are masked out of the policy.

The defaults train for a few minutes. They learn to beat `random` and `easy`; the
stronger bots need much more. Section 6 lists ideas.
""")
    code(c, r'''
@dataclasses.dataclass
class Config:
    updates: int = 60                 # raise this a lot for real training
    n_games: int = 64                 # games played in parallel
    rollout_len: int = 16             # moves per game per update
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    clip_eps: float = 0.2
    epochs: int = 4
    minibatch: int = 256
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    opponents: dict = dataclasses.field(default_factory=lambda: {"random": 0.4, "easy": 0.4, "medium": 0.15, "hard": 0.05})
    seed: int = 0


class Game:
    """One game: your net plays `side`, a bot plays the other color."""

    def __init__(self, rng, cfg):
        names = list(cfg.opponents)
        w = np.array([cfg.opponents[n] for n in names], dtype=float)
        self.opp = names[rng.choice(len(names), p=w / w.sum())]
        self.side = int(rng.integers(1, 3))
        self.board = Board(SPEC)
        self.bot_moves()

    def bot_moves(self):
        while not self.board.done and self.board.player != self.side:
            self.board.play(bots[self.opp].choose(self.board.copy()))


def collect(net, games, cfg, rng, tally):
    """cfg.rollout_len moves in every game. Returns the rollout as tensors."""
    buf = {k: [] for k in ("obs", "mask", "act", "logp", "val", "rew", "done")}
    for _ in range(cfg.rollout_len):
        boards = [g.board for g in games]
        obs = encode(boards).to(device)
        mask = torch.from_numpy(np.stack([b.legal_mask() for b in boards])).to(device)
        with torch.no_grad():
            logits, values = net(obs)
            dist = Categorical(logits=logits.masked_fill(~mask, float("-inf")))
            act = dist.sample()
        rew = np.zeros(len(games), dtype=np.float32)
        done = np.zeros(len(games), dtype=np.float32)
        for i, (g, a) in enumerate(zip(games, act.tolist())):
            g.board.play(a)
            g.bot_moves()
            if g.board.done:
                w = g.board.winner
                rew[i] = 0.0 if w == 0 else (1.0 if w == g.side else -1.0)
                done[i] = 1.0
                tally.setdefault(g.opp, np.zeros(3, int))[0 if rew[i] > 0 else 1 if rew[i] == 0 else 2] += 1
                games[i] = Game(rng, cfg)
        for k, v in (("obs", obs), ("mask", mask), ("act", act), ("logp", dist.log_prob(act)), ("val", values),
                     ("rew", torch.from_numpy(rew).to(device)), ("done", torch.from_numpy(done).to(device))):
            buf[k].append(v)
    with torch.no_grad():
        _, last_value = net(encode([g.board for g in games]).to(device))
    return {k: torch.stack(v) for k, v in buf.items()}, last_value


def gae(rew, val, done, last_value, gamma, lam):
    adv = torch.zeros_like(rew)
    running = torch.zeros_like(last_value)
    for t in reversed(range(rew.shape[0])):
        next_v = last_value if t == rew.shape[0] - 1 else val[t + 1]
        nd = 1.0 - done[t]
        running = rew[t] + gamma * next_v * nd - val[t] + gamma * lam * nd * running
        adv[t] = running
    return adv, adv + val


def ppo_update(net, opt, roll, adv, ret, cfg):
    obs, mask, act, old_logp = (roll[k].flatten(0, 1) for k in ("obs", "mask", "act", "logp"))
    adv, ret = adv.flatten(), ret.flatten()
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    net.train()
    for _ in range(cfg.epochs):
        perm = torch.randperm(len(obs), device=device)
        for s in range(0, len(obs), cfg.minibatch):
            i = perm[s:s + cfg.minibatch]
            logits, v = net(obs[i])
            dist = Categorical(logits=logits.masked_fill(~mask[i], float("-inf")))
            ratio = torch.exp(dist.log_prob(act[i]) - old_logp[i])
            pg = -torch.min(ratio * adv[i], ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv[i]).mean()
            loss = pg + cfg.vf_coef * ((v - ret[i]) ** 2).mean() - cfg.ent_coef * dist.entropy().mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            opt.step()
    net.eval()


def train(cfg, net):
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    games = [Game(rng, cfg) for _ in range(cfg.n_games)]
    tally, t0 = {}, time.time()
    for update in range(1, cfg.updates + 1):
        net.eval()
        roll, last_value = collect(net, games, cfg, rng, tally)
        adv, ret = gae(roll["rew"], roll["val"], roll["done"], last_value, cfg.gamma, cfg.gae_lambda)
        ppo_update(net, opt, roll, adv, ret, cfg)
        if update % 10 == 0:
            line = "  ".join(f"{n} {t[0]}/{t[1]}/{t[2]}" for n, t in sorted(tally.items()))
            print(f"update {update:4d}  {time.time() - t0:5.0f}s  wins/draws/losses: {line}", flush=True)
            tally.clear()
    return net


cfg = Config()
net = train(cfg, Net(**NET_CONFIG).to(device))
''')
    md(c, r"""
## 5. How good is it?

`evaluate` plays your network against each bot: every opening twice, once with
each color, the way the grader plays. Then watch a game: the next cell writes an
HTML page that replays games move by move.
""")
    code(c, r'''
def evaluate(player, n_openings=10, seed=2):
    openings = make_openings(SPEC, n_openings, seed=seed)
    scores = {}
    for name, bot in bots.items():
        r = play_match(player, bot, SPEC, openings)
        scores[name] = r.score
        print(r)
    print(f"mean score {np.mean(list(scores.values())):.3f}")
    return scores


me = NetPlayer(net, "you")
scores = evaluate(me)
''')
    code(c, r'''
recs = [record_game(SPEC, me, bots["medium"], op) for op in make_openings(SPEC, 2, seed=3)]
recs += [record_game(SPEC, bots["medium"], me, op) for op in make_openings(SPEC, 2, seed=3)]
write_html(recs, "replay_a.html", "Your net vs medium")
print("wrote replay_a.html: open it in a browser")
''')
    md(c, r"""
## 6. Make it better

The template is a start, not a good bot. Ideas, roughly from cheap to expensive:

* **Train longer, against harder bots.** Raise `updates`, and move the
  `opponents` weights toward `medium` and `hard` as your net improves (a
  curriculum). Against strong bots from the start, a new net almost never wins,
  so it gets almost no learning signal.
* **Self-play.** Add frozen copies of your own net as opponents (see
  notebook b). The held-out bots are not the four you have.
* **Learn from games.** Record many games between the bots and train the
  network to copy the moves of the side that **won** (supervised learning).
  Then improve it with RL.
* **A bigger network.** You have up to 200 million parameters, but a move must
  take at most 1 second on one CPU core. Measure before you grow it.
* **Search at play time.** `agent.py` may run a search (for example MCTS) that
  calls your network, within the time limit.
* **Use the symmetry.** A position and its mirror image (columns reversed) are
  worth the same.

## 7. Hand in

The next cell writes the folder `submission_a/` and checks it like the grader
does. Hand in that folder (zip it). If you change `agent.py` (for example, to
add a search), edit `AGENT_PY` below or the file in the folder, and run
`check` again.
""")
    code(c, HAND_IN, __AGENT_PY__=repr(AGENT_PY.strip("\n") + "\n"), **S)
    return c


# --------------------------------------------------------------------------- notebook b

def notebook_b():
    c = []
    S = dict(__SPEC__="CONNECT6", __MODEL__="model_b", __FOLDER__="submission_b")
    md(c, r"""
# Part b: Connect 6 by self-play

Train a network to play **Connect 6** on a **13 × 13** board (6 in a row wins), by
playing against itself. You get no bots to learn from: two gauge bots only
measure your progress.

**How we grade:** every team's bot plays every other team's bot. Each pairing
plays the same set of short openings, each twice (once with each color). Win 1,
draw 0.5, loss 0. Your result is your score over all your games. So you must
beat the other teams, not only your own past versions.

**This notebook gives you:**
1. the game and the network template;
2. self-play PPO that saves a new version of your network every few updates;
3. a **tournament harness**: your versions play each other and the gauges, with
   a table and Elo ratings;
4. scrimmages against other teams' submissions;
5. the hand-in.

The same rules as in Part a hold for your bot (200 million parameters, moves
from your network, 1 second per move on one CPU core).

**Reading:** [Self-play (Hugging Face Deep RL course)](https://huggingface.co/learn/deep-rl-course/en/unit7/self-play)
""")
    code(c, SETUP, **S)
    code(c, r'''
gauges = load_packed("bots.bin", "connect6_gauges", seed=0)    # easy6, hard6: only to measure progress
b = Board(SPEC)
for col in [6, 6, 7, 5]:
    b.play(col)
print(b.render())
''')
    md(c, r"""
## 1. Your network

The next cell writes `model_b.py`. Change it as you like and reload.
""")
    code(c, MODEL, **S)
    code(c, NET_DEMO, **S)
    md(c, r"""
## 2. Self-play with PPO

The learner plays many games at once. In each game its opponent is a frozen copy
of an earlier version, taken from a **pool**. Reward: +1 win, -1 loss, 0 draw, at
the end. Every `snapshot_every` updates, the current net is saved to
`checkpoints/` and joins the pool.

**The pool only ever picks the newest version.** That is "naive" self-play, and
it can go round in circles: version 5 beats version 4, version 6 beats version 5,
and nobody beats version 3 any more. Fixing it is part of the task.
""")
    code(c, r'''
CKPT = pathlib.Path("checkpoints")
CKPT.mkdir(exist_ok=True)


class Pool:
    """Frozen past versions of the learner."""

    def __init__(self):
        self.nets, self.names = [], []

    def add(self, net, name):
        frozen = copy.deepcopy(net).eval()
        for p in frozen.parameters():
            p.requires_grad_(False)
        self.nets.append(frozen)
        self.names.append(name)

    def sample(self, rng):
        # TODO: always the newest version. Try all versions, or prefer the ones you lose to.
        return len(self.nets) - 1


class Game:
    def __init__(self, pool, rng):
        self.board = Board(SPEC)
        for col in make_openings(SPEC, 1, plies=2, seed=int(rng.integers(1 << 30)))[0]:
            self.board.play(col)                 # a short random opening, for variety
        self.side = int(rng.integers(1, 3))
        self.opp = pool.sample(rng)


def opponent_turns(games, pool, temperature=1.0):
    """One move for the opponent in every game where it is to move (batched per opponent)."""
    by_opp = {}
    for g in games:
        if not g.board.done and g.board.player != g.side:
            by_opp.setdefault(g.opp, []).append(g)
    for k, gs in by_opp.items():
        boards = [g.board for g in gs]
        with torch.no_grad():
            logits, _ = pool.nets[k](encode(boards).to(device))
        mask = torch.from_numpy(np.stack([b.legal_mask() for b in boards])).to(device)
        cols = Categorical(logits=logits.masked_fill(~mask, float("-inf")) / temperature).sample().tolist()
        for g, col in zip(gs, cols):
            g.board.play(col)


@dataclasses.dataclass
class Config:
    updates: int = 30                 # raise this a lot for real training
    n_games: int = 64
    rollout_len: int = 16
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    clip_eps: float = 0.2
    epochs: int = 4
    minibatch: int = 256
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    snapshot_every: int = 10
    seed: int = 0


def collect(net, games, pool, cfg, rng, tally):
    buf = {k: [] for k in ("obs", "mask", "act", "logp", "val", "rew", "done")}
    opponent_turns(games, pool)
    for _ in range(cfg.rollout_len):
        boards = [g.board for g in games]
        obs = encode(boards).to(device)
        mask = torch.from_numpy(np.stack([b.legal_mask() for b in boards])).to(device)
        with torch.no_grad():
            logits, values = net(obs)
            dist = Categorical(logits=logits.masked_fill(~mask, float("-inf")))
            act = dist.sample()
        for g, a in zip(games, act.tolist()):
            g.board.play(a)
        opponent_turns(games, pool)
        rew = np.zeros(len(games), dtype=np.float32)
        done = np.zeros(len(games), dtype=np.float32)
        for i, g in enumerate(games):
            if g.board.done:
                w = g.board.winner
                rew[i] = 0.0 if w == 0 else (1.0 if w == g.side else -1.0)
                done[i] = 1.0
                tally[0 if rew[i] > 0 else 1 if rew[i] == 0 else 2] += 1
                games[i] = Game(pool, rng)
        opponent_turns(games, pool)
        for k, v in (("obs", obs), ("mask", mask), ("act", act), ("logp", dist.log_prob(act)), ("val", values),
                     ("rew", torch.from_numpy(rew).to(device)), ("done", torch.from_numpy(done).to(device))):
            buf[k].append(v)
    with torch.no_grad():
        _, last_value = net(encode([g.board for g in games]).to(device))
    return {k: torch.stack(v) for k, v in buf.items()}, last_value


def gae(rew, val, done, last_value, gamma, lam):
    adv = torch.zeros_like(rew)
    running = torch.zeros_like(last_value)
    for t in reversed(range(rew.shape[0])):
        next_v = last_value if t == rew.shape[0] - 1 else val[t + 1]
        nd = 1.0 - done[t]
        running = rew[t] + gamma * next_v * nd - val[t] + gamma * lam * nd * running
        adv[t] = running
    return adv, adv + val


def ppo_update(net, opt, roll, adv, ret, cfg):
    obs, mask, act, old_logp = (roll[k].flatten(0, 1) for k in ("obs", "mask", "act", "logp"))
    adv, ret = adv.flatten(), ret.flatten()
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    net.train()
    for _ in range(cfg.epochs):
        perm = torch.randperm(len(obs), device=device)
        for s in range(0, len(obs), cfg.minibatch):
            i = perm[s:s + cfg.minibatch]
            logits, v = net(obs[i])
            dist = Categorical(logits=logits.masked_fill(~mask[i], float("-inf")))
            ratio = torch.exp(dist.log_prob(act[i]) - old_logp[i])
            pg = -torch.min(ratio * adv[i], ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv[i]).mean()
            loss = pg + cfg.vf_coef * ((v - ret[i]) ** 2).mean() - cfg.ent_coef * dist.entropy().mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            opt.step()
    net.eval()


def save_version(net, name):
    torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, CKPT / f"{name}.pt")


def train(cfg, net, pool=None, version=0):
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    if pool is None:
        pool = Pool()
        pool.add(net, f"v{version}")
        save_version(net, f"v{version}")
    games = [Game(pool, rng) for _ in range(cfg.n_games)]
    tally, t0 = np.zeros(3, int), time.time()
    for update in range(1, cfg.updates + 1):
        net.eval()
        roll, last_value = collect(net, games, pool, cfg, rng, tally)
        adv, ret = gae(roll["rew"], roll["val"], roll["done"], last_value, cfg.gamma, cfg.gae_lambda)
        ppo_update(net, opt, roll, adv, ret, cfg)
        if update % cfg.snapshot_every == 0:
            version += 1
            save_version(net, f"v{version}")
            pool.add(net, f"v{version}")
            w, d, l = tally
            print(f"update {update:5d}  {time.time() - t0:5.0f}s  vs pool +{w} ={d} -{l}  -> saved v{version}", flush=True)
            tally[:] = 0
    return net, pool, version


cfg = Config()
net, pool, version = train(cfg, Net(**NET_CONFIG).to(device))
''')
    md(c, r"""
## 3. The tournament harness

Every saved version plays every other version and the two gauges, on the same
openings, with both colors. You get points, Elo ratings (`easy6` fixed at 1000)
and the score matrix. Read the matrix, not only the Elo: in self-play, version A
can beat B, B beat C, and C beat A. After the short default run, expect a noisy
table where later versions do not always beat earlier ones (not even `v0`). That
is the problem to solve.
""")
    code(c, r'''
def load_versions(every=1):
    paths = sorted(CKPT.glob("v*.pt"), key=lambda p: int(p.stem[1:]))[::every]
    players = {}
    for p in paths:
        v = Net(**NET_CONFIG).to(device)
        v.load_state_dict(torch.load(p, map_location=device, weights_only=True))
        players[p.stem] = NetPlayer(v.eval(), p.stem)
    return players


players = {**load_versions(), "random": RandomPlayer(seed=0), **gauges}
table = round_robin(players, SPEC, make_openings(SPEC, 4, seed=7), anchor=("easy6", 1000.0))
print(table)
names = table.names
print("score of row against column:")
print("          " + " ".join(f"{n[:8]:>8}" for n in names))
for i, n in enumerate(names):
    print(f"{n[:8]:>8}  " + " ".join("       -" if i == j else f"{table.matrix[i, j]:8.2f}" for j in range(len(names))))
''')
    md(c, r"""
### Watch a game

The next cell writes an HTML page with games of your newest version against `hard6`.
""")
    code(c, r'''
latest = NetPlayer(net, f"v{version}")
recs = [record_game(SPEC, latest, gauges["hard6"], op) for op in make_openings(SPEC, 1, seed=3)]
recs += [record_game(SPEC, gauges["hard6"], latest, op) for op in make_openings(SPEC, 1, seed=3)]
write_html(recs, "replay_b.html", f"v{version} vs hard6")
print("wrote replay_b.html: open it in a browser")
''')
    md(c, r"""
## 4. Play other teams before the final

Teams may swap submission folders during the event. Put them in `scrimmage/`
(one folder per team, each with its `agent.py`), and run a tournament against
your newest version.
""")
    code(c, r'''
from connectk.agents import load_agent

scrim = pathlib.Path("scrimmage")
others = {}
if scrim.exists():
    for folder in sorted(p for p in scrim.iterdir() if (p / "agent.py").exists()):
        others[folder.name] = load_agent(folder)
if others:
    print(round_robin({"mine": NetPlayer(net, "mine"), **others, **gauges}, SPEC, make_openings(SPEC, 4, seed=11)))
else:
    print("no submission folders in scrimmage/ yet")
''')
    md(c, r"""
## 5. Make it better

* **The pool.** Sample all past versions, or favor the ones you lose to
  (prioritized fictitious self-play). Keep the gauges or a random player in the
  pool too.
* **Variety.** The random openings keep games different. More variety
  (temperature, more opening moves) makes a sturdier bot.
* **More compute.** More games in parallel, longer runs, a GPU.
* **A bigger network** (up to 200 million parameters, but 1 second per move on a CPU).
* **Search.** MCTS on top of your network, in training (to make better targets,
  as in AlphaZero) and at play time (within the time limit).
* **Choose your hand-in with the tournament,** not by "newest". The newest
  version is not always the strongest.

## 6. Hand in

The next cell writes the folder `submission_b/` from `net` (change it to your best
version) and checks it like the grader does. Hand in that folder (zip it).
""")
    code(c, HAND_IN, __AGENT_PY__=repr(AGENT_PY.strip("\n") + "\n"), **S)
    return c


def build(cells):
    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    for i, cell in enumerate(nb.cells):
        cell["id"] = f"cell-{i:02d}"        # fixed ids keep the files stable for --check
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"},
                   "accelerator": "GPU"}
    return nbformat.writes(nb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    out = {"contest/notebook_a.ipynb": build(notebook_a()), "contest/notebook_b.ipynb": build(notebook_b())}
    stale = []
    for rel, text in out.items():
        path = ROOT / rel
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(rel)
        else:
            path.write_text(text)
            print(f"wrote {rel}")
    if stale:
        sys.exit(f"out of date: {', '.join(stale)}; run python3 hidden/build_contest.py")


if __name__ == "__main__":
    main()
