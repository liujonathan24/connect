"""Reference Part A trainer: the shipped ppo.py with its problems fixed.

    cd part_a_connect6
    python3 ../hidden/solutions/part_a/ppo_fixed.py --out runs/ref                # PPO only
    python3 ../hidden/solutions/part_a/ppo_fixed.py --bc_steps 4000 --out runs/ref_bc   # imitation first

What changed compared with ppo.py (see ANSWERS.md for why):
  * reward: +1 win, -1 loss, 0 draw, at the end of the game only
  * illegal columns masked in the policy, in the rollout and in the update
  * gamma 0.99, lambda 0.95, lr 3e-4, clip 0.2, 4 epochs, minibatch 256,
    entropy 0.01, value coef 0.5, grad clip 0.5, normalized advantages
  * both colors; bot_c (the cheater) removed; frozen copies of the agent
    join the opponent pool, for variety
  * optional: imitation of the clean offline games (house and honest public
    bots), with mirror images, before PPO
"""

import argparse
import collections
import copy
import dataclasses
import gzip
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path.cwd()))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from connectk.game import Board  # noqa: E402
from connectk.net import ConnectNet, encode_batch, encode_relative, relative_grid, save_net  # noqa: E402
from env import SPEC, VecBotEnv, load_opponents  # noqa: E402
from hidden.solutions.part_a.clean_data import check_game  # noqa: E402
from ppo import compute_gae, evaluate, policy_step, ppo_update  # noqa: E402


@dataclasses.dataclass
class Config:
    updates: int = 400
    n_envs: int = 64
    rollout_len: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    clip_eps: float = 0.2
    epochs: int = 4
    minibatch: int = 256
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_adv: bool = True
    mask_illegal: bool = True
    agent_side: str = "random"
    opponents: tuple = ("bot_a", "bot_b", "bot_d")
    snapshot_every: int = 50      # add a frozen copy of the agent to the opponents
    eval_every: int = 50
    eval_openings: int = 4
    bc_steps: int = 0
    seed: int = 0
    threads: int = 4
    out: str = "runs/ref"


def reward_fn(info):
    if info["winner"] == info["agent"]:
        return 1.0
    if info["winner"]:
        return -1.0
    return 0.0


class FrozenNet:
    """A frozen copy of the agent, as an opponent bot (samples from its policy)."""

    def __init__(self, net, name, seed):
        self.net = copy.deepcopy(net).eval()
        self.name = name
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def choose(self, board):
        logits, _ = self.net(torch.from_numpy(encode_batch([board])))
        logits = logits[0].numpy().astype(np.float64)
        logits[~board.legal_mask()] = -np.inf
        p = np.exp(logits - logits.max())
        return int(self.rng.choice(len(p), p=p / p.sum()))

    def play(self, board):
        board.play(self.choose(board))


def offline_positions(path="data/offline_games.jsonl.gz", winners_only=True):
    """(grids, moves, outcomes) from clean games without bot_c.

    With winners_only, the policy data keeps only the moves of the side that
    won (both sides in a draw) and never bot_a's moves (it plays at random
    20% of the time). Copying every player mixes styles that disagree.
    """
    grids, moves, zs = [], [], []
    with gzip.open(path, "rt") as f:
        for line in f:
            g = json.loads(line)
            if check_game(g) is not None or "bot_c" in (g["p1"], g["p2"]):
                continue
            names = {1: g["p1"], 2: g["p2"]}
            b = Board(SPEC)
            for p, _, c in g["moves"]:
                keep = not winners_only or ((g["winner"] in (0, p)) and names[p] != "bot_a")
                if keep:
                    grids.append(relative_grid(b))
                    moves.append(c)
                    zs.append(0.0 if g["winner"] == 0 else (1.0 if g["winner"] == p else -1.0))
                b.play(c)
    return np.array(grids, np.int8), np.array(moves), np.array(zs, np.float32)


def imitation(net, steps, rng, batch=256, lr=1e-3):
    g, m, z = offline_positions()
    print(f"imitation on {len(g)} positions", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    for step in range(1, steps + 1):
        i = rng.integers(len(g), size=batch)
        gg, mm = g[i], m[i].copy()
        flip = rng.random(batch) < 0.5
        gg = np.where(flip[:, None, None], gg[:, :, ::-1], gg)
        mm[flip] = SPEC.cols - 1 - mm[flip]
        logits, v = net(torch.from_numpy(encode_relative(gg)))
        loss = F.cross_entropy(logits, torch.from_numpy(mm)) + F.mse_loss(v, torch.from_numpy(z[i]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"  imitation step {step}: loss {loss.item():.3f}", flush=True)


def train(cfg):
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.threads)
    out = pathlib.Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    all_bots = load_opponents(seed=cfg.seed)
    opponents = {n: all_bots[n] for n in cfg.opponents}
    net = ConnectNet(SPEC)
    if cfg.bc_steps:
        imitation(net, cfg.bc_steps, rng)
        save_net(net, out / "bc.pt")
        res = evaluate(net, all_bots, cfg.eval_openings)
        print("  after imitation: " + "  ".join(f"{n} {r.score:.2f}" for n, r in res.items()), flush=True)
    env = VecBotEnv(cfg.n_envs, opponents, agent_side=cfg.agent_side, seed=cfg.seed)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    boards = env.reset()
    tally = collections.defaultdict(lambda: np.zeros(3, int))
    t0 = time.time()
    for update in range(1, cfg.updates + 1):
        obs_l, mask_l, act_l, logp_l, val_l, rew_l, done_l = [], [], [], [], [], [], []
        net.eval()
        for _ in range(cfg.rollout_len):
            actions, logp, values, obs, masks = policy_step(net, boards, cfg, "cpu")
            boards, dones, infos = env.step(actions.numpy())
            for info in infos:
                if info["winner"] or info["draw"]:
                    tally[info["opponent"]][0 if info["winner"] == info["agent"] else 1 if info["draw"] else 2] += 1
            obs_l.append(obs); mask_l.append(masks); act_l.append(actions); logp_l.append(logp)
            val_l.append(values)
            rew_l.append(torch.tensor([reward_fn(i) for i in infos], dtype=torch.float32))
            done_l.append(torch.from_numpy(dones.astype(np.float32)))
        with torch.no_grad():
            _, last_v = net(torch.from_numpy(encode_batch(boards)))
        adv, ret = compute_gae(torch.stack(rew_l), torch.stack(val_l), torch.stack(done_l), last_v,
                               cfg.gamma, cfg.gae_lambda)
        net.train()
        stats = ppo_update(net, opt, (torch.cat(obs_l), torch.cat(mask_l), torch.cat(act_l), torch.cat(logp_l),
                                      adv.flatten(), ret.flatten()), cfg)
        if update % 10 == 0:
            games = " ".join(f"{n}:{t[0]}/{t[1]}/{t[2]}" for n, t in sorted(tally.items()))
            print(f"[{update:4d}] {time.time() - t0:6.0f}s ent {stats['entropy']:.2f} v {stats['v_loss']:.3f} "
                  f"W/D/L {games}", flush=True)
            tally.clear()
        if cfg.snapshot_every and update % cfg.snapshot_every == 0:
            name = f"self_{update}"
            snap = FrozenNet(net, name, cfg.seed + update)
            for e in env.envs:
                e.opponents[name] = snap
                e.names = list(e.opponents)
                w = np.array([1.0 if not n.startswith("self_") else 0.5 for n in e.names])
                e.weights = w / w.sum()
        if update % cfg.eval_every == 0 or update == cfg.updates:
            save_net(net, out / f"net_{update:05d}.pt")
            save_net(net, out / "submission.pt")
            res = evaluate(net, all_bots, cfg.eval_openings)
            print("  eval: " + "  ".join(f"{n} {r.score:.2f}" for n, r in res.items()), flush=True)
    return net


def main():
    ap = argparse.ArgumentParser()
    for f in dataclasses.fields(Config):
        if f.type is bool:
            ap.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"), default=f.default)
        elif f.name == "opponents":
            ap.add_argument("--opponents", nargs="+", default=list(f.default))
        else:
            ap.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    args = ap.parse_args()
    train(Config(**{k: (tuple(v) if k == "opponents" else v) for k, v in vars(args).items()}))


if __name__ == "__main__":
    main()
