"""PPO baseline for Part A (Connect 6 on 13 x 13).

    python3 ppo.py                      # train with the settings in Config
    python3 ppo.py --updates 50 --out runs/try1

This file runs, but it does not learn much yet:

  * reward_fn() returns 0 for every move. Write it.
  * Many settings in Config are poor. Find them and fix them.
  * It trains against all four bots, and one of them is bad data.

You may change anything in this file (and write new files). You may not
change the network or its input encoding (connectk/net.py): the grader loads
your weights into that exact network.
"""

import argparse
import dataclasses
import pathlib
import time
from collections import defaultdict

import numpy as np
import torch
from torch.distributions import Categorical

from connectk.arena import make_openings, play_match
from connectk.net import ConnectNet, NetPlayer, encode_batch, mask_batch, save_net
from env import SPEC, VecBotEnv, load_opponents


@dataclasses.dataclass
class Config:
    updates: int = 300            # PPO updates (one rollout each)
    n_envs: int = 32              # games played in parallel
    rollout_len: int = 32         # agent moves per game per rollout
    gamma: float = 0.5
    gae_lambda: float = 0.0
    lr: float = 3e-2
    clip_eps: float = 0.9
    epochs: int = 30              # passes over each rollout
    minibatch: int = 32
    ent_coef: float = 0.0
    vf_coef: float = 10.0
    max_grad_norm: float = 1e9
    normalize_adv: bool = False
    mask_illegal: bool = False    # if False, an illegal move ends the game as a loss
    agent_side: str = "first"     # "first", "second" or "random"
    opponents: tuple = ("bot_a", "bot_b", "bot_c", "bot_d")
    eval_every: int = 25
    eval_openings: int = 4
    seed: int = 0
    threads: int = 4
    out: str = "runs/ppo"


def reward_fn(info):
    """Reward for one agent move. Called once per move, with the env's info.

    info["winner"], info["agent"], info["draw"], info["illegal"] tell you how the
    game stands after your move and the bot's answer. info["after_agent"] is
    the board right after your move.
    """
    return 0.0  # TODO


def policy_step(net, boards, cfg, device):
    """Sample one action per board. Returns actions, log-probs, values, obs, masks."""
    obs = torch.from_numpy(encode_batch(boards)).to(device)
    masks = torch.from_numpy(mask_batch(boards)).to(device)
    with torch.no_grad():
        logits, values = net(obs)
        if cfg.mask_illegal:
            logits = logits.masked_fill(~masks, float("-inf"))
        dist = Categorical(logits=logits)
        actions = dist.sample()
        logp = dist.log_prob(actions)
    return actions, logp, values, obs, masks


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """rewards, values, dones: (T, N). Returns advantages and returns, (T, N)."""
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(last_value)
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        last = delta + gamma * lam * not_done * last
        adv[t] = last
    return adv, adv + values


def ppo_update(net, opt, batch, cfg):
    obs, masks, actions, old_logp, adv, ret = batch
    n = obs.shape[0]
    if cfg.normalize_adv:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    stats = defaultdict(list)
    for _ in range(cfg.epochs):
        perm = torch.randperm(n)
        for start in range(0, n, cfg.minibatch):
            idx = perm[start:start + cfg.minibatch]
            logits, values = net(obs[idx])
            if cfg.mask_illegal:
                logits = logits.masked_fill(~masks[idx], float("-inf"))
            dist = Categorical(logits=logits)
            logp = dist.log_prob(actions[idx])
            ratio = torch.exp(logp - old_logp[idx])
            s1 = ratio * adv[idx]
            s2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv[idx]
            pg_loss = -torch.min(s1, s2).mean()
            v_loss = ((values - ret[idx]) ** 2).mean()
            entropy = dist.entropy().mean()
            loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            opt.step()
            stats["pg_loss"].append(pg_loss.item())
            stats["v_loss"].append(v_loss.item())
            stats["entropy"].append(entropy.item())
            stats["clip_frac"].append(((ratio - 1).abs() > cfg.clip_eps).float().mean().item())
    return {k: float(np.mean(v)) for k, v in stats.items()}


def evaluate(net, opponents, n_openings):
    """Strict matches (the referee checks every move) against each bot."""
    player = NetPlayer(net, "agent")
    openings = make_openings(SPEC, n_openings, plies=2, seed=12345)
    return {name: play_match(player, bot, SPEC, openings) for name, bot in opponents.items()}


def train(cfg):
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.threads)
    device = "cpu"
    out = pathlib.Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    all_bots = load_opponents(seed=cfg.seed)
    opponents = {n: all_bots[n] for n in cfg.opponents}
    env = VecBotEnv(cfg.n_envs, opponents, agent_side=cfg.agent_side, seed=cfg.seed)
    net = ConnectNet(SPEC).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    boards = env.reset()
    tally = defaultdict(lambda: np.zeros(4, int))   # per bot: wins, draws, losses, illegal
    t0 = time.time()
    for update in range(1, cfg.updates + 1):
        buf = defaultdict(list)
        net.eval()
        for _ in range(cfg.rollout_len):
            actions, logp, values, obs, masks = policy_step(net, boards, cfg, device)
            boards, dones, infos = env.step(actions.cpu().numpy())
            rewards = [reward_fn(info) for info in infos]
            for info in infos:
                if info["winner"] or info["draw"] or info["illegal"]:
                    t = tally[info["opponent"]]
                    t[0 if info["winner"] == info["agent"] else 1 if info["draw"] else 2] += 1
                    t[3] += info["illegal"]
            buf["obs"].append(obs)
            buf["masks"].append(masks)
            buf["actions"].append(actions)
            buf["logp"].append(logp)
            buf["values"].append(values)
            buf["rewards"].append(torch.tensor(rewards, dtype=torch.float32))
            buf["dones"].append(torch.from_numpy(dones.astype(np.float32)))
        with torch.no_grad():
            _, last_value = net(torch.from_numpy(encode_batch(boards)).to(device))
        values = torch.stack(buf["values"])
        adv, ret = compute_gae(torch.stack(buf["rewards"]), values, torch.stack(buf["dones"]),
                               last_value, cfg.gamma, cfg.gae_lambda)
        batch = (torch.cat(buf["obs"]), torch.cat(buf["masks"]), torch.cat(buf["actions"]),
                 torch.cat(buf["logp"]), adv.flatten(), ret.flatten())
        net.train()
        stats = ppo_update(net, opt, batch, cfg)
        if update % 5 == 0 or update == 1:
            games = " ".join(f"{n}:{t[0]}/{t[1]}/{t[2]}" + (f"(ill {t[3]})" if t[3] else "")
                             for n, t in sorted(tally.items()))
            print(f"[{update:4d}] {time.time() - t0:6.0f}s  pg {stats['pg_loss']:+.3f}  v {stats['v_loss']:.3f}  "
                  f"ent {stats['entropy']:.2f}  clip {stats['clip_frac']:.2f}  W/D/L {games}", flush=True)
            tally.clear()
        if update % cfg.eval_every == 0 or update == cfg.updates:
            save_net(net, out / f"net_{update:05d}.pt")
            save_net(net, out / "submission.pt")
            res = evaluate(net, all_bots, cfg.eval_openings)
            print("  eval: " + "  ".join(f"{n} {r.score:.2f}" for n, r in res.items()), flush=True)
    return net


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in dataclasses.fields(Config):
        kind = f.type if f.type in (int, float, str) else None
        if f.type is bool or f.type == "bool":
            parser.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"), default=f.default)
        elif f.name == "opponents":
            parser.add_argument("--opponents", nargs="+", default=list(f.default))
        else:
            parser.add_argument(f"--{f.name}", type=kind or type(f.default), default=f.default)
    args = parser.parse_args()
    cfg = Config(**{k: (tuple(v) if k == "opponents" else v) for k, v in vars(args).items()})
    train(cfg)


if __name__ == "__main__":
    main()
