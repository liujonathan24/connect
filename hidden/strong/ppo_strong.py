"""PPO for the strong nets: play the bots (and past copies of itself) directly.

    python3 hidden/strong/ppo_strong.py --game connect5 --init runs/c5h_last.pt \\
        --tactics runs/tactics5.npz --out runs/c5p

Starts from a trained net. Opponents: the strong-net bot pool (train_strong.
bot_pool) plus frozen copies of the agent, added every --snapshot_every
updates. Illegal columns are masked. Reward +1 win, -1 loss, 0 draw. With
--tactics, every update also adds the tactical blunder penalty on a batch
from make_tactics.py, so the agent keeps its tactics while PPO changes it.
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import copy  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.distributions import Categorical  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from connectk.arena import make_openings  # noqa: E402
from connectk.game import SPECS, Board  # noqa: E402
from connectk.net import ConnectNet, encode_batch, mask_batch, save_net  # noqa: E402
from hidden.strong.distill import tactics_loss  # noqa: E402
from hidden.strong.train_strong import bot_pool, evaluate  # noqa: E402


class Frozen:
    """A frozen copy of the agent as an opponent (samples from its policy)."""

    def __init__(self, net, name, seed):
        self.net = copy.deepcopy(net).eval()
        self.name = name
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def choose(self, board):
        logits = self.net(torch.from_numpy(encode_batch([board])))[0][0].numpy().astype(np.float64)
        logits[~board.legal_mask()] = -np.inf
        p = np.exp(logits - logits.max())
        return int(self.rng.choice(len(p), p=p / p.sum()))


class Game:
    def __init__(self, spec, opponents, weights, rng):
        self.board = Board(spec)
        for c in make_openings(spec, 1, plies=2, seed=int(rng.integers(1 << 30)))[0]:
            self.board.play(c)
        self.side = int(rng.integers(1, 3))
        self.opp = opponents[rng.choice(len(opponents), p=weights)]
        self.opp_moves()

    def opp_moves(self):
        while not self.board.done and self.board.player != self.side:
            self.board.play(self.opp.choose(self.board.copy()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", default="connect5", choices=["connect5", "connect6"])
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tactics", default=None)
    ap.add_argument("--tactic_weight", type=float, default=2.0)
    ap.add_argument("--updates", type=int, default=2000)
    ap.add_argument("--n_games", type=int, default=128)
    ap.add_argument("--rollout_len", type=int, default=16)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--ent", type=float, default=0.003)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--snapshot_every", type=int, default=100)
    ap.add_argument("--self_weight", type=float, default=0.3, help="share of games against frozen copies")
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    spec = SPECS[args.game]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    net = ConnectNet(spec)
    net.load_state_dict(torch.load(args.init, weights_only=True))
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    tactics = None
    if args.tactics:
        d = np.load(args.tactics)
        tactics = (d["grids"], d["bad"])
    _, bots = bot_pool(spec, args.seed)
    snaps = [Frozen(net, "self_0", args.seed)]

    def pool():
        opps = bots + snaps
        w = np.array([(1 - args.self_weight) / len(bots)] * len(bots) + [args.self_weight / len(snaps)] * len(snaps))
        return opps, w / w.sum()

    opps, w = pool()
    games = [Game(spec, opps, w, rng) for _ in range(args.n_games)]
    log = open(out / "log.txt", "a")
    t0 = time.time()
    tally = np.zeros(3, int)

    def say(msg):
        msg = f"[{time.time() - t0:7.0f}s] {msg}"
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    for update in range(1, args.updates + 1):
        net.eval()
        buf = {k: [] for k in ("obs", "mask", "act", "logp", "val", "rew", "done")}
        for _ in range(args.rollout_len):
            boards = [g.board for g in games]
            obs = torch.from_numpy(encode_batch(boards))
            mask = torch.from_numpy(mask_batch(boards))
            with torch.no_grad():
                logits, values = net(obs)
                dist = Categorical(logits=logits.masked_fill(~mask, float("-inf")))
                act = dist.sample()
            rew = np.zeros(len(games), np.float32)
            done = np.zeros(len(games), np.float32)
            for i, (g, a) in enumerate(zip(games, act.tolist())):
                g.board.play(a)
                g.opp_moves()
                if g.board.done:
                    wnr = g.board.winner
                    rew[i] = 0.0 if wnr == 0 else (1.0 if wnr == g.side else -1.0)
                    done[i] = 1.0
                    tally[0 if rew[i] > 0 else 1 if rew[i] == 0 else 2] += 1
                    games[i] = Game(spec, opps, w, rng)
            for k, v in (("obs", obs), ("mask", mask), ("act", act), ("logp", dist.log_prob(act)), ("val", values),
                         ("rew", torch.from_numpy(rew)), ("done", torch.from_numpy(done))):
                buf[k].append(v)
        with torch.no_grad():
            _, last_v = net(torch.from_numpy(encode_batch([g.board for g in games])))
        rews, vals, dones = torch.stack(buf["rew"]), torch.stack(buf["val"]), torch.stack(buf["done"])
        adv = torch.zeros_like(rews)
        run = torch.zeros_like(last_v)
        for t in reversed(range(args.rollout_len)):
            nv = last_v if t == args.rollout_len - 1 else vals[t + 1]
            nd = 1.0 - dones[t]
            delta = rews[t] + args.gamma * nv * nd - vals[t]
            run = delta + args.gamma * args.lam * nd * run
            adv[t] = run
        ret = (adv + vals).flatten()
        adv = adv.flatten()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        obs, mask = torch.cat(buf["obs"]), torch.cat(buf["mask"])
        act, old_logp = torch.cat(buf["act"]), torch.cat(buf["logp"])
        net.train()
        n = len(obs)
        for _ in range(args.epochs):
            perm = torch.randperm(n)
            for s in range(0, n, args.minibatch):
                i = perm[s:s + args.minibatch]
                logits, v = net(obs[i])
                dist = Categorical(logits=logits.masked_fill(~mask[i], float("-inf")))
                ratio = torch.exp(dist.log_prob(act[i]) - old_logp[i])
                pg = -torch.min(ratio * adv[i], ratio.clamp(1 - args.clip, 1 + args.clip) * adv[i]).mean()
                loss = pg + args.vf * ((v - ret[i]) ** 2).mean() - args.ent * dist.entropy().mean()
                if tactics is not None:
                    loss = loss + args.tactic_weight * tactics_loss(net, tactics, 256, rng)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
        if update % 10 == 0:
            wn, dr, ls = tally
            say(f"update {update}: games +{wn} ={dr} -{ls} (score {(wn + dr / 2) / max(1, tally.sum()):.3f})")
            tally[:] = 0
        if update % args.snapshot_every == 0:
            snaps.append(Frozen(net, f"self_{update}", args.seed + update))
            opps, w = pool()
        if update % args.eval_every == 0:
            save_net(net, out / f"upd_{update:05d}.pt")
            save_net(net, out / "last.pt")
            scores = evaluate(net, spec, n_openings=8)
            say(f"eval at update {update}: " + "  ".join(f"{k} {s:.2f}" for k, s in scores.items())
                + f"  mean {np.mean(list(scores.values())):.3f}")


if __name__ == "__main__":
    main()
