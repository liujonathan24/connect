"""Imitation training on one or more .npz files (g: relative grids, t: targets, z: results).

    python3 hidden/strong/train_imitation.py --game connect6 --data runs/winners6.npz runs/offline6.npz \\
        --init part_a_connect6/runs/calib/reference2/bc.pt --steps 20000 --out runs/c6_imit.pt

Policy: cross-entropy to t. Value: MSE to z. Mirror images at random.
Cosine learning-rate decay. Saves the net every --save_every steps and at
the end, and prints the score against the eval panel (train_strong.panel).
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from connectk.game import SPECS  # noqa: E402
from connectk.net import ConnectNet, encode_relative, save_net  # noqa: E402
from hidden.strong.train_strong import evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", default="connect6", choices=sorted(SPECS))
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--init", default=None)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--save_every", type=int, default=5000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = SPECS[args.game]
    torch.set_num_threads(args.threads)
    parts = [np.load(p) for p in args.data]
    g = np.concatenate([d["g"] for d in parts])
    t = np.concatenate([d["t"] for d in parts]).astype(np.float32)
    z = np.concatenate([d["z"] for d in parts]).astype(np.float32)
    print(f"{len(g)} positions from {len(parts)} files", flush=True)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    net = ConnectNet(spec)
    if args.init:
        net.load_state_dict(torch.load(args.init, weights_only=True))
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        i = rng.integers(len(g), size=args.batch)
        gg, tt = g[i], t[i]
        flip = rng.random(args.batch) < 0.5
        gg = np.where(flip[:, None, None], gg[:, :, ::-1], gg)
        tt = np.where(flip[:, None], tt[:, ::-1], tt)
        logits, v = net(torch.from_numpy(encode_relative(gg)))
        p_loss = -(torch.from_numpy(np.ascontiguousarray(tt)) * F.log_softmax(logits, -1)).sum(-1).mean()
        loss = p_loss + args.vf * F.mse_loss(v, torch.from_numpy(z[i]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % args.save_every == 0 or step == args.steps:
            save_net(net, args.out)
            net.eval()
            s = evaluate(net, spec, n_openings=4)
            net.train()
            print(f"step {step}: policy {p_loss.item():.3f} ({time.time() - t0:.0f}s)  eval "
                  + "  ".join(f"{k} {x:.2f}" for k, x in s.items()) + f"  mean {np.mean(list(s.values())):.3f}", flush=True)


if __name__ == "__main__":
    main()
