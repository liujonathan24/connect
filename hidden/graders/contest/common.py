"""Shared by the contest graders: play matches between submissions and bots in worker processes.

Every job loads its submissions itself (connectk.agents.load_agent), with one
CPU thread and no GPU, so a team's bot plays the way PROBLEM.md promises.
A move slower than connectk.agents.MOVE_LIMIT loses that game (the referee
treats the exception like an illegal move).
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")          # graders play on CPU
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import multiprocessing as mp  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

STRONG = ROOT / "hidden/strong"
HELDOUT_BOTS = ROOT / "hidden/heldout_bots"


def team_name(folder):
    folder = pathlib.Path(folder).resolve()
    return folder.parent.name if folder.name.startswith("submission") else folder.name


def make_player(ref, spec):
    """ref: ("team", folder) | ("bot", roster, name, seed) | ("net", path, sims, name) | ("agent", folder, name).

    "agent" is a held-out bot in the submission format (hidden/heldout_bots/): it
    plays like a submission, but without the move timer.
    """
    import torch
    torch.set_num_threads(1)
    from connectk.agents import TimedAgent, load_agent
    kind = ref[0]
    if kind == "team":
        return TimedAgent(load_agent(ref[1]), team_name(ref[1]))
    if kind == "bot":
        from hidden.bots import heldout
        return {b.name: b for b in getattr(heldout, ref[1])(seed=ref[3])}[ref[2]]
    if kind == "agent":
        agent = load_agent(ref[1])
        agent.name = ref[2]
        return agent
    if kind == "net":
        from connectk.net import NetPlayer
        from hidden.strong.bignet import load_model
        from hidden.strong.players import NeuralMCTS
        net = load_model(spec, ref[1])
        return NeuralMCTS(net, ref[2], ref[3]) if ref[2] else NetPlayer(net, ref[3])
    raise ValueError(ref)


def _job(args):
    a_ref, b_ref, spec_name, openings = args
    from connectk.arena import play_match
    from connectk.game import SPECS
    spec = SPECS[spec_name]
    try:
        a = make_player(a_ref, spec)
    except Exception as exc:                  # a team that cannot load loses every game of the job
        return a_ref, b_ref, None, f"{type(exc).__name__}: {exc}"
    b = make_player(b_ref, spec)
    r = play_match(a, b, spec, openings, keep_games=True)
    timeouts = sum(1 for a_first, g in r.games if g.reason.startswith("error: MoveTimeout") and
                   g.winner == (2 if a_first else 1))
    stats = {"wins": r.wins, "draws": r.draws, "losses": r.losses, "timeouts": timeouts,
             "illegal": r.illegal_a - timeouts}
    for side, p in (("a", a), ("b", b)):
        if hasattr(p, "moves"):
            stats[f"{side}_moves"], stats[f"{side}_seconds"], stats[f"{side}_slowest"] = p.moves, p.seconds, p.slowest
    return a_ref, b_ref, stats, None


def run_jobs(jobs, workers):
    if workers > 1 and sys.platform.startswith("linux"):
        with mp.get_context("fork").Pool(workers) as pool:
            yield from pool.imap_unordered(_job, jobs)
    else:
        yield from map(_job, jobs)
