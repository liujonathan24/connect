# Connect 5 / Connect 6: an RL hackathon problem

Teams train a neural network to play Connect-k with gravity: the rules of
Connect 4 on a bigger board, with a longer line to win.

| Part | Game | Task | Result |
|---|---|---|---|
| A | Connect 6, 13 × 13 | Online RL against 4 given bots (one of them cheats), a broken PPO trainer, optional poisoned offline games | Score against held-out bots |
| B | Connect 5, 9 × 9 | Self-play, with a notebook and a tournament harness for past versions | Placement in a tournament between all teams |

In both parts the network is fixed (`connectk/net.py`): every team trains the
same architecture with the same input encoding, and only hands in weights.
At test time the network plays greedily (the legal column with the highest
policy logit) with no search.

## The game

Two players drop stones into columns; a stone falls to the lowest empty
cell. The first player with k in a row (across, up or diagonal) wins. A full
board is a draw.

```sh
pip install -r requirements.txt
python3 -m connectk play --game connect5                  # you vs a random player
python3 -m connectk watch random random --game connect6   # write replay.html
python3 -m unittest discover tests                        # about 1 minute
```

`connectk/` holds the rules (`game.py`), the fixed network (`net.py`), the
strict referee and openings (`arena.py`), round robins with Elo
(`tournament.py`), the packed-bot loader (`packed.py`) and an HTML replay
viewer (`viewer.py`).

## Part A: Connect 6 against unseen bots

Students get four bots in a packed file. `bot_c` cheats, but only on a live
board: in the training environment a bot plays its own move on the board,
and `bot_c` sometimes drops two stones, puts a stone in mid-air, or turns an
opponent stone into its own. Against the strict referee it plays fair. So
its games teach an agent that good play loses. Teams must find it from data.

The shipped `ppo.py` runs but cannot learn: `reward_fn()` returns 0, and the
settings are poor (`gamma` 0.5, `lr` 3e-2, clip 0.9, 30 epochs, no
advantage normalization, no masking, first player only).

`data/offline_games.jsonl.gz` holds 5,472 games: house bots (strong, styles
from the held-out pool), the public bots, and `bot_c` games. 1,006 records
are bad: 665 `bot_c` games with rule breaks, plus planted poison (moves off
the board, on occupied or floating cells, out of turn, after the end, and
wrong winner labels). A replay under the rules finds all of them.

Grading: `hidden/graders/part_a/grade.py` plays each submission against the
held-out pool (12 rule-based bots and a trained network), on fixed openings
with both colors, and reports the mean score per bot.

## Part B: Connect 5 by self-play

Students get `selfplay.ipynb`: self-play PPO against a pool of past
versions (the baseline always picks the newest one, which is the flaw to
fix), a tournament harness that rates all saved versions with Elo, replays,
scrimmages against other teams' files, and the hand-in. Two packed gauge
bots measure progress.

The final: `hidden/graders/part_b/final_tournament.py` runs a round robin of
all teams plus fixed anchors (random, two rule-based bots, a trained
network), and ranks teams by points.

## Running the event

```sh
python3 hidden/make_handout.py --part a --zip ~/connect6-handout
python3 hidden/make_handout.py --part b --zip ~/connect5-handout
# ... teams work ...
python3 hidden/graders/part_a/grade.py submissions_a/*.pt
python3 hidden/graders/part_b/final_tournament.py --replay final.html submissions_b/*.pt
```

`hidden/README.md` lists everything in `hidden/`, including how to rebuild
the bots, the data, the notebook and the strong networks. Keep `hidden/` out
of the handouts (the handout maker does this).

## Layout

```
connectk/             game, fixed network, referee, tournaments, viewer, CLI
part_a_connect6/      Part A: problem, questions, env, PPO baseline, evaluate, bots, offline data
part_b_connect5/      Part B: problem, questions, self-play notebook, gauge bots
hidden/               held-out bots, graders, answers, reference solutions, strong nets, builders
tests/                rules, referee, network, bots, environment, data
demo/connect-replays.html  some recorded games (open in a browser)
```
