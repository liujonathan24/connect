# Part A: Connect 6 against bots you have not met

Train a neural network to play **Connect 6** on a **13 × 13** board. We grade
it against a set of opponent bots that you never see.

## The game

The rules are the rules of Connect 4. Two players take turns to drop a stone
into a column. The stone falls to the lowest empty cell. The first player with
**6 in a row** (across, up, or diagonal) wins. A full board is a draw.

```
python3 -m connectk play --game connect6          # play it in the terminal
```

## What you get

| File | What it is |
|---|---|
| `connectk/` | The game, the fixed network, the referee, tournaments, a replay viewer. Do not change `connectk/net.py`. |
| `env.py` | A training environment: your agent against the four bots. |
| `opponents.bin` | Four opponent bots: `bot_a`, `bot_b`, `bot_c`, `bot_d`. |
| `ppo.py` | A PPO trainer. It runs, but it does not learn (see below). |
| `evaluate.py` | Plays your network against the four bots, like the grader does. |
| `data/offline_games.jsonl.gz` | 5,472 recorded games between bots. Optional. |
| `QUESTIONS.md` | Questions to answer in your write-up. |

## The network is fixed

Everybody trains the same network: `ConnectNet` in `connectk/net.py` (six 3×3
convolutions with 64 channels, a policy head with one logit per column, and a
value head), with the input encoding in `encode()`. You choose the weights.
Nothing else.

The grader plays your network **greedily**: the legal column with the highest
policy logit. There is no search and no randomness at test time. So
everything your agent knows must be in the weights. (You may use search,
other networks, or anything else **during training**.)

## The four bots

`env.load_opponents()` gives you four bots. They are black boxes: play them,
record them, measure them. Do not unpack `opponents.bin` or read its code;
treat the bots as opponents on a remote server. That is a rule of the event.

**One of the four bots is bad data.** Find out which one, show how you know,
and decide what to do about it.

## The trainer

`python3 ppo.py` trains with PPO against all four bots. As shipped:

* `reward_fn()` returns 0 for every move. Your agent can learn nothing until
  you write it.
* Many settings in `Config` are poor choices.
* It trains on all four bots, bad one included.

Change anything you want in `ppo.py`, or write your own trainer.

## The offline games

`data/offline_games.jsonl.gz` has one game per line:

```json
{"id": 17, "p1": "bot_b", "p2": "house_3", "moves": [[1, 0, 6], [2, 1, 6], ...], "winner": 1}
```

Each move is `[player, row, col]` (row 0 is the bottom). `winner` is 1, 2, or
0 for a draw. The `house_*` players are bots that we use internally. You may
use these games in any way, or not at all. **Not every record is correct.**

## Scoring

Submit one file, `submission.pt`, made with:

```python
from connectk.net import save_net
save_net(net, "submission.pt")
```

We check that it loads into `ConnectNet(CONNECT6)` with `strict=True`. Then it
plays our held-out bots: every bot plays the same set of short random
openings, each opening twice (once with each color).

* Win 1, draw 0.5, loss 0.
* Your score is the **mean over the held-out bots** of your score against
  each bot. Each bot weighs the same.
* The held-out bots are a mix of weak, medium and strong players. Some play
  in ways that none of the four public bots do.

Also hand in short answers to `QUESTIONS.md`.

## Hints

* Check that `evaluate.py` gives a mean score near 0 for an untrained network
  before you trust any number from training.
* A greedy network that never blocks an immediate threat loses to every
  competent bot, however well it attacks.
* The second player has a real disadvantage in this game. Train with both
  colors.
