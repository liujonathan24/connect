# Part B: Connect 5 by self-play, then a tournament

Train a neural network to play **Connect 5** on a **9 × 9** board by playing
against itself. At the end, all teams' networks meet in one tournament.
Your placement in that tournament is your result.

## The game

The rules are the rules of Connect 4. Two players take turns to drop a stone
into a column; it falls to the lowest empty cell. The first player with
**5 in a row** (across, up, or diagonal) wins. A full board is a draw.

```
python3 -m connectk play --game connect5
```

## What you get

| File | What it is |
|---|---|
| `selfplay.ipynb` | Start here. Self-play PPO, a tournament harness for your versions, replays, hand-in. |
| `connectk/` | The game, the fixed network, the referee, tournaments, a replay viewer. Do not change `connectk/net.py`. |
| `gauges.bin` | Two fixed bots, `gauge_easy` and `gauge_hard`, to measure progress against. |
| `QUESTIONS.md` | Questions to answer in your write-up. |

You get no data and no strong opponents. Everything your network learns, it
learns from its own games.

## The network is fixed

Everybody trains the same `ConnectNet` from `connectk/net.py`, with the same
input encoding. You choose the weights, nothing else. In the tournament, the
network plays **greedily**: the legal column with the highest policy logit.
There is no search at test time. In training you may use anything.

## The final tournament

* Hand in one file: `submission.pt` (`save_net(net, "submission.pt")`). It
  must load into `ConnectNet(CONNECT5)` with `strict=True`, or you are out.
* Every team plays every other team. Each pairing plays the same set of short
  random openings, each one twice (once with each color).
* Win 1, draw 0.5, loss 0.
* A few fixed anchor players also take part (a random player, some bots, a
  strong network of ours). They fix the Elo scale.
* **Placement** is by total points over all your games. Ties go to the
  better Elo, then to the head-to-head result.

Your network will meet styles it never saw in self-play. Before the final,
you may swap `.pt` files with other teams and play scrimmages (section 5 of
the notebook). A network that only beats its own past versions may not beat
anybody else.

## How to start

1. Run the notebook top to bottom (a few minutes on a laptop CPU).
2. Look at the tournament table: did later versions beat earlier ones? Did
   any version beat `gauge_hard`?
3. Read section 6 of the notebook for ideas, and the questions.
