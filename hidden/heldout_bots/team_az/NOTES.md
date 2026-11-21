# Notes: what we tried and what worked

## Summary

Both bots use the same design: a small residual network and an AlphaZero-style
MCTS (PUCT) at play time. The network gives the move priors and the position
values. The search uses only the game rules. A move stops at a time budget of
0.75 s on one core (the slowest move we measured in full games was 0.76 s).

## Network (`azcore.py`)

* Base input: 5 planes from the view of the side to move: my stones, the
  opponent's stones, the landing cell of each column, all ones (to see the
  edges), and "the side to move is the first player" (parity matters in
  gravity games).
* Part b adds 5 input planes computed from the rules (like the template's
  landing cells): cells that complete k in a row now (for me / for the
  opponent), cells on lines that need two more stones (me / opponent), and
  the row parity (odd/even rows, for zugzwang). The network still decides.
* Body: residual blocks (two 3x3 convolutions with batch norm).
  Part a: 64 channels, 6 blocks (498,632 parameters).
  Part b: 48 channels, 5 blocks (224,084 parameters).
* Policy head: one logit per cell; the logit of a column is read at its
  landing cell (fully convolutional).
* Value head: win/draw/loss logits. Part a: a flat linear head. Part b: global
  mean and max pooling (a flat head memorized the results of the games).
* At load time, the batch norms are folded into the convolutions (5% faster).

## Search

* PUCT with c = 1.8, first-play urgency = parent value - 0.25, leaf batches
  of 8 with virtual loss, and tree reuse between moves.
* Exact one-ply tactics from the rules: if the side to move can win now, only
  the winning moves are searched (proven win); if the opponent has two or
  more immediate wins, the node is a proven loss; if the opponent has one,
  the block is the only move searched. Proven nodes need no network call.
  This fixed blunders where a young net did not see a simple open line.
* MCTS-solver at play time: proven results move up the tree (a node is won if
  one move leads to a proven loss of the opponent; it is lost or drawn when
  all its moves are proven). We checked it against brute-force negamax on 60
  random endgames: every proven value was exact, and every chosen move kept
  the exact value.

## Training

Part a (Connect 5):
1. Supervised pretraining on 12,000 recorded games between the given bots
   (random openings, 3% random moves): policy = the moves of the side that won
   (and of `hard` in drawn games), value = the result. Greedy net: 0.33 vs
   `hard`; with MCTS (200 simulations): 0.68 vs `hard`.
2. AlphaZero self-play from that network (720 iterations, about 6 hours): 75%
   self-play, 25% games against `hard`/`medium` (only our own moves are
   training targets). With MCTS (200 simulations), iteration 120 scored 0.95
   vs `hard` and 0.975 vs `medium`; later iterations kept beating earlier ones
   head-to-head (300 vs 180: 0.69; 400 vs 300: 0.54).

Part b (Connect 6): AlphaZero self-play from zero (about 270 iterations over
three runs; the final run with the rule-feature planes had 169), with 20% of the games
against the gauges in the pool (as the notebook suggests). Only our own
search moves are training targets; nothing is copied from the gauges.
The last two learning-rate drops gave a clear gain in the last hour.

Self-play details: playout-cap randomization (25% of the moves get a full
search of 128-160 simulations and give a policy target; the rest get a fast
search of 32-40), Dirichlet noise at the root of the full searches,
temperature for the first 8-10 moves, random openings of 0-4 plies from the
middle columns (like the grader), mirror augmentation. Value target =
0.5 * game result + 0.5 * root value of the search.

Compute: one machine with 4 CPU cores and no GPU, about 6 hours of training.

## Final results (the hand-in folders, loaded like the grader, 0.75 s per move)

| Bot | Opponent | Wins / draws / losses | Score |
|---|---|---|---|
| submission_a | hard | 19 / 0 / 1 | 0.95 |
| submission_a | medium | 20 / 0 / 0 | 1.00 |
| submission_b | hard6 | 7 / 2 / 3 | 0.67 |
| submission_b | easy6 | 12 / 0 / 0 | 1.00 |

No illegal moves and no errors; the slowest move was 0.84 s (limit 2 s).
`python3 -m connectk check` passes for both folders.

## What did not work / lessons

* In part b the value head overfit badly at first (92% accuracy on games it
  trained on, 41% on new games): all positions of a game share one label, and
  most results are decided late. Pooling, mixing in the search value, more
  weight decay and, most of all, the rule-feature planes fixed it: near the
  end of the games, the correlation of the value with the result rose from
  about 0.1 to 0.3-0.4.
* Without the exact one-ply tactics, the young part-b net lost to `easy6`
  by missing open lines on the bottom row.
* `hard6` wins many endgames by zugzwang (threat parity). This is the main
  limit of our part-b bot.
