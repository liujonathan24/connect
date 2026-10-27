# Strong networks: what worked, what did not

The hard bots are our trained networks with an MCTS on top
(`players.py`, `NeuralMCTS`). They play in the Part A held-out pool
(h13_neural with 100 simulations, h14_net greedy) and as Part B anchors
(anchor_neural with 200 simulations, anchor_net greedy).

All numbers: score against the strongest rule-based bots (win 1, draw 0.5),
measured with `pick_best.py` (40–60 games per bot). Connect 5 panel:
gauge_hard, anchor_search, anchor_heuristic. Connect 6 panel: bot_d, h07,
h08, h09, h10. 4 CPU cores, no GPU.

## Final

| | Connect 5 | Connect 6 |
|---|---|---|
| net, greedy | 0.55 | 0.56 |
| net + MCTS (200 / 100 simulations) | 0.75 | 0.75 |

## The recipe that worked: winners-only imitation

1. Make games between strong bots and keep only the moves of the side that
   won (both sides in a draw), never the moves of weak bots
   (`make_winner_data.py`; Connect 6: 20,000 house-bot games, 1.9M
   positions; Connect 5: 40,000 games, 1.1M positions). For Connect 6 also
   add the clean winners' positions of the Part A offline games.
2. Train the fixed network on them (`train_imitation.py`, cross-entropy to
   the move, MSE to the result, mirror images, cosine learning rate;
   15,000–30,000 steps).
3. Pick the step with `pick_best.py`, copy it to `connectN_strong.pt`.
4. At play time, put the MCTS on top (`players.py`).

```sh
python3 hidden/strong/make_winner_data.py --game connect6 --games 20000 --out hidden/strong/runs/winners6.npz
python3 hidden/strong/train_imitation.py --game connect6 --data hidden/strong/runs/winners6.npz \
    hidden/strong/runs/offline6.npz --init <a first imitation net> --steps 30000 --lr 5e-4 --out hidden/strong/runs/c6_imit.pt
python3 hidden/strong/pick_best.py --game connect6 --openings 20 <checkpoints>
```

(`offline6.npz` is `offline_positions()` of `solutions/part_a/ppo_fixed.py`
saved as g/t/z; the first imitation net was `ppo_fixed.py --bc_steps 8000`.)

## What did not work (kept for reference)

| Method | Script | Result (greedy) |
|---|---|---|
| imitation of all teacher moves (winners and losers) | `train_strong.py` imitation phase | C5 0.31, C6 0.25 |
| expert iteration with a 1-move lookahead teacher | `train_strong.py --teacher lookahead` | C5 0.29–0.46, flat |
| AlphaZero-style MCTS self-play | `train_strong.py --teacher mcts` | C5 0.35 -> 0.43, then flat; C6 went down |
| MCTS labels on the net's own games (DAgger), sharpened, with tactical penalties | `distill.py`, `make_tactics.py` | C5 0.33–0.44; tactical losses fell, other losses rose |
| PPO against the bots from a trained net | `ppo_strong.py` | C5 0.28–0.31 |
| imitation of the C alpha-beta engine | `ab_distill.py`, `engine/abengine.c` | C5 0.10–0.18 (worse) |

Lessons:

* The MCTS player was always much stronger than the greedy net it was built
  on (0.72 vs 0.44). Distilling a search into a greedy policy needs far more
  data than this CPU could make in hours.
* Most losses of the early greedy nets were two-move tactical blunders. The
  network can learn those tactics (`make_tactics.py` data took its blunder
  rate on unseen critical positions from 27% to 6%), but that alone did not
  raise its score.
* The engine is correct (0 mismatches against an exact solver on 150
  Connect 4 endgames) and as strong as the MCTS teacher, but its moves come
  from deep calculation with a crude evaluation; a net that copies them
  without the calculation plays worse.
* What finally worked was plain imitation with the right filter: only the
  moves of players who went on to win. Each step of the strong bots'
  play is a pattern the net can learn, and the filter removes the moves
  that lost.
