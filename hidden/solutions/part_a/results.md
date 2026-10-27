# Part A: measured reference scores

All scores come from `hidden/graders/part_a/grade.py --openings 8` (16 games
per held-out bot, both colors; win 1, draw 0.5). The held-out pool has 14
bots: h01–h12 (rule-based) and h13/h14 (our trained network with and without
a search). Runs on a 4-core CPU, no GPU.

| Submission | How it was made | Score |
|---|---|---|
| untrained | `ConnectNet(CONNECT6)`, random weights | 0.138 |
| shipped baseline | `ppo.py` exactly as shipped, 50 updates (~45 min) | 0.141 |
| reference v1, imitation | `ppo_fixed.py`: imitation of every fair player in the clean offline games, 3,000 steps | 0.234 |
| reference v1, + PPO | the same, then 150 PPO updates with fixed settings | 0.176 |
| **reference v2, imitation** | `ppo_fixed.py --bc_steps 8000`: winners' moves only, no bot_a, 8,000 steps (~30 min) | **0.654** |
| reference v2, + PPO | the same, then 50 / 100 / 200 PPO updates | 0.473 / 0.433 / 0.507 |
| our strong net (greedy) | `hidden/strong/connect6_strong.pt` at the time | 0.397 |

Per bot, reference v2 imitation: random 1.00, sloppy 1.00, novice 0.81,
stacker 0.78, edge 0.75, rusher 0.88, balanced 0.78, turtle 0.28, planner
0.25, search 0.44, search_def 0.53, mixer 0.78, h13 (net + search) 0.38,
h14 (net) 0.50.

## What these numbers say

* **The floor is about 0.14.** An untrained net still beats the random bot
  and the sloppy bot sometimes. The shipped baseline does not move off the
  floor: with a zero reward it cannot learn.
* **Fixed PPO from scratch stays near the floor.** The agent loses almost
  every game against even the weakest public bot, its value head learns
  "always -1", and the advantages are near 0: a sparse-reward problem. Teams
  need a way out: an imitation start, reward shaping, or a curriculum
  (self-play or weak opponents first).
* **The offline data is the strongest lever,** if it is cleaned and used
  well. Copying every player gives 0.23; copying only the winners (and never
  the weak bot_a) gives 0.65. This rewards the intended skills: finding the
  bad data, and thinking about what to imitate.
* **Naive PPO on top of imitation lowers the score** (0.65 -> 0.43–0.51 here).
  The value head starts untrained, so the first advantages are noise, and a
  learning rate that suits training from scratch is too high for fine-tuning.
  Teams that keep improving after imitation do something about it (a lower
  learning rate, value-head warm-up, a KL term to the imitation policy, or
  early stopping on a validation set of opponents).

## Score bands for grading

| Score | Reading |
|---|---|
| below 0.20 | no real learning (floor 0.14) |
| 0.20–0.40 | partial: something learns, but weak against medium bots |
| 0.40–0.60 | good: beats the weak and medium bots most of the time |
| above 0.60 | excellent: our reference imitation level or better |
