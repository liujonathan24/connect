# Part A: answers and grading guide

Numbers marked (measured) come from the scripts in this folder; see
`results.md` for the runs.

## Rewards

**1. Zero reward.** Every return and every advantage is 0 (GAE of all-zero
rewards gives advantage = -value at first, and the value head quickly fits
0). The policy gradient is then 0, so the policy only moves by noise from
the value loss through the shared body. Signs in the log: `pg` and `v` near
0 after a few updates, and win rates that do not change. With the shipped
learning rate the noise steps are large, so the entropy also falls to 0 in
a few updates (one column every time, often illegal moves). A good answer
reads this off the printed losses.

A follow-up that strong teams find: even with a correct ±1 reward, PPO from
scratch barely moves against these bots. The agent almost never wins, its
value head learns "always -1", and the advantages vanish (`results.md`).

**2. Terminal reward and shaping.** +1 win, -1 loss, 0 draw (illegal move:
-1, or mask it). Shaping for "4 in a row" or "blocked a threat" can change
the optimal policy: the agent may farm shaped reward (make many open 4s,
never finish) or prefer long games. Potential-based shaping,
F(s, s') = γΦ(s') − Φ(s), cannot change the optimal policy (Ng, Harada,
Russell 1999): the shaped terms telescope along any trajectory, so every
policy's return shifts by the same −Φ(s0). Φ can be any board evaluation
(for example, a count of open lines). Full marks: names the
potential-based form and why it is safe.

**3. Illegal moves.** A loss for illegal moves wastes samples (early on, a
large share of games end at once, column 0 fills, ...) and the agent must
learn the rules from the reward. Masking (logits of full columns set to
−inf) removes the problem. The mask must be applied to the logits both when
sampling in the rollout and when recomputing log-probs in the update, with
the same mask (store it with the observation). Otherwise the ratio
π_new(a|s)/π_old(a|s) compares two different distributions. Note: masking
also matches the grader, which only considers legal columns.

## The trainer

**4. Settings.** Expected list (the shipped value, then the problem):

| Setting | Shipped | Problem | Reasonable |
|---|---|---|---|
| reward | 0 | nothing to learn | ±1 at the end |
| `gamma` | 0.5 | a reward 30 agent moves away is scaled by 0.5^30 ≈ 1e-9; games last 20-80 agent moves | 0.99-1.0 |
| `gae_lambda` | 0.0 | one-step TD only; with a bad value function early on, no signal reaches early moves | 0.9-0.95 |
| `lr` | 3e-2 | far too high for Adam on a conv net; the policy collapses or diverges | 1e-4 to 1e-3 |
| `clip_eps` | 0.9 | ratios from 0.1 to 1.9 pass: almost no trust region | 0.1-0.3 |
| `epochs` | 30 | 30 passes on one small batch: overfits the batch, ratios run away | 3-10 |
| `minibatch` | 32 | noisy gradients, many steps per update with 30 epochs | 128-512 |
| `ent_coef` | 0 | early collapse to one column | 0.003-0.02 |
| `vf_coef` | 10 | the value loss dominates the shared body | 0.5-1 |
| `max_grad_norm` | 1e9 | no clipping; one bad batch can wreck the net | 0.5-1 |
| `normalize_adv` | False | advantage scale changes with the reward and value | True |
| `mask_illegal` | False | see 3 | True |
| `agent_side` | first | never learns to play second, but the grader uses both colors | random |
| `opponents` | all four | includes the cheater | drop bot_c |

Full marks: at least the reward, gamma, lr, clip/epochs, masking, sides and
the cheater, each with a reason. Bonus: noticing that 30 epochs with
minibatch 32 also make every update slow (about 1,000 gradient steps per
1,024 samples; about 50 s per update on one CPU core).

**5. Colors.** A first-player-only agent is much weaker as the second
player: it has never seen positions with an odd number of stones where it
must defend first. Measure with `evaluate.py` per color (or play_match with
`keep_games=True`). In our runs the first-player-only agent dropped by
roughly half its score as the second player (measured, see results.md).

## The bad bot

**6. Which bot.** `bot_c`. Evidence (any two, at least one from games):

* Rule check in the environment (`detect_cheater.py`, test 1). After the
  agent's move (`info["after_agent"]`) and after the bot's answer (the
  returned board), a fair answer adds exactly one stone of the bot's color,
  on the landing cell of its column, and changes nothing else. Over 60 games:
  bot_a, bot_b, bot_d: 0 breaks; bot_c: 56 stones off the landing cell, 34
  turns with more than one stone, 1 opponent stone changed (measured).
* Strict referee vs trusting environment (test 2). The same fair player
  (`bot_d`) scores 0.78 against bot_c under the strict referee but 0.12 in the
  environment (measured). Fair bots score about the same in both.
* The offline games: replay each game under the rules. 665 of 688 games with
  bot_c fail; about 13% of games of any other player fail (planted poison
  and their games against bot_c) (measured, `clean_data.py`).
* Floating stones visible in the final boards, an impossible stone count
  (bot has two more stones than the agent), win rates far above its
  strict-referee strength.

Win rates alone ("bot_c beats us most") are not enough: a bot can just be
strong. Grade down if that is the only evidence.

**7. What it does.** On a live board only, per turn:

* teleport: if it can complete 6 in a row at any empty cell, including one in
  mid-air, it puts the stone there (90% of the time);
* steal: if the opponent threatens an immediate win, it turns one of the
  opponent's stones on that line into its own (70%);
* extra: 4% of its turns, it drops a second stone.

Otherwise it plays a solid heuristic (lines, defense, one-reply lookahead).
Teams do not need these exact rates; a description of the three behaviors
with rough frequencies is full marks.

**8. Effect on training.** The agent gets −1 after positions where it played
well (it blocked a threat; the cheater then stole a stone or won in
mid-air). The value head learns that good defensive positions are lost, and
the policy gradient pushes down the moves that were right. The observed
transitions also do not follow the rules, so the value function learns
dynamics that do not exist in the test. The effect is worst exactly on the
skill that matters most against strong held-out bots: defense. Bonus: notice
that the cheater's games also add variance (advantages with random sign).

**9. Drop or repair.** Dropping is the simple, correct default: the three
fair bots and self-play give enough data. Repair is possible (the
environment could replace the cheating move with a referee check, or
episodes with a rule break could be masked out of the update: cut the
episode at the break and bootstrap from the value), but costs work for
little gain. Either answer is fine with a reason. "Keep it, it is a strong
opponent" without a fix is wrong.

## The offline games

**10. Cleaning.** Replay each game with `Board.play` and reject at the first
problem (`clean_data.py`). Counts (first problem per game): wrong_turn 606
(mostly bot_c's extra stones), floating 157 (bot_c teleports and planted),
occupied 80 (bot_c steals and planted), off_board 60, wrong_winner 60,
after_end 42; total 1,005 of 5,472. The manifest in `hidden/data` labels
every bad game; the reference checker catches all 1,005 with no false alarms.
Expect teams to find each kind; a team that only checks for off-board moves
misses most of the bad data. Give credit for checking the winner label by
replay.

**11. Using them.** Good uses: imitation learning (cross-entropy on the
house bots' moves) as a warm start before PPO; training the value head on
the game results; both with mirror images. What to imitate matters a lot:
in our runs, copying every fair player gave a held-out score of 0.20, and
copying only the moves of the side that won (and never bot_a, which plays
at random 20% of the time) gave 0.64 (`results.md`). Full marks need some
choice of this kind, with a reason. Pitfall: PPO's update assumes the
data came from the current policy (it uses π_old(a|s) in the ratio). Offline
moves come from other policies; you do not know their probabilities, and
the importance ratios would be wrong or undefined. So offline data goes into
a separate supervised step (or an off-policy method), not into the PPO
batch. Bonus: noting that imitation of the house bots biases the agent
toward the held-out styles, which helps the score.

## Generalization

**12. Unseen bots.** Credit for: training against several bots plus frozen
copies of itself (a league), randomized openings, both colors, imitation of
the house bots, mirror augmentation, early stopping on a validation set of
opponents (for example, train on two fair bots and validate on the third).
The held-out pool has styles not in the public bots (stacking, edge play,
alpha-beta search, a trained network); a team that overfits to bot_a/b/d
loses to those.

**13. Symmetry.** Reversing the columns maps a position to an equivalent
one, and column c to cols−1−c. Use it to double the data (augmentation), or
to average the policy over both images (not allowed at test time here,
since the net plays alone, but fine for training targets). The cost: none
in data; the network must still learn it (it is not built in).

**14. Games needed.** For a score near 0.6 the standard error per game is
about sqrt(0.6·0.4/n) (less with draws). To tell 0.60 from 0.65 (a gap of
0.05) with about two standard errors, you need 2·sqrt(0.24/n) ≤ 0.05/√2 for
a difference of two estimates, so n ≈ 770 games per agent; about 400 if one
side is a fixed known number. Full marks: any honest calculation of this
kind, and a statement of how many games were actually played.

## Grading guide

* Score (the main result): from `grade.py`. See `results.md` for reference
  points and score bands: the floor is about 0.13 (untrained), fixed PPO from
  scratch stays near it (sparse reward), winners-only imitation of the clean
  offline games reaches about 0.64, and naive PPO on top of it lowers it.
* Write-up: 14 questions. Core: 1, 3, 4, 6, 8, 10. A team that finds the
  cheater with game-level evidence and cleans the data correctly has done the
  heart of the problem, even with a modest score.
