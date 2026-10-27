# Part B: answers and grading guide

## Self-play

**1. Naive self-play (newest version only).** Failure modes:

* Cycling: version n+1 learns to beat version n, forgets how to beat n−1,
  and the population goes around in circles (rock, paper, scissors). Seen as
  a score matrix that is not "upper triangular": later versions lose to some
  earlier ones.
* Forgetting / narrow play: the learner only sees the newest opponent's
  style; skills that the newest version no longer needs (for example,
  defending a rare plan) fade. Seen as a later version scoring worse than an
  earlier one against `gauge_hard` or `random`.
* Collapse: both sides settle into the same few openings; the policy
  becomes near-deterministic and stops exploring.

Full marks: two modes and evidence from their own tournament tables (a
version that loses to an older one, Elo that goes down, a gauge score that
goes down).

**2. Pool.** Good answers: uniform over all past versions (fictitious
self-play), prioritized fictitious self-play (sample versions in proportion to
how often they beat the learner, for example weight (1 − p_win)²), a mix of
newest and past versions, fixed bots in the pool. Any choice with a reason
and a measurement is fine.

**3. Non-stationarity.** The opponent is part of the environment, so the
environment changes when the pool changes. The value function's targets
shift (a position that was won against v3 may be lost against v7), so value
estimates lag, which biases GAE advantages. PPO also assumes the data comes
from one fixed environment per batch. Remedies: add versions slowly, sample
from a big pool (the mixture changes slowly), larger batches.

## The tournament harness

**4. Elo and transitivity.** Elo (Bradley-Terry) assumes P(A beats B)
depends only on r_A − r_B, so A > B and B > C imply A > C. Self-play makes
cycles; then Elo averages them away and can rank a version high that loses
to the version you will meet. In the score matrix, look for a cycle: A beats
B, B beats C, C beats A.

**5. Openings.** Greedy nets are deterministic: the same pair from the same
position gives the same game every time, so 100 games would be 1 game
repeated. The harness starts games from short random openings, and plays
each opening with both colors, so the first-player advantage cancels.
Without it, a match is one game (or two), and the first-player advantage
decides it.

**6. Which version.** The newest is not always the strongest (cycling,
forgetting, noise in the last updates). Choose by a tournament against
many opponents (past versions, gauges, other teams' files), not only the
last one. A strong answer looks at the score against a wide field, not
only at Elo.

**7. Games needed.** For a true score of 0.55 the standard error after n
games is about sqrt(0.55·0.45/n) ≈ 0.5/sqrt(n). To be about 2 standard errors
from 0.5: 0.05 ≥ 2·0.5/sqrt(n), so n ≥ 400 games for that one pairing.

## Making it play better

**8. Search in training.** Expert iteration: at each move, run a search
(a one-move lookahead with the value head, or MCTS), play its choice, and
train the policy head to predict the search's choice (cross-entropy toward
the search's visit counts or its best move), and the value head toward the
game result. The network then plays the improved policy without search.
`hidden/strong/train_strong.py` does this with a one-move lookahead
(immediate win, immediate loss, value of each reply).

**9. Symmetry.** Mirror the columns: the same position, with column c
mapped to cols−1−c, and the same value. Use it to double the data, or to
average targets.

**10. First player.** With openings played with both colors, the first
player's advantage shows as a difference between a team's score as player
1 and as player 2. For strong players, Connect 4 (7 × 6) is a first-player
win; Connect 5 on 9 × 9 has no published solution we know of. To find out:
a solver (alpha-beta with a transposition table, threat-space search) on the
real game, or evidence from strong self-play (first-player win rate rising
as play gets stronger). An answer that says "unknown, here is how I would
check, here is what my data suggests" is full marks.

## The final

**11. Robust or exploit.** In a round robin against many teams, robust play
usually wins: exploiting one team's weakness gains a few games but can
open weaknesses against others. Exploitation can pay when you know the
field (scrimmages), but you face every team. Credit for any answer backed by
scrimmage data.

**12. Ten times more compute.** For example: MCTS-based expert iteration with
more simulations, larger and more diverse pools, longer runs. Credit for
concrete, ranked plans.

## Grading guide

* Placement in the final (`final_tournament.py`) is the result.
* Write-up: 12 questions; core: 1, 4, 5, 6, 8.
* `results.md` (part A folder) has reference strengths for the gauges, the
  anchors and our strong network.
