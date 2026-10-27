# Part A questions

Answer in your write-up. Short answers are fine; give numbers where the
question asks for them, and say how you measured them.

## Rewards

1. As shipped, `reward_fn()` returns 0 for every move. What does PPO do with
   that reward? Look at the losses it prints and explain them.
2. Write a reward for a win, a loss and a draw. Then consider extra rewards
   for good moves (making 4 or 5 in a row, blocking a threat). Can such
   "shaping" change which policy is optimal? Give a form of shaping that
   cannot, and say why.
3. As shipped, an illegal move ends the game as a loss. Compare that with
   masking illegal columns in the policy. If you mask, where must the mask go
   so that the PPO ratio stays correct?

## The trainer

4. List each setting in `Config` that you changed. For each one, say what
   goes wrong with the shipped value. (Hint: how many moves does a game last,
   and what does `gamma = 0.5` do to a reward that comes 30 moves later?)
5. Your agent plays both colors in the test. How does a run that only trains
   as the first player do as the second player? Measure it.

## The bad bot

6. Which of the four bots is bad data? Give at least two independent pieces
   of evidence. At least one must come from the games themselves, not from
   win rates.
7. What exactly does that bot do? How often?
8. What happens to a policy and to a value function trained on games against
   it? Think about what the agent is taught about positions where it has, in
   fact, played well.
9. Is it better to drop that bot from training, or to keep it and repair the
   data? Defend your choice.

## The offline games

10. Clean the offline games. Which kinds of bad records did you find, and how
    many of each? Describe your checks.
11. Did the offline games help you? If you used them, how (imitation of the
    moves, training the value head, something else)? PPO is on-policy: what
    goes wrong if you put these games straight into the PPO update?

## Generalization

12. The grader uses bots you have never played. What did you do so that your
    agent does not just learn to beat the three bots you trained against?
13. The board looks the same in a mirror. How can you use that symmetry in
    training? What does it cost?
14. How many games do you need to tell a true score of 0.60 from 0.65 with
    some confidence? How many did you play for your final evaluation?
