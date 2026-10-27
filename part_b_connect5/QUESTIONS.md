# Part B questions

Answer in your write-up. Give numbers where you can, and say how you
measured them.

## Self-play

1. The baseline always plays its newest version. Describe two ways this can
   fail. Did you see either in your tournament tables? Show the evidence.
2. What opponent pool did you use in the end (all past versions, a
   prioritized choice, fixed bots mixed in, something else)? Why?
3. In self-play the opponent changes while the learner trains. What does
   that do to the value function, and to PPO's advantage estimates?

## The tournament harness

4. Elo assumes that each player's strength is one number. When is that false
   in self-play? How do you see it in the score matrix?
5. Two greedy networks always play the same game from the same position. How
   does the harness still get useful statistics? What would go wrong without
   that?
6. How did you choose which version to hand in? Why is "the newest" not
   always right?
7. How many games does the harness need to rank two versions whose true
   scores against each other are 55% and 45%? Did you play that many?

## Making it play better

8. Your network plays without search in the final. If you used search in
   training, describe how its decisions got into the network's weights.
9. How did you use the mirror symmetry of the board?
10. How does the first player's advantage show up in your results? Is
    Connect 5 on 9 × 9 a win for the first player? Explain how you would try
    to find out.

## The final

11. In the final you meet other teams' networks. Is it better to be robust
    (do well against everyone) or to exploit the teams you expect to meet?
    What did you do?
12. What would you do with ten times more compute?
