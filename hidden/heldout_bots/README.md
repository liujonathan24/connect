# Held-out bots in the submission format

Bots in the submission format (a folder with `agent.py`), used as held-out
opponents. `team_az` is a test-solve of the contest by another agent (not a
human team). The graders load them without the move timer.

| Folder | Where it plays |
|---|---|
| `team_az/submission_a` | Part a held-out bot `n_team_az`: a 0.5M-parameter residual net with its own MCTS, 0.75 s per move. The hardest bot in the pool (it scored 0.94 on the pool before it joined). |
| `team_az/submission_b` | Part b reference `ref_team_az` (not counted in S_b): the same design for Connect 6. |

`team_az/NOTES.md` is that agent's own description of how it trained the bots.
