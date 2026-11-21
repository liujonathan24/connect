# Connect 5 / Connect 6: an RL hackathon problem

Teams train neural networks to play Connect-k with gravity: the rules of
Connect 4 on a bigger board, with a longer line to win.

## The contest version (`contest/`)

The problem as written in `contest/PROBLEM.md`:

| Part | Game | Task | Points |
|---|---|---|---|
| a | Connect 5, 9 × 9 | Train a network (any design, under 200M parameters) to beat bots; four bots given | 5T × score against held-out bots |
| b | Connect 6, 13 × 13 | Self-play; play the other teams | 10T × score in a round robin between teams |

A submission is a folder with `agent.py` (`make_agent(folder)` returns an
object with `.net` and `.choose(board)`); 1 second per move on one CPU core.

```sh
python3 hidden/make_contest.py ~/contest-handout      # connect5.zip, notebook_a.ipynb, notebook_b.ipynb, PROBLEM.md
# ... teams work, then hand in submission_a/ and submission_b/ ...
python3 hidden/graders/contest/grade_a.py --workers 32 teams/*/submission_a
python3 hidden/graders/contest/tournament_b.py --workers 32 --reference teams/*/submission_b
```

Our trained networks are held out completely: they play only inside the
graders (Part a's hardest held-out bots, Part b's non-counting reference),
and the handout builder refuses to package them. Submissions are code: run
the graders in a sandbox (a container without network access).

Reference points (Part a, `grade_a.py`): the notebook template after its
2-minute default run scores 0.055; our standard net, greedy, scores 0.601.

## The earlier version (`part_a_connect6/`, `part_b_connect5/`)

Also complete: Part A plays Connect 6 against four given bots, one of which
cheats, with a broken PPO baseline and poisoned offline data; Part B is Connect
5 self-play. Every team uses one fixed network. Details below.

## The game

Two players drop stones into columns; a stone falls to the lowest empty
cell. The first player with k in a row (across, up or diagonal) wins. A full
board is a draw.

```sh
pip install -r requirements.txt
python3 -m connectk play --game connect5                  # you vs a random player
python3 -m connectk watch random random --game connect6   # write replay.html
python3 -m unittest discover tests                        # about 1 minute
```

`connectk/` holds the rules (`game.py`), the fixed network (`net.py`), the
strict referee and openings (`arena.py`), round robins with Elo
(`tournament.py`), the packed-bot loader (`packed.py`) and an HTML replay
viewer (`viewer.py`).

## Part A: Connect 6 against unseen bots

Students get four bots in a packed file. `bot_c` cheats, but only on a live
board: in the training environment a bot plays its own move on the board,
and `bot_c` sometimes drops two stones, puts a stone in mid-air, or turns an
opponent stone into its own. Against the strict referee it plays fair. So
its games teach an agent that good play loses. Teams must find it from data.

The shipped `ppo.py` runs but cannot learn: `reward_fn()` returns 0, and the
settings are poor (`gamma` 0.5, `lr` 3e-2, clip 0.9, 30 epochs, no
advantage normalization, no masking, first player only).

`data/offline_games.jsonl.gz` holds 5,472 games: house bots (strong, styles
from the held-out pool), the public bots, and `bot_c` games. 1,005 records
are bad: 665 `bot_c` games with rule breaks, plus planted poison (moves off
the board, on occupied or floating cells, out of turn, after the end, and
wrong winner labels). A replay under the rules finds all of them.

Grading: `hidden/graders/part_a/grade.py` plays each submission against the
held-out pool, on fixed openings with both colors, and reports the mean score
per bot. The pool: 12 rule-based bots of many styles, and our strong network
twice: alone (greedy, like a submission) and with an MCTS on top (the hard
bot; it scores 0.75 against our five strongest rule-based bots). Reference
scores are in `hidden/solutions/part_a/results.md`: untrained 0.13, shipped
baseline 0.14, winners-only imitation of the clean offline games 0.64.

## Part B: Connect 5 by self-play

Students get `selfplay.ipynb`: self-play PPO against a pool of past
versions (the baseline always picks the newest one, which is the flaw to
fix), a tournament harness that rates all saved versions with Elo, replays,
scrimmages against other teams' files, and the hand-in. Two packed gauge
bots measure progress.

The final: `hidden/graders/part_b/final_tournament.py` runs a round robin of
all teams plus fixed anchors (random, two rule-based bots, our strong network
alone and with an MCTS on top), and ranks teams by points. The strong network
alone scores 0.55 against our three strongest Connect 5 bots; with the search
on top, 0.75.

You can play our strong networks in a browser (`demo/play-connect.html`), and
teams can do the same with their own: `python3 -m connectk webplay my.pt`.

## Running the event

```sh
python3 hidden/make_handout.py --part a --zip ~/connect6-handout
python3 hidden/make_handout.py --part b --zip ~/connect5-handout
# ... teams work ...
python3 hidden/graders/part_a/grade.py submissions_a/*.pt
python3 hidden/graders/part_b/final_tournament.py --replay final.html submissions_b/*.pt
```

`hidden/README.md` lists everything in `hidden/`, including how to rebuild
the bots, the data, the notebook and the strong networks. Keep `hidden/` out
of the handouts (the handout maker does this).

## Layout

```
connectk/             game, fixed network, referee, tournaments, viewer, CLI
part_a_connect6/      Part A: problem, questions, env, PPO baseline, evaluate, bots, offline data
part_b_connect5/      Part B: problem, questions, self-play notebook, gauge bots
hidden/               held-out bots, graders, answers, reference solutions, strong nets, builders
tests/                rules, referee, network, bots, environment, data
demo/connect-replays.html  some recorded games (open in a browser)
demo/play-connect.html     play against our strong nets in a browser
```
