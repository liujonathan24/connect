# Connect 5 and Connect 6

Hello! This problem walks you through training a small neural network to play
Connect 5 and Connect 6. In Part a, you train a neural network to defeat a
series of bots. Part b introduces self-play: your network learns by playing
against itself, and then it plays the other teams' networks.

## The game

The rules are the rules of Connect 4 on a bigger board. Two players take turns
to drop a stone into a column; the stone falls to the lowest empty cell. The
first player with k stones in a row (across, up, or diagonal) wins. A full
board is a draw.

| Game | Board | To win | Part |
|---|---|---|---|
| Connect 5 | 9 columns × 9 rows | 5 in a row | a |
| Connect 6 | 13 columns × 13 rows | 6 in a row | b |

## What you get

| File | What it is |
|---|---|
| `connect5.zip` | The code: the game (`connectk/`), the bots (`bots.bin`), this text, `requirements.txt` |
| `notebook_a.ipynb` | Part a: a network template, PPO training against the bots, evaluation, replays, hand-in |
| `notebook_b.ipynb` | Part b: self-play training, a tournament harness for your versions, scrimmages, hand-in |

**Setup.** Put `connect5.zip` next to the notebooks; the first cell of each
notebook unpacks it. On your own machine: `pip install -r requirements.txt
jupyter`. On Google Colab: upload `connect5.zip` together with the notebook,
and choose a GPU runtime if you have one.

Useful commands (after unpacking):

```
python3 -m connectk play --game connect5                          # play the game in a terminal
python3 -m connectk check submission_a --game connect5            # check a submission like the grader
python3 -m connectk check submission_b --game connect6
```

## Part a (5T points)

In `connect5.zip` and `notebook_a.ipynb`, we provide template code that trains
an initial neural network and plays it against four Connect 5 bots: `random`,
`easy`, `medium` and `hard`. Change the training code and the algorithm to
make the best Connect 5 bot you can. The model must stay under 200 million
parameters.

**Grading.** Your bot plays a set of **held-out bots** that you do not get:
about a dozen, with many styles, and several much stronger than `hard`. Every
held-out bot plays the same short random openings, each opening twice (once
with each color). A win counts 1, a draw 0.5, a loss 0. Your score `S_a` is
your mean score over all these games, from 0 to 1.

**Points:** 5T × `S_a`.

Resources:

* [Reinforcement learning fundamentals](https://medium.com/@812jugalgajjar/reinforcement-learning-fundamentals-understanding-the-framework-42963a7a5b40)

## Part b (up to 10T points)

`notebook_b.ipynb` holds starter code for self-play, this time for Connect 6.
You may change the code in any way. Your bot is graded by how well it plays
against the other teams' bots.

**Grading.** A round robin: every team's bot plays every other team's bot.
Each pairing plays the same short random openings, each opening twice (once
with each color). A win counts 1, a draw 0.5, a loss 0. Your score `S_b` is
your mean score over all your games, from 0 to 1.

**Points:** 10T × `S_b`.

You get no bots to learn from in this part, only two gauge bots (`easy6` and
`hard6`) to measure your progress. Before the final, you may swap submission
folders with other teams and play scrimmages (section 4 of the notebook).

Resources:

* [Self-play (Hugging Face Deep RL course)](https://huggingface.co/learn/deep-rl-course/en/unit7/self-play)

## Rules for every bot you hand in

1. **Format.** A submission is a folder with an `agent.py` that defines
   `make_agent(folder)`. It returns an object with:
   * `net`: a `torch.nn.Module` that holds **all** the parameters your bot uses;
   * `choose(board)`: returns the column to play (`board` is a `connectk.Board`).

   The folder may hold more files (weights, `model.py`, a config). The hand-in
   cell of each notebook writes a working folder for you.
2. **Size.** At most **200 million parameters** in `agent.net`, and a folder of
   at most 1 GB.
3. **Your network decides.** The moves must come from your network. A search
   on top of it (for example MCTS that calls your network) is allowed. A
   hand-written game engine or evaluation that picks the moves is not. Do not
   unpack or copy the bots in `bots.bin`.
4. **Time.** We play on a CPU (one core, no GPU). Aim for at most **1 second
   per move**. A move that takes more than **2 seconds loses that game**.
5. **Packages.** Your bot may use Python 3.10+, `numpy` and `torch` only, and no
   network access.
6. **Fair play.** An illegal move, or a crash, loses that game. Check your
   folder with `python3 -m connectk check <folder> --game <game>` before you
   hand it in.

## Hand in

* `submission_a/` (Connect 5) and `submission_b/` (Connect 6), as one zip.
* Optional: a short note on what you tried and what worked.

## Hints

* Against strong bots, a new network almost never wins, so it gets almost no
  learning signal. Start against weak opponents and make them harder as you
  improve.
* Learning from recorded games works: play many games between bots and train
  your network to copy the moves of the side that **won**.
* A bigger network is not free: you have 1 second per move on a CPU.
* Measure with many games: a score from 10 games is mostly noise.
