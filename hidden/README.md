# hidden/: graders, answers, held-out bots, builders

Keep this folder away from teams. `make_handout.py` never copies it.

```
make_handout.py                 makes the clean folder for Part A or Part B
graders/part_a/grade.py         scores Part A submissions against the held-out bots
graders/part_b/final_tournament.py   the Part B final: all teams + anchors, round robin
bots/engine.py                  bot engine: heuristic, alpha-beta search, the cheater
bots/public.py                  the bots teams get (packed into the .bin files)
bots/heldout.py                 the Part A held-out pool and the Part B anchors
build_blobs.py                  packs bots/public.py into part_*/{opponents,gauges}.bin
build_notebook.py               writes part_b_connect5/selfplay.ipynb (edit cells there)
make_offline_data.py            writes the Part A offline games and data/offline_manifest.json
live.py                         games on a live board, recorded cell by cell (shows rule breaks)
data/offline_manifest.json      what is wrong with each offline game
strong/train_strong.py          trains the strong networks (imitation, then expert iteration)
strong/mcts.py                  batched AlphaZero-style MCTS, the teacher for expert iteration
strong/connect6_strong.pt       strong Connect 6 network (held-out bot h13_neural)
strong/connect5_strong.pt       strong Connect 5 network (final anchor anchor_neural)
solutions/part_a/ANSWERS.md     answers and grading guide, Part A
solutions/part_a/results.md     measured scores: baseline, reference, strong net
solutions/part_a/clean_data.py  replays the offline games and rejects bad ones
solutions/part_a/detect_cheater.py   finds bot_c from play alone
solutions/part_a/ppo_fixed.py   the PPO trainer with every problem fixed
solutions/part_b/ANSWERS.md     answers and grading guide, Part B
```

## Rebuild

Everything is deterministic. After a change to the bots, the data or the
notebook cells:

```sh
python3 hidden/build_blobs.py
python3 hidden/build_notebook.py
python3 hidden/make_offline_data.py          # ~8 minutes on 4 cores
python3 -m unittest discover tests           # checks that all of these agree
```

The strong networks take hours on a CPU. Imitation first, then expert
iteration with the MCTS teacher, from the imitation weights:

```sh
python3 hidden/strong/train_strong.py --game connect5 --out hidden/strong/runs/c5 --iters 0
python3 hidden/strong/train_strong.py --game connect5 --out hidden/strong/runs/c5m \
    --resume hidden/strong/runs/c5/bc.pt --seed_data hidden/strong/runs/c5/teacher.npz \
    --teacher mcts --sims 96 --games_per_iter 64 --steps_per_iter 60 --buffer 100000 --lr 5e-4
# the same for connect6, with --sims 64 --games_per_iter 48 --buffer 120000
```

Pick the final weights with a large match against the bots (the in-run
evals use only 8 games per bot), then copy them to
`hidden/strong/connect5_strong.pt` and `hidden/strong/connect6_strong.pt`.

## Check the graders

```sh
cd part_a_connect6 && python3 ppo.py --updates 5 --out runs/smoke && cd ..
python3 hidden/graders/part_a/grade.py part_a_connect6/runs/smoke/submission.pt hidden/strong/connect6_strong.pt
python3 hidden/graders/part_b/final_tournament.py --openings 4 hidden/strong/connect5_strong.pt part_b_connect5/checkpoints/v4.pt
```
