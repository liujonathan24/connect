"""Games on a live board, recorded by looking at what changed.

The training environment of Part A lets bots play their own move on the live
board (bot.play(board)). This module does the same for two bots and records
every changed cell, so a rule break shows up in the record:

    drop      a stone on the landing cell of its column (a fair move)
    teleport  a stone on an empty cell that is not a landing cell
    steal     a stone where the opponent had one
    extra     a second new stone in the same turn
"""

import numpy as np

from connectk.game import Board


def diff_moves(before, after, mover):
    """List of [player, row, col, kind] for the cells that changed in one turn."""
    out = []
    heights = before.heights
    changed = np.argwhere(before.grid != after.grid)
    # fair drops first, so a second stone in the same turn is the "extra" one
    changed = sorted(changed.tolist(), key=lambda rc: (before.grid[rc[0], rc[1]] != 0, rc[0] != heights[rc[1]]))
    new_stones = 0
    for r, c in changed:
        was, now = int(before.grid[r, c]), int(after.grid[r, c])
        if now == 0:
            kind = "remove"
        elif was != 0:
            kind = "steal"
        elif new_stones >= 1:
            kind = "extra"
        elif r == heights[c]:
            kind = "drop"
        else:
            kind = "teleport"
        new_stones += was == 0 and now != 0
        out.append([now or mover, r, c, kind])
    return out


def play_live(first, second, spec, opening=(), max_turns=None):
    """Both bots play on the live board. Returns (moves, winner, board)."""
    board = Board(spec)
    moves = []
    for c in opening:
        p = board.player
        r = board.play(c)
        moves.append([p, r, c, "drop"])
    bots = {1: first, 2: second}
    turns = 0
    while not board.done and (max_turns is None or turns < max_turns):
        mover = board.player
        before = board.copy()
        bots[mover].play(board)
        board.resync(player=3 - mover)
        moves.extend(diff_moves(before, board, mover))
        turns += 1
    return moves, int(board.winner), board
