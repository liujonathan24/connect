"""The bots students get (packed into a blob by hidden/build_blobs.py).

Part A (connect6): four opponents. bot_c cheats on a live board.
Part B (connect5): two honest gauges to measure progress against.
"""

from .engine import CheaterBot, HeuristicBot


def _s(seed, i):
    return None if seed is None else seed * 7919 + i


def connect6_public(seed=None):
    return {
        # weak: random moves now and then, misses blocks
        "bot_a": HeuristicBot("bot_a", _s(seed, 1), epsilon=0.2, p_block=0.6, temperature=0.5),
        # attacker: builds its own lines, defends little
        "bot_b": HeuristicBot("bot_b", _s(seed, 2), attack=[0, 1, 6, 36, 216, 1296, 7776],
                              defense=[0, 0.3, 1.5, 8, 40, 200, 0]),
        # balanced and solid; cheats when it gets the live board
        "bot_c": CheaterBot("bot_c", _s(seed, 3), attack=[0, 1, 4, 16, 64, 256, 1024],
                            defense=[0, 1, 4, 16, 64, 256, 0], lookahead=0.3, temperature=0.2),
        # defender: spoils the opponent's lines and looks one reply ahead
        "bot_d": HeuristicBot("bot_d", _s(seed, 4), attack=[0, 1, 3, 9, 27, 81, 243],
                              defense=[0, 2, 8, 32, 128, 512, 0], lookahead=0.5),
    }


def connect5_gauges(seed=None):
    return {
        "gauge_easy": HeuristicBot("gauge_easy", _s(seed, 11), epsilon=0.25, p_block=0.7, temperature=0.5),
        "gauge_hard": HeuristicBot("gauge_hard", _s(seed, 12), attack=[0, 1, 4, 16, 64, 256],
                                   defense=[0, 1, 4, 16, 64, 0], lookahead=0.5),
    }
