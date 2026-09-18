"""
Deterministic, counter-based randomness for the search.

Contract shared by MCTS.py and every game logic:
  random_seed == 0  -> TRUE randomness (np.random), used for the real games
                       (Arena, Coach self-play).
  random_seed != 0  -> DETERMINISTIC: the outcome of a chance event is a pure
                       function of (random_seed, counter). Used inside the tree so
                       that a "universe" is one reproducible chance stream.

Why a hash and not an LCG: the previous generators used a multiplier a with
a ≡ 1 (mod m) (1981 mod 6 = 1, 4594591 mod m = 1 for most deck sizes), which
degenerates into (seed + counter) mod m: a cyclic counter. Marginals were
uniform but the joint law was wrong (after face k always came face k+1, two
draws sharing one counter were perfectly correlated, universes were mere phase
shifts of the same cycle). A splitmix64-style mixer gives decorrelated outputs
for consecutive counters and for different seeds, at the same cost.
"""
import warnings
import numpy as np
from numba import njit

# The mixer below multiplies uint64 values that are MEANT to overflow (mod
# 2**64) -- that wraparound IS the splitmix64 mixing step, not a bug. Compiled
# (@njit) code wraps silently, matching C semantics. Interpreted mode (e.g.
# NUMBA_DISABLE_JIT=1, used for debugging without paying compile time) runs
# real numpy scalar arithmetic instead, which numpy warns about by default.
# The computed value is identical either way; only this warning is spurious.
warnings.filterwarnings('ignore', message='overflow encountered in scalar multiply',
                        category=RuntimeWarning)

_M1 = np.uint64(0x9E3779B97F4A7C15)
_M2 = np.uint64(0xBF58476D1CE4E5B9)
_M3 = np.uint64(0x94D049BB133111EB)


@njit(cache=True, nogil=True)
def hashed_draw(random_seed, counter, m):
    """Uniform integer in [0, m) as a pure function of (random_seed, counter).

    counter: any int64 that changes between successive chance events of one
    stream (a per-game draw counter, a hash of the deck state, ...). Two draws
    that must be independent MUST use different counters.
    """
    x = np.uint64(np.int64(random_seed)) * _M1 + np.uint64(np.int64(counter))
    x ^= x >> np.uint64(30)
    x *= _M2
    x ^= x >> np.uint64(27)
    x *= _M3
    x ^= x >> np.uint64(31)
    return int(x % np.uint64(m))


@njit(cache=True, nogil=True)
def stream_seed(base_seed, stream_index, universes):
    """Seed of chance stream number `stream_index` for the current move.

    universes  > 0 : `universes` distinct streams, cycled (stream_index % universes)
    universes  < 0 : unlimited, every index is its own stream
    universes == 0 : legacy deterministic mode, single fixed stream (seed -1)
    The result is never 0 (0 means true randomness in the game logic).
    """
    if universes == 0:
        return np.int64(-1)
    idx = stream_index % universes if universes > 0 else stream_index
    # 2**31-1 keeps the value in a comfortable int range for the game logic
    return np.int64(1 + hashed_draw(base_seed, np.int64(idx), 2147483647))
