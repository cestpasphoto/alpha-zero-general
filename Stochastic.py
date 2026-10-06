"""
Deterministic, counter-based randomness for the search.

Contract shared by every game logic (getNextState / make_move):
  random_seed == 0  -> TRUE randomness (np.random), used for the real games
                       (Arena, Coach self-play).
  random_seed != 0  -> DETERMINISTIC: the outcome of a chance event is a pure
                       function of (random_seed, counter). Used inside the tree so
                       that a "universe" is one reproducible chance stream.

A splitmix64-style mixer gives decorrelated outputs for consecutive counters and
for different seeds. Do not replace it with an LCG whose multiplier is 1 modulo
the range: that degenerates into a cyclic counter.
"""
import warnings
import numpy as np
from numba import njit

# The uint64 overflow below IS the mixing step. Compiled code wraps silently;
# interpreted mode (NUMBA_DISABLE_JIT=1) computes the same value but warns.
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
