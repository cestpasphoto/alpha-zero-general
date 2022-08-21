import numpy as np
from numba import njit

NO_MOVE = 4
NO_BUILD = 4

NO_GOD     = 0
APOLLO     = 1
MINOTAUR   = 2
ATLAS      = 3
HEPHAESTUS = 4
ARTEMIS    = 5
DEMETER    = 6
HERMES     = 7
PAN        = 8
NB_GODS = 9

@njit(cache=True, fastmath=True, nogil=True)
def _decode_action(action):
	worker, action_ = divmod(action, NB_GODS*9*9)
	power, action_ = divmod(action_, 9*9)
	move_direction, build_direction = divmod(action_, 9)

	return worker, power, move_direction, build_direction

@njit(cache=True, fastmath=True, nogil=True)
def _encode_action(worker, power, move_direction, build_direction):
	action = NB_GODS*9*9*worker + 9*9*power + 9*move_direction + build_direction
	return action


def _generate_permutation(permutation):
	result = []
	for i in range(NB_GODS*2*9*9):
		worker, power, move_direction, build_direction = _decode_action(i)
		new_move_direction, new_build_direction = permutation[move_direction], permutation[build_direction]
		new_i = _encode_action(worker, power, new_move_direction, new_build_direction)
		result.append(new_i)
	return result

# Rotated directions
#   0  1  2       2  5  8
#   3  4  5  <--- 1  4  7
#   6  7  8       0  3  6
rotation_core = np.array([6,3,0,7,4,1,8,5,2], dtype=np.int8)
rotation = np.array(_generate_permutation(rotation_core), dtype=np.int8)

# FlippedLR directions
#   0  1  2       2  1  0
#   3  4  5  <--- 5  4  3
#   6  7  8       8  7  6
flipLR_core   = np.array([2,1,0,5,4,3,8,7,6], dtype=np.int8)
flipLR = np.array(_generate_permutation(flipLR_core), dtype=np.int8)


# FlippedUD directions
#   0  1  2       6  7  8
#   3  4  5  <--- 3  4  5
#   6  7  8       0  1  2
flipUD_core   = np.array([6,7,8,3,4,5,0,1,2], dtype=np.int8)
flipUD = np.array(_generate_permutation(flipUD_core), dtype=np.int8)
