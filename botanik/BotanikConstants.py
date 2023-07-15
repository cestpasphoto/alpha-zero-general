import numpy as np
# from numba import njit

# Colors
EMPTY  = 0
SOURCE = 1
BLUE   = 2
YELLOW = 3
GREEN  = 4
RED    = 5
BLACK  = 6

# Type
PIPE2_ANGLE    = 0
PIPE2_STRAIGHT = 1
PIPE3   = 2
PIPE4   = 3
PLANT   = 4
VEGET   = 5
MECABOT = 6

cards_generic = [
#   Col #Flo Type N  E  S  W
	[-1,  0,  0,  0, 1, 1, 0],
	[-1,  0,  0,  0, 1, 1, 0],
	[-1,  1,  0,  0, 1, 1, 0],
	[-1,  0,  1,  1, 0, 1, 0],
	[-1,  0,  1,  1, 0, 1, 0],
	[-1,  1,  1,  1, 0, 1, 0],
	[-1,  0,  2,  0, 1, 1, 1],
	[-1,  0,  2,  0, 1, 1, 1],
	[-1,  1,  2,  0, 1, 1, 1],
	[-1,  0,  3,  1, 1, 1, 1],
	[-1,  3,  4,  0, 0, 1, 0],
	[-1,  3,  5,  0, 0, 1, 0],
	[-1,  0,  6,  0, 0, 0, 0],
]

def gen_all_cards():
	np_cards_generic = np.array(cards_generic, dtype=np.int8)
	result = np.tile(np_cards_generic, (5,1,1))
	for c in range(5):
		result[c,:,0] = c+2
	return result

np_all_cards = gen_all_cards()
