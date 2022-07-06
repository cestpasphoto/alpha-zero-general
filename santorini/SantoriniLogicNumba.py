import numpy as np
from numba import njit
import numba

# @njit(cache=True, fastmath=True, nogil=True)
def observation_size():
	return (25, 2) # True size is 5,5,2 but other functions expects 2-dim answer

# @njit(cache=True, fastmath=True, nogil=True)
def action_size():
	return 2*8*8

# @njit(cache=True, fastmath=True, nogil=True)
def max_score_diff():
	return 3-0

# STATE is 5x5x2
#	First dimension 5x5 locates the workers (1&2 for current, -1&-2 for opponent)
#	Second dimension 5x5 lists current level
#	Initial status is
#		0  0  0  0  0        0  0  0  0  0
#		0  0  1  0  0        0  0  0  0  0
#		0 -1  0 -2  0        0  0  0  0  0
#		0  0  2  0  0        0  0  0  0  0
#		0  0  0  0  0        0  0  0  0  0
#
# ACTION is bitfield
# 	field W: which worker used (2 - 1b)
#	field M: to which direction moving such worker (8 - 3b)
# 	field B: in which direction the worker builds (8 - 3b)
#	W MMM BBB
#
# DIRECTION
#
#	0  1  2
#   3  -  4
#   5  6  7

# spec = [
# 	('state'        		, numba.int8[:,:,:]),
# ]
# @numba.experimental.jitclass(spec)
class Board():
	def __init__(self, num_players):
		self.state = np.zeros((5,5,2), dtype=np.int8)
		self.init_game()

	def get_score(self, player):
		highest_level = 0
		# Highest level of the 2 workers
		if player == 0:
			for i in np.ndindex(5,5):
				worker, level = self.state[i]
				if worker > 0 and level > highest_level:
						highest_level = level
		else:
			for i in np.ndindex(5,5):
				worker, level = self.state[i]
				if worker < 0 and level > highest_level:
						highest_level = level
		return highest_level

	def init_game(self):
		self.state = np.zeros((5,5,2), dtype=np.int8)
		# Place workers
		self.state[2,1,0], self.state[2,3,0] =  1,  2 # current player
		self.state[1,2,0], self.state[3,2,0] = -1, -2 # opponent
		
	def get_state(self):
		return self.state

	def valid_moves(self, player):
		actions = np.zeros(2*8*8, dtype=np.bool_)
		for worker in range(2):
			worker_id = (worker+1) * (1 if player == 0 else -1)
			worker_old_position = self._get_worker_position(worker_id)
			for move_direction in range(8):
				worker_new_position = self._apply_direction(worker_old_position, move_direction)
				if not self._able_to_move_worker_to(worker_old_position, worker_new_position):
					continue
				for build_direction in range(8):
					build_position = self._apply_direction(worker_new_position, build_direction)
					if not self._able_to_build(build_position, ignore=worker_id):
						continue
					actions[build_direction+8*move_direction+8*8*worker] = True
		return actions

	def make_move(self, move, player, deterministic):
		# Decode move
		worker, move_ = divmod(move, 8*8)
		worker_id = (worker+1) * (1 if player == 0 else -1)
		move_direction, build_direction = divmod(move_, 8)

		worker_old_position = self._get_worker_position(worker_id)
		worker_new_position = self._apply_direction(worker_old_position, move_direction)
		self.state[worker_old_position[0], worker_old_position[1], 0] = 0
		self.state[worker_new_position[0], worker_new_position[1], 0] = worker_id

		build_position = self._apply_direction(worker_new_position, build_direction)
		self.state[build_position[0], build_position[1], 1] += 1

		return 1-player

	def check_end_game(self):
		if self.get_score(0) == 3 or self.valid_moves(1).sum() == 0:	# P0 wins
			return np.array([1, 0], dtype=np.float32)
		if self.get_score(1) == 3:										# P1 wins
			return np.array([0, 1], dtype=np.float32)
		return np.array([0, 0], dtype=np.float32)						# no winner yet

	def swap_players(self, nb_swaps):
		# Since only 2 players, ignore nb_swaps
		self.state[:,:,0] = -self.state[:,:,0]

	def get_symmetries(self, policy, valid_actions):
		symmetries = [(self.state.copy(), policy.copy(), valid_actions.copy())]
		# Permute worker 1 and 2
		# Permute worker -1 and -2
		# Permute worker 1 and 2
		# Rotate 90°, 180°, 270°
		# Mirror vertically, horizontally

		# for tier in range(3):
		# 	for permutation in np_cards_symmetries:
		# 		symmetries.append((self.state.copy(), new_policy, new_valid_actions))

		return symmetries

	def get_round(self):
		return self.state[:,:,1].sum()

	def copy_state(self, state, copy_or_not):
		if self.state is state and not copy_or_not:
			return
		self.state = state.copy() if copy_or_not else state

	def _get_worker_position(self, searched_worker):
		for i in np.ndindex(5, 5, 1):
			worker = self.state[i]
			if worker == searched_worker:
				return i[:2]
		print(f'Should not happen gwp {searched_worker}')
		return (-1, -1)

	def _apply_direction(self, position, direction):
		delta = DIRECTIONS[direction]
		return (position[0]+delta[0], position[1]+delta[1])

	def _able_to_move_worker_to(self, old_position, new_position):
		if not (0<=new_position[0]<5 and 0<=new_position[1]<5):		# Out of grid?
			return False

		if self.state[new_position[0], new_position[1], 0] != 0:	# Cell already used by another worker?
			return False

		new_level = self.state[new_position[0], new_position[1], 1]
		if new_level > 3:											# Dome in future position?
			return False

		old_level = self.state[old_position[0], old_position[1], 1]
		if new_level > old_level + 1:								# Future level much higher than current level?
			return False

		return True

	# Check whether possible at position, ignoring worker 'ignore' (in case such worker is meant to have moved)
	def _able_to_build(self, position, ignore=0):
		if not (0<=position[0]<5 and 0<=position[1]<5):					# Out of grid?
			return False

		if not self.state[position[0], position[1], 0] in [0, ignore]:	# Cell already used by another worker?
			return False

		if self.state[position[0], position[1], 1] > 3:					# Dome in future position?
			return False

		return True


DIRECTIONS = np.array([
	(-1,-1),
	( 0,-1),
	( 1,-1),
	(-1, 0),
	( 1, 0),
	(-1, 1),
	( 0, 1),
	( 1, 1),
], dtype=np.int8)