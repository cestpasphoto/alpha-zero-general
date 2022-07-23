import numpy as np
from numba import njit
import numba

# 0: 2x2 workers set at an arbitrary position before 1st move
# 1: 2x2 workers set at a random position before 1st move
# 2: No worker pre-set, each player has to chose their position
INIT_METHOD = 1

@njit(cache=True, fastmath=True, nogil=True)
def observation_size():
	return (25, 2) # True size is 5,5,2 but other functions expects 2-dim answer

@njit(cache=True, fastmath=True, nogil=True)
def action_size():
	return 2*8*8

@njit(cache=True, fastmath=True, nogil=True)
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
# DIRECTIONS
#
#	0  1  2
#   3  -  4
#   5  6  7

DIRECTIONS = np.array([
	(-1,-1),
	(-1, 0),
	(-1, 1),
	( 0,-1),
	( 0, 1),
	( 1,-1),
	( 1, 0),
	( 1, 1),
], dtype=np.int8)

# def _generate_permutation(permutation):
# 	result = []
# 	for i in range(2*8*8):
# 		worker, move_ = divmod(i, 8*8)
# 		move_direction, build_direction = divmod(move_, 8)
# 		new_move_direction, new_build_direction = permutation[move_direction], permutation[build_direction]
# 		new_i = new_build_direction + 8*new_move_direction + 8*8*worker
# 		result.append(new_i)
# 	print(result)
#
## Rotated directions
##   0  1  2       2  4  7
##   3  -  4  <--- 1  -  6
##   5  6  7       0  3  5
# rotation = np.array([5,3,0,6,1,7,4,2], dtype=np.int8)
# _generate_permutation(rotation)
#
## FlippedLR directions
##   0  1  2       2  1  0
##   3  -  4  <--- 4  -  3
##   5  6  7       7  6  5
# flipLR   = np.array([2,1,0,4,3,7,6,5], dtype=np.int8)
# _generate_permutation(flipLR)
#
## FlippedUD directions
##   0  1  2       5  6  7
##   3  -  4  <--- 3  -  4
##   5  6  7       0  1  2
# flipUD   = np.array([5,6,7,3,4,0,1,2], dtype=np.int8)
# _generate_permutation(flipUD)

rotation = np.array(
	[45, 43, 40, 46, 41, 47, 44, 42, 29, 27, 24, 30, 25, 31, 28, 26, 5, 3, 0, 6, 1, 7, 4, 2, 53, 51, 48, 54, 49, 55, 52, 50, 13, 11, 8, 14, 9, 15, 12, 10, 61, 59, 56, 62, 57, 63, 60, 58, 37, 35, 32, 38, 33, 39, 36, 34, 21, 19, 16, 22, 17, 23, 20, 18, 109, 107, 104, 110, 105, 111, 108, 106, 93, 91, 88, 94, 89, 95, 92, 90, 69, 67, 64, 70, 65, 71, 68, 66, 117, 115, 112, 118, 113, 119, 116, 114, 77, 75, 72, 78, 73, 79, 76, 74, 125, 123, 120, 126, 121, 127, 124, 122, 101, 99, 96, 102, 97, 103, 100, 98, 85, 83, 80, 86, 81, 87, 84, 82]
	, dtype=np.int8)
flipLR = np.array(
	[18, 17, 16, 20, 19, 23, 22, 21, 10, 9, 8, 12, 11, 15, 14, 13, 2, 1, 0, 4, 3, 7, 6, 5, 34, 33, 32, 36, 35, 39, 38, 37, 26, 25, 24, 28, 27, 31, 30, 29, 58, 57, 56, 60, 59, 63, 62, 61, 50, 49, 48, 52, 51, 55, 54, 53, 42, 41, 40, 44, 43, 47, 46, 45, 82, 81, 80, 84, 83, 87, 86, 85, 74, 73, 72, 76, 75, 79, 78, 77, 66, 65, 64, 68, 67, 71, 70, 69, 98, 97, 96, 100, 99, 103, 102, 101, 90, 89, 88, 92, 91, 95, 94, 93, 122, 121, 120, 124, 123, 127, 126, 125, 114, 113, 112, 116, 115, 119, 118, 117, 106, 105, 104, 108, 107, 111, 110, 109]
	, dtype=np.int8)
flipUD = np.array(
	[45, 46, 47, 43, 44, 40, 41, 42, 53, 54, 55, 51, 52, 48, 49, 50, 61, 62, 63, 59, 60, 56, 57, 58, 29, 30, 31, 27, 28, 24, 25, 26, 37, 38, 39, 35, 36, 32, 33, 34, 5, 6, 7, 3, 4, 0, 1, 2, 13, 14, 15, 11, 12, 8, 9, 10, 21, 22, 23, 19, 20, 16, 17, 18, 109, 110, 111, 107, 108, 104, 105, 106, 117, 118, 119, 115, 116, 112, 113, 114, 125, 126, 127, 123, 124, 120, 121, 122, 93, 94, 95, 91, 92, 88, 89, 90, 101, 102, 103, 99, 100, 96, 97, 98, 69, 70, 71, 67, 68, 64, 65, 66, 77, 78, 79, 75, 76, 72, 73, 74, 85, 86, 87, 83, 84, 80, 81, 82]
	, dtype=np.int8)

spec = [
	('state'        		, numba.int8[:,:,:]),
]
@numba.experimental.jitclass(spec)
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
		if INIT_METHOD == 0:
			# Predefined places
			self.state[2,1,0], self.state[2,3,0] =  1,  2 # current player
			self.state[1,2,0], self.state[3,2,0] = -1, -2 # opponent
		elif INIT_METHOD == 1:
			# Random places
			init_places = np.random.choice(5*5, 4, replace=False)
			workers = [1, -1, 2, -2]
			for place, worker in zip(init_places, workers):
				self.state[place//5, place%5, 0] = worker
		elif INIT_METHOD == 2:
			# Players decide
			pass # Nothing to do
		
	def get_state(self):
		return self.state

	def valid_moves(self, player):
		actions = np.zeros(2*8*8, dtype=np.bool_)
		if INIT_METHOD == 2 and np.abs(self.state[:,:,0]).sum() != 6: 	# Not all workers are set, need to chose their position
			for index, value in np.ndenumerate(self.state[:,:,0]):
				actions[ 5*index[0]+index[1] ] = (value == 0)
		else:															# All workers on set, ready to play
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
		if INIT_METHOD == 2 and np.abs(self.state[:,:,0]).sum() != 6:	# Not all workers are set, need to chose their position
			# Search for missing worker to place
			sum_workers = np.abs(self.state[:,:,0]).sum()
			if sum_workers == 0 or sum_workers == 1: 	# 0 -> empty board			, 1 -> -1 already on board
				worker_to_place = 1 if player == 0 else -1
			elif sum_workers == 2 or sum_workers == 4: 	# 2 -> 1 and -1 on board	, 4 -> -1,1,-2 on board
				worker_to_place = 2 if player == 0 else -2
			else:
				assert(False)
			# Put worker at the coded position
			y, x = divmod(move, 5)
			self.state[y,x,0] = worker_to_place
		else:															# All workers on set, ready to play
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
		if INIT_METHOD == 2 and np.abs(self.state[:,:,0]).sum() != 6:	# workers not all set, no winner yet
			return np.array([0, 0], dtype=np.float32)					
		if self.get_score(0) == 3 or self.valid_moves(1).sum() == 0:	# P0 wins
			return np.array([1, -1], dtype=np.float32)
		if self.get_score(1) == 3 or self.valid_moves(0).sum() == 0:	# P1 wins
			return np.array([-1, 1], dtype=np.float32)
		return np.array([0, 0], dtype=np.float32)						# no winner yet

	def swap_players(self, nb_swaps):
		# Since only 2 players, ignore nb_swaps
		self.state[:,:,0] = -self.state[:,:,0]

	def get_symmetries(self, policy, valid_actions):
		symmetries = [(self.state.copy(), policy.copy(), valid_actions.copy())]
		state_backup = self.state.copy()

		# Rotate 90°, 180°, 270°
		def _apply_permutation(permutation, array, array2):
			array_copy, array2_copy = array.copy(), array2.copy()
			for i in range(2*8*8):
				new_i = permutation[i]
				array_copy[new_i], array2_copy[new_i] = array[i], array2[i]
			return array_copy, array2_copy

		rotated_policy, rotated_actions = policy, valid_actions
		for i in range(3):
			self.state = np.rot90(self.state)
			rotated_policy, rotated_actions = _apply_permutation(rotation, rotated_policy, rotated_actions)
			symmetries.append((self.state.copy(), rotated_policy.copy(), rotated_actions.copy()))
		self.state = state_backup.copy()

		# Mirror horizontally, vertically
		flipped_state = np.fliplr(self.state).copy()
		flipped_policy, flipped_actions = _apply_permutation(flipLR, policy, valid_actions)
		symmetries.append((flipped_state, flipped_policy, flipped_actions))
		self.state = state_backup.copy()

		flipped_state = np.flipud(self.state).copy()
		flipped_policy, flipped_actions = _apply_permutation(flipUD, policy, valid_actions)
		symmetries.append((flipped_state, flipped_policy, flipped_actions))
		self.state = state_backup.copy()

		if INIT_METHOD == 2 and np.abs(self.state[:,:,0]).sum() != 6:
			return symmetries # workers not all set, stopping here

		# Permute worker 1 and 2
		def _swap_halves(array, middle_index):
			array_copy = array.copy()
			array_copy[:middle_index], array_copy[middle_index:] = array[middle_index:], array[:middle_index]
			return array_copy	
		w1, w2 = self._get_worker_position(1), self._get_worker_position(2)
		self.state[:,:,0][w1], self.state[:,:,0][w2] = 2, 1
		swapped_policy = _swap_halves(policy, 8*8)
		swapped_actions = _swap_halves(valid_actions, 8*8)
		symmetries.append((self.state.copy(), swapped_policy, swapped_actions))
		self.state = state_backup.copy()

		# Permute worker -1 and -2
		wm1, wm2 = self._get_worker_position(-1), self._get_worker_position(-2)
		self.state[:,:,0][wm1], self.state[:,:,0][wm2] = -2, -1
		symmetries.append((self.state.copy(), policy.copy(), valid_actions.copy()))
		self.state = state_backup.copy()

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
