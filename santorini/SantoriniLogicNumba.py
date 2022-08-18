import numpy as np
from numba import njit
import numba
from .SantoriniConstants import _decode_action, _encode_action, rotation, flipLR, flipUD, DIRECTIONS, NO_MOVE, NO_BUILD

# 0: 2x2 workers set at an arbitrary position before 1st move
# 1: 2x2 workers set at a random position before 1st move
# 2: No worker pre-set, each player has to chose their position
INIT_METHOD = 1

@njit(cache=True, fastmath=True, nogil=True)
def observation_size():
	return (25, 3) # True size is 5,5,3 but other functions expects 2-dim answer

@njit(cache=True, fastmath=True, nogil=True)
def action_size():
	return 2*2*9*9

@njit(cache=True, fastmath=True, nogil=True)
def max_score_diff():
	return 3-0

# STATE is 5x5x3
#	First dimension 5x5 locates the workers (1&2 for current, -1&-2 for opponent)
#	Second dimension 5x5 lists current level
#	Initial status is
#		0  0  0  0  0        0  0  0  0  0
#		0  0  1  0  0        0  0  0  0  0
#		0 -1  0 -2  0        0  0  0  0  0
#		0  0  2  0  0        0  0  0  0  0
#		0  0  0  0  0        0  0  0  0  0
#   Third dimension 5x5 contains additional info about god for each player and memory of previous action
#		0,0: god id of player to play (0,1 same with opponent)
#		0,2: info about player to play (0,3 same with opponent)
#		0,4: does player to play just played before?
#
# ACTION is bitfield
#	field P: using god's power or not (2)
# 	field W: which worker used (2)
#	field M: to which direction moving such worker (9)
# 	field B: in which direction the worker builds (9)
#	P W MMM BBB

# DIRECTIONS
#
#	0  1  2
#   3  4  5
#   6  7  8

spec = [
	('state'        		, numba.int8[:,:,:]),
]
@numba.experimental.jitclass(spec)
class Board():
	def __init__(self, num_players):
		self.state = np.zeros((5,5,3), dtype=np.int8)
		self.init_game()

	def get_score(self, player):
		highest_level = 0
		# Highest level of the 2 workers
		if player == 0:
			for i in np.ndindex(5,5):
				worker, level, _ = self.state[i]
				if worker > 0 and level > highest_level:
						highest_level = level
		else:
			for i in np.ndindex(5,5):
				worker, level, _ = self.state[i]
				if worker < 0 and level > highest_level:
						highest_level = level
		return highest_level

	def init_game(self):
		self.state = np.zeros((5,5,3), dtype=np.int8)
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
		actions = np.zeros(2*2*9*9, dtype=np.bool_)
		if INIT_METHOD == 2 and np.abs(self.state[:,:,0]).sum() != 6: 	# Not all workers are set, need to chose their position
			for index, value in np.ndenumerate(self.state[:,:,0]):
				actions[ 5*index[0]+index[1] ] = (value == 0)
		else:															# All workers on set, ready to play
			for worker in range(2):
				worker_id = (worker+1) * (1 if player == 0 else -1)
				worker_old_position = self._get_worker_position(worker_id)
				for move_direction in range(9):
					worker_new_position = self._apply_direction(worker_old_position, move_direction)
					if not self._able_to_move_worker_to(worker_old_position, worker_new_position):
						continue
					if move_direction == NO_MOVE:
						continue
					for build_direction in range(9):
						build_position = self._apply_direction(worker_new_position, build_direction)
						if not self._able_to_build(build_position, ignore=worker_id):
							continue
						if build_direction == NO_BUILD:
							continue
						actions[_encode_action(0, worker, move_direction, build_direction)] = True
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
			_, worker, move_direction, build_direction = _decode_action(move)
			worker_id = (worker+1) * (1 if player == 0 else -1)

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
		if nb_swaps != 1:
			return
		self.state[:,:,0] = -self.state[:,:,0]
		self.state[0,0,2], self.state[0,1,2] = self.state[0,1,2], self.state[0,0,2]
		self.state[0,2,2], self.state[0,3,2] = self.state[0,3,2], self.state[0,2,2]

	def get_symmetries(self, policy, valid_actions):
		symmetries = [(self.state.copy(), policy.copy(), valid_actions.copy())]
		state_backup = self.state.copy()

		# Rotate 90°, 180°, 270°
		def _apply_permutation(permutation, array, array2):
			array_copy, array2_copy = array.copy(), array2.copy()
			for i, new_i in enumerate(permutation):
				array_copy[new_i], array2_copy[new_i] = array[i], array2[i]
			return array_copy, array2_copy

		rotated_policy, rotated_actions = policy, valid_actions
		for i in range(3):
			self.state[:,:,:2] = np.rot90(self.state[:,:,:2])
			rotated_policy, rotated_actions = _apply_permutation(rotation, rotated_policy, rotated_actions)
			symmetries.append((self.state.copy(), rotated_policy.copy(), rotated_actions.copy()))
		self.state = state_backup.copy()

		# Mirror horizontally, vertically
		self.state[:,:,:2] = np.fliplr(self.state[:,:,:2]).copy()
		flipped_policy, flipped_actions = _apply_permutation(flipLR, policy, valid_actions)
		symmetries.append((self.state, flipped_policy, flipped_actions))
		self.state = state_backup.copy()

		self.state[:,:,:2] = np.flipud(self.state[:,:,:2]).copy()
		flipped_policy, flipped_actions = _apply_permutation(flipUD, policy, valid_actions)
		symmetries.append((self.state, flipped_policy, flipped_actions))
		self.state = state_backup.copy()

		if INIT_METHOD == 2 and np.abs(self.state[:,:,0]).sum() != 6:
			return symmetries # workers not all set, stopping here

		# Permute worker 1 and 2, n = nb actions per worker
		def _swap_workers(array, n):
			array_copy = array.copy()
			array_copy[   :  n], array_copy[  n:2*n] = array[  n:2*n], array[   :  n]
			array_copy[2*n:3*n], array_copy[3*n:   ] = array[3*n:   ], array[2*n:3*n]
			return array_copy	
		w1, w2 = self._get_worker_position(1), self._get_worker_position(2)
		self.state[:,:,0][w1], self.state[:,:,0][w2] = 2, 1
		swapped_policy = _swap_workers(policy, 9*9)
		swapped_actions = _swap_workers(valid_actions, 9*9)
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
