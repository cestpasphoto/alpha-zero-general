import numpy as np
from numba import njit
import numba
from .SantoriniConstants import *
from .SantoriniConstants import _decode_action, _encode_action

# 0: 2x2 workers set at an arbitrary position before 1st move
# 1: 2x2 workers set at a random position before 1st move
# 2: No worker pre-set, each player has to chose their position
# 3: same as 1 plus randomly chosing god power
INIT_METHOD = 3

@njit(cache=True, fastmath=True, nogil=True)
def observation_size():
	return (25, 3) # True size is 5,5,3 but other functions expects 2-dim answer

@njit(cache=True, fastmath=True, nogil=True)
def action_size():
	return NB_GODS*2*9*9

@njit(cache=True, fastmath=True, nogil=True)
def max_score_diff():
	return 3-0

# STATE is 5x5x3
#	First dimension 5x5 locates the workers (1&2 for current, -1&-2 for opponent)
#	Second dimension 5x5 lists current level
#   Third dimension 5x5 contains additional info. Cells 0-11: information about god power used
#   	by player to play, data is info on previous move, or move to avoid, or ... depending
#  		on god. This data is stored in cell = god id. Reseted when starting a new round.
#   	Cells 11-22: same with opponent. Cell 23: general information (nb of rounds)
#	Initial status in INIT_METHOD=0 is:
#		0  0  0  0  0        0  0  0  0  0        0  0  0  0  0
#		0  0  1  0  0        0  0  0  0  0        0  0  0  0  0
#		0 -1  0 -2  0        0  0  0  0  0        0  0  0  0  0
#		0  0  2  0  0        0  0  0  0  0        0  0  0  0  0
#		0  0  0  0  0        0  0  0  0  0        0  0  0  0  0
#
# ACTION is bitfield
#	field P: god used or no god (11)
# 	field W: which worker used (2)
#	field M: to which direction moving such worker (9)
# 	field B: in which direction the worker builds (9)
#	W P MMM BBB

# DIRECTIONS
#
#	0  1  2
#   3  4  5
#   6  7  8

@njit(cache=True, fastmath=True, nogil=True)
def _position_if_pushed(old, new):
	return (2*new[0] - old[0], 2*new[1] - old[1])

@njit(cache=True, fastmath=True, nogil=True)
def _apply_direction(position, direction):
	DIRECTIONS = [
		(-1,-1),
		(-1, 0),
		(-1, 1),
		( 0,-1),
		( 0, 0),
		( 0, 1),
		( 1,-1),
		( 1, 0),
		( 1, 1),
	]
	delta = DIRECTIONS[direction]
	return (position[0]+delta[0], position[1]+delta[1])

spec = [
	('state'      , numba.int8[:,:,:]),
	('workers'    , numba.int8[:,:]),
	('levels'     , numba.int8[:,:]),
	('gods_power' , numba.int8[:,:]),
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
				worker, level = self.workers[i], self.levels[i]
				if worker > 0 and level > highest_level:
						highest_level = level
		else:
			for i in np.ndindex(5,5):
				worker, level = self.workers[i], self.levels[i]
				if worker < 0 and level > highest_level:
						highest_level = level
		return highest_level

	def init_game(self):
		self.copy_state(np.zeros((5,5,3), dtype=np.int8), copy_or_not=False)

		# Place workers
		if INIT_METHOD == 0:
			# Predefined places
			self.workers[2,1], self.workers[2,3] =  1,  2 # current player
			self.workers[1,2], self.workers[3,2] = -1, -2 # opponent
		elif INIT_METHOD == 1 or INIT_METHOD == 3:
			# Random places
			init_places = np.random.choice(5*5, 4, replace=False)
			workers_list = [1, -1, 2, -2]
			for place, worker in zip(init_places, workers_list):
				self.workers[place//5, place%5] = worker

			if INIT_METHOD == 3:
				if NB_GODS > 1:
					gods = np.random.choice(NB_GODS-1, 2, replace=False)+1
				else:
					gods = [0,0]
				self.gods_power.flat[gods[0]+NB_GODS*0] = 1
				self.gods_power.flat[gods[1]+NB_GODS*1] = 1

		elif INIT_METHOD == 2:
			# Players decide
			pass # Nothing to do
		
	def get_state(self):
		return self.state

	def valid_moves(self, player):
		actions = np.zeros(action_size(), dtype=np.bool_)
		if INIT_METHOD == 2 and np.abs(self.workers).sum() != 6: 		# Not all workers are set, need to chose their position
			for index, value in np.ndenumerate(self.workers):
				actions[ 5*index[0]+index[1] ] = (value == 0)
		else:															# All workers on set, ready to play
			### For optimization purpose, duplicated code for each god
			### NO GOD ###
			if self.gods_power.flat[NO_GOD+NB_GODS*player] > 0:
				for worker in range(2):
					worker_id = (worker+1) * (1 if player == 0 else -1)
					worker_old_position = self._get_worker_position(worker_id)
					for move_direction in range(9):
						if move_direction == NO_MOVE:
							continue
						worker_new_position = _apply_direction(worker_old_position, move_direction)
						if not self._able_to_move_worker_to(worker_old_position, worker_new_position, player):
							continue
						for build_direction in range(9):
							if build_direction == NO_BUILD:
								continue
							build_position = _apply_direction(worker_new_position, build_direction)
							if not self._able_to_build(build_position, ignore=worker_id):
								continue
							actions[_encode_action(worker, NO_GOD, move_direction, build_direction)] = True

			### APOLLO ###
			elif self.gods_power.flat[APOLLO+NB_GODS*player] > 0:
				for worker in range(2):
					worker_id = (worker+1) * (1 if player == 0 else -1)
					worker_old_position = self._get_worker_position(worker_id)
					for move_direction in range(9):
						if move_direction == NO_MOVE:
							continue
						worker_new_position = _apply_direction(worker_old_position, move_direction)
						if not self._able_to_move_worker_to(worker_old_position, worker_new_position, player):
							if self._able_to_move_worker_to(worker_old_position, worker_new_position, player, swap_with_opponent=True):
								use_power = True
							else:
								continue
						else:
							use_power = False
						for build_direction in range(9):
							if build_direction == NO_BUILD:
								continue
							build_position = _apply_direction(worker_new_position, build_direction)
							if not self._able_to_build(build_position, ignore=worker_id):
								continue
							actions[_encode_action(worker, APOLLO if use_power else NO_GOD, move_direction, build_direction)] = True

			### MINOTAUR ###
			elif self.gods_power.flat[MINOTAUR+NB_GODS*player] > 0:
				for worker in range(2):
					worker_id = (worker+1) * (1 if player == 0 else -1)
					worker_old_position = self._get_worker_position(worker_id)
					for move_direction in range(9):
						if move_direction == NO_MOVE:
							continue
						worker_new_position = _apply_direction(worker_old_position, move_direction)
						if not self._able_to_move_worker_to(worker_old_position, worker_new_position, player):
							if self._able_to_move_worker_to(worker_old_position, worker_new_position, player, push_opponent=True):
								use_power = True
							else:
								continue
						else:
							use_power = False
						for build_direction in range(9):
							if build_direction == NO_BUILD:
								continue
							build_position = _apply_direction(worker_new_position, build_direction)
							if not self._able_to_build(build_position, ignore=worker_id):
								continue
							actions[_encode_action(worker, MINOTAUR if use_power else NO_GOD, move_direction, build_direction)] = True

			### ATLAS ###
			elif self.gods_power.flat[ATLAS+NB_GODS*player] > 0:
				for worker in range(2):
					worker_id = (worker+1) * (1 if player == 0 else -1)
					worker_old_position = self._get_worker_position(worker_id)
					for move_direction in range(9):
						if move_direction == NO_MOVE:
							continue
						worker_new_position = _apply_direction(worker_old_position, move_direction)
						if not self._able_to_move_worker_to(worker_old_position, worker_new_position, player):
							continue
						for build_direction in range(9):
							if build_direction == NO_BUILD:
								continue
							build_position = _apply_direction(worker_new_position, build_direction)
							if self._able_to_build(build_position, ignore=worker_id):
								actions[_encode_action(worker, NO_GOD, move_direction, build_direction)] = True

							if self._able_to_build(build_position, ignore=worker_id, dome_with_god=True):
								actions[_encode_action(worker, ATLAS, move_direction, build_direction)] = True

			### HEPHAESTUS ###
			elif self.gods_power.flat[HEPHAESTUS+NB_GODS*player] > 0:
				for worker in range(2):
					worker_id = (worker+1) * (1 if player == 0 else -1)
					worker_old_position = self._get_worker_position(worker_id)
					for move_direction in range(9):
						if move_direction == NO_MOVE:
							continue
						worker_new_position = _apply_direction(worker_old_position, move_direction)
						if not self._able_to_move_worker_to(worker_old_position, worker_new_position, player):
							continue
						for build_direction in range(9):
							if build_direction == NO_BUILD:
								continue
							build_position = _apply_direction(worker_new_position, build_direction)
							if self._able_to_build(build_position, ignore=worker_id):
								actions[_encode_action(worker, NO_GOD, move_direction, build_direction)] = True

							if self._able_to_build(build_position, ignore=worker_id, two_levels_no_dome=True):
								actions[_encode_action(worker, HEPHAESTUS, move_direction, build_direction)] = True

			else:
				print(f'Should not happen vm , {player}')
				# breakpoint()

		return actions

	def make_move(self, move, player, deterministic):
		opponent_to_play_next = True

		if INIT_METHOD == 2 and np.abs(self.workers).sum() != 6:	# Not all workers are set, need to chose their position
			# Search for missing worker to place
			sum_workers = np.abs(self.workers).sum()
			if sum_workers == 0 or sum_workers == 1: 	# 0 -> empty board			, 1 -> -1 already on board
				worker_to_place = 1 if player == 0 else -1
			elif sum_workers == 2 or sum_workers == 4: 	# 2 -> 1 and -1 on board	, 4 -> -1,1,-2 on board
				worker_to_place = 2 if player == 0 else -2
			else:
				assert(False)
			# Put worker at the coded position
			y, x = divmod(move, 5)
			self.workers[y,x] = worker_to_place
		else:														# All workers on set, ready to play
			# Decode move
			worker, power, move_direction, build_direction = _decode_action(move)
			worker_id = (worker+1) * (1 if player == 0 else -1)

			if power == NO_GOD:
				worker_old_position = self._get_worker_position(worker_id)
				worker_new_position = _apply_direction(worker_old_position, move_direction)
				self.workers[worker_old_position], self.workers[worker_new_position] = 0, worker_id
				build_position = _apply_direction(worker_new_position, build_direction)
				self.levels[build_position] += 1
			elif power == APOLLO:
				worker_old_position = self._get_worker_position(worker_id)
				worker_new_position = _apply_direction(worker_old_position, move_direction)
				# Swap
				self.workers[worker_old_position], self.workers[worker_new_position] = self.workers[worker_new_position], self.workers[worker_old_position]
				build_position = _apply_direction(worker_new_position, build_direction)
				self.levels[build_position] += 1
			elif power == MINOTAUR:
				worker_old_position = self._get_worker_position(worker_id)
				worker_new_position = _apply_direction(worker_old_position, move_direction)
				# Push
				new_opponent_position = _position_if_pushed(worker_old_position, worker_new_position)
				self.workers[worker_old_position], self.workers[worker_new_position], self.workers[new_opponent_position] = 0, self.workers[worker_old_position], self.workers[worker_new_position]
				build_position = _apply_direction(worker_new_position, build_direction)
				self.levels[build_position] += 1
			elif power == ATLAS:
				worker_old_position = self._get_worker_position(worker_id)
				worker_new_position = _apply_direction(worker_old_position, move_direction)
				self.workers[worker_old_position], self.workers[worker_new_position] = 0, worker_id
				build_position = _apply_direction(worker_new_position, build_direction)
				self.levels[build_position] = 4 # Dome
			elif power == HEPHAESTUS:
				worker_old_position = self._get_worker_position(worker_id)
				worker_new_position = _apply_direction(worker_old_position, move_direction)
				self.workers[worker_old_position], self.workers[worker_new_position] = 0, worker_id
				build_position = _apply_direction(worker_new_position, build_direction)
				self.levels[build_position] += 2 # Two levels at once
			else:
				print(f'Should not happen mm {power} ({move})')

		if opponent_to_play_next:
			return 1-player
		else:
			return player

	def check_end_game(self):
		if INIT_METHOD == 2 and np.abs(self.workers).sum() != 6:	# workers not all set, no winner yet
			return np.array([0, 0], dtype=np.float32)					
		if self.get_score(0) == 3 or self.valid_moves(1).sum() == 0:	# P0 wins
			return np.array([1, -1], dtype=np.float32)
		if self.get_score(1) == 3 or self.valid_moves(0).sum() == 0:	# P1 wins
			return np.array([-1, 1], dtype=np.float32)
		return np.array([0, 0], dtype=np.float32)						# no winner yet

	def swap_players(self, nb_swaps):
		if nb_swaps != 1:
			return
		# Swap workers
		self.workers[:,:] = -self.workers
		# Swap gods
		# self.gods_power.flat[0:NB_GODS], self.gods_power.flat[NB_GODS:2*NB_GODS] = self.gods_power.flat[NB_GODS:2*NB_GODS], self.gods_power.flat[0:NB_GODS]
		gods_power_copy = self.gods_power.copy()
		for i in range(2*NB_GODS):
			self.gods_power.flat[i] = gods_power_copy.flat[(i+NB_GODS)%(2*NB_GODS)]

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
			self.workers[:,:] = np.rot90(self.workers)
			self.levels[:,:] = np.rot90(self.levels)
			rotated_policy, rotated_actions = _apply_permutation(rotation, rotated_policy, rotated_actions)
			symmetries.append((self.state.copy(), rotated_policy.copy(), rotated_actions.copy()))
		self.state[:,:,:] = state_backup.copy()

		# Mirror horizontally, vertically
		self.workers[:,:] = np.fliplr(self.workers)
		self.levels[:,:] = np.fliplr(self.levels)
		flipped_policy, flipped_actions = _apply_permutation(flipLR, policy, valid_actions)
		symmetries.append((self.state.copy(), flipped_policy, flipped_actions))
		self.state[:,:,:] = state_backup.copy()

		self.workers[:,:] = np.flipud(self.workers).copy()
		self.levels[:,:] = np.flipud(self.levels).copy()
		flipped_policy, flipped_actions = _apply_permutation(flipUD, policy, valid_actions)
		symmetries.append((self.state.copy(), flipped_policy, flipped_actions))
		self.state[:,:,:] = state_backup.copy()

		if INIT_METHOD == 2 and np.abs(self.state[:,:,0]).sum() != 6:
			return symmetries # workers not all set, stopping here

		# Permute worker 1 and 2
		def _swap_workers(array, half_size):
			array_copy = array.copy()
			array_copy[:half_size], array_copy[half_size:] = array[half_size:], array[:half_size]
			return array_copy	
		w1, w2 = self._get_worker_position(1), self._get_worker_position(2)
		self.workers[w1], self.workers[w2] = 2, 1
		swapped_policy = _swap_workers(policy, action_size()//2)
		swapped_actions = _swap_workers(valid_actions, action_size()//2)
		symmetries.append((self.state.copy(), swapped_policy, swapped_actions))
		self.state[:,:,:] = state_backup.copy()

		# Permute worker -1 and -2
		wm1, wm2 = self._get_worker_position(-1), self._get_worker_position(-2)
		self.workers[wm1], self.workers[wm2] = -2, -1
		symmetries.append((self.state.copy(), policy.copy(), valid_actions.copy()))
		self.state[:,:,:] = state_backup.copy()

		return symmetries

	def get_round(self):
		return self.levels.sum()

	def copy_state(self, state, copy_or_not):
		if self.state is state and not copy_or_not:
			return
		self.state = state.copy() if copy_or_not else state

		self.workers    = self.state[:,:,0]
		self.levels     = self.state[:,:,1]
		self.gods_power = self.state[:,:,2]

	def _get_worker_position(self, searched_worker):
		for i in np.ndindex(5, 5):
			if self.workers[i] == searched_worker:
				return i
		# breakpoint()
		print(f'Should not happen gwp {searched_worker}')
		return (-1, -1)

	# Same function as after because @jitclass doesn't support recursive function
	def _able_to_move_worker_to(self, old_position, new_position, player, swap_with_opponent=False, push_opponent=False):
		if not (0<=new_position[0]<5 and 0<=new_position[1]<5):		# Out of grid?
			return False

		if self.workers[new_position] != 0:		# Cell already used by another worker?
			opponents = [-1, -2] if player == 0 else [1, 2]
			if (swap_with_opponent or push_opponent) and (self.workers[new_position] in opponents):
				if push_opponent: # Check opponent future position if he's pushed
					if not self._able_to_push_opponent(_position_if_pushed(old_position, new_position)):
						return False
			else:
				return False

		new_level = self.levels[new_position]
		if new_level > 3:						# Dome in future position?
			return False

		old_level = self.levels[old_position]
		if new_level > old_level + 1:			# Future level much higher than current level?
			return False

		return True

	# Same function as before because @jitclass doesn't support recursive function
	def _able_to_push_opponent(self, new_position):
		if not (0<=new_position[0]<5 and 0<=new_position[1]<5):		# Out of grid?
			return False

		if self.workers[new_position] != 0:		# Cell already used by another worker?
			return False

		new_level = self.levels[new_position]
		if new_level > 3:						# Dome in future position?
			return False

		# Future level much higher than current level?
		# Not tested in this mode
		return True

	# Check whether possible at position, ignoring worker 'ignore' (in case such worker is meant to have moved)
	def _able_to_build(self, position, ignore=0, two_levels_no_dome=False, dome_with_god=False):
		assert not(two_levels_no_dome and dome_with_god)
		if not (0<=position[0]<5 and 0<=position[1]<5):											# Out of grid?
			return False

		if not self.workers[position] in [0, ignore]:											# Cell already used by another worker?
			return False

		if self.levels[position] >= (2 if two_levels_no_dome else (3 if dome_with_god else 4)):	# Dome in future position?
			return False

		return True