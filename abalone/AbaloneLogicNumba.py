import numpy as np
from numba import njit
import numba

# Starting layout: 0 = Classic, 1 = Belgian Daisy, 2 = German Daisy.
# Either a scalar (fixed layout) or a tuple: one entry is then drawn per game,
# repeat an entry to weight it, e.g. (1, 2, 0, 1, 2).
INITIAL_LAYOUT              = 1    # Use (1, 2, 0, 1, 2) for training
ENABLE_DYNAMIC_KOMI         = False # Must disable during games, but good to enable at the beginning of training
MAX_PLIES                   = 300  # Hard cap on game length in plies (stored on 2 int8 cells, max 16383)
NO_EJECTION_LIMIT           = 60   # Game ends after this many consecutive plies without ejection (max 127)

# =============================================================================
# BOARD DESCRIPTION
# =============================================================================
# The Abalone board is represented on an axial grid of 9x9.
# Playable cells (61 cells) satisfy the condition: 4 <= r + q <= 12
#
# State dimension: (9, 9, 4)
#   z=0 : Current player's marbles (1 if present, 0 otherwise)
#   z=1 : Opponent's marbles (1 if present, 0 otherwise)
#   z=2 : Board mask (1 if cell is playable, 0 otherwise)
#   z=3 : Miscellaneous data (misc):
#         (0, 0) : Current player's score (ejected marbles, 0 to 6)
#         (0, 1) : Opponent's score (0 to 6)
#         (0, 2) : Ply counter, low byte (0 to 127)
#         (0, 3) : Komi (only if ENABLE_DYNAMIC_KOMI): 1 = current player wins ties, 0 = opponent wins ties
#         (0, 4) : Ply counter, high byte (ply = low + 128 * high)
#         (0, 5) : Plies since last ejection (0 to NO_EJECTION_LIMIT)
#
# A game ends when a player has ejected 6 marbles, or when MAX_PLIES or
# NO_EJECTION_LIMIT is reached. In the latter case the higher score wins, then
# komi decides (if enabled), otherwise the game is a draw.
#
# =============================================================================
# ACTION DESCRIPTION
# =============================================================================
# Action space size: 3402 (9 * 9 * 42)
# An action is decomposed into:
#   - An "Anchor" cell (r, q)
#   - A group size (1, 2, or 3 marbles)
#   - An alignment axis (0: East, 1: South-East, 2: South-West)
#   - A movement direction (0 to 5)
# The Anchor is always the marble in the group closest to the axis origin 
# (minimum 'r', and if tied, minimum 'q').

# Plain-Python normalisation (scalar -> 1-tuple) so that numba sees a homogeneous tuple
_LAYOUT_POOL = tuple(int(x) for x in INITIAL_LAYOUT) if isinstance(INITIAL_LAYOUT, (tuple, list)) else (int(INITIAL_LAYOUT),)

@njit(cache=True, fastmath=True, nogil=True)
def observation_size():
	return (9, 9, 4)

@njit(cache=True, fastmath=True, nogil=True)
def action_size():
	return 3402 # 9 * 9 * 42

# Axial directions: (delta_r, delta_q)
DIRECTIONS = np.array([
	[ 0,  1], # 0: East
	[ 1,  0], # 1: South-East
	[ 1, -1], # 2: South-West
	[ 0, -1], # 3: West
	[-1,  0], # 4: North-West
	[-1,  1]  # 5: North-East
], dtype=np.int8)

@njit(cache=True, fastmath=True, nogil=True, inline='always')
def _encode_action(r, q, size, axis, direction):
	if size == 1:
		plane = direction
	elif size == 2:
		plane = 6 + axis * 6 + direction
	else:
		plane = 24 + axis * 6 + direction
	return r * 9 * 42 + q * 42 + plane

@njit(cache=True, fastmath=True, nogil=True, inline='always')
def _decode_action(action):
	plane = action % 42
	q = (action // 42) % 9
	r = action // (42 * 9)
	direction = plane % 6
	if plane < 6:
		size, axis = 1, 0
	elif plane < 24:
		size, axis = 2, (plane - 6) // 6
	else:
		size, axis = 3, (plane - 24) // 6
	return r, q, size, axis, direction

@njit(cache=True, fastmath=True, nogil=True, inline='always')
def is_on_board(r, q, mask):
	if 0 <= r < 9 and 0 <= q < 9:
		return mask[r, q] == 1
	return False

# =============================================================================
# SYMMETRY PRE-CALCULATION
# =============================================================================
@njit(cache=True)
def _build_action_symmetries():
	# Precomputes a map [rotation, flip, action_id] -> symmetric_action_id
	sym = np.zeros((6, 2, 3402), dtype=np.int32)
	for rot in range(6):
		for flip in range(2):
			for a in range(3402):
				r, q, size, axis, d = _decode_action(a)
				
				# 1. Transform all marbles in the group
				mr = np.zeros(size, dtype=np.int32)
				mq = np.zeros(size, dtype=np.int32)
				for i in range(size):
					nr = r + i * DIRECTIONS[axis, 0]
					nq = q + i * DIRECTIONS[axis, 1]
					
					if flip == 1: # Reflect across vertical axis
						nr, nq = nr, 12 - nr - nq
					for _ in range(rot): # Rotate 60° CW around center (4,4)
						nnr = nq + nr - 4
						nnq = 8 - nr
						nr, nq = nnr, nnq
					mr[i], mq[i] = nr, nq
					
				# 2. Find new anchor (min r, then min q)
				min_i = 0
				for i in range(1, size):
					if mr[i] < mr[min_i] or (mr[i] == mr[min_i] and mq[i] < mq[min_i]):
						min_i = i
				new_r, new_q = mr[min_i], mq[min_i]
				
				# 3. Find new axis
				new_axis = 0
				if size > 1:
					other_i = 1 if min_i == 0 else 0
					dr = mr[other_i] - new_r
					dq = mq[other_i] - new_q
					
					if dr == 0 and dq > 0: new_axis = 0
					elif dr > 0 and dq == 0: new_axis = 1
					elif dr > 0 and dq < 0: new_axis = 2
					
				# 4. Transform direction
				new_d = d
				if flip == 1:
					new_d = [3, 2, 1, 0, 5, 4][new_d]
				new_d = (new_d + rot) % 6
				
				# 5. Encode mapping
				sym[rot, flip, a] = _encode_action(new_r, new_q, size, new_axis, new_d)
	return sym

# Compute map once at module initialization
ACTION_SYMMETRIES = _build_action_symmetries()

# =============================================================================
# BOARD CLASS
# =============================================================================
spec = [
	('state', numba.int8[:,:,:]),
	('my_marbles', numba.int8[:,:]),
	('opp_marbles', numba.int8[:,:]),
	('board_mask', numba.int8[:,:]),
	('misc', numba.int8[:,:]),
]

@numba.experimental.jitclass(spec)
class Board():
	def __init__(self, num_players):
		self.state = np.zeros(observation_size(), dtype=np.int8)
		self.init_game()

	def init_game(self):
		self.copy_state(np.zeros(observation_size(), dtype=np.int8), copy_or_not=False)
		
		# Initialize valid board mask
		for r in range(9):
			for q in range(9):
				if 4 <= r + q <= 12:
					self.board_mask[r, q] = 1

		# ====================================================================
		# STARTING LAYOUT CONFIGURATION
		# ====================================================================
		layout = _LAYOUT_POOL[np.random.randint(len(_LAYOUT_POOL))]
		if layout == 0:
			# Classic Layout
			self.opp_marbles[0, 4:9] = 1
			self.opp_marbles[1, 3:9] = 1
			self.opp_marbles[2, 4:7] = 1

			self.my_marbles[8, 0:5] = 1
			self.my_marbles[7, 0:6] = 1
			self.my_marbles[6, 2:5] = 1

		elif layout == 1:
			# Belgian Daisy - Clusters closer to the edges
			# Opponent (White)
			self.opp_marbles[0, 4:6] = 1
			self.opp_marbles[1, 3:6] = 1
			self.opp_marbles[2, 3:5] = 1
			
			self.opp_marbles[6, 4:6] = 1
			self.opp_marbles[7, 3:6] = 1
			self.opp_marbles[8, 3:5] = 1

			# Current Player (Black)
			self.my_marbles[0, 7:9] = 1
			self.my_marbles[1, 6:9] = 1
			self.my_marbles[2, 6:8] = 1
			
			self.my_marbles[6, 1:3] = 1
			self.my_marbles[7, 0:3] = 1
			self.my_marbles[8, 0:2] = 1

		elif layout == 2:
			# German Daisy - Clusters one row away from the edges
			# Opponent (White): H4-H5, G3-G5, F3-F4 and D6-D7, C5-C7, B5-B6
			self.opp_marbles[1, 3:5] = 1
			self.opp_marbles[2, 2:5] = 1
			self.opp_marbles[3, 2:4] = 1

			self.opp_marbles[5, 5:7] = 1
			self.opp_marbles[6, 4:7] = 1
			self.opp_marbles[7, 4:6] = 1

			# Current Player (Black): H8-H9, G7-G9, F7-F8 and D2-D3, C1-C3, B1-B2
			self.my_marbles[1, 7:9] = 1
			self.my_marbles[2, 6:9] = 1
			self.my_marbles[3, 6:8] = 1

			self.my_marbles[5, 1:3] = 1
			self.my_marbles[6, 0:3] = 1
			self.my_marbles[7, 0:2] = 1


		if ENABLE_DYNAMIC_KOMI:
			# misc[0, 3] = 1 means the CURRENT player wins ties
			# misc[0, 3] = 0 means the OPPONENT wins ties
			self.misc[0, 3] = np.random.randint(2)

	def get_state(self):
		return self.state

	def copy_state(self, state, copy_or_not):
		if self.state is state and not copy_or_not:
			return
		self.state = state.copy() if copy_or_not else state

		self.my_marbles  = self.state[:, :, 0]
		self.opp_marbles = self.state[:, :, 1]
		self.board_mask  = self.state[:, :, 2]
		self.misc        = self.state[:, :, 3]

	def get_round(self):
		return np.int32(self.misc[0, 2]) + 128 * np.int32(self.misc[0, 4])

	def _increment_round(self):
		# Ply counter spread over 2 int8 cells to go beyond 127 without overflow
		if self.misc[0, 2] >= 127:
			self.misc[0, 2] = 0
			self.misc[0, 4] += 1
		else:
			self.misc[0, 2] += 1

	def get_score(self, player):
		return self.misc[0, 0] if player == 0 else self.misc[0, 1]

	def valid_moves(self, player):
		actions = np.zeros(action_size(), dtype=np.bool_)
		opp = 1 - player

		for r in range(9):
			for q in range(9):
				if self.state[r, q, player] == 0:
					continue

				# Size 1 moves
				for d in range(6):
					nr, nq = r + DIRECTIONS[d, 0], q + DIRECTIONS[d, 1]
					if is_on_board(nr, nq, self.board_mask) and self.state[nr, nq, player] == 0 and self.state[nr, nq, opp] == 0:
						actions[_encode_action(r, q, 1, 0, d)] = True

				# Size 2 & 3 moves (Pruning search space dynamically)
				for axis in range(3):
					r1, q1 = r + DIRECTIONS[axis, 0], q + DIRECTIONS[axis, 1]
					if not is_on_board(r1, q1, self.board_mask) or self.state[r1, q1, player] == 0:
						continue 

					sizes_to_check = [2]
					r2, q2 = r1 + DIRECTIONS[axis, 0], q1 + DIRECTIONS[axis, 1]
					if is_on_board(r2, q2, self.board_mask) and self.state[r2, q2, player] == 1:
						sizes_to_check.append(3)

					for size in sizes_to_check:
						for d in range(6):
							is_inline = (d == axis) or (d == (axis + 3) % 6)
							
							if not is_inline:
								broadside_valid = True
								for i in range(size):
									tr, tq = r + i * DIRECTIONS[axis, 0] + DIRECTIONS[d, 0], q + i * DIRECTIONS[axis, 1] + DIRECTIONS[d, 1]
									if not is_on_board(tr, tq, self.board_mask) or self.state[tr, tq, player] == 1 or self.state[tr, tq, opp] == 1:
										broadside_valid = False
										break
								if broadside_valid:
									actions[_encode_action(r, q, size, axis, d)] = True
							else:
								if d == axis: 
									front_r = r + (size - 1) * DIRECTIONS[axis, 0]
									front_q = q + (size - 1) * DIRECTIONS[axis, 1]
								else:         
									front_r, front_q = r, q
								
								tr, tq = front_r + DIRECTIONS[d, 0], front_q + DIRECTIONS[d, 1]
								
								if not is_on_board(tr, tq, self.board_mask):
									continue
								if self.state[tr, tq, player] == 1:
									continue
								if self.state[tr, tq, opp] == 0:
									actions[_encode_action(r, q, size, axis, d)] = True 
									continue

								opp_count = 0
								curr_r, curr_q = tr, tq
								sumito_valid = False

								while True:
									if not is_on_board(curr_r, curr_q, self.board_mask):
										if opp_count > 0: sumito_valid = True 
										break
									if self.state[curr_r, curr_q, opp] == 1:
										opp_count += 1
										if opp_count >= size: break 
										curr_r += DIRECTIONS[d, 0]
										curr_q += DIRECTIONS[d, 1]
									elif self.state[curr_r, curr_q, player] == 1:
										break 
									else:
										sumito_valid = True 
										break

								if sumito_valid:
									actions[_encode_action(r, q, size, axis, d)] = True
		return actions

	def make_move(self, move, player, random_seed):
		r, q, size, axis, d = _decode_action(move)
		is_inline = (d == axis) or (d == (axis + 3) % 6)
		opp = 1 - player
		ejected = False

		if size == 1 or not is_inline:
			for i in range(size):
				cr = r + i * DIRECTIONS[axis, 0]
				cq = q + i * DIRECTIONS[axis, 1]
				tr, tq = cr + DIRECTIONS[d, 0], cq + DIRECTIONS[d, 1]
				self.state[cr, cq, player] = 0
				self.state[tr, tq, player] = 1
		else:
			if d == axis:
				front_r = r + (size - 1) * DIRECTIONS[axis, 0]
				front_q = q + (size - 1) * DIRECTIONS[axis, 1]
				back_r, back_q = r, q
			else:
				front_r, front_q = r, q
				back_r = r + (size - 1) * DIRECTIONS[axis, 0]
				back_q = q + (size - 1) * DIRECTIONS[axis, 1]
			
			tr, tq = front_r + DIRECTIONS[d, 0], front_q + DIRECTIONS[d, 1]

			if is_on_board(tr, tq, self.board_mask) and self.state[tr, tq, opp] == 1:
				curr_r, curr_q = tr, tq
				while is_on_board(curr_r, curr_q, self.board_mask) and self.state[curr_r, curr_q, opp] == 1:
					curr_r += DIRECTIONS[d, 0]
					curr_q += DIRECTIONS[d, 1]
				
				self.state[tr, tq, opp] = 0 
				if is_on_board(curr_r, curr_q, self.board_mask):
					self.state[curr_r, curr_q, opp] = 1 
				else:
					# Opponent marble pushed off the board: mover scores
					self.misc[0, player] += 1
					ejected = True

			self.state[back_r, back_q, player] = 0
			self.state[tr, tq, player] = 1

		self._increment_round()
		if ejected:
			self.misc[0, 5] = 0
		elif self.misc[0, 5] < 127:
			self.misc[0, 5] += 1
		return 1 - player

	def check_end_game(self, next_player):
		if self.misc[0, 0] >= 6:
			return np.array([1.0, -1.0], dtype=np.float32)
		if self.misc[0, 1] >= 6:
			return np.array([-1.0, 1.0], dtype=np.float32)
			
		if self.get_round() >= MAX_PLIES or self.misc[0, 5] >= NO_EJECTION_LIMIT:
			if self.misc[0, 0] > self.misc[0, 1]: return np.array([1.0, -1.0], dtype=np.float32)
			if self.misc[0, 1] > self.misc[0, 0]: return np.array([-1.0, 1.0], dtype=np.float32)
			if ENABLE_DYNAMIC_KOMI:
				if self.misc[0, 3] == 1:
					return np.array([1.0, -1.0], dtype=np.float32)
				else:
					return np.array([-1.0, 1.0], dtype=np.float32)
			return np.array([0.001, 0.001], dtype=np.float32)
			
		return np.array([0.0, 0.0], dtype=np.float32)

	def swap_players(self, nb_swaps):
		if nb_swaps % 2 == 1:
			# Swap board layers
			temp_marbles = self.my_marbles.copy()
			self.my_marbles[:, :] = self.opp_marbles
			self.opp_marbles[:, :] = temp_marbles
			
			temp_score = self.misc[0, 0]
			self.misc[0, 0] = self.misc[0, 1]
			self.misc[0, 1] = temp_score

			if ENABLE_DYNAMIC_KOMI:
				self.misc[0, 3] = 1 - self.misc[0, 3]

	def get_symmetries(self, policy, valids):
		symmetries = []
		
		for rot in range(6):
			for flip in range(2):
				# 1. Transform board state
				new_state = np.zeros_like(self.state)
				for r in range(9):
					for q in range(9):
						if self.board_mask[r, q] == 1:
							nr, nq = r, q
							if flip == 1:
								nr, nq = nr, 12 - nr - nq
							for _ in range(rot):
								nnr = nq + nr - 4
								nnq = 8 - nr
								nr, nq = nnr, nnq
							new_state[nr, nq, :] = self.state[r, q, :]
				# Copy misc layer without transformations
				new_state[:, :, 3] = self.state[:, :, 3].copy()
				
				# 2. Transform policy and valids arrays
				new_policy = np.zeros_like(policy)
				new_valids = np.zeros_like(valids)
				
				for a in range(3402):
					if valids[a]: # Optimization: only remap valid actions
						mapped_a = ACTION_SYMMETRIES[rot, flip, a]
						new_policy[mapped_a] = policy[a]
						new_valids[mapped_a] = valids[a]
						
				symmetries.append((new_state, new_policy, new_valids))
				
		return symmetries