import numpy as np
from numba import njit
import numba

from .BotanikConstants import *

############################## BOARD DESCRIPTION ##############################
# Board is described by a 36x5x7 array (depends on MACHINE_SIZE). Each card is
# represented using 1 line of 7 values:
#####   0      1      2      3     4      5     6
##### color #flowers type  north  east  south   west
# color is either EMPTY (no card), SOURCE (source card), BLUE, YELLOW, ...
# #flowers is number of flowers on the card: 0 or 1 or 3
# type is a value identifying type of card (0-3 for pipes, 4 for plant, 5 for
#      vegetable and 6 for mecabot)
# n/e/s/w is either 0 or 1 depending whether a pipe is open in this direction
# Special cards:
# Empty card is    0 0 0 0 0 0 0
# Mecabot is       C 0 6 0 0 0 0
# Source card is   1 0 0 0 0 1 0
#
# Here is the description of each line of the board, assuming MACHINE_SIZE=5.
# For readibility, we defined "shortcuts" that actually are views (numpy name)
# of overal board.
##### Index  Shortcut              Subindex	 Meaning
#####   0    self.misc                0      z=0: Round
#####                                        z=1: Status within round
#####                                        z=2: Main player
#####                                        z=3: Nb of open pipes in P0 machine
#####                                        z=4: Nb of open pipes in P1 machine
#####                                 1      z=0: P0 score
#####                                        z=1: P1 score
#####                                3,4     Bitfield of available cards, see below
#####   1    self.arrival_cards      0-2     Cards in arrival zone
#####   2    self.p0_register        0-4     Cards of player 0's register
#####   3    self.p1_register        0-4     Cards of player 1's register
#####   4    self.middle_reg         0-4     Visible cards on middle row
#####   5    self.freed_cards        0,1     Player 0's cards to complete the machine (mecabot always on first slot)
#####                                2,3     Player 1's cards to complete the machine
#####  6-10  self.p0_machine                 25 slots of P0 machine (y is 1st coord, x is 2nd)
##### 11-15  self.p1_machine                 25 slots of P1 machine
##### 16-20  self.p0_optim_neighbors         Not for NN - Same size as machine, stating if cell has neighbors
##### 21-25  self.p1_optim_neighbors         Not for NN - Same size as machine, stating if cell has neighbors
##### 26-30  self.p0_optim_needpipes         Not for NN - Same size as machine, stating if card needs to have pipe
##### 31-35  self.p1_optim_needpipes         Not for NN - Same size as machine, stating if card needs to have pipe
#
# Status in self.misc[0,1] is one of the following values, see comments in
# BotanikConstants.py file:
# PLAYER_TO_PUT_TO_REGISTER, OTHERP_TO_EXPAND_MACHINE, OTHERP_TO_SWAP_MECABOT,
# MAINPL_TO_EXPAND_MACHINE, MAINPL_TO_SWAP_MECABOT
# "Main player" refers to the last player who put a card from arrival to
# register. And "other player" is the other player :)
#
# Bitfield of available cards is size 2*7 of bytes:
#           COL 0    COL 1    COL 2    COL 3    COL 4    COL 5    COL 6
# ROW 0   abcdefgh ........ ........ ........ ........ ........ ........
# ROW 1   ijklm... ........ ........ ........ ........ ........ ........
# Bit 'a' is 1 if blue card 0 is available, same for bit 'b' for blue card 1,
# bit 'c' for blue card 2, ... until bit 'm' for blue card 12. Then same for
# yellow on col 1, etc: more generally color X is on col X-2
# Columns 5 and 6 are unused

############################## ACTION DESCRIPTION #############################
# We coded 236 actions, again taking some shortcuts and assuming MACHINE_SIZE=5
# Here is description of each action:
##### Index  Meaning
#####   0    Move card 0 from arrival zone to slot 0 of player register
#####   1    Move card 0 from arrival zone to slot 1 of player register
#####  ...   
#####   5    Move card 1 from arrival zone to slot 0 of player register
#####  ...   
#####  10    Move card 2 from arrival zone to slot 0 of player register
#####  ...   
#####  14    Move card 2 from arrival zone to slot 4 of player register
#####  15    Move card 0 from arrival zone to slot 0 of middle row
#####  ...   
#####  29    Move card 2 from arrival zone to slot 4 of middle row
#####  30    Swap mecabot with card on slot 0 of middle row
#####  ...   
#####  34    Swap mecabot with card on slot 4 of middle row
#####  35    Use freed card 0 to expand machine on slot 0
#####  36    Use freed card 0 to expand machine on slot 0, turned 90° clockwise
#####  37    Use freed card 0 to expand machine on slot 0, turned 180°
#####  38    Use freed card 0 to expand machine on slot 0, turned 270°
#####  39    Use freed card 0 to expand machine on slot 1
#####  ...
##### 134    Use freed card 0 to expand machine on slot 24, turned 270°
##### 135    Use freed card 1 to expand machine on slot 0
#####  ...
##### 234    Use freed card 1 to expand machine on slot 24, turned 270°
##### 235    Throw away freed cards (nowhere to put on machine)

@njit(cache=True, fastmath=True, nogil=True)
def observation_size():
	return ((6 + 6*NB_ROWS_FOR_MACH), 5, 7) # True size is 36,5,7 but other functions expects 2-dim answer

@njit(cache=True, fastmath=True, nogil=True)
def action_size():
	return 36 + 8 * MACHINE_SIZE*MACHINE_SIZE

mask = np.array([4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint16)

@njit(cache=True, fastmath=True, nogil=True)
def my_packbits(array):
	product = np.multiply(array.astype(np.uint16), mask[:len(array)])
	return product.sum()

@njit(cache=True, fastmath=True, nogil=True)
def my_unpackbits(value):
	return (np.bitwise_and(value, mask) != 0).astype(np.uint8)

@njit(cache=True, fastmath=True, nogil=True)
def my_random_choice(prob):
	result = np.searchsorted(np.cumsum(prob), np.random.random(), side="right")
	return result

@njit(cache=True, fastmath=True, nogil=True)
def packedUint_to_int8(x):
	dual_int16 = divmod(np.int16(x), 256)
	cast_to_npint8 = [np.int8(x-256 if x > 127 else x) for x in dual_int16]
	return cast_to_npint8

@njit(cache=True, fastmath=True, nogil=True)
def int8_to_packedUint(arr):
	return np.uint64(256) * (arr[0]+256 if arr[0] < 0 else arr[0]) + (arr[1]+256 if arr[1] < 0 else arr[1])


spec = [
	('state'         		, numba.int8[:,:,:]),
	('misc'          		, numba.int8[:,:]),
	('arrival_cards' 		, numba.int8[:,:]),
	('p0_register'   		, numba.int8[:,:]),
	('p1_register'   		, numba.int8[:,:]),
	('middle_reg'    		, numba.int8[:,:]),
	('freed_cards'   		, numba.int8[:,:]),
	('p0_machine'    		, numba.int8[:,:,:]),
	('p0_optim_neighbors'	, numba.int8[:,:,:]),
	('p0_optim_needpipes'	, numba.int8[:,:,:]),
	('p1_machine'    		, numba.int8[:,:,:]),
	('p1_optim_neighbors'	, numba.int8[:,:,:]),
	('p1_optim_needpipes'	, numba.int8[:,:,:]),
]
@numba.experimental.jitclass(spec)
class Board():
	def __init__(self, num_players):
		self.state = np.zeros(observation_size(), dtype=np.int8)
		self.init_game()

	def get_score(self, player):
		return self.misc[1, player]

	def init_game(self):
		self.copy_state(np.zeros(observation_size(), dtype=np.int8), copy_or_not=False)
		# Set all cards as available
		enable_all_cards = packedUint_to_int8(my_packbits(np.ones(len(mask), dtype=np.int8)))
		for color in range(5):
			self.misc[3:, color] = enable_all_cards
		# Draw 5 cards for middle row
		for i in range(5):
			self.middle_reg[i,:] = self._draw_cards(1)[0,:]
		self._draw_cards_to_arrival_zone()
		# Init machines with a source card
		self._init_machines()
	
	def copy_state(self, state, copy_or_not):
		if self.state is state and not copy_or_not:
			return
		self.state = state.copy() if copy_or_not else state

		self.misc          = self.state[0    ,:,:]
		self.arrival_cards = self.state[1    ,:,:]
		self.p0_register   = self.state[2    ,:,:]
		self.p1_register   = self.state[3    ,:,:]
		self.middle_reg    = self.state[4    ,:,:]
		self.freed_cards   = self.state[5    ,:,:]
		self.p0_machine         = np.ascontiguousarray(self.state[6                   :6+  NB_ROWS_FOR_MACH,:,:]).reshape(-1)[:MACHINE_SIZE*MACHINE_SIZE*7].reshape(MACHINE_SIZE, MACHINE_SIZE, 7)
		self.p1_machine         = np.ascontiguousarray(self.state[6+  NB_ROWS_FOR_MACH:6+2*NB_ROWS_FOR_MACH,:,:]).reshape(-1)[:MACHINE_SIZE*MACHINE_SIZE*7].reshape(MACHINE_SIZE, MACHINE_SIZE, 7)
		self.p0_optim_neighbors = np.ascontiguousarray(self.state[6+2*NB_ROWS_FOR_MACH:6+3*NB_ROWS_FOR_MACH,:,:]).reshape(-1)[:MACHINE_SIZE*MACHINE_SIZE*7].reshape(MACHINE_SIZE, MACHINE_SIZE, 7)
		self.p1_optim_neighbors = np.ascontiguousarray(self.state[6+3*NB_ROWS_FOR_MACH:6+4*NB_ROWS_FOR_MACH,:,:]).reshape(-1)[:MACHINE_SIZE*MACHINE_SIZE*7].reshape(MACHINE_SIZE, MACHINE_SIZE, 7)
		self.p0_optim_needpipes = np.ascontiguousarray(self.state[6+4*NB_ROWS_FOR_MACH:6+5*NB_ROWS_FOR_MACH,:,:]).reshape(-1)[:MACHINE_SIZE*MACHINE_SIZE*7].reshape(MACHINE_SIZE, MACHINE_SIZE, 7)
		self.p1_optim_needpipes = np.ascontiguousarray(self.state[6+5*NB_ROWS_FOR_MACH:6+6*NB_ROWS_FOR_MACH,:,:]).reshape(-1)[:MACHINE_SIZE*MACHINE_SIZE*7].reshape(MACHINE_SIZE, MACHINE_SIZE, 7)
		# Warning: np.ascontiguousarray may return a copy in general case. but
		# in this particular case it returns a view. And Numba needs it to
		# ensure that reshape is done on a contiguous array.
		# See this test code:
		#   base_ptr = self.state.ctypes.data
		#   total_bytes_big = self.state.size # itemsize = 1 for np.int8
		#   view_ptr = self.p1_optim_needpipes.ctypes.data
		#   print((view_ptr >= base_ptr) and (view_ptr < base_ptr + total_bytes_big))

	def valid_moves(self, player):
		result = np.zeros(action_size(), dtype=np.bool_)
		if self.misc[0,1] == PLAYER_TO_PUT_TO_REGISTER:
			result[  :30] = self._valid_register(player)
		elif self.misc[0,1] in [MAINPL_TO_SWAP_MECABOT, OTHERP_TO_SWAP_MECABOT]:
			result[30:35] = self._valid_swap_mecabot(player)
		elif self.misc[0,1] in [MAINPL_TO_EXPAND_MACHINE, OTHERP_TO_EXPAND_MACHINE]:
			result[35:-1] = self._valid_expand_mach(player)
			if not np.any(result[35:-1]):
				result[-1] = True # If no other move possible, allow to throw cards away
		return result

	def make_move(self, move, player, random_seed):
		if move < 15:
			self._move_to_register(move, player)
		elif move < 30:
			self._move_to_middle_row_and_unlink(move, player)
		elif move < 35:
			self._swap_mecabot(move, player)
		elif move < action_size()-1:
			self._expand_machine(move, player)
		elif move < action_size():
			self._throw_cards_away(move, player)
		
		new_state, main_player = self.misc[0,1], self.misc[0,2]
		# Update arrival zone if needed
		if new_state == PLAYER_TO_PUT_TO_REGISTER:
			if (_is_empty_card(self.arrival_cards[0,:]) and _is_empty_card(self.arrival_cards[1,:]) and _is_empty_card(self.arrival_cards[2,:])):
				self._draw_cards_to_arrival_zone()

		# Update number of rounds + main player, and select player for next action
		if new_state == PLAYER_TO_PUT_TO_REGISTER:
			self.misc[0,0] += 1
			main_player = 1 - main_player
			self.misc[0,2] = main_player
			return main_player
		if new_state in [MAINPL_TO_EXPAND_MACHINE, MAINPL_TO_SWAP_MECABOT]:
			return main_player
		else:
			return 1-main_player

	def get_state(self):
		return self.state

	def check_end_game(self, next_player):
		# Check there still are any available cards in bitfield, or any freed card to use
		if np.any(self.misc[3:5, :] != 0) or np.any(_are_not_empty_cards(self.arrival_cards[:3,:])) or np.any(_are_not_empty_cards(self.freed_cards[:4,:])):
			return np.array([0, 0], dtype=np.float32) # No winner yet

		if   self.misc[1,0] > self.misc[1,1]:
			return np.array([1, -1], dtype=np.float32)
		elif self.misc[1,0] < self.misc[1,1]:
			return np.array([-1, 1], dtype=np.float32)
		else:
			# Same score, look at nb of cards in machines
			nb_cards_p0 = np.count_nonzero(self.p0_machine[:,:,0])
			nb_cards_p1 = np.count_nonzero(self.p1_machine[:,:,0])
			if   nb_cards_p0 > nb_cards_p1:
				return np.array([1, -1], dtype=np.float32)
			elif nb_cards_p0 < nb_cards_p1:
				return np.array([-1, 1], dtype=np.float32)
			return np.array([0.01, 0.01], dtype=np.float32)

	def swap_players(self, nb_swaps):
		if nb_swaps != 1:
			print(f'Someone requested {nb_swaps} swaps, which is not 1')
			return

		# Swap registers of P0 and P1
		p0_copy = self.p0_register.copy()
		self.p0_register[:] = self.p1_register[:]
		self.p1_register[:] = p0_copy[:]

		# Swap freed cards
		p0_freed_copy = self.freed_cards[:2,:].copy()
		self.freed_cards[:2,:] = self.freed_cards[2:4,:]
		self.freed_cards[2:4,:] = p0_freed_copy[:]

		# Update misc
		if self.misc[0,1] > PLAYER_TO_PUT_TO_REGISTER:
			self.misc[0,1] = (self.misc[0,1]+1)%4 + 1
		self.misc[0,2] = 1 - self.misc[0,2]
		self.misc[1,0], self.misc[1,1] = self.misc[1,1], self.misc[1,0]

		# Swap machines
		machine_copy = self.p0_machine.copy()
		self.p0_machine[:] = self.p1_machine[:]
		self.p1_machine[:] = machine_copy[:]
		machine_copy = self.p0_optim_neighbors.copy()
		self.p0_optim_neighbors[:] = self.p1_optim_neighbors[:]
		self.p1_optim_neighbors[:] = machine_copy[:]
		machine_copy = self.p0_optim_needpipes.copy()
		self.p0_optim_needpipes[:] = self.p1_optim_needpipes[:]
		self.p1_optim_needpipes[:] = machine_copy[:]

	def get_symmetries(self, policy, valids):
		# Always called on canonical board, meaning player = 0
		# In all symmetries, no need to update the "optim" vectors as they are not used by NN
		symmetries = [(self.state.copy(), policy.copy(), valids.copy())]
		state_backup, policy_backup, valids_backup = symmetries[0]

		# Apply horizontal symmetry on a machine (swap cells and change directions)
		def _horizontal_symmetry_machine(machine):
			m = MACHINE_SIZE-1
			for y in range(MACHINE_SIZE):
				for x in range((MACHINE_SIZE+1)//2):
					if m-x != x:
						# Swap cells
						machine[y,x,:], machine[y,m-x,:] = machine[y,m-x,:], machine[y,x,:].copy()
						# Change directions
						machine[y, m-x, EAST], machine[y, m-x, WEST] = machine[y, m-x, WEST], machine[y, m-x, EAST]
						machine[y,   x, EAST], machine[y,   x, WEST] = machine[y,   x, WEST], machine[y,   x, EAST]
					else:
						machine[y,   x, EAST], machine[y,   x, WEST] = machine[y,   x, WEST], machine[y,   x, EAST]
					
		# Apply horizontal symmetry on policy+valids (swap value while taking care of new directions)
		def _horizontal_symmetry_polval(policy, valids, freed_cards):
			m, mm = MACHINE_SIZE-1, MACHINE_SIZE*MACHINE_SIZE
			for y in range(MACHINE_SIZE):
				for x in range((MACHINE_SIZE+1)//2):
					for card_i in range(2):
						card_type = freed_cards[card_i, 2]
						for orient in range(4):
							# equivalent orientation after mirroring depends on card's type
							new_orient = ([1,0,3,2] if card_type == PIPE2_ANGLE else [0,3,2,1])[orient]
							action1 = 35 + 4* (mm*card_i + (MACHINE_SIZE*y + x))
							action2 = 35 + 4* (mm*card_i + (MACHINE_SIZE*y + m-x))
							if m-x != x:
								policy[action2+new_orient], policy[action1+new_orient] = policy_backup[action1+orient], policy_backup[action2+orient]
								valids[action2+new_orient], valids[action1+new_orient] = valids_backup[action1+orient], valids_backup[action2+orient]
							else:
								policy[action1+new_orient] = policy_backup[action1+orient]
								valids[action1+new_orient] = valids_backup[action1+orient]

		# Swap freedcard 0 and 1 (assuming not empty)
		def _swap_freed(freed_cards, policy, valids):
			m, mm = MACHINE_SIZE-1, MACHINE_SIZE*MACHINE_SIZE
			# Update policy and valids
			for yx in range(mm):
				for orient in range(4):
					action1 = 35 + 4*(25*0 + yx) + orient
					action2 = 35 + 4*(25*1 + yx) + orient
					policy[action2], policy[action1] = policy_backup[action1], policy_backup[action2]
					valids[action2], valids[action1] = valids_backup[action1], valids_backup[action2]
			# Update freed_cards
			freed_cards[0,:], freed_cards[1,:] = freed_cards[1,:], freed_cards[0,:].copy()

		# Permutation is a list ([0,2,1] for instance to swap last 2 items)
		def _permute_arrival(permutation, arrival_cards, policy, valids):
			arrival_init = arrival_cards.copy()
			for i, new_i in enumerate(permutation):
				arrival_cards[new_i] = arrival_init[i]
				policy[5*new_i:5*new_i+5], policy[5*new_i+15:5*new_i+15+5] = policy_backup[5*i:5*i+5], policy_backup[5*i+15:5*i+15+5]
				valids[5*new_i:5*new_i+5], valids[5*new_i+15:5*new_i+15+5] = valids_backup[5*i:5*i+5], valids_backup[5*i+15:5*i+15+5]

		# Permutation is a list ([1,2,3,4,0] for instance to move all items to the right)
		def _permute_registers(permutation, r0, r1, rM, policy, valids):
			r0_init, r1_init, rM_init = r0.copy(), r1.copy(), rM.copy()
			for i, new_i in enumerate(permutation):
				r0[new_i], r1[new_i], rM[new_i] = r0_init[i], r1_init[i], rM_init[i]
				for z in range(7):
					policy[z*5+new_i] = policy_backup[z*5+i]
					valids[z*5+new_i] = valids_backup[z*5+i]

		# Change colors in all cards of array, except if empty card or source card
		def _roll_colors_1d(array, nroll):
			for i in range(array.shape[0]):
				color = array[i,0]
				if color != EMPTY and color != SOURCE:
					array[i,0] = ((color - 2) + nroll) % 5 + 2
		def _roll_colors_2d(array, nroll):
			for i in range(array.shape[0]):
				for j in range(array.shape[1]):
					color = array[i,j,0]
					if color != EMPTY and color != SOURCE:
						array[i,j,0] = ((color - 2) + nroll) % 5 + 2

		# Left/right symmetries of machine 0, and policy/valids
		_horizontal_symmetry_machine(self.p0_machine)
		_horizontal_symmetry_polval(policy, valids, self.freed_cards)
		symmetries.append((self.state.copy(), policy.copy(), valids.copy()))
		self.state[:,:,:], policy, valids = state_backup.copy(), policy_backup.copy(), valids_backup.copy()

		# Left/right symmetries of machine 1, no change on policy/valids
		_horizontal_symmetry_machine(self.p1_machine)
		symmetries.append((self.state.copy(), policy.copy(), valids.copy()))
		self.state[:,:,:], policy, valids = state_backup.copy(), policy_backup.copy(), valids_backup.copy()

		# Swap 2 freed cards
		if np.all(_are_not_empty_cards(self.freed_cards[:2])):
			_swap_freed(self.freed_cards, policy, valids)
			symmetries.append((self.state.copy(), policy.copy(), valids.copy()))
			self.state[:,:,:], policy, valids = state_backup.copy(), policy_backup.copy(), valids_backup.copy()

		# Permutations on arrival zone
		for permut in permutations_arrival:
			_permute_arrival(permut, self.arrival_cards, policy, valids)
			symmetries.append((self.state.copy(), policy.copy(), valids.copy()))
			self.state[:,:,:], policy, valids = state_backup.copy(), policy_backup.copy(), valids_backup.copy()

		# Some permutations on registers
		for permut in permutations_registers:
			_permute_registers(permut, self.p0_register, self.p1_register, self.middle_reg, policy, valids)
			symmetries.append((self.state.copy(), policy.copy(), valids.copy()))
			self.state[:,:,:], policy, valids = state_backup.copy(), policy_backup.copy(), valids_backup.copy()

		# Swap colors (no need change bitfield)
		for color_roll in [2, 4]:
			_roll_colors_1d(self.arrival_cards, color_roll)
			_roll_colors_1d(self.p0_register, color_roll)
			_roll_colors_1d(self.p1_register, color_roll)
			_roll_colors_1d(self.middle_reg, color_roll)
			_roll_colors_1d(self.freed_cards, color_roll)
			_roll_colors_2d(self.p0_machine, color_roll)
			_roll_colors_2d(self.p1_machine, color_roll)
			symmetries.append((self.state.copy(), policy.copy(), valids.copy()))
			self.state[:,:,:], policy, valids = state_backup.copy(), policy_backup.copy(), valids_backup.copy()

		return symmetries

	def get_round(self):
		return self.misc[0,0]

	def _draw_cards(self, how_many):
		result = np.zeros((how_many, 7), dtype=np.int8)
		# Translate list of available cards to a simple format
		available_cards = np.zeros((5, 13), dtype=np.bool_)
		bitfield = self.misc[3:5, :].astype(np.uint8)
		for color in range(5):
			available_cards[color, :] = my_unpackbits(int8_to_packedUint(bitfield[:, color]))

		for i in range(how_many):
			if available_cards.sum() == 0:
				if i != 0:
					raise Exception(f'This should not happen: could take {i-1} cards to arrival but not 3')
				return None
			# Choose random card amongst available ones
			available_cards_flat = available_cards.flatten()
			choice = my_random_choice(available_cards_flat / available_cards_flat.sum())
			choice = divmod(choice, 13)

			available_cards[choice] = False
			result[i,:] = np_all_cards[choice[0], choice[1], :]

		# Update bitfield
		for color in range(5):
			self.misc[3:, color] = packedUint_to_int8(my_packbits(available_cards[color, :]))
		return result

	def _draw_cards_to_arrival_zone(self):
		cards = self._draw_cards(3)
		if cards is not None:
			self.arrival_cards[:3,:] = cards

	def _valid_register(self, player):
		result = np.zeros(30, dtype=np.bool_)
		is_slot_empty = _are_empty_cards(self.p0_register) if player == 0 else _are_empty_cards(self.p1_register)
		arrivalcard_is_not_empty = _are_not_empty_cards(self.arrival_cards[:3,:])

		# Actual computation for player side of register
		for i in range(3):
			if arrivalcard_is_not_empty[i]:
				card = self.arrival_cards[i,:]
				match_color = (self.middle_reg[:,0] == card[0])
				match_type  = (self.middle_reg[:,2] == card[2])
				# print(f'{i}: {is_slot_empty} {np.logical_or(match_color, match_type)}')
				result[5*i:5*(i+1)] = np.logical_and(is_slot_empty, np.logical_or(match_color, match_type))

		# Actual computation for middle row
		for i in range(3):
			result[15+5*i:15+5*(i+1)] = arrivalcard_is_not_empty[i]

		return result

	def _valid_swap_mecabot(self, player):
		result = _are_not_mecabots(self.middle_reg)
		return result

	def _valid_expand_mach(self, player):
		result = np.zeros(8*MACHINE_SIZE*MACHINE_SIZE, dtype=np.bool_)
		machine = self.p0_machine if player == 0 else self.p1_machine
		optim_neighbors = self.p0_optim_neighbors if player == 0 else self.p1_optim_neighbors
		optim_needpipes = self.p0_optim_needpipes if player == 0 else self.p1_optim_needpipes
		
		admissible_cells = np.argwhere(optim_neighbors[:,:,0])
		nb_open_pipes = _compute_open_pipes(machine)

		for freed_card_slot in range(2):
			freed_card = self.freed_cards[2*player+freed_card_slot]
			if not _is_empty_card(freed_card):
				for y,x in admissible_cells:
					 result_4orient = _check_card_on_machine(freed_card, y, x, optim_needpipes[y,x,:], optim_neighbors[y,x,:], nb_open_pipes)
					 yx = x + MACHINE_SIZE*(y + MACHINE_SIZE*freed_card_slot)
					 result[(yx)*4:(yx+1)*4] = result_4orient

		return result
	
	def _move_to_register(self, move, player):
		card_i, register_slot = divmod(move, 5)
		# Move card from arrival to player register
		if player == 0:
			self.p0_register[register_slot,:] = self.arrival_cards[card_i, :]
		else:
			self.p1_register[register_slot,:] = self.arrival_cards[card_i, :]
		self.arrival_cards[card_i, :] = 0
	
	def _move_to_middle_row_and_unlink(self, move, player):
		card_i, middlerow_slot = divmod(move-15, 5)
		# Move card from arrival to middle
		self.middle_reg[middlerow_slot,:] = self.arrival_cards[card_i, :]
		self.arrival_cards[card_i, :] = 0

		self._free_card_if_needed(middlerow_slot)

	def _free_card_if_needed(self, slot):
		# Move register card(s) if no more link with middle row
		middle_color, middle_type = self.middle_reg[slot, 0], self.middle_reg[slot, 2]
		for p, reg in [(0, self.p0_register), (1, self.p1_register)]:
			player_color, player_type = reg[slot, 0], reg[slot, 2]
			if not _is_empty_card(reg[slot,:]) and player_color != middle_color and player_type != middle_type:
				# Check where to move
				if   _is_empty_card(self.freed_cards[2*p+0, :]):
					new_slot = 0
				elif _is_empty_card(self.freed_cards[2*p+1, :]):
					new_slot = 1
				else:
					# Super rare case, for instance
					# 1. P0 put a card that frees a mecabot for P0 and another one for P1
					# 2. P0 swaps its mecabot with a card that frees a normal card for P0 and P1
					# 3. P1 swaps its mecabot (step 0) with a card that frees a normal card for P1
					# At the end, P1 freed 1 card at step 2, and swapped 1 + freed 1 at step 3
					# 3 cards for P1 whereas there are 2 slots... While this issue is being fixed,
					# let's use slot 1
					new_slot = 1
					#raise Exception(f'All free slots are busy for player {p}')

				# Move card
				self.freed_cards[2*p+new_slot, :] = reg[slot, :]
				reg[slot, :] = 0
				# print(f'=== Card {slot} of P{p} was unlinked, moved to {new_slot} - {player_color},{middle_color} - {player_type},{middle_type}')

				# Update state
				is_p_the_main_player = (p == self.misc[0,2])
				if _is_mecabot(self.freed_cards[2*p+new_slot, :]):
					new_status = MAINPL_TO_SWAP_MECABOT if is_p_the_main_player else OTHERP_TO_SWAP_MECABOT
					# Mecabot needs to be the 1st freed card
					if new_slot != 0:
						# print('🎁 Plus this card is a mecabot and I need to swap with other free cards 🎁')
						mecabot_copy = self.freed_cards[2*p+new_slot, :].copy()
						self.freed_cards[2*p+new_slot, :] = self.freed_cards[2*p, :]
						self.freed_cards[2*p, :] = mecabot_copy[:]
					else:
						# print('🎁 Plus this card is a mecabot 🎁')
						pass
				else:
					new_status = MAINPL_TO_EXPAND_MACHINE if is_p_the_main_player else OTHERP_TO_EXPAND_MACHINE
				self.misc[0,1] = max(self.misc[0,1], new_status) # Higher values have higher priority

	def _swap_mecabot(self, move, player):
		middlerow_slot = move-30
		mecabot_copy = self.freed_cards[2*player, :].copy() # Mecabot is always 1st freed card
		self.freed_cards[2*player, :] = self.middle_reg[middlerow_slot, :]
		self.middle_reg[middlerow_slot, :] = mecabot_copy[:]

		if _is_mecabot(self.freed_cards[2*player+1, :]):
			raise Exception('Rare case I assumed never happening: player was having 2 mecabots freed')

		# Define new state
		if self.misc[0,1] == MAINPL_TO_SWAP_MECABOT:
			self.misc[0,1] = MAINPL_TO_EXPAND_MACHINE
		elif self.misc[0,1] == OTHERP_TO_SWAP_MECABOT:
			self.misc[0,1] = OTHERP_TO_EXPAND_MACHINE
		else:
			raise Exception('Initial state before moving a mecabot is unexpected')

		# Checkf if it frees a card
		self._free_card_if_needed(middlerow_slot)

	def _expand_machine(self, move, player):
		card_i, move_ = divmod(move-35, 4*MACHINE_SIZE*MACHINE_SIZE)
		slot, orient = divmod(move_, 4)
		slot_y, slot_x = divmod(slot, MACHINE_SIZE)
		machine = self.p0_machine if player == 0 else self.p1_machine
		optim_neighbors = self.p0_optim_neighbors if player == 0 else self.p1_optim_neighbors
		optim_needpipes = self.p0_optim_needpipes if player == 0 else self.p1_optim_needpipes
		
		# Put oriented card in machine, and update internals
		machine[slot_y, slot_x, :] = self.freed_cards[2*player + card_i, :]
		machine[slot_y, slot_x, NORTH:] = np.roll(machine[slot_y, slot_x, NORTH:], orient)
		self.freed_cards[2*player + card_i, :] = 0
		self._update_optims(slot_y, slot_x, machine, optim_neighbors, optim_needpipes)

		# Shift other freed card to 1st slot if needed
		if card_i == 0 and not _is_empty_card(self.freed_cards[2*player+1, :]):
			self.freed_cards[2*player, :] = self.freed_cards[2*player+1, :]
			self.freed_cards[2*player+1, :] = 0

		# Update score
		self.misc[1, player] = _compute_score(machine)

		# Decide next state
		mainpl = self.misc[0,2]
		otherp = 1 - mainpl
		if   _is_mecabot(self.freed_cards[2*mainpl, :]):
			self.misc[0,1] = MAINPL_TO_SWAP_MECABOT
			raise Exception('This situation should not happen since MECABOT has highest priority than expanding machine')
		elif not _is_empty_card(self.freed_cards[2*mainpl, :]):
			self.misc[0,1] = MAINPL_TO_EXPAND_MACHINE
		elif _is_mecabot(self.freed_cards[2*otherp, :]):
			self.misc[0,1] = OTHERP_TO_SWAP_MECABOT
		elif not _is_empty_card(self.freed_cards[2*otherp, :]):
			self.misc[0,1] = OTHERP_TO_EXPAND_MACHINE
		else:
			self.misc[0,1] = PLAYER_TO_PUT_TO_REGISTER

	def _init_machines(self):
		src_y, src_x = MACHINE_SIZE//3, MACHINE_SIZE//2
		# Init machines with a source card
		self.p0_machine[src_y,src_x,:], self.p1_machine[src_y,src_x,:] = SOURCE_CARD, SOURCE_CARD
		self.misc[0,3], self.misc[0,4] = 1, 1
		# Init optim arrays
		self._update_optims(src_y, src_x, self.p0_machine, self.p0_optim_neighbors, self.p0_optim_needpipes)
		self._update_optims(src_y, src_x, self.p1_machine, self.p1_optim_neighbors, self.p1_optim_needpipes)

	def _update_optims(self, y, x, machine, optim_neighbors, optim_needpipes):
		for orient, (dy, dx) in zip(range(NORTH, WEST+1), [(-1,0), (0,1), (1,0), (0,-1)]):
			if 0<=y+dy<MACHINE_SIZE and 0<=x+dx<MACHINE_SIZE:
				opposite_orient = (orient-3 + 2) % 4 +3
				# Neighbor cells (if empty) are new candidates for next cards
				optim_neighbors[y+dy, x+dx, 0] = _is_empty_card(machine[y+dy, x+dx, :])
				# Neighbor cells have me as new neighbor
				optim_neighbors[y+dy, x+dx, opposite_orient] = 1
				# Future neighbors need to have pipe if I have one (or need not have if I don't)
				optim_needpipes[y+dy, x+dx, opposite_orient] = (machine[y, x, orient] > 0)
		# And I am not a candidate anymore (code could be less overkill but easier for debug)
		optim_neighbors[y, x, :] = 0
		optim_needpipes[y, x, :] = 0

	def _throw_cards_away(self, move, player):
		self.freed_cards[2*player:2*player+2, :] = 0

		# Decide next state
		mainpl = self.misc[0,2]
		otherp = 1 - mainpl
		if   _is_mecabot(self.freed_cards[2*mainpl, :]):
			self.misc[0,1] = MAINPL_TO_SWAP_MECABOT
			raise Exception('This situation should not happen since MECABOT has highest priority than expanding machine')
		elif not _is_empty_card(self.freed_cards[2*mainpl, :]):
			self.misc[0,1] = MAINPL_TO_EXPAND_MACHINE
		elif _is_mecabot(self.freed_cards[2*otherp, :]):
			self.misc[0,1] = OTHERP_TO_SWAP_MECABOT
		elif not _is_empty_card(self.freed_cards[2*otherp, :]):
			self.misc[0,1] = OTHERP_TO_EXPAND_MACHINE
		else:
			self.misc[0,1] = PLAYER_TO_PUT_TO_REGISTER

@njit(cache=True, fastmath=True, nogil=True)
def _is_empty_card(card):
	return (card[0] == EMPTY)

@njit(cache=True, fastmath=True, nogil=True)
def _are_empty_cards(cards):
	return (cards[:,0] == EMPTY)

@njit(cache=True, fastmath=True, nogil=True)
def _are_not_empty_cards(cards):
	return (cards[:,0] != EMPTY)

@njit(cache=True, fastmath=True, nogil=True)
def _is_mecabot(card):
	return (card[2] == MECABOT)

@njit(cache=True, fastmath=True, nogil=True)
def _are_mecabots(cards):
	return (cards[:,2] == MECABOT)

@njit(cache=True, fastmath=True, nogil=True)
def _are_not_mecabots(cards):
	return (cards[:,2] != MECABOT)

@njit(cache=True, fastmath=True, nogil=True)
def _compute_open_pipes(machine):
	nb_open_pipes, m = 0, MACHINE_SIZE-1
	for y in range(5):
		for x in range(5):
			if not _is_empty_card(machine[y, x, :]):
				if y>0 and _is_empty_card(machine[y-1, x, :]) and machine[y, x, NORTH] > 0:
					nb_open_pipes += 1
				if x<m and _is_empty_card(machine[y, x+1, :]) and machine[y, x, EAST] > 0:
					nb_open_pipes += 1
				if y<m and _is_empty_card(machine[y+1, x, :]) and machine[y, x, SOUTH] > 0:
					nb_open_pipes += 1
				if x>0 and _is_empty_card(machine[y, x-1, :]) and machine[y, x, WEST] > 0:
					nb_open_pipes += 1
	return nb_open_pipes

@njit(cache=True, fastmath=True, nogil=True)
def _check_card_on_machine(card, y, x, optim_needpipes, optim_neighbors, initial_open_pipes):
	result = np.zeros(4, dtype=np.bool_)
	orient_range = range(4)
	m = MACHINE_SIZE-1
	if card[2] == PIPE2_STRAIGHT:
		orient_range = range(2)
	elif card[2] == PIPE4:
		orient_range = range(1)

	for orient in orient_range:
		oriented_card = np.roll(card[NORTH:], orient)
		# Check discontinuity of pipes of new card with its neighbours
		pipes                = np.multiply(oriented_card, np.array([y>0, x<m, y<m, x>0])) # Do not count pipes out of bounds
		pipes_with_neighbors = np.multiply(oriented_card, optim_neighbors[NORTH:])
		matching_pipes = np.all(pipes_with_neighbors == optim_needpipes[NORTH:])
		if matching_pipes:
			# Check if there still are open pipes
			card_pipes = pipes.sum() 
			closed_pipes = pipes_with_neighbors.sum()
			open_pipes = card_pipes - closed_pipes
			if initial_open_pipes - closed_pipes + open_pipes > 0:
				result[orient] = True
			# else False
		# else False
	return result

@njit(cache=False, fastmath=True, nogil=True)
def _compute_score(machine):
	visited = np.zeros((machine.shape[0], machine.shape[1]), dtype=np.bool_)
	labels  = np.ones((machine.shape[0], machine.shape[1]), dtype=np.int8) * 99
	equivalencies        = [set([0]) for i in range(0)] # Empty list but with inferrable type
	nb_cards_per_label   = [0 for i in range(0)]        # Empty list but with inferrable type
	nb_flowers_per_label = [0 for i in range(0)]        # Empty list but with inferrable type
	src_y, src_x = MACHINE_SIZE//3, MACHINE_SIZE//2
	_dfs(machine, src_y, src_x, visited, labels, equivalencies, nb_cards_per_label, nb_flowers_per_label)

	# print(labels)
	# print(equivalencies)

	total_score = 0
	visited_1d = np.zeros(len(equivalencies), dtype=np.bool_)
	for connex_area in range(1, len(equivalencies)):
		nb_cards, nb_flowers = _score_sum(equivalencies, nb_cards_per_label, nb_flowers_per_label, visited_1d, set([connex_area]))
		total_score += nb_cards+nb_flowers if nb_cards >= 3 else nb_flowers
		# print(f'{nb_cards} {nb_flowers} -> {nb_cards if nb_cards >= 3 else 0} {nb_flowers}   - {total_score=}')
	return total_score

# Implement the "faster scanning version" in
# https://en.wikipedia.org/wiki/Connected-component_labeling#Two-pass
@njit(cache=False, fastmath=True, nogil=True)
def _dfs(machine, y, x, visited, labels, equivalencies, nb_cards_per_label, nb_flowers_per_label):
	visited[y, x] = True
	neighbors = []
	m = MACHINE_SIZE-1
	if y>0 and machine[y, x, NORTH] > 0:
		neighbors.append((y-1, x))
	if x<m and machine[y, x, EAST] > 0:
		neighbors.append((y, x+1))
	if y<m and machine[y, x, SOUTH] > 0:
		neighbors.append((y+1, x))
	if x>0 and machine[y, x, WEST] > 0:
		neighbors.append((y, x-1))
	nb_flowers = machine[y, x, 1]

	# First pass: Find the neighbor (of same color) with the smallest label and
	# assign it to the current element. If there are no neighbors (of same
	# color), uniquely label the current element
	neighbors_labels = [labels[ny,nx] for ny, nx in neighbors if machine[ny, nx, 0] == machine[y, x, 0]]
	new_label = min(neighbors_labels + [99])
	if new_label == 99:
		new_label = len(equivalencies)
		equivalencies.append(set([new_label]))
		nb_cards_per_label.append(1)
		nb_flowers_per_label.append(int(nb_flowers))
	else:
		for i in neighbors_labels:
			if i != 99:
				equivalencies[i].add(new_label)
		nb_cards_per_label[new_label] += 1
		nb_flowers_per_label[new_label] += int(nb_flowers)
	labels[y, x] = new_label

	# Second pass: recurrence with neighbors, whatever their colors
	for ny, nx in neighbors:
		if not _is_empty_card(machine[ny, nx, :]) and not visited[ny, nx]:
			_dfs(machine, ny, nx, visited, labels, equivalencies, nb_cards_per_label, nb_flowers_per_label)

@njit(cache=True, fastmath=True, nogil=True)
def _score_sum(equivalencies, nb_cards_per_label, nb_flowers_per_label, visited, to_visit):
	nb_cards, nb_flowers = 0, 0
	for i in to_visit:
		if not visited[i]:
			nb_cards, nb_flowers = nb_cards+nb_cards_per_label[i], nb_flowers+nb_flowers_per_label[i]
			visited[i] = True
			# Recurrence
			deeper_result = _score_sum(equivalencies, nb_cards_per_label, nb_flowers_per_label, visited, equivalencies[i])
			nb_cards, nb_flowers = nb_cards+deeper_result[0], nb_flowers+deeper_result[1]

	return (nb_cards, nb_flowers)
	
	