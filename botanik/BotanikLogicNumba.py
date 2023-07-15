import numpy as np
# from numba import njit
# import numba

from .BotanikConstants import np_all_cards, EMPTY
from .BotanikDisplay import card_to_str

############################## BOARD DESCRIPTION ##############################
# Board is described by a 5x5x7 array. Each card is represented using 1 line of
# 6 values:
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
# Source card is   1 0 0 0 0 0 0
#
# Here is the description of each line of the board. For readibility, we defined
# "shortcuts" that actually are views (numpy name) of overal board.
##### Index  Shortcut              Subindex	 Meaning
#####   0    self.misc                0      Round on z=0, status within round on z=1
#####                                3,4     Bitfield of available cards, see below
#####   1    self.temp_cards         0-2     Cards in arrival zone
#####                                3,4     Cards to complete machines
#####   2    self.p0_register        0-4     Cards of player 0's register
#####   3    self.p1_register        0-4     Cards of player 1's register
#####   4    self.middle_reg         0-4     Visible cards on middle row
#
# Status (in self.misc) is 0 if need to draw cards for arrival zone, 1 if
# someone needs to chose one card from arrival zone and put it on register, 2
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
# We coded 30 actions, taking some shortcuts ...
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
#####  16    Move card 0 from arrival zone to slot 1 of middle row
#####  ...   
#####  20    Move card 1 from arrival zone to slot 0 of middle row
#####  ...   
#####  25    Move card 2 from arrival zone to slot 0 of middle row
#####  ...   
#####  29    Move card 2 from arrival zone to slot 4 of middle row


def observation_size():
	return (5*5, 7) # True size is 4,5,6 but other functions expects 2-dim answer

def action_size():
	return 30

mask = np.array([4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint16)

def my_packbits(array):
	product = np.multiply(array.astype(np.uint16), mask[:len(array)])
	return product.sum()

def my_unpackbits(value):
	return (np.bitwise_and(value.astype(np.uint16), mask) != 0).astype(np.uint8)

def my_random_choice(prob):
	result = np.searchsorted(np.cumsum(prob), np.random.random(), side="right")
	return result

class Board():
	def __init__(self, num_players):
		self.state = None
		self.init_game()

	def get_score(self, player):
		return 0

	def init_game(self):
		self.copy_state(np.zeros((5,5,7), dtype=np.int8), copy_or_not=False)
		# Set all cards as available
		enable_all_cards = divmod(my_packbits(np.ones(len(mask), dtype=np.int8)), 256)
		for color in range(5):
			self.misc[3:, color] = enable_all_cards
		# Draw 5 cards for P0, 5 for P1 and 5 for middle row
		for i in range(5):
			cards = self._draw_cards(3)
			self.p0_register[i,:] = cards[0,:]
			self.p1_register[i,:] = cards[1,:]
			self.middle_reg[i,:]  = cards[2,:]
		self._draw_cards_to_arrival_zone()
	
	def copy_state(self, state, copy_or_not):
		if self.state is state and not copy_or_not:
			return
		self.state = state.copy() if copy_or_not else state

		self.misc        = self.state[0 ,:,:]
		self.temp_cards  = self.state[1 ,:,:]
		self.p0_register = self.state[2 ,:,:]
		self.p1_register = self.state[3 ,:,:]
		self.middle_reg  = self.state[4 ,:,:]
		
	def valid_moves(self, player):
		result = np.zeros(30, dtype=np.bool_)
		result[  :30] = self._valid_register(player)
		return result

	def make_move(self, move, player, deterministic):
		if move < 15:
			self._move_to_register(move, player)
		elif move < 30:
			self._move_to_middle_row(move)
		
		# Update arrival zone if needed
		if (self.temp_cards[0,0] == EMPTY and self.temp_cards[1,0] == EMPTY and self.temp_cards[2,0] == EMPTY):
			self._draw_cards_to_arrival_zone()

		# Update internal state
		self.misc[0][0] += 1
		return (player+1)%2

	def get_state(self):
		return self.state

	def check_end_game(self, next_player):
		return np.array([0, 0], dtype=np.float32)								# no winner yet

	def swap_players(self, nb_swaps):
		pass

	def get_symmetries(self, policy, valid_actions):
		pass

	def get_round(self):
		return self.misc.flat[0][0]

	def _draw_cards(self, how_many):
		result = np.zeros((how_many, 7), dtype=np.int8)
		# Translate list of available cards to a simple format
		available_cards = np.zeros((5, 13), dtype=np.bool_) 
		for color in range(5):
			available_cards[color, :] = my_unpackbits(256*self.misc[3, color].astype(np.uint8) + self.misc[4, color].astype(np.uint8))

		for i in range(how_many):
			# Choose random card amongst available ones
			if available_cards.sum() == 0:
				print('Error, empty deck')
				breakpoint()
				return None
			choice = my_random_choice(available_cards.flat[:] / available_cards.sum())
			choice = divmod(choice, 13)

			available_cards[choice] = False
			result[i,:] = np_all_cards[choice[0], choice[1], :]

		# Update bitfield
		for color in range(5):
			self.misc[3:, color] = divmod(my_packbits(available_cards[color, :]), 256)
		return result

	def _draw_cards_to_arrival_zone(self):
		self.temp_cards[:3,:] = self._draw_cards(3)

	def _valid_register(self, player):
		result = np.zeros(30, dtype=np.bool_)
		is_slot_empty = (self.p0_register[:,0] == EMPTY) if player == 0 else (self.p1_register[:,0] == EMPTY)
		arrivalcard_is_not_empty = (self.temp_cards[:3,0] != EMPTY)

		# Actual computation for player side of register
		for i in range(3):
			if not arrivalcard_is_not_empty[i]:
				result[5*i:5*(i+1)] = False
			else:
				card = self.temp_cards[i,:]
				match_color = (self.middle_reg[:,0] == card[0])
				match_type  = (self.middle_reg[:,2] == card[2])
				print(f'{i}: {is_slot_empty} {np.logical_or(match_color, match_type)}')
				result[5*i:5*(i+1)] = np.logical_and(is_slot_empty, np.logical_or(match_color, match_type))

		# Actual computation for middle row
		for i in range(3):
			result[15+5*i:15+5*(i+1)] = arrivalcard_is_not_empty[i]

		return result

	def _move_to_register(self, move, player):
		card_i, register_slot = divmod(move, 5)
		if player == 0:
			self.p0_register[register_slot,:] = self.temp_cards[card_i, :]
		else:
			self.p1_register[register_slot,:] = self.temp_cards[card_i, :]
		self.temp_cards[card_i, :] = 0
	
	def _move_to_middle_row(self, move):
		card_i, middlerow_slot = divmod(move-15, 5)
		self.middle_reg[middlerow_slot,:] = self.temp_cards[card_i, :]
		self.temp_cards[card_i, :] = 0