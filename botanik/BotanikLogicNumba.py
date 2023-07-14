import numpy as np
# from numba import njit
# import numba

############################## BOARD DESCRIPTION ##############################
# Board is described by a 6x5x6 array. Each card is represented using 1 line of
# 6 values:
#####   0      1      2      3      4      5
##### color #flowers north  east  south   west
# color is either EMPTY (no card), SOURCE (source card), BLUE, YELLOW, ...
# #flowers is number of flowers on the card: 0 or 1 or 3
# n/e/s/w is either 0 or 1 depending whether a pipe is open in this direction
# Mecabots have -1 for 4 last values.
#
# Here is the description of each line of the board. For readibility, we defined
# "shortcuts" that actually are views (numpy name) of overal board.
##### Index  Shortcut              Subindex	 Meaning
#####   0    self.misc                0      Round
#####                                1-2     Stores temp cards
#####                                3-4     Not used
#####   1    self.p0_cards           0-4     Cards of player 0
#####   2    self.p1_cards           0-4     Cards of player 1
#####   3    self.middle_row         0-4     Visible cards on middle row
#####  4,5   self.used_cards                 Bitfield listing if card i was drawn or not

############################## ACTION DESCRIPTION #############################

def observation_size():
	return (6*5, 6) # True size is 6,5,6 but other functions expects 2-dim answer

def action_size():
	return 1

class Board():
	def __init__(self, num_players):
		self.state = np.zeros((6,5,6), dtype=np.int8)
		self.init_game()

	def get_score(self, player):
		return 0

	def init_game(self):
		self.copy_state(np.zeros((6,5,6), dtype=np.int8), copy_or_not=False)
		
	def get_state(self):
		return self.state

	def valid_moves(self, player):
		result = np.zeros(1, dtype=np.bool_)
		result[0] = 1
		return result

	def make_move(self, move, player, deterministic):
		return (player+1)%2

	def check_end_game(self, next_player):
		return np.array([0, 0], dtype=np.float32)								# no winner yet

	def swap_players(self, nb_swaps):
		pass

	def get_symmetries(self, policy, valid_actions):
		pass

	def get_round(self):
		return self.misc.flat[0][0]

	def copy_state(self, state, copy_or_not):
		if self.state is state and not copy_or_not:
			return
		self.state = state.copy() if copy_or_not else state

		self.misc       = self.state[0 ,:,:]
		self.p0_cards   = self.state[1 ,:,:]
		self.p1_cards   = self.state[2 ,:,:]
		self.middle_row = self.state[3 ,:,:]
		self.used_cards = self.state[4:,:,:]
