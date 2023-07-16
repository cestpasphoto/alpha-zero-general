import numpy as np
from numba import njit
import numba

from .BotanikConstants import *
from .BotanikDisplay import card_to_str

############################## BOARD DESCRIPTION ##############################
# Board is described by a 6x5x7 array. Each card is represented using 1 line of
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
#####   0    self.misc                0      Round on z=0, status within round on z=1, main player on z=2
#####                                3,4     Bitfield of available cards, see below
#####   1    self.arrival_cards      0-2     Cards in arrival zone
#####   2    self.p0_register        0-4     Cards of player 0's register
#####   3    self.p1_register        0-4     Cards of player 1's register
#####   4    self.middle_reg         0-4     Visible cards on middle row
#####   5    self.freed_cards        0,1     Player 0's cards to complete the machine (mecabot always on first slot)
#####                                2,3     Player 1's cards to complete the machine
#
# "Main player" refers to the last player who put a card from arrival to
# register. And "other player" is the other player :)
# Status in self.misc[0,1] is one of the following values, see comments in
# BotanikConstants.py file:
# PLAYER_TO_PUT_TO_REGISTER, OTHERP_TO_EXPAND_MACHINE, OTHERP_TO_SWAP_MECABOT,
# MAINPL_TO_EXPAND_MACHINE, MAINPL_TO_SWAP_MECABOT
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
# We coded 37 actions, taking some shortcuts ...
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
#####  35    Use freed card 0 to expand machine
#####  36    Use freed card 1 to expand machine

@njit(cache=True, fastmath=True, nogil=True)
def observation_size():
	return (6*5, 7) # True size is 4,5,6 but other functions expects 2-dim answer

@njit(cache=True, fastmath=True, nogil=True)
def action_size():
	return 37

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

spec = [
	('state'         , numba.int8[:,:,:]),
	('misc'          , numba.int8[:,:]),
	('arrival_cards' , numba.int8[:,:]),
	('p0_register'   , numba.int8[:,:]),
	('p1_register'   , numba.int8[:,:]),
	('middle_reg'    , numba.int8[:,:]),
	('freed_cards'   , numba.int8[:,:]),
]
@numba.experimental.jitclass(spec)
class Board():
	def __init__(self, num_players):
		self.state = np.zeros((6,5,7), dtype=np.int8)
		self.init_game()

	def get_score(self, player):
		return 0

	def init_game(self):
		self.copy_state(np.zeros((6,5,7), dtype=np.int8), copy_or_not=False)
		# Set all cards as available
		enable_all_cards = divmod(my_packbits(np.ones(len(mask), dtype=np.int8)), 256)
		for color in range(5):
			self.misc[3:, color] = enable_all_cards
		# Draw 5 cards for middle row
		for i in range(5):
			self.middle_reg[i,:] = self._draw_cards(1)[0,:]
		self._draw_cards_to_arrival_zone()
	
	def copy_state(self, state, copy_or_not):
		if self.state is state and not copy_or_not:
			return
		self.state = state.copy() if copy_or_not else state

		self.misc          = self.state[0 ,:,:]
		self.arrival_cards = self.state[1 ,:,:]
		self.p0_register   = self.state[2 ,:,:]
		self.p1_register   = self.state[3 ,:,:]
		self.middle_reg    = self.state[4 ,:,:]
		self.freed_cards   = self.state[5 ,:,:]
		
	def valid_moves(self, player):
		result = np.zeros(37, dtype=np.bool_)
		if self.misc[0,1] == PLAYER_TO_PUT_TO_REGISTER:
			result[  :30] = self._valid_register(player)
		elif self.misc[0,1] in [MAINPL_TO_SWAP_MECABOT, OTHERP_TO_SWAP_MECABOT]:
			result[30:35] = self._valid_swap_mecabot(player)
		elif self.misc[0,1] in [MAINPL_TO_EXPAND_MACHINE, OTHERP_TO_EXPAND_MACHINE]:
			result[35:  ] = self._valid_expand_mach(player)
		return result

	def make_move(self, move, player, deterministic):
		if move < 15:
			self._move_to_register(move, player)
		elif move < 30:
			self._move_to_middle_row_and_unlink(move, player)
		elif move < 35:
			self._swap_mecabot(move, player)
		elif move < 37:
			self._expand_machine(move, player)
		
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
		return np.array([0.01, 0.01], dtype=np.float32) # Ex aequo for this draft implementation

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

	def get_symmetries(self, policy, valid_actions):
		pass

	def get_round(self):
		return self.misc[0,0]

	def _draw_cards(self, how_many):
		result = np.zeros((how_many, 7), dtype=np.int8)
		# Translate list of available cards to a simple format
		available_cards = np.zeros((5, 13), dtype=np.bool_)
		bitfield = self.misc[3:5, :].astype(np.uint8)
		for color in range(5):
			available_cards[color, :] = my_unpackbits(256*bitfield[0, color] + bitfield[1, color])

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
			self.misc[3:, color] = divmod(my_packbits(available_cards[color, :]), 256)
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
			if not arrivalcard_is_not_empty[i]:
				result[5*i:5*(i+1)] = False
			else:
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
		freed_cards = self.freed_cards[2*player:2*(player+1),:]
		result = np.logical_and(_are_not_empty_cards(freed_cards), _are_not_mecabots(freed_cards))
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
					raise Exception(f'All free slots are busy for player {p}')

				# Move card
				self.freed_cards[2*p+new_slot, :] = reg[slot, :]
				reg[slot, :] = 0
				print(f'=== Card {slot} of P{p} was unlinked, moved to {new_slot} - {player_color},{middle_color} - {player_type},{middle_type}')

				# Update state
				is_p_the_main_player = (p == self.misc[0,2])
				if _is_mecabot(self.freed_cards[2*p+new_slot, :]):
					new_status = MAINPL_TO_SWAP_MECABOT if is_p_the_main_player else OTHERP_TO_SWAP_MECABOT
					# Mecabot needs to be the 1st freed card
					if new_slot != 0:
						print('🎁 Plus this card is a mecabot and I need to swap with other free cards 🎁')
						mecabot_copy = self.freed_cards[2*p+new_slot, :].copy()
						self.freed_cards[2*p+new_slot, :] = self.freed_cards[2*p, :]
						self.freed_cards[2*p, :] = mecabot_copy[:]
					else:
						print('🎁 Plus this card is a mecabot 🎁')
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
		slot = (move-35)
		self.freed_cards[2*player + slot, :] = 0 # Just cancel the card for now

		# Shift other freed card if any
		if slot == 0 and not _is_empty_card(self.freed_cards[2*player+1, :]):
			self.freed_cards[2*player, :] = self.freed_cards[2*player+1, :]
			self.freed_cards[2*player+1, :] = 0

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
