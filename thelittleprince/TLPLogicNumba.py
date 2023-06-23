import numpy as np
from numba import njit
import numba
import random

############################## BOARD DESCRIPTION ##############################
# Board is described by a 55x15 array (1st dim is larger with 4+ players).
# Each card is described by a line 1x15, each of 15 attributes are listed
# below(see "List of attributes"). Here is the description of each line of
# the board. For readibility, we defined "shortcuts" that actually are views
# (numpy name) of overal board.
##### Index  Shortcut              	Meaning
#####   0    self.round_and_state  	Round number on row 0, current player in row 1, bitfield of who can play on row 2, and rows 3-12 are bitfield representing remaining cards
#####  1-3   self.market      		Cards ready to be picked by players
#####  4-6   self.players_score		Score for each player, attribute by attribute
#####  7-54  self.players_cards		Description of 16 Player0 cards, then 16 Player1 cards
# Indexes above are assuming 3 players, you can have more details in copy_state().

############################## ACTION DESCRIPTION #############################
# There are n*n actions (n being nb of players). Here is description of each action:
##### Index  Meaning
#####   0    Take card 0, and designate player 0 as next player
#####   1    Take card 0, and designate player 1 as next player
#####  ...
#####   n    Take card 1, and designate player 0 as next player
#####  n+1   Take card 1, and designate player 1 as next player
#####  ...
#####  n*n-1 Take card n-1, and designate player n-1 as next player
# Next player can be current player, when selecting last card. Type of card is
# randomly chosen by logic not by players.

# REPEATABLE_RANDOM like in splendor/SplendorLogicNumba.py doesn't seem to help
# for this game, probably because too much randomness...

@njit(cache=True, fastmath=True, nogil=True)
def observation_size(num_players):
	return (18*num_players + 1, 15) # 2nd dimension is card attributes (like fox, sunset, ...)

@njit(cache=True, fastmath=True, nogil=True)
def action_size(num_players):
	return num_players*num_players

@njit(cache=True, fastmath=True, nogil=True)
def my_random_choice_and_normalize(prob):
	normalized_prob = prob / prob.sum()
	result = np.searchsorted(np.cumsum(normalized_prob), np.random.random(), side="right")
	return result

mask = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint8)

@njit(cache=True, fastmath=True, nogil=True)
def my_packbits(array):
	product = np.multiply(array.astype(np.uint8), mask[:len(array)])
	return product.sum()

@njit(cache=True, fastmath=True, nogil=True)
def my_unpackbits(value):
	return (np.bitwise_and(value, mask) != 0).astype(np.uint8)

@njit(cache=True, fastmath=True, nogil=True)
def slots_in_planet(card_type):
	if   card_type == EMPTY:
		raise Exception('you cannot take an empty card')
	elif card_type == CENTER:
		possible_slots = [5, 6, 9, 10]
	elif card_type == UPHILL_EDGE:
		possible_slots = [1, 7, 8, 14]
	elif card_type == DOWNHILL_EDGE:
		possible_slots = [2, 4, 11, 13]
	else: # >= CORNER
		possible_slots = [0, 3, 12, 15]
	return possible_slots

spec = [
	('num_players'         , numba.int8),
	('current_player_index', numba.int8),

	('state'            , numba.int8[:,:]),
	('round_and_state'  , numba.int8[:]),
	('market'           , numba.int8[:,:]),
	('players_score'    , numba.int8[:,:]),
	('players_cards'    , numba.int8[:,:]),
]
@numba.experimental.jitclass(spec)
class Board():
	def __init__(self, num_players):
		self.num_players = num_players
		self.current_player_index = 0
		self.state = np.zeros(observation_size(self.num_players), dtype=np.int8)
		self.init_game()

	def get_score(self, player):
		return self.players_score[player, :].sum()

	def init_game(self):
		self.copy_state(np.zeros(observation_size(self.num_players), dtype=np.int8), copy_or_not=False)

		# Initialize list of players who can play this turn
		self.round_and_state[2] = my_packbits(np.ones(self.num_players, dtype=np.bool_))
		# Initialize available cards
		self.round_and_state[3:13] = my_packbits(np.ones(8, dtype=np.bool_))
		# Initialise market
		self._fill_market_if_needed()
		
	def get_state(self):
		return self.state

	def valid_moves(self, player):
		result = np.zeros(self.num_players*self.num_players, dtype=np.bool_)
		who_can_play = my_unpackbits(self.round_and_state[2])[:self.num_players]
		who_can_play[player] = False
		if not np.any(who_can_play): # means end of turn, so current play will play again
			who_can_play[player] = True
		can_be_picked = (self.market[:, CARD_TYPE] != EMPTY)
		for p in range(self.num_players):
			if who_can_play[p]:
				for i in range(self.num_players):
					if can_be_picked[i]:
						result[i*self.num_players + p] = True
		return result

	def make_move(self, move, player, deterministic):
		card_to_take, player_delta = divmod(move, self.num_players)
		next_player = (player + player_delta) % self.num_players
		self._take_card(card_to_take, player)
		self._update_score(player)
		self._fill_market_if_needed()
		self._player_cant_play_again_this_turn(player)

		self.round_and_state[0] += 1
		self.round_and_state[1] = next_player
		return next_player

	def copy_state(self, state, copy_or_not):
		if self.state is state and not copy_or_not:
			return
		self.state = state.copy() if copy_or_not else state
		n = self.num_players
		self.round_and_state = self.state[0           ,:] # 1    # Round number on row 0, current player in row 1, bitfield of who can play on row 2, and rows 3-12 are bitfield representing remaining cards
		self.market          = self.state[1    :n+1   ,:] # n    # Cards ready to be picked by players
		self.players_score   = self.state[n+1  :2*n+1 ,:] # n    # Current player score, attribute by attribute
		self.players_cards   = self.state[2*n+1:18*n+1,:] # n*16 # Players' cards: P0-c0, P0-c1, ..., P0-c15, P1-c0, P1-c1, ..., Pn-C15

	def check_end_game(self):
		if self.get_round() < 16 * self.num_players:
			return np.full(self.num_players, 0., dtype=np.float32)
		
		scores = np.array([self.get_score(p) for p in range(self.num_players)], dtype=np.int8)
		score_max = scores.max()
		single_winner = ((scores == score_max).sum() == 1)
		winners = [(1. if single_winner else 0.01) if s == score_max else -1. for s in scores]
		return np.array(winners, dtype=np.float32)

	# if n=1, transform P0 to Pn, P1 to P0, ... and Pn to Pn-1
	# else do this action n times
	def swap_players(self, nb_swaps):
		def _roll_in_place_axis0(array, shift):
			tmp_copy = array.copy()
			size0 = array.shape[0]
			for i in range(size0):
				array[i,:] = tmp_copy[(i+shift)%size0,:]
		_roll_in_place_axis0(self.players_score, 1 *nb_swaps)
		_roll_in_place_axis0(self.players_cards, 16*nb_swaps)
		# Update current player
		self.round_and_state[1] = (self.round_and_state[1] - nb_swaps + self.num_players) % self.num_players
		# Update list of players who can play
		who_can_play = my_unpackbits(self.round_and_state[2])[:self.num_players]
		self.round_and_state[2] = my_packbits(np.roll(who_can_play, -nb_swaps))

	def get_symmetries(self, policy, valid_actions):
		n = self.num_players

		# permute randomly players listed in list 'players'
		def _permute_players(players, input_state, input_pi, input_v):
			np_players, shuffled_players = np.array(players), np.array(players)
			np.random.shuffle(shuffled_players)
			# use similar code to copy_state()
			return_state = input_state.copy()
			input_round_and_state , input_score , input_cards  = input_state [0,:], input_state [n+1:2*n+1,:], input_state [2*n+1:18*n+1,:]
			return_round_and_state, return_score, return_cards = return_state[0,:], return_state[n+1:2*n+1,:], return_state[2*n+1:18*n+1,:]
			who_can_play = my_unpackbits(input_round_and_state[2])[:n]
			return_who_can_play = who_can_play.copy()
			# Permute input_state (score, cards and round_and_state)
			for i in range(np_players.size):
				old_player, new_player = np_players[i], shuffled_players[i]
				return_score[new_player] = input_score[old_player]
				return_cards[new_player*16:(new_player+1)*16, :] = input_cards[old_player*16:(old_player+1)*16, :]
				return_who_can_play[new_player] = who_can_play[old_player]
			return_round_and_state[2] = my_packbits(return_who_can_play)
			
			# Permute policy and valid actions
			return_pi, return_v = input_pi.copy(), input_v.copy()
			for i in range(np_players.size):
				old_player, new_player = np_players[i], shuffled_players[i]
				for card in range(n):
					old_index_action, new_index_action = card * n + old_player, card * n + new_player
					return_pi[new_index_action] = input_pi[old_index_action]
					return_v[new_index_action]  = input_v[old_index_action]

			return return_state, return_pi, return_v

		# permute randomly market cards listed in list 'market_cards'
		def _permute_cards_market(market_cards, input_state, input_pi, input_v):
			np_market_cards, shuffled_market_cards = np.array(market_cards), np.array(market_cards)
			np.random.shuffle(shuffled_market_cards)
			return_state, return_pi, return_v = input_state.copy(), input_pi.copy(), input_v.copy()
			# use similar code to copy_state()
			input_market  = input_state [1:n+1,:]
			return_market = return_state[1:n+1,:]
			for i in range(np_market_cards.size):
				old_card, new_card = np_market_cards[i], shuffled_market_cards[i]
				return_market[new_card, :] = input_market[old_card, :]
				for player in range(n):
					old_index_action, new_index_action = old_card * n + player, new_card * n + player
					return_pi[new_index_action] = input_pi[old_index_action]
					return_v[new_index_action]  = input_v[old_index_action]
			return return_state, input_pi, input_v

		# permute randomly cards listed in list 'planet_cards' in planet of player 'player'
		def _permute_cards_planet(planet_cards, player, input_state, input_pi, input_v):
			np_planet_cards, shuffled_planet_cards = np.array(planet_cards), np.array(planet_cards)
			np.random.shuffle(shuffled_planet_cards)
			return_state = input_state.copy()
			# use similar code to copy_state()
			input_cards  = input_state [2*n+1:18*n+1,:]
			return_cards = return_state[2*n+1:18*n+1,:]
			for i in range(np_planet_cards.size):
				old_card, new_card = np_planet_cards[i], shuffled_planet_cards[i]
				return_cards[16*player + new_card, :] = input_cards[16*player + old_card, :]
			return return_state, input_pi, input_v

		def _add_to_list_no_duplicate(s, p, v, list_):
			for s_, p_, v_ in list_:
				if np.array_equal(s_, s): # we should compare p and v too, but if state is same, then policy+valids should be same
					return False
			list_.append((s, p, v))
			return True

		symmetries = [(self.state.copy(), policy.copy(), valid_actions.copy())]

		# Permute players (those who played this turn on one hand, those who hasn't played yet on another hand)
		current_player = self.round_and_state[1]
		who_can_play = my_unpackbits(self.round_and_state[2])[:self.num_players]
		players_who_played        = [ i for i in range(self.num_players) if not who_can_play[i] and i != current_player ]
		players_who_havent_played = [ i for i in range(self.num_players) if     who_can_play[i] and i != current_player ]
		for i in range(self.num_players): # arbitary number of symmetries
			new_state, new_policy, new_valids = _permute_players(players_who_played       , self.state, policy, valid_actions)
			new_state, new_policy, new_valids = _permute_players(players_who_havent_played, new_state, new_policy, new_valids)
			_add_to_list_no_duplicate(new_state, new_policy, new_valids, symmetries)
		
		# Permute remaining cards in market and cards in planet
		for i in range(self.num_players): # arbitary number of symmetries
			list_cards_market = [i for i in range(self.num_players) if self.market[i, CARD_TYPE] != EMPTY]
			new_state, new_policy, new_valids = _permute_cards_market(list_cards_market, self.state, policy, valid_actions)
			for player in range(self.num_players):
				for card_type in range(1, 5):
					list_of_cards_in_planet = [i for i in range(16) if self.players_cards[16*player + i, CARD_TYPE]//25 == card_type]
					new_state, new_policy, new_valids = _permute_cards_planet(list_of_cards_in_planet, player, new_state, new_policy, new_valids)
			_add_to_list_no_duplicate(new_state, new_policy, new_valids, symmetries)

		return symmetries

	def get_round(self):
		return self.round_and_state[0]

	def _take_card(self, i, p):
		# decide slot in current player planet
		best_slot = -1
		for slot in slots_in_planet(self.market[i, CARD_TYPE]):
			if self.players_cards[16*p + slot, CARD_TYPE] == EMPTY:
				best_slot = 16*p + slot
				break
		# take card now
		self.players_cards[best_slot, :] = self.market[i, :]
		self.market[i, :] = 0

		# Put cards face down if more 3 baobabs
		if self.players_cards[16*p:16*(p+1), BAOBAB].sum() >= 3:
			for card in range(16):
				if self.players_cards[16*p+card, BAOBAB] >= 1:
					self.players_cards[16*p+card, :CARD_TYPE] = 0
					self.players_cards[16*p+card, FACE_DOWN] = 1


	def _update_score(self, p):
		def _compute_score(character, sum_attributes):
			if   character == NONE:
				return
			elif character == VAIN_MAN:
				self.players_score[p, SNAKE] += 4*sum_attributes[SNAKE]
			elif character == GEOGRAPHER:
				for card in range(16):
					if card not in slots_in_planet(CORNER) and self.players_cards[16*p + card, VOLCANO] == 0:
						self.players_score[p, VOLCANO] += 1
			elif character == ASTRONOMER:
				self.players_score[p, SUNSET] += 2*sum_attributes[SUNSET]
			elif character == KING:
				roses_score = [0, 14, 7, 0]
				self.players_score[p, ROSE] += roses_score[ min(sum_attributes[ROSE], 3) ]
			elif character == LAMPLIGHTER:
				self.players_score[p, LAMPPOST] += sum_attributes[LAMPPOST]
			elif character == HUNTER:
				self.players_score[p, SNAKE]    += 3 if sum_attributes[SNAKE   ]>0 else 0
				self.players_score[p, ELEPHANT] += 3 if sum_attributes[ELEPHANT]>0 else 0
				# Give 3 points if any sheep specy exist, but not for each of them
				if   sum_attributes[SHEEP_WHITE]>0:
					self.players_score[p, SHEEP_WHITE] += 3
				elif  sum_attributes[SHEEP_GREY]>0:
					self.players_score[p, SHEEP_GREY] += 3
				elif sum_attributes[SHEEP_BROWN]>0:
					self.players_score[p, SHEEP_BROWN] += 3
			elif character == DRUNKARD:
				self.players_score[p, BAOBAB] += 3*sum_attributes[FACE_DOWN]
			elif character == BUSINESSMAN_W:
				self.players_score[p, SHEEP_WHITE] += 2*sum_attributes[SHEEP_WHITE]
			elif character == BUSINESSMAN_G:
				self.players_score[p, SHEEP_GREY]  += 3*sum_attributes[SHEEP_GREY]
			elif character == BUSINESSMAN_B:
				self.players_score[p, SHEEP_BROWN] += 5*sum_attributes[SHEEP_BROWN]
			elif character == GARDENER:
				self.players_score[p, BAOBAB] += 7*sum_attributes[BAOBAB]
			elif character == TURKISH:
				self.players_score[p, BIG_STAR] += sum_attributes[BIG_STAR]
			elif character == LITTLE_PRINCE:
				if   sum_attributes[SHEEP_WHITE]>0:
					self.players_score[p, SHEEP_WHITE] += 3
				if  sum_attributes[SHEEP_GREY]>0:
					self.players_score[p, SHEEP_GREY] += 3
				if sum_attributes[SHEEP_BROWN]>0:
					self.players_score[p, SHEEP_BROWN] += 3
				self.players_score[p, BOX] += sum_attributes[BOX]
			else:
				print('Unknown character ' + str(character))

			# Volcanoes
			nb_volcanoes = [self.players_cards[16*p_:16*(p_+1), VOLCANO].sum() for p_ in range(self.num_players)]
			max_volcanoes = max(nb_volcanoes)
			for p_ in range(self.num_players):
				# Ugly to write in FACE_DOWN row, but couldn't find proper way without impacting processing time
				self.players_score[p_, FACE_DOWN] = -max_volcanoes if nb_volcanoes[p_] == max_volcanoes else 0

		sum_attributes = self.players_cards[16*p:16*(p+1), :].sum(axis=0)
		self.players_score[p, :] = 0
		for character_slot in slots_in_planet(CORNER):
			card_type = self.players_cards[16*p + character_slot, CARD_TYPE]
			character = max(card_type - CORNER, 0)
			_compute_score(character, sum_attributes)

	def _fill_market_if_needed(self):
		if np.any(self.market[:, CARD_TYPE] != EMPTY) or np.all(self.players_cards[:, CARD_TYPE] > 0):
			return
		# Market is empty, need to refill it. First, chose randomly one of 4 categories of cards
		type_with_room_player0 = [
			self.players_cards[10, CARD_TYPE] == EMPTY, # Does player 0 has last card of type CENTER
			self.players_cards[14, CARD_TYPE] == EMPTY, # Does player 0 has last card of type UPHILL_EDGE
			self.players_cards[13, CARD_TYPE] == EMPTY, # Does player 0 has last card of type DOWNHILL_EDGE
			self.players_cards[15, CARD_TYPE] == EMPTY, # Does player 0 has last card of type CORNER
		]
		# the simpler code below is not supported by numba
		#type_with_room_player0 = [self.players_cards[location, CARD_TYPE] == EMPTY for location in [10,14,13,15]]
		card_type = my_random_choice_and_normalize(np.array(type_with_room_player0))
		available_cards = self._available_cards()
		# Chose randomly cards among these categories
		for i in range(self.num_players):
			card_index = my_random_choice_and_normalize(available_cards[20*card_type:20*(card_type+1)])
			self.market[i, :] = np_all_cards[card_type][card_index, :]
			available_cards[20*card_type + card_index] = False
		self._set_available_cards(available_cards)
		# Reset list of players who played during this turn
		self.round_and_state[2] = my_packbits(np.ones(self.num_players, dtype=np.bool_))

	def _available_cards(self):
		available_or_not = np.zeros(80, dtype=np.bool_)
		# Available cards are stored in rows 3-12
		for i in range(10):
			available_or_not[8*i:8*(i+1)] = my_unpackbits(self.round_and_state[i+3])
		return available_or_not

	def _set_available_cards(self, available_cards):
		# Available cards are stored in rows 3-12
		for i in range(10):
			self.round_and_state[i+3] = my_packbits(available_cards[8*i:8*(i+1)])

	def _player_cant_play_again_this_turn(self, player):
		who_can_play = my_unpackbits(self.round_and_state[2])[:self.num_players]
		who_can_play[player] = False
		self.round_and_state[2] = my_packbits(who_can_play)


# List of attributes
FACE_DOWN   = 0
BAOBAB      = 1
VOLCANO     = 2
SUNSET      = 3
ROSE        = 4
LAMPPOST    = 5
BOX         = 6
BIG_STAR    = 7
FOX         = 8
ELEPHANT    = 9
SNAKE       = 10
SHEEP_WHITE = 11
SHEEP_GREY  = 12
SHEEP_BROWN = 13
CARD_TYPE   = 14

# List of card types
EMPTY         = 0 * 25
CENTER        = 1 * 25
UPHILL_EDGE   = 2 * 25
DOWNHILL_EDGE = 3 * 25
CORNER        = 4 * 25

# List of characters
NONE           = 0
VAIN_MAN       = 1
GEOGRAPHER     = 2
ASTRONOMER     = 3
KING           = 4
LAMPLIGHTER    = 5
HUNTER         = 6
DRUNKARD       = 7
BUSINESSMAN_W  = 8
BUSINESSMAN_G  = 9
BUSINESSMAN_B  = 10
GARDENER       = 11
TURKISH        = 12
LITTLE_PRINCE  = 13

#       DOWN BAOB VOLC SUNS ROSE LAMP BOX  STAR FOX  ELEP SNAK SH_W SH_G SH_B TYPE
all_cards = [
	[
		[0  , 1  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 2  , 0  , 1  , CENTER ],
		[0  , 1  , 0  , 0  , 0  , 0  , 0  , 0  , 1  , 0  , 0  , 2  , 0  , 0  , CENTER ],
		[0  , 1  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 2  , 0  , CENTER ],
		[0  , 0  , 1  , 0  , 0  , 2  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , CENTER ],
		[0  , 0  , 1  , 0  , 0  , 1  , 0  , 0  , 1  , 0  , 0  , 1  , 0  , 0  , CENTER ],
		[0  , 0  , 1  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , 0  , 1  , 0  , CENTER ],
		[0  , 0  , 1  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , 1  , 1  , 0  , 0  , CENTER ],
		[0  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 1  , 0  , 0  , 0  , CENTER ],
		[0  , 0  , 0  , 0  , 0  , 3  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , CENTER ],
		[0  , 0  , 0  , 0  , 1  , 2  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , CENTER ],
		[0  , 0  , 0  , 0  , 0  , 2  , 1  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , CENTER ],
		[0  , 0  , 0  , 0  , 0  , 2  , 0  , 0  , 0  , 0  , 1  , 0  , 0  , 0  , CENTER ],
		[0  , 0  , 0  , 0  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , 2  , 0  , 0  , CENTER ],
		[0  , 0  , 0  , 0  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , 1  , 0  , 0  , CENTER ],
		[0  , 0  , 0  , 0  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , 1  , 0  , CENTER ],
		[0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 1  , 0  , 0  , CENTER ],
		[0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , CENTER ],

		[0  , 0  , 1  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , 0  , 1  , 0  , CENTER ],
		[0  , 0  , 1  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , 1  , 1  , 0  , 0  , CENTER ],
		[0  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 1  , 0  , 0  , 0  , CENTER ],
	],
	[
		[0  , 1  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , UPHILL_EDGE ],
		[0  , 1  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 1  , 0  , UPHILL_EDGE ],
		[0  , 1  , 0  , 0  , 1  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , 0  , UPHILL_EDGE ],
		[0  , 1  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 1  , 1  , UPHILL_EDGE ],
		[0  , 1  , 0  , 0  , 0  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , 1  , UPHILL_EDGE ],
		[0  , 0  , 1  , 0  , 0  , 2  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , 0  , UPHILL_EDGE ],
		[0  , 0  , 1  , 1  , 0  , 1  , 0  , 0  , 1  , 0  , 0  , 0  , 1  , 0  , UPHILL_EDGE ],
		[0  , 0  , 1  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , 1  , 1  , 0  , UPHILL_EDGE ],
		[0  , 0  , 1  , 1  , 0  , 1  , 0  , 0  , 0  , 1  , 0  , 1  , 0  , 0  , UPHILL_EDGE ],
		[0  , 0  , 1  , 1  , 0  , 1  , 0  , 0  , 0  , 0  , 1  , 1  , 0  , 0  , UPHILL_EDGE ],
		[0  , 0  , 1  , 0  , 0  , 0  , 0  , 2  , 0  , 0  , 0  , 0  , 0  , 0  , UPHILL_EDGE ],
		[0  , 0  , 1  , 0  , 0  , 0  , 0  , 2  , 0  , 0  , 0  , 0  , 1  , 0  , UPHILL_EDGE ],
		[0  , 0  , 0  , 0  , 0  , 3  , 0  , 0  , 0  , 0  , 1  , 0  , 0  , 0  , UPHILL_EDGE ],
		[0  , 0  , 0  , 1  , 1  , 2  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , UPHILL_EDGE ],
		[0  , 0  , 0  , 1  , 0  , 2  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , UPHILL_EDGE ],
		[0  , 0  , 0  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , 2  , 0  , 0  , UPHILL_EDGE ],
		[0  , 0  , 0  , 0  , 0  , 1  , 2  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , UPHILL_EDGE ],
		[0  , 0  , 0  , 0  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , 1  , 0  , 0  , UPHILL_EDGE ],
		[0  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , UPHILL_EDGE ],
		[0  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , UPHILL_EDGE ],
	],
	[
		[0  , 1  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 1  , 0  , DOWNHILL_EDGE ],
		[0  , 1  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , 0  , 1  , 0  , 0  , 0  , DOWNHILL_EDGE ],
		[0  , 1  , 0  , 0  , 1  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , 0  , DOWNHILL_EDGE ],
		[0  , 1  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 1  , 1  , DOWNHILL_EDGE ],
		[0  , 1  , 0  , 0  , 0  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , 1  , DOWNHILL_EDGE ],
		[0  , 0  , 1  , 0  , 0  , 2  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , 0  , DOWNHILL_EDGE ],
		[0  , 0  , 1  , 1  , 0  , 1  , 0  , 0  , 1  , 0  , 0  , 0  , 1  , 0  , DOWNHILL_EDGE ],
		[0  , 0  , 1  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , 1  , 1  , 0  , DOWNHILL_EDGE ],
		[0  , 0  , 1  , 1  , 0  , 1  , 0  , 0  , 0  , 1  , 0  , 1  , 0  , 0  , DOWNHILL_EDGE ],
		[0  , 0  , 1  , 1  , 0  , 1  , 0  , 0  , 0  , 0  , 1  , 1  , 0  , 0  , DOWNHILL_EDGE ],
		[0  , 0  , 1  , 0  , 0  , 0  , 0  , 2  , 0  , 0  , 0  , 0  , 0  , 0  , DOWNHILL_EDGE ],
		[0  , 0  , 1  , 0  , 0  , 0  , 0  , 2  , 0  , 0  , 0  , 0  , 1  , 0  , DOWNHILL_EDGE ],
		[0  , 0  , 0  , 0  , 0  , 3  , 0  , 0  , 0  , 0  , 1  , 0  , 0  , 0  , DOWNHILL_EDGE ],
		[0  , 0  , 0  , 1  , 1  , 2  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , DOWNHILL_EDGE ],
		[0  , 0  , 0  , 1  , 0  , 2  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , DOWNHILL_EDGE ],
		[0  , 0  , 0  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , 2  , 0  , 0  , DOWNHILL_EDGE ],
		[0  , 0  , 0  , 0  , 0  , 1  , 2  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , DOWNHILL_EDGE ],
		[0  , 0  , 0  , 0  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , 1  , 0  , 0  , DOWNHILL_EDGE ],
		[0  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , DOWNHILL_EDGE ],
		[0  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , DOWNHILL_EDGE ],
	],
	[
		[0  , 0  , 0  , 0  , 0  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , 0  , CORNER + VAIN_MAN ],
		[0  , 0  , 0  , 0  , 0  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , 0  , CORNER + VAIN_MAN ],
		[0  , 0  , 0  , 0  , 0  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , 0  , CORNER + GEOGRAPHER ],
		[0  , 0  , 0  , 0  , 0  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , 0  , CORNER + GEOGRAPHER ],
		[0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , CORNER + ASTRONOMER ],
		[0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , CORNER + ASTRONOMER ],
		[0  , 0  , 0  , 0  , 0  , 0  , 0  , 2  , 0  , 0  , 0  , 0  , 0  , 0  , CORNER + KING ],
		[0  , 0  , 0  , 0  , 0  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , 0  , CORNER + LAMPLIGHTER ],
		[0  , 0  , 0  , 0  , 0  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , 0  , CORNER + LAMPLIGHTER ],
		[0  , 0  , 0  , 0  , 0  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , 0  , CORNER + HUNTER ],
		[0  , 0  , 0  , 0  , 0  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , 0  , CORNER + HUNTER ],
		[0  , 0  , 0  , 0  , 0  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , 0  , CORNER + DRUNKARD ],
		[0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , CORNER + BUSINESSMAN_W ],
		[0  , 0  , 0  , 0  , 0  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , 0  , CORNER + BUSINESSMAN_G ],
		[0  , 0  , 0  , 0  , 0  , 0  , 0  , 2  , 0  , 0  , 0  , 0  , 0  , 0  , CORNER + BUSINESSMAN_B ],
		[0  , 0  , 0  , 0  , 0  , 0  , 0  , 2  , 0  , 0  , 0  , 0  , 0  , 0  , CORNER + GARDENER ],
		[0  , 0  , 0  , 0  , 0  , 0  , 0  , 2  , 0  , 0  , 0  , 0  , 0  , 0  , CORNER + GARDENER ],
		[0  , 0  , 0  , 0  , 0  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , 0  , CORNER + TURKISH ],
		[0  , 0  , 0  , 0  , 0  , 0  , 0  , 1  , 0  , 0  , 0  , 0  , 0  , 0  , CORNER + TURKISH ],
		[0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , CORNER + LITTLE_PRINCE ],
	],
]
#       DOWN BAOB VOLC SUNS ROSE LAMP BOX  STAR FOX  ELEP SNAK SH_W SH_G SH_B TYPE

np_all_cards = np.array(all_cards, dtype=np.int8)
