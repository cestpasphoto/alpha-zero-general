import numpy as np
from numba import njit
import numba
import random
from Stochastic import hashed_draw

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
# (note that definition of next player is relative to current player)
##### Index  Meaning
#####   0    Take card 0, and designate current player      as next player
#####   1    Take card 0, and designate player to the right as next player
#####  ...
#####  n-1   Take card 0, and designate player to the left  as next player
#####  ...
#####   n    Take card 1, and designate current player      as next player
#####  n+1   Take card 1, and designate player to the right as next player
#####  ...
#####  n*n-1 Take card n-1  & designate player to the left  as next player
#####  n*n+t Choose stack t (0=center, 1=uphill edge, 2=downhill edge, 3=character)
#
# OFFICIAL TURN STRUCTURE
#   1. The first player of the turn CHOOSES a stack (a type he still has room
#      for), then n cards of that type are drawn (chance) into the market.
#   2. He takes one card and designates who plays next among those who have not
#      played this turn; and so on.
#   3. The last player takes the last card and designates himself: he opens the
#      next turn by choosing a stack.
# Every turn gives one card of the same type to each player, so all planets
# always have the same number of cards per type: room is the same for everyone.
# The choose-stack phase is identified by an empty market (no extra state).
# A decision with a single legal option is never exposed (make_move resolves
# it): the last card of a turn, and the stack choice once one type is left.

# Chance (market refill) follows the framework contract: random_seed == 0 ->
# np.random, otherwise a deterministic hashed_draw of (seed, round, type, slot).

@njit(cache=True, fastmath=True, nogil=True)
def observation_size(num_players):
	return (18*num_players + 1, 15) # 2nd dimension is card attributes (like fox, sunset, ...)

@njit(cache=True, fastmath=True, nogil=True)
def action_size(num_players):
	return num_players*num_players + NB_CARD_TYPES

mask = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint8)

@njit(cache=True, fastmath=True, nogil=True)
def my_packbits(array):
	product = np.multiply(array.astype(np.uint8), mask[:len(array)])
	return product.sum()

@njit(cache=True, fastmath=True, nogil=True)
def my_unpackbits(value):
	return (np.bitwise_and(value, mask) != 0).astype(np.uint8)

@njit(cache=True, fastmath=True, nogil=True)
def uint_to_int8(x):
	x_ = np.int16(x)
	return np.int8(x_-256 if x_ > 127 else x_)

@njit(cache=True, fastmath=True, nogil=True)
def int8_to_uint(x):
	# reinterpret the int8 byte as uint8 (two's complement): -1 -> 255.
	# The former np.uint64(x) + 256 mixed uint64 and int64, which numba types as float64.
	return np.uint8(np.int16(x) & 0xFF)

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
		self.round_and_state[2] = uint_to_int8(uint_to_int8(my_packbits(np.ones(self.num_players, dtype=np.bool_))))
		# Initialize available cards
		self.round_and_state[3:13] = uint_to_int8(uint_to_int8(my_packbits(np.ones(8, dtype=np.bool_))))
		# Market stays empty: player 0 opens the game by choosing a stack

	def get_state(self):
		return self.state

	def valid_moves(self, player):
		n = self.num_players
		result = np.zeros(action_size(n), dtype=np.bool_)
		if self._market_is_empty(): # start of turn: choose a stack
			room = self._types_with_room(player)
			for t in range(NB_CARD_TYPES):
				result[n*n + t] = room[t]
			return result
		who_can_play = my_unpackbits(int8_to_uint(self.round_and_state[2]))[:self.num_players]
		who_can_play[player] = False
		if not np.any(who_can_play): # means end of turn, so current play will play again
			who_can_play[player] = True
		can_be_picked = (self.market[:, CARD_TYPE] != EMPTY)
		for pdelta in range(self.num_players):
			p = (player + pdelta) % self.num_players
			if who_can_play[p]:
				for i in range(self.num_players):
					if can_be_picked[i]:
						result[i*self.num_players + pdelta] = True
		return result

	def make_move(self, move, player, random_seed):
		next_player = self._apply(move, player, random_seed)
		# Never expose a decision with a single legal option: resolving it here
		# saves a full MCTS search (last card of each turn, last stack type)
		for _guard in range(4 * 16 * self.num_players):
			if self.get_round() >= 16 * self.num_players:
				break
			valids = self.valid_moves(next_player)
			if valids.sum() != 1:
				break
			next_player = self._apply(np.flatnonzero(valids)[0], next_player, random_seed)
		return next_player

	def _apply(self, move, player, random_seed):
		n = self.num_players
		if move >= n*n:             # choose a stack: the market is filled, same player then picks
			self._fill_market(move - n*n, random_seed)
			next_player = player
		else:
			card_to_take, player_delta = divmod(move, n)
			next_player = (player + player_delta) % n
			self._take_card(card_to_take, player)
			self._update_score(player)
			self._player_cant_play_again_this_turn(player)
			self.round_and_state[0] += 1
			# Last card of the turn: everybody can play again, the last taker
			# (who designated himself) opens the next turn
			if self._market_is_empty() and not np.all(self.players_cards[:, CARD_TYPE] != EMPTY):
				self.round_and_state[2] = uint_to_int8(my_packbits(np.ones(n, dtype=np.bool_)))
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
		
		scores = np.array([self.get_score(p) for p in range(self.num_players)], dtype=np.int32) # totals can exceed 127
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
		who_can_play = my_unpackbits(int8_to_uint(self.round_and_state[2]))[:self.num_players]
		self.round_and_state[2] = uint_to_int8(my_packbits(np.roll(who_can_play, -nb_swaps)))

	def get_symmetries(self, policy, valid_actions):
		# No player permutation: the per-player value/Q targets are rolled by Coach
		# with the seat of the mover only, a permutation of the opponents would
		# mislabel their values. Only card permutations, which leave values unchanged.
		n = self.num_players

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
			return return_state, return_pi, return_v

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
				for card in range(16): # placed non-corner cards without volcano (empty slots do not count)
					if card not in slots_in_planet(CORNER) and self.players_cards[16*p + card, CARD_TYPE] != EMPTY and self.players_cards[16*p + card, VOLCANO] == 0:
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

		sum_attributes = self.players_cards[16*p:16*(p+1), :].sum(axis=0)
		self.players_score[p, :] = 0
		for character_slot in slots_in_planet(CORNER):
			card_type = self.players_cards[16*p + character_slot, CARD_TYPE]
			character = max(card_type - CORNER, 0)
			_compute_score(character, sum_attributes)

		# Volcanoes: the penalty depends on ALL planets, so refresh it for every
		# player at every move (it was skipped while the mover had no character).
		# Ugly to write in FACE_DOWN column, but couldn't find proper way without impacting processing time
		nb_volcanoes = [self.players_cards[16*p_:16*(p_+1), VOLCANO].sum() for p_ in range(self.num_players)]
		max_volcanoes = max(nb_volcanoes)
		for p_ in range(self.num_players):
			self.players_score[p_, FACE_DOWN] = -max_volcanoes if nb_volcanoes[p_] == max_volcanoes else 0

	def _market_is_empty(self):
		return np.all(self.market[:, CARD_TYPE] == EMPTY)

	def _types_with_room(self, player):
		# a type has room iff the LAST slot of that type is still empty (slots are
		# filled in slots_in_planet() order); identical for all players
		room = np.zeros(NB_CARD_TYPES, dtype=np.bool_)
		for t in range(NB_CARD_TYPES):
			room[t] = (self.players_cards[16*player + LAST_SLOT[t], CARD_TYPE] == EMPTY)
		return room

	def _fill_market(self, card_type, random_seed):
		# Draw num_players cards of the chosen type. Integer draw on the list of
		# available cards: exact, never lands on an unavailable card (the former
		# float cumsum/searchsorted could, with probability ~1e-16, return an
		# out-of-range index since cumsum(1/k)[-1] < 1 for some k).
		available_cards = self._available_cards()
		# counter: one domain per (round, type), one slot per drawn card (n <= 8)
		base_ctr = (np.int64(np.uint8(self.round_and_state[0])) * NB_CARD_TYPES + card_type) * 8
		for i in range(self.num_players):
			candidates = np.flatnonzero(available_cards[20*card_type:20*(card_type+1)])
			if random_seed == 0:   # np.int64 on both branches, or numba unifies int64/uint64 into float64
				k = np.int64(np.random.randint(0, candidates.size))
			else:
				k = np.int64(hashed_draw(random_seed, base_ctr + i, candidates.size))
			card_index = candidates[k]
			self.market[i, :] = np_all_cards[card_type][card_index, :]
			available_cards[20*card_type + card_index] = False
		self._set_available_cards(available_cards)

	def _available_cards(self):
		available_or_not = np.zeros(80, dtype=np.bool_)
		# Available cards are stored in rows 3-12
		for i in range(10):
			available_or_not[8*i:8*(i+1)] = my_unpackbits(int8_to_uint(self.round_and_state[i+3]))
		return available_or_not

	def _set_available_cards(self, available_cards):
		# Available cards are stored in rows 3-12
		for i in range(10):
			self.round_and_state[i+3] = uint_to_int8(my_packbits(available_cards[8*i:8*(i+1)]))

	def _player_cant_play_again_this_turn(self, player):
		who_can_play = my_unpackbits(int8_to_uint(self.round_and_state[2]))[:self.num_players]
		who_can_play[player] = False
		self.round_and_state[2] = uint_to_int8(my_packbits(who_can_play))


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
NB_CARD_TYPES = 4         # stacks: center, uphill edge, downhill edge, corner (character)
# Last slot of each type in slots_in_planet() order: empty iff the type still has room
LAST_SLOT     = np.array([10, 14, 13, 15], dtype=np.int64)

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
