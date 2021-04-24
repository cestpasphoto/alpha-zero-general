#!/home/best/dev/splendor/venv/bin/python3

from .SplendorLogic import np_all_nobles, np_all_cards_1, np_all_cards_2, np_all_cards_3, len_all_cards, np_different_gems_up_to_2, np_different_gems_up_to_3, np_cards_symmetries, np_reserve_symmetries
import numpy as np
from numba import njit
import numba

ENABLE_ACTION_RESERVE  = True
ENABLE_ACTION_GIVEBACK = True

idx_white, idx_blue, idx_green, idx_red, idx_black, idx_gold, idx_points = range(7)
mask = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint8)

@njit(cache=True, fastmath=True, nogil=True)
def observation_size(num_players):
	return (32+12*num_players, 7)

@njit(cache=True, fastmath=True, nogil=True)
def action_size():
	return 81

@njit(cache=True, fastmath=True, nogil=True)
def my_random_choice(prob):
	result = np.searchsorted(np.cumsum(prob), np.random.random(), side="right")
	return result

@njit(cache=True, fastmath=True, nogil=True)
def my_packbits(array):
	product = np.multiply(array.astype(np.uint8), mask[:len(array)])
	return product.sum()

@njit(cache=True, fastmath=True, nogil=True)
def my_unpackbits(value):
	return (np.bitwise_and(value, mask) != 0).astype(np.uint8)

@njit(cache=True, fastmath=True, nogil=True)
def np_all_axis1(x):
	out = np.ones(x.shape[0], dtype=np.bool8)
	for i in range(x.shape[1]):
		out = np.logical_and(out, x[:, i])
	return out


spec = [
	('num_players'         , numba.int8),
	('current_player_index', numba.int8),
	('num_gems_in_play'    , numba.int8),
	('num_nobles'          , numba.int8),
	('max_moves'           , numba.int8),
	('score_win'           , numba.int8),

	('state'           , numba.int8[:,:]),
	('bank'            , numba.int8[:,:]),
	('cards_tiers'     , numba.int8[:,:]),
	('nb_deck_tiers'   , numba.int8[:,:]),
	('nobles'          , numba.int8[:,:]),
	('players_gems'    , numba.int8[:,:]),
	('players_nobles'  , numba.int8[:,:]),
	('players_cards'   , numba.int8[:,:]),
	('players_reserved', numba.int8[:,:]),
]
@numba.experimental.jitclass(spec)
class Board():
	def __init__(self, num_players):
		n = num_players
		self.num_players = n
		self.current_player_index = 0
		self.num_gems_in_play = {2: 4, 3: 5, 4: 7}[n]
		self.num_nobles = {2:3, 3:4, 4:5}[n]
		self.max_moves = 126
		self.score_win = 15
		self.state = np.zeros(observation_size(self.num_players), dtype=np.int8)
		self.init_game()

	def get_score(self, player):
		card_points  = self.players_cards[player, idx_points]
		noble_points = self.players_nobles[player*3:player*3+3, idx_points].sum()
		return card_points + noble_points

	def init_game(self):
		self.copy_state(np.zeros(observation_size(self.num_players), dtype=np.int8), copy_or_not=False)

		# Bank
		self.bank[:] = np.array([[self.num_gems_in_play]*5 + [5, 0]], dtype=np.int8)
		# Decks
		for tier in range(3):
			nb_deck_cards_per_color = len_all_cards[tier]
			# HOW MANY cards per color are in deck of tier 0, pratical for NN
			self.nb_deck_tiers[2*tier,:5] = nb_deck_cards_per_color
			# WHICH cards per color are in deck of tier 0, pratical for logic
			self.nb_deck_tiers[2*tier+1,:5] = my_packbits(np.ones(nb_deck_cards_per_color, dtype=np.int8))
		# Tiers
		for tier in range(3):
			for index in range(4):
				self._fill_new_card(tier, index, False)
		# Nobles
		nobles_indexes = np.random.choice(len(np_all_nobles), size=self.num_nobles, replace=False)
		for i, index in enumerate(nobles_indexes):
			self.nobles[i, :] = np_all_nobles[index]

	def get_state(self):
		return self.state

	def valid_moves(self, player):
		result = np.zeros(81, dtype=np.bool_)
		result[0         :12]            = self._valid_buy(player)
		result[12        :12+15]         = self._valid_reserve(player)
		result[12+15     :12+15+3]       = self._valid_buy_reserve(player)
		result[12+15+3   :12+15+3+30]    = np.concatenate((self._valid_get_gems(player) , self._valid_get_gems_identical(player)))
		result[12+15+3+30:12+15+3+30+20] = np.concatenate((self._valid_give_gems(player), self._valid_give_gems_identical(player)))
		result[80] = True #empty move
		return result

	def make_move(self, move, player, deterministic):
		if   move < 12:
			self._buy(move, player, deterministic)
		elif move < 12+15:
			self._reserve(move-12, player, deterministic)
		elif move < 12+15+3:
			self._buy_reserve(move-12-15, player)
		elif move < 12+15+3+30:
			self._get_gems(move-12-15-3, player)
		elif move < 12+15+3+30+20:
			self._give_gems(move-12-15-3-30, player)
		else:
			pass # empty move
		self.bank[0][idx_points] += 1 # Count number of rounds

	def copy_state(self, state, copy_or_not):
		if self.state is state and not copy_or_not:
			return
		self.state = state.copy() if copy_or_not else state
		n = self.num_players
		self.bank             = self.state[0     :1      ,:]	# 1
		self.cards_tiers      = self.state[1     :25     ,:]	# 2*12
		self.nb_deck_tiers    = self.state[25    :31     ,:]	# 6
		self.nobles           = self.state[31    :32+n   ,:]	# N+1
		self.players_gems     = self.state[32+n  :32+2*n ,:]	# N
		self.players_nobles   = self.state[32+2*n:32+5*n ,:]	# 3*N
		self.players_cards    = self.state[32+5*n:32+6*n ,:]	# N
		self.players_reserved = self.state[32+6*n:32+12*n,:]	# 6*N

	def check_end_game(self):
		if self.bank[0][idx_points] % self.num_players != 0: # Check only when 1st player is about to play
			return False, [False, False]
		
		scores = [self.get_score(p) for p in range(self.num_players)]
		score_max = max(scores)
		end = (score_max >= self.score_win) or (self.bank[0][idx_points] >= self.max_moves)
		winners = [(s == score_max) for s in scores] if end else [False, False]
		return end, winners

	def swap_players(self):
		def _swap(array, half_size):
			tmp_copy = array[:half_size,:].copy()
			array[:half_size,:] = array[half_size:2*half_size,:]
			array[half_size:2*half_size,:] = tmp_copy
		_swap(self.players_gems, 1)
		_swap(self.players_cards, 1)
		_swap(self.players_nobles, 3)
		_swap(self.players_reserved, 6)

	def get_symmetries(self, policy, valid_actions):
		def _swap_cards(cards, permutation):
			full_permutation = [2*p+i for p in permutation for i in range(2)]
			cards_copy = cards.copy()
			for i in range(len(permutation)*2):
				cards[i, :] = cards_copy[full_permutation[i], :]
		def _copy_and_permute(array, permutation, start_index):
			new_array = array.copy()
			for i, p in enumerate(permutation):
				new_array[start_index+i] = array[start_index+p]
			return new_array
		def _copy_and_permute2(array, permutation, start_index, other_start_index):
			new_array = array.copy()
			for i, p in enumerate(permutation):
				new_array[start_index      +i] = array[start_index      +p]
				new_array[other_start_index+i] = array[other_start_index+p]
			return new_array

		symmetries = [(self.state.copy(), policy.copy(), valid_actions.copy())]
		# Permute common cards within same tier
		for tier in range(3):
			for permutation in np_cards_symmetries:
				cards_tiers_backup = self.cards_tiers.copy()
				_swap_cards(self.cards_tiers[8*tier:8*tier+8, :], permutation)
				new_policy = _copy_and_permute2(policy, permutation, 4*tier, 12+4*tier)
				new_valid_actions = _copy_and_permute2(valid_actions, permutation, 4*tier, 12+4*tier)
				symmetries.append((self.state.copy(), new_policy, new_valid_actions))
				self.cards_tiers[:] = cards_tiers_backup
		
		# Permute reserved cards
		for player in range(self.num_players):
			nb_reserved_cards = self._nb_of_reserved_cards(player)
			for permutation in np_reserve_symmetries[nb_reserved_cards]:
				if permutation[0] < 0:
					continue
				players_reserved_backup = self.players_reserved.copy()
				_swap_cards(self.players_reserved[6*player:6*player+6, :], permutation)
				if player == 0:
					new_policy = _copy_and_permute(policy, permutation, 12+15)
					new_valid_actions = _copy_and_permute(valid_actions, permutation, 12+15)
				else:
					new_policy = policy.copy()
					new_valid_actions = valid_actions.copy()
				symmetries.append((self.state.copy(), new_policy, new_valid_actions))
				self.players_reserved[:] = players_reserved_backup

		return symmetries

	def get_round(self):
		return self.bank[0][idx_points]

	def _get_deck_card(self, tier):
		nb_remaining_cards_per_color = self.nb_deck_tiers[2*tier,:5]
		if nb_remaining_cards_per_color.sum() == 0: # no more cards
			return None
		
		# First we chose color randomly, then we pick a card 
		color = my_random_choice(nb_remaining_cards_per_color/nb_remaining_cards_per_color.sum())
		remaining_cards = my_unpackbits(self.nb_deck_tiers[2*tier+1, color])
		card_index = my_random_choice(remaining_cards/remaining_cards.sum())
		# Update internals
		remaining_cards[card_index] = 0
		self.nb_deck_tiers[2*tier+1, color] = my_packbits(remaining_cards)
		self.nb_deck_tiers[2*tier, color] -= 1

		if tier == 0:
			card = np_all_cards_1[color][card_index]
		elif tier == 1:
			card = np_all_cards_2[color][card_index]
		else:
			card = np_all_cards_3[color][card_index]
		return card

	def _fill_new_card(self, tier, index, deterministic):
		self.cards_tiers[8*tier+2*index:8*tier+2*index+2] = 0
		if not deterministic:
			card = self._get_deck_card(tier)
			if card is not None:
				self.cards_tiers[8*tier+2*index:8*tier+2*index+2] = card

	def _buy_card(self, card0, card1, player):
		card_cost = card0[:5]
		player_gems = self.players_gems[player][:5]
		player_cards = self.players_cards[player][:5]
		missing_colors = np.maximum(card_cost - player_gems - player_cards, 0).sum()
		# Apply changes
		paid_gems = np.minimum(np.maximum(card_cost - player_cards, 0), player_gems)
		player_gems -= paid_gems
		self.bank[0][:5] += paid_gems
		self.players_gems[player][idx_gold] -= missing_colors
		self.bank[0][idx_gold] += missing_colors
		self.players_cards[player] += card1

		self._give_nobles_if_earned(player)

	def _valid_buy(self, player):
		cards_cost = self.cards_tiers[:2*12:2,:5]

		player_gems = self.players_gems[player][:5]
		player_cards = self.players_cards[player][:5]
		missing_colors = np.maximum(cards_cost - player_gems - player_cards, 0).sum(axis=1)
		enough_gems_and_gold = missing_colors <= self.players_gems[player][idx_gold]
		not_empty_cards = cards_cost.sum(axis=1) != 0

		return np.logical_and(enough_gems_and_gold, not_empty_cards).astype(np.int8)

	def _buy(self, i, player, deterministic):
		tier, index = divmod(i, 4)
		self._buy_card(self.cards_tiers[2*i], self.cards_tiers[2*i+1], player)
		self._fill_new_card(tier, index, deterministic)

	def _valid_reserve(self, player):
		if not ENABLE_ACTION_RESERVE:
			return np.zeros(12+3, dtype=np.int8)
		not_empty_cards = np.vstack((self.cards_tiers[:2*12:2,:5], self.nb_deck_tiers[::2, :5])).sum(axis=1) != 0

		allowed_reserved_cards = 3
		#if self.bank[0][idx_points] > 40 or self.bank[0][idx_points] % 2 == 1:
		#	allowed_reserved_cards = 3
		#elif self.bank[0][idx_points] > 20:
		#	allowed_reserved_cards = 2
		#else:
		#	allowed_reserved_cards = 1
		empty_slot = (self.players_reserved[6*player+2*(allowed_reserved_cards-1)+1][:5].sum() == 0)
		return np.logical_and(not_empty_cards, empty_slot).astype(np.int8)

	def _reserve(self, i, player, deterministic):
		# Detect empty reserve slot
		reserve_slots = [6*player+2*i for i in range(3)]
		for slot in reserve_slots:
			if self.players_reserved[slot,:5].sum() == 0:
				empty_slot = slot
				break
		
		if i < 12: # reserve visible card
			tier, index = divmod(i, 4)
			self.players_reserved[empty_slot:empty_slot+2] = self.cards_tiers[8*tier+2*index:8*tier+2*index+2]
			self._fill_new_card(tier, index, deterministic)
		else:      # reserve from deck
			if not deterministic:
				tier = i - 12
				self.players_reserved[empty_slot:empty_slot+2] = self._get_deck_card(tier)
		
		if self.bank[0][idx_gold] > 0 and self.players_gems[player].sum() <= 9:
			self.players_gems[player][idx_gold] += 1
			self.bank[0][idx_gold] -= 1

	def _valid_buy_reserve(self, player):
		card_index = np.arange(3)
		cards_cost = self.players_reserved[6*player+2*card_index,:5]

		player_gems = self.players_gems[player][:5]
		player_cards = self.players_cards[player][:5]
		missing_colors = np.maximum(cards_cost - player_gems - player_cards, 0).sum(axis=1)
		enough_gems_and_gold = missing_colors <= self.players_gems[player][idx_gold]
		not_empty_cards = cards_cost.sum(axis=1) != 0

		return np.logical_and(enough_gems_and_gold, not_empty_cards).astype(np.int8)

	def _buy_reserve(self, i, player):
		start_index = 6*player+2*i
		self._buy_card(self.players_reserved[start_index], self.players_reserved[start_index+1], player)
		# shift remaining reserve to the beginning
		if i < 2:
			self.players_reserved[start_index:6*player+4] = self.players_reserved[start_index+2:6*player+6]
		self.players_reserved[6*player+4:6*player+6] = 0 # empty last reserve slot

	def _valid_get_gems(self, player):
		gems = np_different_gems_up_to_3[:,:5]
		enough_in_bank = np_all_axis1((self.bank[0][:5] - gems) >= 0)
		not_too_many_gems = self.players_gems[player].sum() + gems.sum(axis=1) <= 10
		result = np.logical_and(enough_in_bank, not_too_many_gems).astype(np.int8)
		return result

	def _valid_get_gems_identical(self, player):
		colors = np.arange(5)
		enough_in_bank = self.bank[0][colors] >= 4
		not_too_many_gems = self.players_gems[player].sum() + 2 <= 10
		result = np.logical_and(enough_in_bank, not_too_many_gems).astype(np.int8)
		return result

	def _get_gems(self, i, player):
		if i < np_different_gems_up_to_3.shape[0]: # Different gems
			gems = np_different_gems_up_to_3[i][:5]
		else:                                      # 2 identical gems
			color = i - np_different_gems_up_to_3.shape[0]
			gems = np.zeros(5, dtype=np.int8)
			gems[color] = 2
		self.bank[0][:5] -= gems
		self.players_gems[player][:5] += gems

	def _valid_give_gems(self, player):
		if not ENABLE_ACTION_GIVEBACK:
			return np.zeros(np_different_gems_up_to_2.shape[0], dtype=np.int8)
		gems = np_different_gems_up_to_2[:,:5]
		result = np_all_axis1((self.players_gems[player][:5] - gems) >= 0).astype(np.int8)
		return result

	def _valid_give_gems_identical(self, player):
		if not ENABLE_ACTION_GIVEBACK:
			return np.zeros(5, dtype=np.int8)
		colors = np.arange(5)
		return (self.players_gems[player][colors] >= 2).astype(np.int8)

	def _give_gems(self, i, player):
		if i < np_different_gems_up_to_2.shape[0]: # Different gems
			gems = np_different_gems_up_to_2[i][:5]
		else:                                      # 2 identical gems
			color = i - np_different_gems_up_to_2.shape[0]
			gems = np.zeros(5, dtype=np.int8)
			gems[color] = 2
		self.bank[0][:5] += gems
		self.players_gems[player][:5] -= gems

	def _give_nobles_if_earned(self, player):
		for i_noble in range(3):
			noble = self.nobles[i_noble][:5]
			if noble.sum() > 0 and np.all(self.players_cards[player][:5] >= noble):
				self.players_nobles[3*player+i_noble] = self.nobles[i_noble]
				self.nobles[i_noble] = 0

	def _nb_of_reserved_cards(self, player):
		for card in range(3):
			if self.players_reserved[6*player+2*card,:5].sum() == 0:
				return card # slot 'card' is empty, there are 'card' cards
		return 3
