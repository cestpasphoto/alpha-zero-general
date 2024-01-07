from .SplendorLogic import np_all_nobles, np_all_cards_1, np_all_cards_2, np_all_cards_3, len_all_cards, np_different_gems_up_to_2, np_different_gems_up_to_3, np_cards_symmetries, np_reserve_symmetries
import numpy as np
from numba import njit
import numba

############################## BOARD DESCRIPTION ##############################
# Board is described by a 56x7 array (1st dim is larger with 3-4 players)
# Gems and Nobles are represented using 1 line. Each is described by 7 values
#####   0      1      2      3      4      5      6
##### White  Blue   Green  Red    Black  Gold   Points 
#
# Cards are represented using 2 lines
#####          0      1      2      3      4      5      6
##### line0: White  Blue   Green  Red    Black    -      -     [ cost ]
##### line1: White  Blue   Green  Red    Black    -   Points   [ gain ]
# First line describes what is needed to buy the card, second line describes
# the card offers once bought (one of the color has value 1, others are 0)
#
# Here is the description of each line of the board. For readibility, we defined
# "shortcuts" that actually are views (numpy name) of overal board.
##### Index  Shortcut              Meaning
#####   0    self.bank             Number of gems in bank, including gold
#####  1-2   self.cards_tiers      Description of 1st visible card (bottom left)
#####  3-4       =                 Description of 2nd visible card (bottom)
#####  5-6       =                 Description of 3rd visible card (bottom)
#####  7-8       =                 Description of 4th visible card (bottom right)
#####  9-10      =                 Description of 5th visible card (middle left)
#####  ...       =
#####  25    self.nb_deck_tiers    How many cards of each color in bottom deck
#####  26        =                 Bitfield listing which cards are in bottom deck
#####  27        =                 How many cards of each color in middle deck
#####  ...       =
#####  31    self.nobles           Description of 1st noble in bank
#####  32        =                 Description of 2nd noble in bank
#####  33        =                 Description of 3rd noble in bank
#####  34    self.players_gems     Number of gems owned by Player0, including gold
#####  35        =                 Number of gems owned by Player1, including gold
#####  36    self.players_nobles   Description of 1st noble owned by Player0
#####  ...       =
#####  39        =                 Description of 1st noble owned by Player1
#####  ...       =
#####  42    self.players_cards    Number of cards owned by Player0, including sum of ther points
#####  43        =                 Number of cards owned by Player1, including sum of ther points
##### 44-45  self.players_reserved Description of 1st reserved card by Player0
#####  ...       =
##### 50-51      =                 Description of 1st reserved card by Player1
#####  ...       =
# Indexes above are assuming 2 players, you can have more details in copy_state().
# When a noble is won, its value is defined to zero, and other nobles stay in
# place (not shifted). The won noble uses same slot index on player side.
# When a card slot is empty (bank or player reserve), both lines are set to zero.

############################## ACTION DESCRIPTION #############################
# We coded 81 actions, taking some shortcuts on combinations of gems that can be
# got or that can be given back, and forbidding to simultaneously get gems and
# give some back.
# Here is description of each action:
##### Index  Meaning
#####   0    Buy 1st visible card (bottom left)
#####   1    Buy 2nd visible card (bottom)
#####   2    Buy 3rd visible card (bottom)
#####   3    Buy 4th visible card (bottom right)
#####   4    Buy 5th visible card (middle left)
#####  ...
#####   12   Reserve 1st visible card
#####  ...
#####   24   Reserve blindly card for bottom deck
#####   25   Reserve blindly card for middle deck
#####   26   Reserve blindly card for top deck
#####   27   Buy 1st card from player's reserve
#####   28   Buy 2nd card from player's reserve
#####   29   Buy 3rd card from player's reserve
#####   30   Get 1st combination of different gems (up to 3)
#####   31   Get 2nd combination of different gems (up to 3)
#####  ...
#####   55   Get 2 identical gems of color 0 = white
#####  ...
#####   60   Give back 1st combination of different gems (up to 2)
#####  ...
#####   75   Give back 2 identical gems of color 0 = white
#####  ...
#####   80   No action, pass
# List of combinations of gems for actions 30-79 are in variables
# list_different_gems_up_to_2 and list_different_gems_up_to_3 in file SplendorLogic

# Full random actions "breaks" exploration of MCTS tree (same action applied to
# same state may lead to different state). By having repeatable behavior but
# still kind of random (that I called "repeatable randomness"), the exploration
# of MCTS tree go deeper and training data is more relevant. Even on truly
# random pit, the training result behaves better than before.
# Enable it ONLY FOR TRAINING.

idx_white, idx_blue, idx_green, idx_red, idx_black, idx_gold, idx_points = range(7)
mask = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint8)
mask2 = 2**(5*np.arange(idx_gold))

@njit(cache=True, fastmath=True, nogil=True)
def observation_size(num_players):
	return (32 + 10*num_players + num_players*num_players, 7)

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
	('max_moves'           , numba.uint8),
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
		self.max_moves = 62 * num_players
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
			self.nb_deck_tiers[2*tier,:idx_gold] = nb_deck_cards_per_color
			# WHICH cards per color are in deck of tier 0, pratical for logic
			self.nb_deck_tiers[2*tier+1,:idx_gold] = my_packbits(np.ones(nb_deck_cards_per_color, dtype=np.int8))
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

		return (player+1)%self.num_players

	def copy_state(self, state, copy_or_not):
		if self.state is state and not copy_or_not:
			return
		self.state = state.copy() if copy_or_not else state
		n = self.num_players
		self.bank             = self.state[0         :1          ,:]	# 1
		self.cards_tiers      = self.state[1         :25         ,:]	# 2*12
		self.nb_deck_tiers    = self.state[25        :31         ,:]	# 6
		self.nobles           = self.state[31        :32+n       ,:]	# N+1
		self.players_gems     = self.state[32+n      :32+2*n     ,:]	# N
		self.players_nobles   = self.state[32+2*n    :32+3*n+n*n ,:]	# N*(N+1)
		self.players_cards    = self.state[32+3*n+n*n:32+4*n+n*n ,:]	# N
		self.players_reserved = self.state[32+4*n+n*n:32+10*n+n*n,:]	# 6*N

	def check_end_game(self):
		if self.get_round() % self.num_players != 0: # Check only when 1st player is about to play
			return np.full(self.num_players, 0., dtype=np.float32)
		
		scores = np.array([self.get_score(p) for p in range(self.num_players)], dtype=np.float32)
		score_max = scores.max()
		end = (score_max >= self.score_win) or (self.get_round() >= self.max_moves)
		if not end:
			return np.full(self.num_players, 0., dtype=np.float32)
		who_has_won = (scores == score_max)
		several_winners = (who_has_won.sum() > 1)
		# Resolve tie by applying penalty in function of nb of cards
		if several_winners:
			for p in range(self.num_players):
				scores[p] -= self._nb_of_cards(p) / 100.
				score_max = scores.max()
				who_has_won = (scores == score_max)
				several_winners = (who_has_won.sum() > 1)
		
		return np.where(who_has_won > 0, 0.01 if several_winners else 1., -1.).astype(np.float32)

	# if n=1, transform P0 to Pn, P1 to P0, ... and Pn to Pn-1
	# else do this action n times
	def swap_players(self, nb_swaps):
		def _roll_in_place_axis0(array, shift):
			tmp_copy = array.copy()
			size0 = array.shape[0]
			for i in range(size0):
				array[i,:] = tmp_copy[(i+shift)%size0,:]
		_roll_in_place_axis0(self.players_gems    , 1*nb_swaps)
		_roll_in_place_axis0(self.players_nobles  , 3*nb_swaps)
		_roll_in_place_axis0(self.players_cards   , 1*nb_swaps)
		_roll_in_place_axis0(self.players_reserved, 6*nb_swaps)

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
		return self.bank[0].astype(np.uint8)[idx_points]

	def _get_deck_card(self, tier, deterministic):
		nb_remaining_cards_per_color = self.nb_deck_tiers[2*tier,:idx_gold]
		if nb_remaining_cards_per_color.sum() == 0: # no more cards
			return None
		
		if deterministic == 0:
			# First we chose color randomly, then we pick a card 
			color = my_random_choice(nb_remaining_cards_per_color/nb_remaining_cards_per_color.sum())
			remaining_cards = my_unpackbits(self.nb_deck_tiers[2*tier+1, color])
			card_index = my_random_choice(remaining_cards/remaining_cards.sum())
			# print(f'TRUE RANDOM')

		elif deterministic == -1:
			# empty card
			print('SHOULD NOT HAPPEN - deterministic = -1')
		
		else:
			# https://stackoverflow.com/questions/3062746/special-simple-random-number-generator
			# m=avail_people_id.size, c=0, a=2*3*5*7*9*11*13*17+1
			remaining_cards_all = [ (c,i) for c in range(5) for i,b in enumerate(my_unpackbits(self.nb_deck_tiers[2*tier+1, c])) if b]
			seed = (self.nb_deck_tiers[2*tier+1, :idx_gold].astype(np.uint8) * mask2).sum()
			fake_random_index = (4594591 * (deterministic+seed)) % len(remaining_cards_all)
			# print(f'{fake_random_index=} {seed=}')
			color, card_index = remaining_cards_all[fake_random_index]
			remaining_cards = my_unpackbits(self.nb_deck_tiers[2*tier+1, color])

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
		if deterministic != -1:
			card = self._get_deck_card(tier, deterministic)
			if card is not None:
				self.cards_tiers[8*tier+2*index:8*tier+2*index+2] = card

	def _buy_card(self, card0, card1, player):
		card_cost = card0[:idx_gold]
		player_gems = self.players_gems[player][:idx_gold]
		player_cards = self.players_cards[player][:idx_gold]
		missing_colors = np.maximum(card_cost - player_gems - player_cards, 0).sum()
		# Apply changes
		paid_gems = np.minimum(np.maximum(card_cost - player_cards, 0), player_gems)
		player_gems -= paid_gems
		self.bank[0][:idx_gold] += paid_gems
		self.players_gems[player][idx_gold] -= missing_colors
		self.bank[0][idx_gold] += missing_colors
		self.players_cards[player] += card1

		self._give_nobles_if_earned(player)

	def _valid_buy(self, player):
		cards_cost = self.cards_tiers[:2*12:2,:idx_gold]

		player_gems = self.players_gems[player][:idx_gold]
		player_cards = self.players_cards[player][:idx_gold]
		missing_colors = np.maximum(cards_cost - player_gems - player_cards, 0).sum(axis=1)
		enough_gems_and_gold = missing_colors <= self.players_gems[player][idx_gold]
		not_empty_cards = cards_cost.sum(axis=1) != 0

		return np.logical_and(enough_gems_and_gold, not_empty_cards).astype(np.int8)

	def _buy(self, i, player, deterministic):
		tier, index = divmod(i, 4)
		self._buy_card(self.cards_tiers[2*i], self.cards_tiers[2*i+1], player)
		self._fill_new_card(tier, index, deterministic)

	def _valid_reserve(self, player):
		not_empty_cards = np.vstack((self.cards_tiers[:2*12:2,:idx_gold], self.nb_deck_tiers[::2, :idx_gold])).sum(axis=1) != 0

		allowed_reserved_cards = 3
		empty_slot = (self.players_reserved[6*player+2*(allowed_reserved_cards-1)+1][:idx_gold].sum() == 0)
		return np.logical_and(not_empty_cards, empty_slot).astype(np.int8)

	def _reserve(self, i, player, deterministic):
		# Detect empty reserve slot
		reserve_slots = [6*player+2*i for i in range(3)]
		for slot in reserve_slots:
			if self.players_reserved[slot,:idx_gold].sum() == 0:
				empty_slot = slot
				break
		
		if i < 12: # reserve visible card
			tier, index = divmod(i, 4)
			self.players_reserved[empty_slot:empty_slot+2] = self.cards_tiers[8*tier+2*index:8*tier+2*index+2]
			self._fill_new_card(tier, index, deterministic)
		else:      # reserve from deck
			if deterministic != -1:
				tier = i - 12
				self.players_reserved[empty_slot:empty_slot+2] = self._get_deck_card(tier, deterministic)
		
		if self.bank[0][idx_gold] > 0 and self.players_gems[player].sum() <= 9:
			self.players_gems[player][idx_gold] += 1
			self.bank[0][idx_gold] -= 1

	def _valid_buy_reserve(self, player):
		card_index = np.arange(3)
		cards_cost = self.players_reserved[6*player+2*card_index,:idx_gold]

		player_gems = self.players_gems[player][:idx_gold]
		player_cards = self.players_cards[player][:idx_gold]
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
		gems = np_different_gems_up_to_3[:,:idx_gold]
		enough_in_bank = np_all_axis1((self.bank[0][:idx_gold] - gems) >= 0)
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
			gems = np_different_gems_up_to_3[i][:idx_gold]
		else:                                      # 2 identical gems
			color = i - np_different_gems_up_to_3.shape[0]
			gems = np.zeros(5, dtype=np.int8)
			gems[color] = 2
		self.bank[0][:idx_gold] -= gems
		self.players_gems[player][:idx_gold] += gems

	def _valid_give_gems(self, player):
		gems = np_different_gems_up_to_2[:,:idx_gold]
		result = np_all_axis1((self.players_gems[player][:idx_gold] - gems) >= 0).astype(np.int8)
		return result

	def _valid_give_gems_identical(self, player):
		colors = np.arange(5)
		return (self.players_gems[player][colors] >= 2).astype(np.int8)

	def _give_gems(self, i, player):
		if i < np_different_gems_up_to_2.shape[0]: # Different gems
			gems = np_different_gems_up_to_2[i][:idx_gold]
		else:                                      # 2 identical gems
			color = i - np_different_gems_up_to_2.shape[0]
			gems = np.zeros(5, dtype=np.int8)
			gems[color] = 2
		self.bank[0][:idx_gold] += gems
		self.players_gems[player][:idx_gold] -= gems

	def _give_nobles_if_earned(self, player):
		for i_noble in range(self.num_nobles):
			noble = self.nobles[i_noble][:idx_gold]
			if noble.sum() > 0 and np.all(self.players_cards[player][:idx_gold] >= noble):
				self.players_nobles[self.num_nobles*player+i_noble] = self.nobles[i_noble]
				self.nobles[i_noble] = 0

	def _nb_of_reserved_cards(self, player):
		for card in range(3):
			if self.players_reserved[6*player+2*card,:idx_gold].sum() == 0:
				return card # slot 'card' is empty, there are 'card' cards
		return 3

	def _nb_of_cards(self, player):
		return self.players_cards[player, :idx_gold].sum()
