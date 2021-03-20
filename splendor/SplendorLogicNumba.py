#!./venv/bin/python3

from SplendorLogic import all_nobles, all_cards_1, all_cards_2, all_cards_3, list_different_gems_up_to_2, list_different_gems_up_to_3, cards_symmetries, reserve_symmetries
import numpy as np

import time

from numba import jit, njit, jit_module
from numba.experimental import jitclass
from numba import int8, float32 


np_all_cards_1 = np.array(all_cards_1, dtype=np.int8)
np_all_cards_2 = np.array(all_cards_2, dtype=np.int8)
np_all_cards_3 = np.array(all_cards_3, dtype=np.int8)
np_all_nobles  = np.array(all_nobles , dtype=np.int8)
all_cards_len = np.array([len(all_cards_1[0]), len(all_cards_2[0]), len(all_cards_3[0])], dtype=np.int8)
np_cards_symmetries = np.array(cards_symmetries)
idx_white, idx_blue, idx_green, idx_red, idx_black, idx_gold, idx_points = range(7)
mask = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint8)

def rand_choice_nb(prob):
	result = np.searchsorted(np.cumsum(prob), np.random.random(), side="right")
	# print('numba:', prob, ' ', result)
	return result

def my_packbits(array):
	product = np.multiply(array.astype(np.uint8), mask[:len(array)])
	return product.sum()

def my_unpackbits(value):
	return (np.bitwise_and(value, mask) != 0).astype(np.uint8)


def copy_state(state):
	n = 2
	bank             = state[0     :1      ,:]	# 1
	cards_tiers      = state[1     :25     ,:]	# 2*12
	nb_deck_tiers    = state[25    :31     ,:]	# 6
	nobles           = state[31    :32+n   ,:]	# N+1
	players_gems     = state[32+n  :32+2*n ,:]	# N
	players_nobles   = state[32+2*n:32+5*n ,:]	# 3*N
	players_cards    = state[32+5*n:32+6*n ,:]	# N
	players_reserved = state[32+6*n:32+12*n,:]	# 6*N

	# for tier in range(3):
	# 	nb_deck_cards_per_color = [8,6,4][tier]
	# 	# HOW MANY cards per color are in deck of tier 0, pratical for NN
	# 	nb_deck_tiers[2*tier,:5] = nb_deck_cards_per_color
	# 	# WHICH cards per color are in deck of tier 0, pratical for logic
	# 	nb_deck_tiers[2*tier+1,:5] = my_packbits(np.full(nb_deck_cards_per_color, 1, dtype=np.uint8))

	return bank, cards_tiers, nb_deck_tiers, nobles, players_gems, players_nobles, players_cards, players_reserved 

def _get_deck_card(tier, nb_deck_tiers):
	nb_remaining_cards_per_color = nb_deck_tiers[2*tier,:5]
	if nb_remaining_cards_per_color.sum() == 0: # no more cards
		return None
	
	# First we chose color randomly, then we pick a card 
	color = rand_choice_nb(nb_remaining_cards_per_color/nb_remaining_cards_per_color.sum())
	remaining_cards = my_unpackbits(nb_deck_tiers[2*tier+1, color])
	card_index = rand_choice_nb(remaining_cards/remaining_cards.sum())
	# Update internals
	remaining_cards[card_index] = 0
	nb_deck_tiers[2*tier+1, color] = my_packbits(remaining_cards)
	nb_deck_tiers[2*tier, color] -= 1

	if tier == 0:
		card = np_all_cards_1[color][card_index]
	elif tier == 1:
		card = np_all_cards_2[color][card_index]
	else:
		card = np_all_cards_3[color][card_index]
	return card

def _fill_new_card(tier, index, cards_tiers, nb_deck_tiers):
	cards_tiers[8*tier+2*index:8*tier+2*index+2] = 0
	card = _get_deck_card(tier, nb_deck_tiers)
	if card is not None:
		cards_tiers[8*tier+2*index:8*tier+2*index+2] = card

def _give_nobles_if_earned(player, nobles, players_cards, players_nobles):
	for i_noble in range(3):
		noble = nobles[i_noble][:5]
		if noble.sum() > 0 and np.all(players_cards[player][:5] >= noble):
			players_nobles[3*player+i_noble] = nobles[i_noble]
			nobles[i_noble] = 0

def _buy_card(card0, card1, player, players_gems, players_cards, nobles, bank, players_nobles):
	card_cost = card0[:5]
	player_gems = players_gems[player][:5]
	player_cards = players_cards[player][:5]
	missing_colors = np.maximum(card_cost - player_gems - player_cards, 0).sum()
	# Apply changes
	paid_gems = np.minimum(np.maximum(card_cost - player_cards, 0), player_gems)
	player_gems -= paid_gems
	bank[0][:5] += paid_gems
	players_gems[player][idx_gold] -= missing_colors
	bank[0][idx_gold] += missing_colors
	players_cards[player] += card1

	_give_nobles_if_earned(player, nobles, players_cards, players_nobles)


def _buy(i, player, cards_tiers, players_gems, players_cards, nobles, nb_deck_tiers, bank, players_nobles):
	tier, index = divmod(i, 4)
	_buy_card(cards_tiers[2*i], cards_tiers[2*i+1], player, players_gems, players_cards, nobles, bank, players_nobles)
	_fill_new_card(tier, index, cards_tiers, nb_deck_tiers)


def _valid_reserve_optim(player, cards_tiers, nb_deck_tiers, players_reserved):
	not_empty_cards = np.vstack((cards_tiers[:24:2,:5], nb_deck_tiers[::2, :5])).sum(axis=1) != 0
	empty_slot = (players_reserved[6*player+5][:5].sum() == 0)
	return np.logical_and(not_empty_cards, empty_slot).astype(np.int8)



#############################################

np_different_gems_up_to_2 = np.array(list_different_gems_up_to_2, dtype=np.int8)
np_different_gems_up_to_3 = np.array(list_different_gems_up_to_3, dtype=np.int8)


def np_all_axis1(x):
    """Numba compatible version of np.all(x, axis=1)."""
    out = np.ones(x.shape[0], dtype=np.bool8)
    for i in range(x.shape[1]):
        out = np.logical_and(out, x[:, i])
    return out

def valid_moves(player, cards_tiers, players_gems, players_cards, nb_deck_tiers, players_reserved, bank):
	result = np.zeros(81, dtype=np.bool_)
	# 14ms
	result[0         :12]            = _valid_buy_optim(player, cards_tiers, players_gems, players_cards)
	# 9ms
	result[12        :12+15]         = _valid_reserve_optim(player, cards_tiers, nb_deck_tiers, players_reserved)
	# 10ms
	result[12+15     :12+15+3]       = _valid_buy_reserve_optim(player, players_reserved, players_gems, players_cards)
	# 12ms
	result[12+15+3   :12+15+3+30]    = np.concatenate((_valid_get_gems_optim(player, bank, players_gems), _valid_get_gems_identical_optim(player, bank, players_gems)))
	# 10ms
	result[12+15+3+30:12+15+3+30+20] = np.concatenate((_valid_give_gems_optim(player, players_gems), _valid_give_gems_identical_optim(player, players_gems)))
	# 0.2ms
	result[80] = True #empty move
	return result

def make_move(move, player, cards_tiers, players_gems, players_cards, nobles, nb_deck_tiers, bank, players_nobles, players_reserved):
	if   move < 12:
		_buy(move, player, cards_tiers, players_gems, players_cards, nobles, nb_deck_tiers, bank, players_nobles)
	elif move < 12+15:
		_reserve(move-12, player, players_reserved, players_gems, players_cards, cards_tiers, bank, nb_deck_tiers)
	elif move < 12+15+3:
		_buy_reserve(move-12-15, player, players_reserved, players_gems, players_cards, nobles, bank, players_nobles)
	elif move < 12+15+3+30:
		_get_gems(move-12-15-3, player, bank, players_gems)
	elif move < 12+15+3+30+20:
		_give_gems(move-12-15-3-30, player, bank, players_gems)
	else:
		pass # empty move
	bank[0][idx_points] += 1 # Count number of rounds

def get_score(player, players_cards, players_nobles):
	card_points  = players_cards[player, idx_points]
	noble_points = players_nobles[player*3:player*3+3, idx_points].sum()
	return card_points + noble_points

def check_end_game(bank, players_cards, players_nobles):
	if bank[0][idx_points] % 2 != 0: # Check only when 1st player is about to play
		return False, [False, False]
	
	scores = [get_score(p, players_cards, players_nobles) for p in range(2)]
	score_max = max(scores)
	end = (score_max >= 15) or (bank[0][idx_points] >= 126)
	winners = [(s == score_max) for s in scores] if end else [False, False]
	return end, winners

def swap_players(players_gems, players_cards, players_nobles, players_reserved):
	def _swap(array, half_size):
		copy = array[:half_size,:].copy()
		array[:half_size,:] = array[half_size:2*half_size,:]
		array[half_size:2*half_size,:] = copy
	_swap(players_gems, 1)
	_swap(players_cards, 1)
	_swap(players_nobles, 3)
	_swap(players_reserved, 6)

def get_symmetries(policy, valid_actions, state, cards_tiers, players_reserved):
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

	symmetries = [(state.copy(), policy.copy(), valid_actions.copy())]
	# Permute common cards within same tier
	for tier in range(3):
		for permutation in np_cards_symmetries:
			cards_tiers_backup = cards_tiers.copy()
			_swap_cards(cards_tiers[8*tier:8*tier+8, :], permutation)
			new_policy = _copy_and_permute(policy, permutation, 4*tier)
			new_valid_actions = _copy_and_permute(valid_actions, permutation, 4*tier)
			symmetries.append((state.copy(), new_policy, new_valid_actions))
			cards_tiers[:] = cards_tiers_backup
	
	# Permute reserved cards
	for player in range(2):
		nb_reserved_cards = _nb_of_reserved_cards(player, players_reserved)
		if nb_reserved_cards in [2,3]:
			xx = [(1, 0, 2)] if nb_reserved_cards == 2 else [(1, 2, 0), (2, 0, 1)]
			for permutation in xx:
				players_reserved_backup = players_reserved.copy()
				_swap_cards(players_reserved[6*player:6*player+6, :], permutation)
				if player == 0:
					new_policy = _copy_and_permute(policy, permutation, 12+15)
					new_valid_actions = _copy_and_permute(valid_actions, permutation, 12+15)
				else:
					new_policy = policy.copy()
					new_valid_actions = valid_actions.copy()
				symmetries.append((state.copy(), new_policy, new_valid_actions))
				players_reserved[:] = players_reserved_backup

	return symmetries

def get_round(bank):
	return bank[0][idx_points]

def _valid_buy_optim(player, cards_tiers, players_gems, players_cards):
	cards_cost = cards_tiers[:24:2,:5]

	player_gems = players_gems[player][:5]
	player_cards = players_cards[player][:5]
	missing_colors = np.maximum(cards_cost - player_gems - player_cards, 0).sum(axis=1)
	enough_gems_and_gold = missing_colors <= players_gems[player][idx_gold]
	not_empty_cards = cards_cost.sum(axis=1) != 0

	return np.logical_and(enough_gems_and_gold, not_empty_cards).astype(np.int8)

def _reserve(i, player, players_reserved, players_gems, players_cards, cards_tiers, bank, nb_deck_tiers):
	# Detect empty reserve slot
	reserve_slots = [6*player+2*i for i in range(3)]
	# empty_slots = np.flatnonzero(players_reserved[reserve_slots,:5].sum(axis=1) == 0)
	for slot in reserve_slots:
		if players_reserved[slot,:5].sum() != 0:
			empty_slot = slot
			break
	
	if i < 12:
		tier, index = divmod(i, 4)
		players_reserved[empty_slot:empty_slot+2] = cards_tiers[8*tier+2*index:8*tier+2*index+2]
		_fill_new_card(tier, index, cards_tiers, nb_deck_tiers)
	else:
		tier = i - 12
		players_reserved[empty_slot:empty_slot+2] = _get_deck_card(tier, nb_deck_tiers)
	
	if bank[0][idx_gold] > 0 and players_gems[player].sum() <= 9:
		players_gems[player][idx_gold] += 1
		bank[0][idx_gold] -= 1

def _valid_buy_reserve_optim(player, players_reserved, players_gems, players_cards):
	card_index = np.arange(3)
	cards_cost = players_reserved[6*player+2*card_index,:5]

	player_gems = players_gems[player][:5]
	player_cards = players_cards[player][:5]
	missing_colors = np.maximum(cards_cost - player_gems - player_cards, 0).sum(axis=1)
	enough_gems_and_gold = missing_colors <= players_gems[player][idx_gold]
	not_empty_cards = cards_cost.sum(axis=1) != 0

	return np.logical_and(enough_gems_and_gold, not_empty_cards).astype(np.int8)

def _buy_reserve(i, player, players_reserved, players_gems, players_cards, nobles, bank, players_nobles):
	start_index = 6*player+2*i
	_buy_card(players_reserved[start_index], players_reserved[start_index+1], player, players_gems, players_cards, nobles, bank, players_nobles)
	# shift remaining reserve to the beginning
	if i < 2:
		players_reserved[start_index:6*player+4] = players_reserved[start_index+2:6*player+6]
	players_reserved[6*player+4:6*player+6] = 0 # empty last reserve slot

def _valid_get_gems_optim(player, bank, players_gems):
	gems = np_different_gems_up_to_3[:,:5]
	enough_in_bank = np_all_axis1((bank[0][:5] - gems) >= 0)
	not_too_many_gems = players_gems[player].sum() + gems.sum(axis=1) <= 10
	result = np.logical_and(enough_in_bank, not_too_many_gems).astype(np.int8)
	return result

def _valid_get_gems_identical_optim(player, bank, players_gems):
	colors = np.arange(5)
	enough_in_bank = bank[0][colors] >= 4
	not_too_many_gems = players_gems[player].sum() + 2 <= 10
	result = np.logical_and(enough_in_bank, not_too_many_gems).astype(np.int8)
	return result

def _get_gems(i, player, bank, players_gems):
	if i < np_different_gems_up_to_3.shape[0]: # Different gems
		gems = np_different_gems_up_to_3[i][:5]
	else: # 2 identical gems
		color = i - np_different_gems_up_to_3.shape[0]
		gems = np.array([2*int(i==color) for i in range(5)], dtype=np.int8)
	bank[0][:5] -= gems
	players_gems[player][:5] += gems

def _valid_give_gems_optim(player, players_gems):
	gems = np_different_gems_up_to_2[:,:5]
	result = np_all_axis1((players_gems[player][:5] - gems) >= 0).astype(np.int8)
	return result

def _valid_give_gems_identical_optim(player, players_gems):
	colors = np.arange(5)
	return (players_gems[player][colors] >= 2).astype(np.int8)

def _give_gems(i, player, bank, players_gems):
	if i < np_different_gems_up_to_2.shape[0]: # Different gems
		gems = np_different_gems_up_to_2[i][:5]
	else: # 2 identical gems
		color = i - np_different_gems_up_to_2.shape[0]
		gems = np.array([2*int(i==color) for i in range(5)], dtype=np.int8)
	bank[0][:5] += gems
	players_gems[player][:5] -= gems

def _nb_of_reserved_cards(player, players_reserved):
	for card_index in range(3):
		if players_reserved[6*player+2*card_index,:5].sum() == 0:
			return card_index # slot 'card' is empty, there are 'card_index' cards
	return 3

def init_game():
	state = np.zeros((56,7), dtype=np.int8)
	bank, cards_tiers, nb_deck_tiers, nobles, players_gems, players_nobles, players_cards, players_reserved = copy_state(state)

	# Bank
	bank[:] = np.array([[4]*5 + [5, 0]])
	# Decks
	for tier in range(3):
		nb_deck_cards_per_color = all_cards_len[tier]
		# HOW MANY cards per color are in deck of tier 0, pratical for NN
		nb_deck_tiers[2*tier,:5] = nb_deck_cards_per_color
		# WHICH cards per color are in deck of tier 0, pratical for logic
		nb_deck_tiers[2*tier+1,:5] = my_packbits(np.ones(nb_deck_cards_per_color, dtype=np.int8))
	# Tiers
	for tier in range(3):
		for index in range(4):
			_fill_new_card(tier, index, cards_tiers, nb_deck_tiers)
	# Nobles
	nobles_indexes = np.random.choice(len(np_all_nobles), size=3)
	for i, index in enumerate(nobles_indexes):
		nobles[i, :] = np_all_nobles[index]

	return state, bank, cards_tiers, nb_deck_tiers, nobles, players_gems, players_nobles, players_cards, players_reserved

jit_module(nopython=True, fastmath=True, cache=True, nogil=True)

####################################################




# @njit(fastmath=True, cache=True)
# def test_function():
# 	state, bank, cards_tiers, nb_deck_tiers, nobles, players_gems, players_nobles, players_cards, players_reserved = init_game()

# 	for _ in range(10):
		# item = np.random.randint(0, 12)
		#_buy(item, 1, cards_tiers, players_gems, players_cards, nobles, nb_deck_tiers, bank, players_nobles)
		# x = _valid_reserve_optim(1, cards_tiers, nb_deck_tiers, players_reserved)
		# x = valid_moves(1, cards_tiers, players_gems, players_cards, nb_deck_tiers, players_reserved, bank)
		# move = np.random.randint(0, 81)
		# make_move(move, 0, cards_tiers, players_gems, players_cards, nobles, nb_deck_tiers, bank, players_nobles, players_reserved)
		# x = check_end_game(bank, players_cards, players_nobles)
		# swap_players(players_gems, players_cards, players_nobles, players_reserved)
		# policy = np.random.rand(81).astype(np.float32)
		# valid_actions = np.random.randint(0, 2, 81).astype(np.bool_)
		# l = get_symmetries(policy, valid_actions, state, cards_tiers, players_reserved)
		# get_round(bank)
		

# test_function()
# start_time = time.time()
# for _ in range(1000):
# 	test_function()
# print((time.time() - start_time) * 1000)


spec = [
	('bank'            , int8[:,:]),
	('cards_tiers'     , int8[:,:]),
	('nb_deck_tiers'   , int8[:,:]),
	('nobles'          , int8[:,:]),
	('players_gems'    , int8[:,:]),
	('players_nobles'  , int8[:,:]),
	('players_cards'   , int8[:,:]),
	('players_reserved', int8[:,:]),
]

@jitclass(spec)
class MyClassTest(object):
	def __init__(self, value):
		self.bank             = np.zeros((1   ,7), dtype=np.int8)
		self.cards_tiers      = np.zeros((2*12,7), dtype=np.int8)
		self.nb_deck_tiers    = np.zeros((6   ,7), dtype=np.int8)
		self.nobles           = np.zeros((3   ,7), dtype=np.int8)
		self.players_gems     = np.zeros((2   ,7), dtype=np.int8)
		self.players_nobles   = np.zeros((6   ,7), dtype=np.int8)
		self.players_cards    = np.zeros((2   ,7), dtype=np.int8)
		self.players_reserved = np.zeros((12  ,7), dtype=np.int8)


	def copy_state(self, state):
		n = 2
		self.bank             = state[0     :1      ,:]	# 1
		self.cards_tiers      = state[1     :25     ,:]	# 2*12
		self.nb_deck_tiers    = state[25    :31     ,:]	# 6
		self.nobles           = state[31    :32+n   ,:]	# N+1
		self.players_gems     = state[32+n  :32+2*n ,:]	# N
		self.players_nobles   = state[32+2*n:32+5*n ,:]	# 3*N
		self.players_cards    = state[32+5*n:32+6*n ,:]	# N
		self.players_reserved = state[32+6*n:32+12*n,:]	# 6*N

mybag = MyClassTest(21)
mybag.copy_state(np.ones((56,7), dtype=np.int8))
print(mybag.bank)
breakpoint()


# (Pdb) valid_moves.inspect_types()                                                                                                                             
# valid_moves (Literal[int](1), array(int8, 2d, A), array(int8, 2d, A), array(int8, 2d, A), array(int8, 2d, A), array(int8, 2d, A), array(int8, 2d, A))

# copy_state (array(int8, 2d, C),) 