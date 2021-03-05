#!/home/best/dev/splendor/venv/bin/python3

import numpy as np
from colorama import Style, Fore, Back
import random
import itertools

class Board():
	def __init__(self, num_players, state=None):
		n = num_players
		self.num_players = n
		self.current_player_index = 0
		self.num_gems_in_play = {2: 4, 3: 5, 4: 7}[n]
		self.num_nobles = {2:3, 3:4, 4:5}[n]
		self.max_moves = 126
		self.score_win = 15
		self.state = None

		if state is None:
			self.init_game()
		else:
			self.copy_state(state, True)

	def get_score(self, player):
		card_points  = self.players_cards[player, idx_points]
		noble_points = self.players_nobles[player*3:player*3+3, idx_points].sum()
		return card_points + noble_points

	def init_game(self):
		self.copy_state(np.zeros(observation_size(self.num_players), dtype=np.int8), copy_or_not=False)

		# Bank
		self.bank[:] = np.array([[self.num_gems_in_play]*5 + [5, 0]])
		# Decks
		for tier in range(3):
			nb_deck_cards_per_color = len(all_cards[tier][0])
			# HOW MANY cards per color are in deck of tier 0, pratical for NN
			self.nb_deck_tiers[2*tier,:5] = nb_deck_cards_per_color
			# WHICH cards per color are in deck of tier 0, pratical for logic
			self.nb_deck_tiers[2*tier+1,:5] = np.packbits([1]*nb_deck_cards_per_color)[0]
		# Tiers
		for tier in range(3):
			for index in range(4):
				self._fill_new_card(tier, index)
		# Nobles
		self.nobles[:] = np.array(random.sample(all_nobles, self.num_nobles))
		# Players
		self.players_gems[:] = 0
		self.players_nobles[:] = 0
		self.players_cards[:] = 0
		self.players_reserved[:] = 0

	def get_state(self):
		return self.state

	def valid_moves(self, player):
		result = [0]*81
		result[0         :12]            = self._valid_buy_optim(player)
		result[12        :12+15]         = self._valid_reserve_optim(player)
		result[12+15     :12+15+3]       = self._valid_buy_reserve_optim(player)
		result[12+15+3   :12+15+3+30]    = np.concatenate((self._valid_get_gems_optim(player) , self._valid_get_gems_identical_optim(player)))
		result[12+15+3+30:12+15+3+30+20] = np.concatenate((self._valid_give_gems_optim(player), self._valid_give_gems_identical_optim(player)))
		result[80] = 1 #empty move
		return result

	def make_move(self, move, player):
		if   move < 12:
			self._buy(move, player)
		elif move < 12+15:
			self._reserve(move-12, player)
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
		end = (score_max >= 15) or (self.bank[0][idx_points] >= self.max_moves)
		winners = [(s == score_max) for s in scores] if end else [False, False]
		return end, winners

	def swap_players(self):
		def _swap(array, half_size):
			array[list(range(2*half_size)),:] = array[list(range(half_size, 2*half_size))+list(range(0, half_size)),:]
		_swap(self.players_gems, 1)
		_swap(self.players_cards, 1)
		_swap(self.players_nobles, 3)
		_swap(self.players_reserved, 6)

	def get_symmetries(self, policy, valid_actions):
		def _swap_cards(cards, permutation):
			full_permutation = [2*p+i for p in permutation for i in range(2)]
			cards[:len(permutation*2), :] = cards[full_permutation, :]

		symmetries = [(self.get_state().copy(), np.array(policy).astype(np.float32), np.array(valid_actions).astype(np.bool_))]
		# Permute common cards within same tier
		for tier in range(3):
			for permutation in cards_symmetries:
				cards_tiers_backup = self.cards_tiers.copy()
				_swap_cards(self.cards_tiers[8*tier:8*tier+8, :], permutation)
				new_policy = np.array(policy).astype(np.float32)
				new_policy[4*tier:4*tier+4] = new_policy[[i+4*tier for i in permutation]]
				new_valid_actions = np.array(valid_actions).astype(np.bool_)
				new_valid_actions[4*tier:4*tier+4] = new_valid_actions[[i+4*tier for i in permutation]]
				symmetries.append((self.get_state().copy(), new_policy, new_valid_actions))
				self.cards_tiers[:] = cards_tiers_backup
		
		# Permute reserved cards
		for player in range(self.num_players):
			for permutation in reserve_symmetries[self._nb_of_reserved_cards(player=player)]:
				players_reserved_backup = self.players_reserved.copy()
				_swap_cards(self.players_reserved[6*player:6*player+6, :], permutation)
				new_policy = np.array(policy).astype(np.float32)
				new_valid_actions = np.array(valid_actions).astype(np.bool_)
				if player == 0:
					new_policy[12+15:12+15+3] = new_policy[[i+12+15 for i in permutation]]
					new_valid_actions[12+15:12+15+3] = new_valid_actions[[i+12+15 for i in permutation]]
				symmetries.append((self.get_state().copy(), new_policy, new_valid_actions))
				self.players_reserved[:] = players_reserved_backup

		return symmetries

	def _get_deck_card(self, tier):
		assert 0 <= tier < 3
		nb_remaining_cards_per_color = self.nb_deck_tiers[2*tier,:5]
		if nb_remaining_cards_per_color.sum() == 0: # no more cards
			return None
		
		# First we chose color randomly, then we pick a card 
		color = np.random.choice(5, p=nb_remaining_cards_per_color/nb_remaining_cards_per_color.sum())
		remaining_cards = np.unpackbits(self.nb_deck_tiers[2*tier+1, color].astype(np.uint8))
		assert np.all(remaining_cards.sum().size == 1)
		assert np.all(remaining_cards.sum() == nb_remaining_cards_per_color[color])
		card_index = np.random.choice(len(remaining_cards), p=remaining_cards/remaining_cards.sum())
		# Update internals
		assert np.all(remaining_cards[card_index] == 1)
		remaining_cards[card_index] = 0
		self.nb_deck_tiers[2*tier+1, color] = np.packbits(remaining_cards).astype(np.int8)
		self.nb_deck_tiers[2*tier, color] -= 1
		assert np.all(remaining_cards.sum() == self.nb_deck_tiers[2*tier, color])
		assert np.all(self.nb_deck_tiers.dtype == np.int8)

		card = all_cards[tier][color][card_index]
		return card

	def _fill_new_card(self, tier, index):
		self.cards_tiers[8*tier+2*index:8*tier+2*index+2] = 0
		card = self._get_deck_card(tier)
		if card is not None:
			self.cards_tiers[8*tier+2*index:8*tier+2*index+2] = card

	def _can_afford(self, card0, player):
		card_cost = card0[:5]
		player_gems = self.players_gems[player][:5]
		player_cards = self.players_cards[player][:5]
		missing_colors = np.maximum(card_cost - player_gems - player_cards, 0).sum()
		return (missing_colors <= self.players_gems[player][idx_gold])

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

	def _is_empty(self, tier, index):
		if index == 4:
			return (self.nb_deck_tiers[2*tier].sum() == 0)
		else:
			return (self.cards_tiers[8*tier+2*index][:5].sum() == 0)

	def _valid_buy_optim(self, player):
		card_index = np.array(range(12))
		cards_cost = self.cards_tiers[2*card_index,:5]

		player_gems = self.players_gems[player][:5]
		player_cards = self.players_cards[player][:5]
		missing_colors = np.maximum(cards_cost - player_gems - player_cards, 0).sum(axis=1)
		enough_gems_and_gold = missing_colors <= self.players_gems[player][idx_gold]
		not_empty_cards = cards_cost.sum(axis=1) != 0

		return np.logical_and(enough_gems_and_gold, not_empty_cards).astype(np.int8)

	def _buy(self, i, player):
		tier, index = divmod(i, 4)
		self._buy_card(self.cards_tiers[2*i], self.cards_tiers[2*i+1], player)
		self._fill_new_card(tier, index)

	def _valid_reserve_optim(self, player):
		card_index = np.array(range(12))
		not_empty_cards = np.vstack([self.cards_tiers[2*card_index,:5], self.nb_deck_tiers[::2, :5]]).sum(axis=1) != 0
		empty_slot = (self.players_reserved[6*player+5][:5].sum() == 0)
		return np.logical_and(not_empty_cards, empty_slot).astype(np.int8)

	def _reserve(self, i, player):
		# Detect empty reserve slot
		reserve_slots = [6*player+2*i for i in range(3)]
		empty_slots = np.flatnonzero(self.players_reserved[reserve_slots][:5].sum(axis=1) == 0)
		if len(empty_slots) == 0:
			breakpoint()
		empty_slot = reserve_slots[empty_slots[0]]
		
		if i < 12:
			tier, index = divmod(i, 4)
			self.players_reserved[empty_slot:empty_slot+2] = self.cards_tiers[8*tier+2*index:8*tier+2*index+2]
			self._fill_new_card(tier, index)
		else:
			tier = i - 12
			self.players_reserved[empty_slot:empty_slot+2] = self._get_deck_card(tier)
		
		if self.bank[0][idx_gold] > 0 and self.players_gems[player].sum() <= 9:
			self.players_gems[player][idx_gold] += 1
			self.bank[0][idx_gold] -= 1

	def _valid_buy_reserve_optim(self, player):
		card_index = np.array(range(3))
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

	def _valid_get_gems_optim(self, player):
		gems = np.array(list_different_gems_up_to_3)[:,:5]
		enough_in_bank = np.all((self.bank[0][:5] - gems) >= 0, axis=1)
		not_too_many_gems = self.players_gems[player].sum() + gems.sum(axis=1) <= 10
		result = np.logical_and(enough_in_bank, not_too_many_gems).astype(np.int8)
		return result

	def _valid_get_gems_identical_optim(self, player):
		colors = np.array(range(5))
		enough_in_bank = self.bank[0][colors] >= 4
		not_too_many_gems = self.players_gems[player].sum() + 2 <= 10
		result = np.logical_and(enough_in_bank, not_too_many_gems).astype(np.int8)
		return result

	def _get_gems(self, i, player):
		if i < len(list_different_gems_up_to_3): # Different gems
			gems = list_different_gems_up_to_3[i][:5]
		else: # 2 identical gems
			color = i - len(list_different_gems_up_to_3)
			gems = np.array([2*int(i==color) for i in range(5)])
		self.bank[0][:5] -= gems
		self.players_gems[player][:5] += gems

	def _valid_give_gems_optim(self, player):
		gems = np.array(list_different_gems_up_to_2)[:,:5]
		result = np.all((self.players_gems[player][:5] - gems) >= 0, axis=1).astype(np.int8)
		return result

	def _valid_give_gems_identical_optim(self, player):
		colors = np.array(range(5))
		return (self.players_gems[player][colors] >= 2).astype(np.int8)

	def _give_gems(self, i, player):
		if i < len(list_different_gems_up_to_2): # Different gems
			gems = list_different_gems_up_to_2[i][:5]
		else: # 2 identical gems
			color = i - len(list_different_gems_up_to_2)
			gems = np.array([2*int(i==color) for i in range(5)])
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


def observation_size(num_players):
	return (32+12*num_players, 7)

def action_size():
	return 81

def move_to_str(move):
	color_names = ['white', 'blue', 'green', 'red', 'black', 'gold']
	if   move < 12:
	    tier, index = divmod(move, 4)
	    return f'bought from tier {tier} index {index}'
	elif move < 12+15:
	    tier, index = divmod(move-12, 5)
	    if index == 4:
	        return f'reserved from deck of tier {tier}'
	    else:
	        return f'reserved from tier {tier} index {index}'
	elif move < 12+15+3:
	    index = move-12-15
	    return f'bought from reserve {index}'
	elif move < 12+15+3+30:
		i = move - 12-15-3
		if i < len(list_different_gems_up_to_3):
			gems_str = [str(v)+" "+color_names[i] for i, v in enumerate(list_different_gems_up_to_3[i][:5]) if v != 0]
			return f'took {", ".join(gems_str)}'
		else:
			return f'took 2 gems of color {color_names[i-len(list_different_gems_up_to_3)]}'
	elif move < 12+15+3+30+20:
		i = move - 12-15-3-30
		if i < len(list_different_gems_up_to_2):
			gems_str = [str(v)+" "+color_names[i] for i, v in enumerate(list_different_gems_up_to_2[i][:5]) if v != 0]
			return f'gave back {", ".join(gems_str)}'
		else:
			return f'gave back 2 {color_names[i-len(list_different_gems_up_to_2)]}'
	else:
	    return f'empty move'

def move_to_short_str(move):
	color_names = ['white', 'blue', 'green', 'red', 'black', 'gold']
	if   move < 12:
	    tier, index = divmod(move, 4)
	    return f'buy tier{tier}-card{index}'
	elif move < 12+15:
	    tier, index = divmod(move-12, 5)
	    if index == 4:
	        return f'rsv t{tier}-deck'
	    else:
	        return f'rsv t{tier}-c{index}'
	elif move < 12+15+3:
	    index = move-12-15
	    return f'buy {index} in reserve'
	elif move < 12+15+3+30:
		i = move - 12-15-3
		if i < len(list_different_gems_up_to_3):
			gems_str = [ light_colors[i] + "  " + Style.RESET_ALL  for i, v in enumerate(list_different_gems_up_to_3[i][:5]) if v != 0]
			return f'{" ".join(gems_str)}'
		else:
			return f'{light_colors[i-len(list_different_gems_up_to_3)] + "    " + Style.RESET_ALL}'
	elif move < 12+15+3+30+20:
		i = move - 12-15-3-30
		if i < len(list_different_gems_up_to_2):
			gems_str = [ light_colors[i] + "  " + Style.RESET_ALL for i, v in enumerate(list_different_gems_up_to_2[i][:5]) if v != 0]
			return f'give {" ".join(gems_str)}'
		else:
			return f'give {light_colors[i-len(list_different_gems_up_to_2)] + "    " + Style.RESET_ALL}'
	else:
	    return f'empty'

def row_to_str(row, n=2):
	if row < 1:
		return 'bank'
	if row < 25:
		tier, index = divmod(row-1, 4)
		cost_or_value = ((row-1)%2 == 0)
		return f'Card in tier {tier} index {index//2} ' + ('cost' if cost_or_value else 'value')
	if row < 28:
		return f'Nb cards in deck of tier {row-25}'
	if row < 29+n:
		return f'Nobles num {row-28}'
	if row < 29+2*n:
		return f'Nb of gems of player {row-29-n}/{n}'
	if row < 29+5*n:
		player, index = divmod(row-29-2*n, 3)
		return f'Noble {index} earned by player {player}/{n}'
	if row < 29+6*n:
		return f'Cards of player {row-29-5*n}/{n}'
	if row < 29+12*n:
		player, index = divmod(row-29-6*n, 6)
		cost_or_value = (index%2 == 0)
		return f'Reserve {index//2} of player {player}/{n} ' + ('cost' if cost_or_value else 'value')
	return f'unknown row {row}'

def _gen_list_of_different_gems(max_num_gems):
	gems = [ np.array([int(i==c) for i in range(7)], dtype=np.int8) for c in range(5) ]
	results = []
	for n in range(1, max_num_gems+1):
		results += [ sum(comb) for comb in itertools.combinations(gems, n) ]
	return results
list_different_gems_up_to_3 =  _gen_list_of_different_gems(3)
list_different_gems_up_to_2 =  _gen_list_of_different_gems(2)

# cards_symmetries = itertools.permutations(range(4))
cards_symmetries   = [(1, 3, 0, 2), (2, 0, 3, 1), (3, 2, 1, 0)]
reserve_symmetries = [
	[], 					# 0 card in reserve
	[], 					# 1 card
	[(1, 0, 2)],			# 2 cards
	[(1, 2, 0), (2, 0, 1)], # 3 cards
]

##### END OF CLASS #####

idx_white, idx_blue, idx_green, idx_red, idx_black, idx_gold, idx_points = range(7)
light_colors = [
	Back.LIGHTWHITE_EX + Fore.BLACK,	# white
	Back.LIGHTBLUE_EX + Fore.WHITE,		# blue
	Back.LIGHTGREEN_EX + Fore.BLACK,	# green
	Back.LIGHTRED_EX + Fore.BLACK,		# red
	Back.LIGHTBLACK_EX + Fore.WHITE,	# black
	Back.LIGHTYELLOW_EX + Fore.BLACK,	# gold
]
strong_colors = [
	Back.WHITE + Fore.BLACK,	# white
	Back.BLUE + Fore.BLACK,		# blue
	Back.GREEN + Fore.BLACK,	# green
	Back.RED + Fore.BLACK,		# red
	Back.BLACK + Fore.WHITE,	# black
	Back.YELLOW + Fore.BLACK,	# gold
]

#    W Blu G  R  Blk  Point
all_nobles = [
	[0, 0, 4, 4, 0, 0, 3],
	[0, 0, 0, 4, 4, 0, 3],
	[0, 4, 4, 0, 0, 0, 3],
	[4, 0, 0, 0, 4, 0, 3],
	[4, 4, 0, 0, 0, 0, 3],
	[3, 0, 0, 3, 3, 0, 3],
	[3, 3, 3, 0, 0, 0, 3],
	[0, 0, 3, 3, 3, 0, 3],
	[0, 3, 3, 3, 0, 0, 3],
	[3, 3, 0, 0, 3, 0, 3],
]

#             COST                      GAIN
#         W Blu G  R  Blk        W Blu G  R  Blk  Point
all_cards_1 = [
	[
		[[0, 0, 0, 0, 3, 0, 0], [0, 1, 0, 0, 0, 0, 0]],
		[[1, 0, 0, 0, 2, 0, 0], [0, 1, 0, 0, 0, 0, 0]],
		[[0, 0, 2, 0, 2, 0, 0], [0, 1, 0, 0, 0, 0, 0]],
		[[1, 0, 2, 2, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0]],
		[[0, 1, 3, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0]],
		[[1, 0, 1, 1, 1, 0, 0], [0, 1, 0, 0, 0, 0, 0]],
		[[1, 0, 1, 2, 1, 0, 0], [0, 1, 0, 0, 0, 0, 0]],
		[[0, 0, 0, 4, 0, 0, 0], [0, 1, 0, 0, 0, 0, 1]],
	],
	[
		[[3, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]],
		[[0, 2, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]],
		[[2, 0, 0, 2, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]],
		[[2, 0, 1, 0, 2, 0, 0], [0, 0, 0, 1, 0, 0, 0]],
		[[1, 0, 0, 1, 3, 0, 0], [0, 0, 0, 1, 0, 0, 0]],
		[[1, 1, 1, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0]],
		[[2, 1, 1, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0]],
		[[4, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 1]],
	],
	[
		[[0, 0, 3, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0]],
		[[0, 0, 2, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0]],
		[[2, 0, 2, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0]],
		[[2, 2, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0]],
		[[0, 0, 1, 3, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0]],
		[[1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0]],
		[[1, 2, 1, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0]],
		[[0, 4, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 1]],
	],
	[
		[[0, 3, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0]],
		[[0, 0, 0, 2, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0]],
		[[0, 2, 0, 0, 2, 0, 0], [1, 0, 0, 0, 0, 0, 0]],
		[[0, 2, 2, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0]],
		[[3, 1, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0]],
		[[0, 1, 1, 1, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0]],
		[[0, 1, 2, 1, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0]],
		[[0, 0, 4, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 1]],
	],
	[
		[[0, 0, 0, 3, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]],
		[[2, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]],
		[[0, 2, 0, 2, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]],
		[[0, 1, 0, 2, 2, 0, 0], [0, 0, 1, 0, 0, 0, 0]],
		[[1, 3, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]],
		[[1, 1, 0, 1, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0]],
		[[1, 1, 0, 1, 2, 0, 0], [0, 0, 1, 0, 0, 0, 0]],
		[[0, 0, 0, 0, 4, 0, 0], [0, 0, 1, 0, 0, 0, 1]],
	],
]

#             COST                      GAIN
#         W Blu G  R  Blk        W Blu G  R  Blk  Point
all_cards_2 = [
	[
		[[0, 2, 2, 3, 0, 0, 0], [0, 1, 0, 0, 0, 0, 1]],
		[[0, 2, 3, 0, 3, 0, 0], [0, 1, 0, 0, 0, 0, 1]],
		[[0, 5, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 2]],
		[[5, 3, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 2]],
		[[2, 0, 0, 1, 4, 0, 0], [0, 1, 0, 0, 0, 0, 2]],
		[[0, 6, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 3]],
	],
	[
		[[2, 0, 0, 2, 3, 0, 0], [0, 0, 0, 1, 0, 0, 1]],
		[[0, 3, 0, 2, 3, 0, 0], [0, 0, 0, 1, 0, 0, 1]],
		[[0, 0, 0, 0, 5, 0, 0], [0, 0, 0, 1, 0, 0, 2]],
		[[3, 0, 0, 0, 5, 0, 0], [0, 0, 0, 1, 0, 0, 2]],
		[[1, 4, 2, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 2]],
		[[0, 0, 0, 6, 0, 0, 0], [0, 0, 0, 1, 0, 0, 3]],
	],
	[
		[[3, 2, 2, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 1]],
		[[3, 0, 3, 0, 2, 0, 0], [0, 0, 0, 0, 1, 0, 1]],
		[[5, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 2]],
		[[0, 0, 5, 3, 0, 0, 0], [0, 0, 0, 0, 1, 0, 2]],
		[[0, 1, 4, 2, 0, 0, 0], [0, 0, 0, 0, 1, 0, 2]],
		[[0, 0, 0, 0, 6, 0, 0], [0, 0, 0, 0, 1, 0, 3]],
	],
	[
		[[0, 0, 3, 2, 2, 0, 0], [1, 0, 0, 0, 0, 0, 1]],
		[[2, 3, 0, 3, 0, 0, 0], [1, 0, 0, 0, 0, 0, 1]],
		[[0, 0, 0, 5, 0, 0, 0], [1, 0, 0, 0, 0, 0, 2]],
		[[0, 0, 0, 5, 3, 0, 0], [1, 0, 0, 0, 0, 0, 2]],
		[[0, 0, 1, 4, 2, 0, 0], [1, 0, 0, 0, 0, 0, 2]],
		[[6, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 3]],
	],
	[
		[[2, 3, 0, 0, 2, 0, 0], [0, 0, 1, 0, 0, 0, 1]],
		[[3, 0, 2, 3, 0, 0, 0], [0, 0, 1, 0, 0, 0, 1]],
		[[0, 0, 5, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 2]],
		[[0, 5, 3, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 2]],
		[[4, 2, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 2]],
		[[0, 0, 6, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 3]],
	]
]

#             COST                      GAIN
#         W Blu G  R  Blk        W Blu G  R  Blk  Point
all_cards_3 = [
	[
		[[3, 0, 3, 3, 5, 0, 0], [0, 1, 0, 0, 0, 0, 3]],
		[[7, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 4]],
		[[6, 3, 0, 0, 3, 0, 0], [0, 1, 0, 0, 0, 0, 4]],
		[[7, 3, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 5]],
	],
	[
		[[3, 5, 3, 0, 3, 0, 0], [0, 0, 0, 1, 0, 0, 3]],
		[[0, 0, 7, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 4]],
		[[0, 3, 6, 3, 0, 0, 0], [0, 0, 0, 1, 0, 0, 4]],
		[[0, 0, 7, 3, 0, 0, 0], [0, 0, 0, 1, 0, 0, 5]],
	],
	[
		[[3, 3, 5, 3, 0, 0, 0], [0, 0, 0, 0, 1, 0, 3]],
		[[0, 0, 0, 7, 0, 0, 0], [0, 0, 0, 0, 1, 0, 4]],
		[[0, 0, 3, 6, 3, 0, 0], [0, 0, 0, 0, 1, 0, 4]],
		[[0, 0, 0, 7, 3, 0, 0], [0, 0, 0, 0, 1, 0, 5]],
	],
	[
		[[0, 3, 3, 5, 3, 0, 0], [1, 0, 0, 0, 0, 0, 3]],
		[[0, 0, 0, 0, 7, 0, 0], [1, 0, 0, 0, 0, 0, 4]],
		[[3, 0, 0, 3, 6, 0, 0], [1, 0, 0, 0, 0, 0, 4]],
		[[3, 0, 0, 0, 7, 0, 0], [1, 0, 0, 0, 0, 0, 5]],
	],
	[
		[[5, 3, 0, 3, 3, 0, 0], [0, 0, 1, 0, 0, 0, 3]],
		[[0, 7, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 4]],
		[[3, 6, 3, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 4]],
		[[0, 7, 3, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 5]],
	]
]

all_cards = [all_cards_1, all_cards_2, all_cards_3]

def _print_round_and_scores(board):
	print('='*15, f' round {board.bank[0][idx_points]}    ', end='')
	for p in range(board.num_players):
		print(f'{Style.BRIGHT}P{p}{Style.RESET_ALL}: {board.get_score(p)} points  ', end='')
	print('='*15, Style.RESET_ALL)

def _print_nobles(board):
	print(f'{Style.BRIGHT}Nobles: {Style.RESET_ALL}', end='')
	for noble in board.nobles:
		if noble[idx_points] == 0:
			print(f'< {Style.DIM}empty{Style.RESET_ALL} >', end=' ')
		else:
			print(f'< {noble[idx_points]} points ', end='')
			for i, color in enumerate(light_colors):
				if noble[i] != 0:
					print(f'{color} {noble[i]} {Style.RESET_ALL} ', end='')
			print(f'> ', end='')
	print(f'{Style.RESET_ALL}')

def _print_card_line(card, line):
	if card[1,:5].sum() == 0:
		print(f'       ', end=' ')
		return
	card_color = np.flatnonzero(card[1,:5] != 0)[0]
	background = light_colors[card_color]
	print(background, end= '')
	if line == 0:
		print(f'     {Style.BRIGHT}{card[1][idx_points]}{Style.NORMAL} ', end=' ')
	else:
		card_cost = np.flatnonzero(card[0,:5] != 0)
		if line-1 < card_cost.size:
			color = card_cost[line-1]
			value = card[0,color]
			print(f' {light_colors[color]} {value} {background}   ', end=' ')
		else:
			print(f'       ', end=' ')
	print(Style.RESET_ALL, end='   ')

def _print_tiers(board):
	for tier in range(2, -1, -1):
		for line in range(5):
			if line == 3:
				print(f'Tier {tier}: ', end='')
			elif line == 4:
				print(f'  ({board.nb_deck_tiers[2*tier].sum():>2})  ', end='')
			else:
				print(f'        ', end='')
			for i in range(4):
				_print_card_line(board.cards_tiers[8*tier+2*i:8*tier+2*i+2, :], line)
			print()
		print()

def _print_bank(board):
	print(f'{Style.BRIGHT}Bank: {Style.RESET_ALL}', end='')
	for c in range(6):
		print(f'{light_colors[c]} {board.bank[0][c]} {Style.RESET_ALL} ', end='')
	print(f'{Style.RESET_ALL}')

def _print_player(board, p):
	# GEMS + CARTES
	print(f'  {Style.BRIGHT}P{p}: {Style.RESET_ALL}', end='')
	for c in range(6):
		my_gems  = board.players_gems[p][c]
		my_cards = board.players_cards[p][c]
		my_cards = (' ('+str(my_cards)+')') if my_cards!=0 else ''
		print(f'{light_colors[c]} {my_gems}{my_cards} {Style.RESET_ALL} ', end='')
	print(f' total {board.players_gems[p].sum()} gems')

	# NOBLES
	for noble in board.players_nobles[3*p:3*p+3]:
		if noble[idx_points] > 0:
			print(f'< {noble[idx_points]} points ', end='')
			for i, color in enumerate(light_colors):
				if noble[i] != 0:
					print(f'{color} {noble[i]} {Style.RESET_ALL} ', end='')
			print(f'> ', end='')
	if board.players_nobles[3*p:3*p+3, idx_points].sum() > 0:
		print(f'{Style.RESET_ALL}')

	# RESERVED
	for line in range(5):
		print(' '*10, end='')
		for r in range(3):
			reserved = board.players_reserved[6*p+2*r:6*p+2*r+2]
			if reserved[0].sum() != 0:
				_print_card_line(reserved, line)
		if board.players_reserved[6*p:6*p+6, :].sum() > 0:
			print()

def print_board(board):
	_print_round_and_scores(board)
	_print_nobles(board)
	print()
	_print_tiers(board)
	_print_bank(board)
	print()
	for p in range(board.num_players):
		_print_player(board, p)
		print()
