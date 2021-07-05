#!/home/best/dev/splendor/venv/bin/python3

import numpy as np
from numba import njit
import numba

@njit(cache=True, fastmath=True, nogil=True)
def observation_size(num_players):
	return (18 + 20*num_players, 2) # 2nd dimension is to keep history of previous states

@njit(cache=True, fastmath=True, nogil=True)
def action_size():
	return 21

@njit(cache=True, fastmath=True, nogil=True)
def my_random_choice(prob):
	result = np.searchsorted(np.cumsum(prob), np.random.random(), side="right")
	return result

spec = [
	('num_players'         , numba.int8),
	('current_player_index', numba.int8),

	('state'            , numba.int8[:,:]),
	('round'            , numba.int8[:,:]),
	('market'           , numba.int8[:,:]),
	('players_money'    , numba.int8[:,:]),
	('players_cards'    , numba.int8[:,:]),
	('players_monuments', numba.int8[:,:]),
]
@numba.experimental.jitclass(spec)
class Board():
	def __init__(self, num_players):
		self.num_players = num_players
		self.current_player_index = 0
		self.state = np.zeros(observation_size(self.num_players), dtype=np.int8)
		self.init_game()

	def get_score(self, player):
		return np.dot(self.players_monuments[4*player:4*(player+1), 0], monuments_cost)

	def get_wealth(self, player):
		return self.get_score(player) + self.players_money[player, 0]

	def init_game(self):
		self.copy_state(np.zeros(observation_size(self.num_players), dtype=np.int8), copy_or_not=False)

		# self.round[:] = 0
		self.market[:,:] = 6
		self.market[6:9,:] = 4 # Special case with purple cards
		self.players_money[:,:] = 3
		for p in range(self.num_players):
			self.players_cards[15*p + 0,:] = 1
			self.players_cards[15*p + 1,:] = 1
		# self.players_monuments[:,:] = 0
		
	def get_state(self):
		return self.state

	def valid_moves(self, player):
		result = np.zeros(21, dtype=np.bool_)
		result[0   :15]     = self._valid_buy_card(player)
		result[15  :15+4]   = self._valid_buy_monument(player)
		result[15+4:15+4+1] = self._valid_diceagain(player)
		result[20] = True #empty move
		return result

	def make_move(self, move, player, deterministic):
		if self.get_round() >= 126:
			raise Exception('number of rounds too high ' + str(self.get_round()))
		if np.any(self.players_money[:,0] >= 126):
			raise Exception('money is too high ' + str(self.players_money[:,0]))

		# Copy history from row 0 to row 1
		if move != 19:
			for data in [self.market, self.players_money, self.players_cards, self.players_monuments]:
				data[:,1] = data[:,0]
			self.round[1] = self.round[0]
			self.last_dice[1] = self.last_dice[0]
			self.last_played_twice[1] = self.last_played_twice[0]

		# Actual move
		if   move < 15:
			self._buy_card(player, move)
		elif move < 15+4:
			self._buy_monument(player, move-15)
		elif move == 19:
			self._dice_again(player)

		# Increase number of rounds
		if move != 19:
			self.round[0] += 1
			next_player, self.last_played_twice[0] = (player+1)%self.num_players, 0
		else:
			next_player, self.last_played_twice[0] = player                     , 1
		# Roll dice for next player
		self.last_dice[0] = self._roll_dice(next_player)
		self._dice_effect(self.last_dice[0], player_who_rolled=next_player)
		print(f'  Dé P{next_player} = {self.last_dice[0]}')

		return next_player

	def copy_state(self, state, copy_or_not):
		if self.state is state and not copy_or_not:
			return
		self.state = state.copy() if copy_or_not else state
		n = self.num_players
		self.round             = self.state[0              ,:]	# 1
		self.last_dice         = self.state[1              ,:]	# 1
		self.last_played_twice = self.state[2              ,:]	# 1
		self.market            = self.state[3      :18     ,:]	# 15
		self.players_money     = self.state[18     :18+n   ,:]	# n*1
		self.players_cards     = self.state[18+n   :18+16*n,:]	# n*15
		self.players_monuments = self.state[18+16*n:18+20*n,:]	# n*4

	def check_end_game(self):
		scores = np.array([self.get_score(p) for p in range(self.num_players)], dtype=np.int8)
		score_max = scores.max()
		if score_max < monuments_cost.sum():
			return np.full(self.num_players, 0., dtype=np.float32)
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
		_roll_in_place_axis0(self.players_money    , 1 *nb_swaps)
		_roll_in_place_axis0(self.players_cards    , 15*nb_swaps)
		_roll_in_place_axis0(self.players_monuments, 4 *nb_swaps)

	def get_symmetries(self, policy, valid_actions):
		# def _swap_cards(cards, permutation):
		# 	full_permutation = [2*p+i for p in permutation for i in range(2)]
		# 	cards_copy = cards.copy()
		# 	for i in range(len(permutation)*2):
		# 		cards[i, :] = cards_copy[full_permutation[i], :]
		# def _copy_and_permute(array, permutation, start_index):
		# 	new_array = array.copy()
		# 	for i, p in enumerate(permutation):
		# 		new_array[start_index+i] = array[start_index+p]
		# 	return new_array
		# def _copy_and_permute2(array, permutation, start_index, other_start_index):
		# 	new_array = array.copy()
		# 	for i, p in enumerate(permutation):
		# 		new_array[start_index      +i] = array[start_index      +p]
		# 		new_array[other_start_index+i] = array[other_start_index+p]
		# 	return new_array

		pass

	def get_round(self):
		return self.round[0]

	def _valid_buy_card(self, player):
		return np.logical_and(self.players_money[player,0] >= cards_cost, self.market[:,0])

	def _valid_buy_monument(self, player):
		return np.logical_and(self.players_money[player,0] >= monuments_cost, self.players_monuments[4*player:4*(player+1),0] == 0)

	def _valid_diceagain(self, player):
		# player must have 'radio' monument and not have played twice
		return self.players_monuments[4*player+3,0] and self.last_played_twice[0]==0
	
	def _buy_card(self, player, card):
		self.players_money[player,0] -= cards_cost[card]
		self.market[card,0] -= 1
		self.players_cards[15*player + card,0] += 1

	def _buy_monument(self, player, monument):
		self.players_money[player,0] -= monuments_cost[monument]
		self.players_monuments[4*player + monument,0] += 1

	def _dice_again(self, player):
		# Copy history to current row
		for data in [self.market, self.players_money, self.players_cards, self.players_monuments]:
			data[:,0] = data[:,1]
		self.round[0] = self.round[1]

	def _roll_dice(self, player):
		dice = np.random.randint(1, 6)
		if self.players_monuments[4*player+0,0] > 0: # Has he got the train station allowing 2 dices?
			dice += np.random.randint(1, 6)
		return dice

	def _dice_effect(self, result, player_who_rolled):
		def _all_receive_from_bank(card_index, money):
			for p in range(self.num_players):
				self.players_money[p,0] += money * self.players_cards[15*p+card_index,0]

		def _current_receive_from_bank(card_index, money):
			p = player_who_rolled
			self.players_money[p,0] += money * self.players_cards[15*p+card_index,0]

		def _current_give(card_index, money):
			for player in range(player_who_rolled+self.num_players-1, player_who_rolled, -1):
				p = player % self.num_players
				amount = min(money * self.players_cards[15*p+card_index,0], self.players_money[player_who_rolled,0])
				self.players_money[p                ,0] += amount
				self.players_money[player_who_rolled,0] -= amount

		def _stadium():
			# Every player give 2$ to current
			for p in range(self.num_players):
				if p == player_who_rolled:
					continue
				amount = min(self.players_money[p,0], 2)
				self.players_money[p                ,0] -= amount
				self.players_money[player_who_rolled,0] += amount

		def _business_center():
			# Current can swap a building with someone else
			# Let's buy the most expensive one from the richest player
			# Against one of my low probability card
			wealths = np.array([self.get_wealth(p) for p in range(self.num_players)], dtype=np.int8)
			target_player = my_random_choice(wealths == wealths.max())
			target_player_cards_cost = np.multiply(np.minimum(self.players_cards[15*target_player:15*(target_player+1), 0], 1), cards_cost)
			target_player_cards_cost[STADE], target_player_cards_cost[AFFAIRES], target_player_cards_cost[CHAINE] = 0, 0, 0 # Forbid to swap these cards
			target_building = my_random_choice(target_player_cards_cost == target_player_cards_cost.max())
			# Choose a very bad card to swap with
			my_cards_cost = np.multiply(np.minimum(self.players_cards[15*player_who_rolled:15*(player_who_rolled+1), 0], 1), cards_cost)
			for i in range(my_cards_cost.size):
				if my_cards_cost[i] == 0:
					my_cards_cost[i] = 99
			my_building = my_random_choice(my_cards_cost == my_cards_cost.min())
			# Do the swap now
			self.players_cards[15*target_player    +target_building, 0] -= 1
			self.players_cards[15*player_who_rolled+target_building, 0] += 1
			self.players_cards[15*player_who_rolled+my_building, 0]     -= 1
			self.players_cards[15*target_player    +my_building, 0]     += 1

		def _tv_channel():
			# Take 5$ from any player
			# Let's choose someone who has 5$, the richest one if hesitating
			moneys = self.players_money[:,0].copy()
			moneys[player_who_rolled] = 0
			money_max = min(moneys.max(), 5)
			who_has_more_money = np.logical_or(moneys == money_max, moneys >= 5)
			wealths = np.array([self.get_wealth(p) if who_has_more_money[p] else 0 for p in range(self.num_players)], dtype=np.int8)
			target_player = my_random_choice(wealths == wealths.max())
			# Now, take from him
			amount = min(self.players_money[target_player, 0], 5)
			self.players_money[target_player    ,0] -= amount
			self.players_money[player_who_rolled,0] += amount

		if result == 1:
			_all_receive_from_bank(CHAMPS, 1)
		elif result == 2:
			_all_receive_from_bank(FERME, 1)
			_current_receive_from_bank(BOULANGERIE, 1)
		elif result == 3:
			_current_give(CAFE, 1) # give first
			_current_receive_from_bank(BOULANGERIE, 1)
		elif result == 4:
			_current_receive_from_bank(SUPERETTE, 3)
		elif result == 5:
			_all_receive_from_bank(FORET, 1)
		elif result == 6:
			if self.players_cards[15*player_who_rolled+STADE, 0] > 0:
				_stadium()
			if self.players_cards[15*player_who_rolled+AFFAIRES, 0] > 0:
				_business_center()
			if self.players_cards[15*player_who_rolled+CHAINE, 0] > 0:
				_tv_channel()
		elif result == 7:
			_current_receive_from_bank(FROMAGERIE, 3 * self._get_current_cow())
		elif result == 8:
			_current_receive_from_bank(MEUBLES, 3 * self._get_current_gear())
		elif result == 9:
			_current_give(RESTAURANT, 2) # give first
			_all_receive_from_bank(MINE, 5)
		elif result == 10:
			_current_give(RESTAURANT, 2) # give first
			_all_receive_from_bank(VERGER, 3)
		elif result == 11:
			_current_receive_from_bank(MARCHE, 2 * self._get_current_wheat())
		elif result == 12:
			_current_receive_from_bank(MARCHE, 2 * self._get_current_wheat())


	def _get_current_cow(self):
		return self.players_cards[15*self.current_player_index + FERME, 0]

	def _get_current_gear(self):
		return self.players_cards[15*self.current_player_index + FORET, 0] + self.players_cards[15*self.current_player_index + MINE, 0]

	def _get_current_wheat(self):
		return self.players_cards[15*self.current_player_index + CHAMPS, 0] + self.players_cards[15*self.current_player_index + VERGER, 0]


# Index of cards
CHAMPS      = 0 
FERME       = 1
BOULANGERIE = 2
CAFE        = 3
SUPERETTE   = 4
FORET       = 5
STADE       = 6
AFFAIRES    = 7
CHAINE      = 8
FROMAGERIE  = 9
MEUBLES     = 10
MINE        = 11
RESTAURANT  = 12
VERGER      = 13
MARCHE      = 14

# Index of monuments
GARE      = 0
CENTRECOM = 1
RADIO     = 2
PARC      = 3

# Cost of cards
cards_cost = np.array([1, 1, 1, 2, 2, 3, 6, 8, 7, 5, 3, 6, 3, 3, 2], dtype=np.int8)
monuments_cost = np.array([4, 10, 16, 22], dtype=np.int8)