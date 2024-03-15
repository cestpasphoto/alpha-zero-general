import numpy as np
from numba import njit
import numba

############################## BOARD DESCRIPTION ##############################
# Board is described by a 58x2 array (1st dim is larger with 3+ players). 2nd
# dimension stores game history (y=0 is current state, y=1 is previous state)
# The 15 types of cards and 4 types of monuments are defined at the bottom of
# the current file. Here is the description of each line of the board. For
# readibility, we defined "shortcuts" that actually are views(numpy name) of
# overal board.
##### Index  Shortcut              	Meaning
#####   0    self.round  			Round number
#####   1    self.last_dice      	Value of last dice(s) roll (sum if 2 dices)
#####   2    self.player_state		Usually 0. Is 2 if current player plays again (amusement park). Add 1 if player rolls dices again (radio tower)
#####  3-17  self.market			Numbers of remaining cards in main deck for each of 15 card types
##### 18-19  self.players_money		Money for each player
##### 20-49  self.players_cards		Number of cards for each player (P0-card0, P0-card1, ... P1-card0, P1-card1, ...)
##### 50-58  self.players_monuments	Number of monuments for each player
# Indexes above are assuming 2 players, you can have more details in copy_state().
# Limitations of monuments:
#   Train station: 	always roll 2 dices, no question asked
#   Stadium: 		choose the most expensive building from richest player, and
#            		swap it with my cheapest card
# 	TV channel: 	take the $5 from the richest

############################## ACTION DESCRIPTION #############################
# There are 21 actions. Here is description of each action:
##### Index  Meaning
#####   0    Buy a card of type 0 (CHAMPS)
#####   1    Buy a card of type 1 (FERME)
#####  ...
#####  14    Buy a card of type 14 (MARCHE)
#####  15    Buy monument of type 0 (GARE)
#####  ...
#####  18    Buy monument of type 4 (PARC)
#####  19    Roll dice(s) again
#####  20    No move

@njit(cache=True, fastmath=True, nogil=True)
def observation_size(num_players):
	return (18 + 20*num_players, 2) # 2nd dimension is to keep history of previous states

@njit(cache=True, fastmath=True, nogil=True)
def action_size():
	return 21

@njit(cache=True, fastmath=True, nogil=True)
def my_random_choice_and_normalize(prob):
	normalized_prob = prob / prob.sum()
	result = np.searchsorted(np.cumsum(prob), np.random.random(), side="right")
	return result

spec = [
	('num_players'         , numba.int8),
	('current_player_index', numba.int8),

	('state'            , numba.int8[:,:]),
	('round'            , numba.int8[:]),
	('last_dice'        , numba.int8[:]),
	('player_state'     , numba.int8[:]),
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
		return np.multiply(self.players_monuments[4*player:4*(player+1), 0], monuments_cost).sum() # np.dot() not supported by numba

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

	def make_move(self, move, player, random_seed):
		# Actual move
		if   move < 15:
			self._buy_card(player, move)
		elif move < 15+4:
			self._buy_monument(player, move-15)
		elif move == 19:
			self._dice_again(player)
		elif move == 20:
			pass

		# Decide next player and increase number of rounds
		# print(f'P={player}, round={self.round[0]}       ', end='')
		if move == 19: # decide to re-roll dices
			next_player = player
		elif self.player_state[0] >= 2: # player had identical dices values
			self.round[0] += 1
			next_player = player
		else:
			self.round[0] += 1
			next_player = (player+1)%self.num_players
		# print(f'next={next_player}, round={self.round[0]}')

		# Copy history from row 0 to row 1
		if move != 19:
			for data in [self.market, self.players_money, self.players_cards, self.players_monuments]:
				data[:,1] = data[:,0]
			self.round[1] = self.round[0]
			# self.last_dice[1] = self.last_dice[0]
			# self.player_state[1] = self.player_state[0]
			
		# Roll dice for next player
		# print('  ', self.players_money[:,0], end=' ')
		self.last_dice[0], identical_dices = self._roll_dice(next_player)
		self._dice_effect(self.last_dice[0], player_who_rolled=next_player)
		# print('  ', self.players_money[:,0], end=' ')

		# Note down whether player has re-rolled dices or has played a new turn
		self.player_state[0]  = 1 if move == 19      else 0
		self.player_state[0] += 2 if identical_dices else 0

		return next_player

	def copy_state(self, state, copy_or_not):
		if self.state is state and not copy_or_not:
			return
		self.state = state.copy() if copy_or_not else state
		n = self.num_players
		self.round             = self.state[0              ,:]	# 1      # Round number
		self.last_dice         = self.state[1              ,:]	# 1      # Value of last dice(s) roll
		self.player_state      = self.state[2              ,:]	# 1      # Usually 0. Is 2 if current player plays again (amusement park). Add 1 if player rolls dices again (radio tower)
		self.market            = self.state[3      :18     ,:]	# 15     # Numbers of remaining cards in main deck
		self.players_money     = self.state[18     :18+n   ,:]	# n*1    # Numbers of money for each player
		self.players_cards     = self.state[18+n   :18+16*n,:]	# n*15   # Number of cards for each player (P0-card0, P0-card1, ... P1-card0, P1-card1, ...)
		self.players_monuments = self.state[18+16*n:18+20*n,:]	# n*4    # Number of monuments for each player

	def check_end_game(self):
		scores = np.array([self.get_score(p) for p in range(self.num_players)], dtype=np.int8)
		score_max = scores.max()
		if score_max < monuments_cost.sum() and self.get_round() < 126 and np.all(self.players_money[:,0] < 126):
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
		symmetries = [(self.state.copy(), policy.copy(), valid_actions.copy())]
		return symmetries

	def get_round(self):
		return self.round[0]

	def _valid_buy_card(self, player):
		return np.logical_and(self.players_money[player,0] >= cards_cost, self.market[:,0])

	def _valid_buy_monument(self, player):
		return np.logical_and(self.players_money[player,0] >= monuments_cost, self.players_monuments[4*player:4*(player+1),0] == 0)

	def _valid_diceagain(self, player):
		# player must have 'radio' monument and not have played twice
		return self.players_monuments[4*player+3,0] and self.player_state[0]%2 == 0
	
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

	def _roll_dice(self, player_who_rolled):
		dice = np.random.randint(1, 6)
		identical = False
		if self.players_monuments[4*player_who_rolled+0,0] > 0: # Has he got the train station allowing 2 dices?
			dice2 = np.random.randint(1, 6)
			identical = (dice == dice2)
			# print('  Dé P' + str(player_who_rolled) + ' = ' + str(dice) + ' ' + str(dice2) + ('*' if identical else ''), end='')
			dice += dice2
		# else:
		# 	print('  Dé P' + str(player_who_rolled) + ' = ' + str(dice), end='')
		return dice, identical

	def _dice_effect(self, result, player_who_rolled):
		def _all_receive_from_bank(card_index, money):
			for p in range(self.num_players):
				self.players_money[p,0] += money * self.players_cards[15*p+card_index,0]
				# if self.players_cards[15*p+card_index,0]:
				# 	print(f'  P{p} +{money}*{self.players_cards[15*p+card_index,0]} from bank', end='')

		def _current_receive_from_bank(card_index, money, bonus_if_mall=False):
			p = player_who_rolled
			bonus = 1 if bonus_if_mall and (self.players_monuments[4*p + CENTRECOM, 0] > 0) else 0
			self.players_money[p,0] += (money+bonus) * self.players_cards[15*p+card_index,0]
			# if self.players_cards[15*p+card_index,0]:
			# 	print(f'  P{p} +{money+bonus}*{self.players_cards[15*p+card_index,0]} from bank', end='')

		def _current_give(card_index, money, bonus_if_mall=False):
			for player in range(player_who_rolled+self.num_players-1, player_who_rolled, -1):
				p = player % self.num_players
				bonus = 1 if bonus_if_mall and (self.players_monuments[4*p + CENTRECOM, 0] > 0) else 0
				amount = min((money+bonus) * self.players_cards[15*p+card_index,0], self.players_money[player_who_rolled,0])
				self.players_money[p                ,0] += amount
				self.players_money[player_who_rolled,0] -= amount
				# if amount:
				# 	print(f'  P{p} +{amount} from P{player_who_rolled}', end='')

		def _stadium():
			# Every player give 2$ to current
			for p in range(self.num_players):
				if p == player_who_rolled:
					continue
				amount = min(self.players_money[p,0], 2)
				self.players_money[p                ,0] -= amount
				self.players_money[player_who_rolled,0] += amount
				# if amount:
				# 	print(f'  P{player_who_rolled} +{amount} from P{p}', end='')

		def _business_center():
			# Current can swap a building with someone else
			# Let's buy the most expensive one from the richest player
			# Against one of my low cost card
			wealths = np.array([self.get_wealth(p) for p in range(self.num_players)], dtype=np.int8)
			wealths[player_who_rolled] = 0 # Avoid swapping with yourself
			target_player = my_random_choice_and_normalize(wealths == wealths.max())
			target_player_cards_cost = np.multiply(np.minimum(self.players_cards[15*target_player:15*(target_player+1), 0], 1), cards_cost)
			target_player_cards_cost[STADE], target_player_cards_cost[AFFAIRES], target_player_cards_cost[CHAINE] = 0, 0, 0 # Forbid to swap these cards
			target_building = my_random_choice_and_normalize(target_player_cards_cost == target_player_cards_cost.max())
			# Choose a very bad card to swap with
			my_cards_cost = np.multiply(np.minimum(self.players_cards[15*player_who_rolled:15*(player_who_rolled+1), 0], 1), cards_cost)
			for i in range(my_cards_cost.size):
				if my_cards_cost[i] == 0:
					my_cards_cost[i] = 99
			my_building = my_random_choice_and_normalize(my_cards_cost == my_cards_cost.min())
			# Do the swap now
			self.players_cards[15*target_player    +target_building, 0] -= 1
			self.players_cards[15*player_who_rolled+target_building, 0] += 1
			self.players_cards[15*player_who_rolled+my_building, 0]     -= 1
			self.players_cards[15*target_player    +my_building, 0]     += 1
			# print(f'  P{player_who_rolled} swaps B{my_building} with B{target_building}-P{target_player}', end='')

		def _tv_channel():
			# Take 5$ from any player
			# Let's choose someone who has 5$, the richest one if hesitating
			moneys = self.players_money[:,0].copy()
			moneys[player_who_rolled] = 0
			money_max = min(moneys.max(), 5)
			who_has_more_money = np.logical_or(moneys == money_max, moneys >= 5)
			wealths = np.array([self.get_wealth(p) if who_has_more_money[p] else 0 for p in range(self.num_players)], dtype=np.int8)
			target_player = my_random_choice_and_normalize(wealths == wealths.max())
			# Now, take from him
			amount = min(self.players_money[target_player, 0], 5)
			self.players_money[target_player    ,0] -= amount
			self.players_money[player_who_rolled,0] += amount
			# if amount:
			# 	print(f'  P{player_who_rolled} +{amount} from P{target_player}', end='')

		if result == 1:
			_all_receive_from_bank(CHAMPS, 1)
		elif result == 2:
			_all_receive_from_bank(FERME, 1)
			_current_receive_from_bank(BOULANGERIE, 1, bonus_if_mall=True)
		elif result == 3:
			_current_give(CAFE, 1, bonus_if_mall=True) # give first
			_current_receive_from_bank(BOULANGERIE, 1, bonus_if_mall=True)
		elif result == 4:
			_current_receive_from_bank(SUPERETTE, 3, bonus_if_mall=True)
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
			_current_give(RESTAURANT, 2, bonus_if_mall=True) # give first
			_all_receive_from_bank(MINE, 5)
		elif result == 10:
			_current_give(RESTAURANT, 2, bonus_if_mall=True) # give first
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