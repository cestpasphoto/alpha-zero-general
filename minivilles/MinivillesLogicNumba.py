import numpy as np
from numba import njit
import numba
from Stochastic import hashed_draw

############################## BOARD DESCRIPTION ##############################
# Board is described by a 58x2 array (1st dim is larger with 3+ players). 2nd
# dimension stores game history (y=0 is current state, y=1 is previous state)
# The 15 types of cards and 4 types of monuments are defined at the bottom of
# the current file. Here is the description of each line of the board. For
# readibility, we defined "shortcuts" that actually are views(numpy name) of
# overal board.
##### Index  Shortcut              	Meaning
#####   0    self.round  			Round number
#####   1    self.last_dice      	[0] value of last roll (sum if 2 dice, 0 while awaiting the dice choice), [1] number of dice used
#####   2    self.player_state		Bit field: +1 dice were rerolled this turn (radio tower), +2 last roll was a double
#####       	                 	(extra turn only with amusement park), +4 awaiting the 1-or-2 dice choice (train station)
#####  3-17  self.market			Numbers of remaining cards in main deck for each of 15 card types
##### 18-19  self.players_money		Money for each player
##### 20-49  self.players_cards		Number of cards for each player (P0-card0, P0-card1, ... P1-card0, P1-card1, ...)
##### 50-57  self.players_monuments	Number of monuments for each player
# Indexes above are assuming 2 players, you can have more details in copy_state().
# Limitations of monuments:
#   Train station: 	owner chooses to roll 1 or 2 dice (actions 21/22) at the start of each
#   	        	turn; non-owners roll 1 die automatically. A reroll (radio tower) rerolls
#   	        	all dice with the same number of dice as the first roll.
#   Business center: always swap my cheapest non-major establishment with the
#                    most expensive one of the richest opponent (no choice)
#   TV channel:      take $5 from the richest opponent among those having $5
#   Ties are broken deterministically in seat order after the roller (no RNG).

############################## ACTION DESCRIPTION #############################
# There are 23 actions. Here is description of each action:
##### Index  Meaning
#####   0    Buy a card of type 0 (CHAMPS)
#####   1    Buy a card of type 1 (FERME)
#####  ...
#####  14    Buy a card of type 14 (MARCHE)
#####  15    Buy monument of type 0 (GARE)
#####  ...
#####  18    Buy monument of type 3 (RADIO)
#####  19    Roll dice(s) again
#####  20    No move
#####  21    Roll 1 die  (only while awaiting the dice choice)
#####  22    Roll 2 dice (only while awaiting the dice choice)

@njit(cache=True, fastmath=True, nogil=True)
def observation_size(num_players):
	return (18 + 20*num_players, 2) # 2nd dimension is to keep history of previous states

@njit(cache=True, fastmath=True, nogil=True)
def action_size():
	return 23

@njit(cache=True, fastmath=True, nogil=True)
def first_true_after(mask, start):
	# Deterministic, frame-invariant tie-break: first True index in seat order
	# strictly after `start` (wrapping around). Consumes no RNG, so it respects
	# the Stochastic.py contract (outcome fixed by state + action + seed).
	n = mask.size
	for k in range(1, n + 1):
		i = (start + k) % n
		if mask[i]:
			return i
	return -1

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
		total_wealth = self.get_score(player) + self.players_money[player, 0]
		total_wealth = 127 if total_wealth > 127 else total_wealth
		return total_wealth

	def init_game(self):
		self.copy_state(np.zeros(observation_size(self.num_players), dtype=np.int8), copy_or_not=False)

		# self.round[:] = 0
		self.market[:,:] = 6
		self.market[6:9,:] = 4 # Special case with purple cards
		self.players_money[:,:] = 3
		for p in range(self.num_players):
			self.players_cards[15*p + CHAMPS     ,:] = 1
			self.players_cards[15*p + BOULANGERIE,:] = 1 # official rules: wheat field + bakery

		# self.players_monuments[:,:] = 0

		# Very first turn of player 0 (no train station yet, so 1 die is rolled automatically)
		self._start_turn(0, 0, 0)
		
	def get_state(self):
		return self.state

	def valid_moves(self, player):
		result = np.zeros(23, dtype=np.bool_)
		if self.player_state[0] & 4: # must first choose how many dice to roll
			result[21] = True
			result[22] = True
			return result
		result[0   :15]     = self._valid_buy_card(player)
		result[15  :15+4]   = self._valid_buy_monument(player)
		result[15+4:15+4+1] = self._valid_diceagain(player)
		result[20] = True #empty move
		return result

	def make_move(self, move, player, random_seed):
		# Dice choice (train station owner): roll, apply effects, same player then buys
		if move >= 21:
			self._roll_and_apply(player, random_seed, self.player_state[0] & 1, move == 22)
			return player

		# Actual move
		if   move < 15:
			self._buy_card(player, move)
		elif move < 15+4:
			self._buy_monument(player, move-15)
		elif move == 19:
			self._dice_again(player)
		elif move == 20:
			pass

		if move == 19: # reroll ALL dice, same number as the first roll (no new choice, no die kept)
			self._roll_and_apply(player, random_seed, 1, self.last_dice[1] == 2)
			return player
		else:
			if (self.player_state[0] & 2) and self.players_monuments[4*player+PARC, 0] > 0: # doubles + amusement park
				next_player = player
			else:
				next_player = (player+1)%self.num_players
			self.round[0] += 1
			# Copy history from row 0 to row 1 (row 1 = state before the next roll)
			for data in [self.market, self.players_money, self.players_cards, self.players_monuments]:
				data[:,1] = data[:,0]
			self.round[1] = self.round[0]

		self._start_turn(next_player, random_seed, 0)
		return next_player

	def _start_turn(self, player, random_seed, rerolled):
		# Owner of the train station chooses the number of dice; others roll 1 die now
		if self.players_monuments[4*player+GARE, 0] > 0:
			self.last_dice[0], self.last_dice[1] = 0, 0
			self.player_state[0] = 4 + rerolled
		else:
			self._roll_and_apply(player, random_seed, rerolled, False)

	def _roll_and_apply(self, player, random_seed, rerolled, two_dice):
		self.last_dice[0], identical_dices = self._roll_dice(random_seed, rerolled, two_dice)
		self.last_dice[1] = 2 if two_dice else 1
		self._dice_effect(self.last_dice[0], player_who_rolled=player)
		self.player_state[0] = rerolled + (2 if identical_dices else 0)

	def copy_state(self, state, copy_or_not):
		if self.state is state and not copy_or_not:
			return
		self.state = state.copy() if copy_or_not else state
		n = self.num_players
		self.round             = self.state[0              ,:]	# 1      # Round number
		self.last_dice         = self.state[1              ,:]	# 1      # Value of last dice(s) roll
		self.player_state      = self.state[2              ,:]	# 1      # Bit field: 1 = rerolled, 2 = double, 4 = awaiting dice choice
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
		# Base condition: enough money and at least 1 card in the market
		valid = np.logical_and(self.players_money[player,0] >= cards_cost, self.market[:,0] > 0)
		
		# Purple cards constraint: only 1 of each type per player
		if self.players_cards[15*player + STADE, 0] > 0:
			valid[STADE] = False
		if self.players_cards[15*player + AFFAIRES, 0] > 0:
			valid[AFFAIRES] = False
		if self.players_cards[15*player + CHAINE, 0] > 0:
			valid[CHAINE] = False
			
		return valid

	def _valid_buy_monument(self, player):
		return np.logical_and(self.players_money[player,0] >= monuments_cost, self.players_monuments[4*player:4*(player+1),0] == 0)

	def _valid_diceagain(self, player):
		# player must have 'radio tower' monument and not have rerolled yet this turn
		return self.players_monuments[4*player+RADIO,0] > 0 and (self.player_state[0] & 1) == 0
	
	def _buy_card(self, player, card):
		self._add_money(player, -cards_cost[card])
		self.market[card,0] -= 1
		self.players_cards[15*player + card,0] += 1

	def _buy_monument(self, player, monument):
		self._add_money(player, -monuments_cost[monument])
		self.players_monuments[4*player + monument,0] += 1

	def _dice_again(self, player):
		# Copy history to current row
		for data in [self.market, self.players_money, self.players_cards, self.players_monuments]:
			data[:,0] = data[:,1]
		self.round[0] = self.round[1]

	def _roll_dice(self, random_seed, reroll, two_dice):
		# Counters are unique per (round, reroll, die index); hashed_draw(.., m) is in [0, m)
		ctr = (np.int64(np.uint8(self.round[0])) * 2 + (1 if reroll else 0)) * 2
		dice = np.random.randint(1, 7) if random_seed == 0 else 1 + hashed_draw(random_seed, ctr, 6)
		identical = False
		if two_dice:
			dice2 = np.random.randint(1, 7) if random_seed == 0 else 1 + hashed_draw(random_seed, ctr + 1, 6)
			identical = (dice == dice2)
			dice += dice2
		return dice, identical

	def _dice_effect(self, result, player_who_rolled):
		def _all_receive_from_bank(card_index, money):
			for p in range(self.num_players):
				self._add_money(p, money * self.players_cards[15*p+card_index,0])
				# if self.players_cards[15*p+card_index,0]:
				# 	print(f'  P{p} +{money}*{self.players_cards[15*p+card_index,0]} from bank', end='')

		def _current_receive_from_bank(card_index, money, bonus_if_mall=False):
			p = player_who_rolled
			bonus = 1 if bonus_if_mall and (self.players_monuments[4*p + CENTRECOM, 0] > 0) else 0
			self._add_money(p, (money+bonus) * self.players_cards[15*p+card_index,0])
			# if self.players_cards[15*p+card_index,0]:
			# 	print(f'  P{p} +{money+bonus}*{self.players_cards[15*p+card_index,0]} from bank', end='')

		def _current_give(card_index, money, bonus_if_mall=False):
			for player in range(player_who_rolled+self.num_players-1, player_who_rolled, -1):
				p = player % self.num_players
				bonus = 1 if bonus_if_mall and (self.players_monuments[4*p + CENTRECOM, 0] > 0) else 0
				amount = min((money+bonus) * self.players_cards[15*p+card_index,0], self.players_money[player_who_rolled,0])
				self._add_money(p                , -amount)
				self._add_money(player_who_rolled,  amount)
				# if amount:
				# 	print(f'  P{p} +{amount} from P{player_who_rolled}', end='')

		def _stadium():
			# Every player give 2$ to current
			for p in range(self.num_players):
				if p == player_who_rolled:
					continue
				amount = min(self.players_money[p,0], 2)
				self._add_money(p                , -amount)
				self._add_money(player_who_rolled,  amount)
				# if amount:
				# 	print(f'  P{player_who_rolled} +{amount} from P{p}', end='')

		def _business_center():
			# Current can swap a building with someone else
			# Let's buy the most expensive one from the richest player
			# Against one of my low cost card
			wealths = np.array([self.get_wealth(p) for p in range(self.num_players)], dtype=np.int8)
			wealths[player_who_rolled] = -1 # Never target yourself
			target_player = first_true_after(wealths == wealths.max(), player_who_rolled)
			target_player_cards_cost = np.multiply(np.minimum(self.players_cards[15*target_player:15*(target_player+1), 0], 1), cards_cost)
			target_player_cards_cost[STADE], target_player_cards_cost[AFFAIRES], target_player_cards_cost[CHAINE] = 0, 0, 0 # Forbid to swap these cards
			if target_player_cards_cost.max() == 0:
				return # target owns no swappable establishment
			target_building = np.argmax(target_player_cards_cost) # card index: frame-invariant
			# Choose my cheapest non-major establishment to give away
			my_cards_cost = np.multiply(np.minimum(self.players_cards[15*player_who_rolled:15*(player_who_rolled+1), 0], 1), cards_cost)
			my_cards_cost[STADE], my_cards_cost[AFFAIRES], my_cards_cost[CHAINE] = 0, 0, 0
			for i in range(my_cards_cost.size):
				if my_cards_cost[i] == 0:
					my_cards_cost[i] = 99
			if my_cards_cost.min() == 99:
				return # I own no swappable establishment
			my_building = np.argmin(my_cards_cost)
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
			who_has_more_money[player_who_rolled] = False
			wealths = np.array([self.get_wealth(p) if who_has_more_money[p] else -1 for p in range(self.num_players)], dtype=np.int8)
			target_player = first_true_after(wealths == wealths.max(), player_who_rolled)
			if target_player < 0 or target_player == player_who_rolled:
				return
			# Now, take from him
			amount = min(self.players_money[target_player, 0], 5)
			self._add_money(target_player    , -amount)
			self._add_money(player_who_rolled,  amount)
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
			_current_receive_from_bank(FROMAGERIE, 3 * self._get_current_cow(player_who_rolled))
		elif result == 8:
			_current_receive_from_bank(MEUBLES, 3 * self._get_current_gear(player_who_rolled))
		elif result == 9:
			_current_give(RESTAURANT, 2, bonus_if_mall=True) # give first
			_all_receive_from_bank(MINE, 5)
		elif result == 10:
			_current_give(RESTAURANT, 2, bonus_if_mall=True) # give first
			_all_receive_from_bank(VERGER, 3)
		elif result == 11:
			_current_receive_from_bank(MARCHE, 2 * self._get_current_wheat(player_who_rolled))
		elif result == 12:
			_current_receive_from_bank(MARCHE, 2 * self._get_current_wheat(player_who_rolled))

	def _add_money(self, player, money_to_add):
		new_money = self.players_money[player, 0] + np.int16(money_to_add)
		if new_money > 127:
			new_money = 127
		if new_money < 0:
			new_money = 0
		self.players_money[player, 0] = new_money

	def _get_current_cow(self, player_who_rolled):
		return self.players_cards[15*player_who_rolled + FERME, 0]

	def _get_current_gear(self, player_who_rolled):
		return self.players_cards[15*player_who_rolled + FORET, 0] + self.players_cards[15*player_who_rolled + MINE, 0]

	def _get_current_wheat(self, player_who_rolled):
		return self.players_cards[15*player_who_rolled + CHAMPS, 0] + self.players_cards[15*player_who_rolled + VERGER, 0]


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
PARC      = 2 # amusement park (16): extra turn on doubles
RADIO     = 3 # radio tower (22): may reroll once per turn

# Cost of cards
cards_cost = np.array([1, 1, 1, 2, 2, 3, 6, 8, 7, 5, 3, 6, 3, 3, 2], dtype=np.int8)
monuments_cost = np.array([4, 10, 16, 22], dtype=np.int8)